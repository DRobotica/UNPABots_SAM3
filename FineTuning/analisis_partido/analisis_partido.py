# ============================================================
# analisis_partido.py
# SAM3 base (ultralytics) + DINOv2 + tracking de robots
# USO: python analisis_partido/analisis_partido.py "video.mov"
# ============================================================

import os, sys, csv, time
import cv2, torch, numpy as np
from PIL import Image
from sklearn.cluster import DBSCAN
import supervision as sv
from ultralytics.models.sam import SAM3SemanticPredictor
from transformers import AutoImageProcessor, AutoModel

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

MAX_ROBOTS, MAX_MISSED, THRESHOLD_DIST = 4, 8, 70  # configuracion tracking

# Cargar SAM3 base (requiere sam3.pt en raiz del proyecto)
sam_pt = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "sam3.pt"))
sam_predictor = SAM3SemanticPredictor(overrides=dict(conf=0.25, task="segment", mode="predict", model=sam_pt))

# Cargar DINOv2 para embeddings visuales de robots
dino_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
dino_model = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval()

# Anotadores de supervision
mask_annotator = sv.MaskAnnotator(opacity=0.6)
label_annotator = sv.LabelAnnotator()


def get_dino_embedding(image_bgr):
    """Extrae vector de 768 dimensiones (huella digital visual) de un robot"""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(image_rgb)
    inputs = dino_processor(images=pil, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = dino_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten()


def similitud_robots(d1, d2):
    """Similitud coseno entre dos embeddings, normalizada a 0-100"""
    d1, d2 = np.asarray(d1, np.float32), np.asarray(d2, np.float32)
    n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
    if n1 == 0 or n2 == 0: return 0.0
    return (np.dot(d1, d2) / (n1 * n2) + 1.0) * 50.0


def obtener_vertices_cancha(mask):
    """Encuentra poligono de cancha desde mascara de lineas (DBSCAN + convexHull)"""
    h, w = mask.shape; scale = 0.25  # reducir al 25% para ahorrar memoria
    small = cv2.resize(mask.astype(np.uint8) * 255, (int(w * scale), int(h * scale)))
    closed = cv2.morphologyEx(small, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    points = cv2.findNonZero(closed)
    if points is None: return None
    pts = (points.reshape(-1, 2) / scale).astype(np.float32)
    if len(pts) > 5000: pts = pts[np.random.choice(len(pts), 5000, replace=False)]  # limite memoria
    clustering = DBSCAN(eps=25, min_samples=20).fit(pts)
    labels = clustering.labels_
    valid = labels[labels != -1]
    if len(valid) == 0: return None
    # Tomar el cluster mas grande como cancha
    best = np.unique(valid, return_counts=True)[0][np.argmax(np.unique(valid, return_counts=True)[1])]
    filtered = pts[labels == best]
    hull = cv2.convexHull(filtered.astype(np.int32))
    approx = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)
    return approx.reshape(-1, 2)


def agrupar_por_similitud(mat):
    """Agrupa robots visualmente similares en el mismo equipo"""
    n = mat.shape[0]; visitado = np.zeros(n, bool); equipos = []
    for i in range(n):
        if visitado[i]: continue
        eq = [i]; visitado[i] = True
        for j in range(n):
            if i != j and not visitado[j] and mat[i, j] == np.max(mat[i]):
                eq.append(j); visitado[j] = True
        equipos.append(eq)
    return equipos


def clasificar_robots(robots):
    """Frame 0: clasifica robots en 2 equipos por similitud de embeddings"""
    if len(robots) == 0: return np.array([])
    robots = sorted(robots, key=lambda r: r[0])  # ordenar por x
    X = np.array([d for _, _, d in robots])
    n = len(X); mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j: mat[i, j] = similitud_robots(X[i], X[j])
    equipos = agrupar_por_similitud(mat)
    etiquetas = np.zeros(n, int)
    for eid, g in enumerate(equipos):
        for r in g: etiquetas[r] = eid
    return etiquetas


# Variables globales de tracking
robots_info, next_robot_id = {}, 0
datos_csv, datos_vertices = [], []
frame_actual = 0
mascara_cancha_global, vertices_cancha_global = None, None


def detectar_objetos(frame):
    """SAM3: 1 llamada con 3 prompts: white lines, robot, ball"""
    sam_predictor.set_image(frame)
    return sv.Detections.from_ultralytics(sam_predictor(text=["white lines", "robot", "ball"])[0])


def procesar_cancha(frame, sam_det):
    """Extrae mascara y vertices de cancha desde lineas blancas"""
    global mascara_cancha_global, vertices_cancha_global, frame_actual
    h, w = frame.shape[:2]
    mask_total = np.zeros((h, w), np.uint8)
    for i in range(len(sam_det)):
        if sam_det.class_id[i] == 0:  # white lines
            mask_total[sam_det.mask[i]] = 255
    mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    contours, _ = cv2.findContours(mask_total, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mascara_cancha_global = np.zeros_like(mask_total)
    if contours:
        cv2.drawContours(mascara_cancha_global, [max(contours, key=cv2.contourArea)], -1, 255, cv2.FILLED)
        vertices_cancha_global = obtener_vertices_cancha(mask_total)
        if vertices_cancha_global is not None:
            for j, (x, y) in enumerate(vertices_cancha_global):
                datos_vertices.append([frame_actual, j, int(x), int(y)])


def extraer_robots(frame, sam_det):
    """Extrae centroides y embeddings DINOv2 de cada robot"""
    global frame_actual
    robots_frame, centroides = [], []
    for i in range(len(sam_det)):
        if sam_det.class_id[i] == 0: continue           # ignorar lineas
        if sam_det.class_id[i] == 1:                    # robot
            mask = sam_det.mask[i]; ys, xs = np.where(mask)
            if len(xs) == 0: continue
            cx, cy = np.mean(xs), np.mean(ys)           # centroide
            # Filtrar fuera de cancha
            if vertices_cancha_global is not None and \
               cv2.pointPolygonTest(vertices_cancha_global.astype(np.float32), (float(cx), float(cy)), False) < 0:
                continue
            x1, y1, x2, y2 = sam_det.xyxy[i].astype(int)
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0: continue
            robots_frame.append((cx, cy, get_dino_embedding(roi)))
            centroides.append((cx, cy))
        else:                                           # ball
            mask = sam_det.mask[i]; ysb, xsb = np.where(mask)
            if len(xsb) > 0: datos_csv.append([frame_actual, "ball", -1, np.mean(xsb), np.mean(ysb)])
    return robots_frame, centroides


def asignar_ids(robots_frame):
    """Tracking: asigna IDs a robots comparando distancia + embedding"""
    global next_robot_id, robots_info
    asignados = set()
    for rid in robots_info:
        if robots_info[rid]["activo"]: robots_info[rid]["missing_frames"] += 1

    for cx, cy, desc in robots_frame:
        mejor_id, mejor_score = None, 1e9
        for rid, info in robots_info.items():
            if not info["activo"] or rid in asignados: continue
            dist = np.linalg.norm(np.array([cx, cy]) - np.array(info["centro"]))
            sim = similitud_robots(desc, info["embedding"]) / 100.0
            score = 0.7 * dist + 30.0 * (1.0 - sim)  # 70% distancia, 30% embedding
            if score < mejor_score and dist < THRESHOLD_DIST:
                mejor_score, mejor_id = score, rid

        if mejor_id is not None:
            # Actualizar robot existente
            robots_info[mejor_id].update({"centro": (cx, cy), "embedding": desc, "missing_frames": 0, "activo": True})
            asignados.add(mejor_id)
        elif next_robot_id < MAX_ROBOTS:
            # Nuevo robot
            robots_info[next_robot_id] = {"centro": (cx, cy), "embedding": desc, "equipo": -1, "activo": True, "missing_frames": 0}
            asignados.add(next_robot_id); next_robot_id += 1


def dibujar(frame, sam_det):
    """Dibuja mascaras, IDs de robot, cancha, y guarda CSV"""
    global frame_actual
    annotated = mask_annotator.annotate(scene=frame.copy(), detections=sam_det)
    annotated = label_annotator.annotate(scene=annotated, detections=sam_det)
    if mascara_cancha_global is not None: annotated[mascara_cancha_global > 0] = (0, 255, 255)  # cyan cancha
    if vertices_cancha_global is not None: cv2.polylines(annotated, [vertices_cancha_global.astype(np.int32)], True, (255, 0, 255), 3)  # magenta poligono
    for rid, info in robots_info.items():
        if not info["activo"]: continue
        cx, cy = info["centro"]; eq = info["equipo"]
        color = (0, 0, 255) if eq == 0 else (0, 255, 255)  # rojo equipo 0, amarillo equipo 1
        cv2.circle(annotated, (int(cx), int(cy)), 8, color, -1)
        cv2.putText(annotated, f"R{rid}", (int(cx) + 10, int(cy) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        datos_csv.append([frame_actual, f"robot_{rid}", eq, cx, cy])
    return annotated


def procesar_frame(frame, _):
    """Callback principal: llamado por supervision.process_video en cada frame"""
    global frame_actual, robots_info, next_robot_id
    sam_det = detectar_objetos(frame)                     # 1. SAM3
    procesar_cancha(frame, sam_det)                        # 2. Cancha
    robots_frame, _ = extraer_robots(frame, sam_det)      # 3. Robots + DINOv2
    if frame_actual == 0:
        etiquetas = clasificar_robots(robots_frame)       # 4a. Clasificar equipos (frame 0)
        for i, (cx, cy, desc) in enumerate(robots_frame):
            robots_info[len(robots_info)] = {"centro": (cx, cy), "embedding": desc, "equipo": etiquetas[i] if i < len(etiquetas) else 0, "activo": True, "missing_frames": 0}
        next_robot_id = len(robots_info)
    else:
        asignar_ids(robots_frame)                          # 4b. Tracking
    annotated = dibujar(frame, sam_det)                    # 5. Dibujar
    for rid in list(robots_info.keys()):
        if robots_info[rid]["missing_frames"] > MAX_MISSED: robots_info[rid]["activo"] = False  # marcar inactivo
    print(f"Frame {frame_actual}: {len(robots_frame)} robots")
    frame_actual += 1
    return annotated


def main():
    video_path = sys.argv[1] if len(sys.argv) > 1 else "IMG_9796.MOV"
    if not os.path.exists(video_path): print(f"No encontrado: {video_path}"); return
    out_v = f"analisis_{os.path.basename(video_path)}"
    sv.process_video(source_path=video_path, target_path=out_v, callback=procesar_frame)
    base = os.path.splitext(os.path.basename(video_path))[0]
    # Guardar CSVs
    with open(f"centroides_{base}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["frame", "robot_id", "equipo", "cx", "cy"]); w.writerows(datos_csv)
    with open(f"vertices_{base}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["frame", "vertice", "x", "y"]); w.writerows(datos_vertices)
    print(f"Completado: {out_v}")


if __name__ == "__main__":
    main()
