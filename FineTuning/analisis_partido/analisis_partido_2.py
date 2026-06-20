# ============================================================
# analisis_partido_2.py
# SAM3 fine-tuning (HuggingFace) + DINOv2 + tracking + NMS
# USO: python analisis_partido/analisis_partido_2.py "video.mov"
# ============================================================

import os, sys, csv, time
import cv2, torch, numpy as np
from PIL import Image
from sklearn.cluster import DBSCAN
import supervision as sv
from transformers import AutoImageProcessor, AutoModel, Sam3Processor, Sam3Model

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Tus 5 clases entrenadas (mismo orden que en train_sam3.py)
MIS_CLASES = ["limites", "pelota", "porteria amarilla", "porteria azul", "robot"]
CLASS_COLORS = {
    "limites": (0, 255, 255), "pelota": (0, 255, 0),
    "porteria amarilla": (0, 255, 255), "porteria azul": (255, 0, 0), "robot": (0, 0, 255),
}
MAX_ROBOTS, MAX_MISSED, THRESHOLD_DIST = 4, 10, 80

# Cargar el modelo fine-tuning (HuggingFace)
_model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "soccer_sam3_final"))
sam_processor = Sam3Processor.from_pretrained(_model_dir)
sam_model = Sam3Model.from_pretrained(_model_dir).to(device).eval()

# DINOv2 para embeddings de robots
dino_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
dino_model = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval()


def nms(instances, iou_threshold=0.5):
    """Non-Maximum Suppression: elimina mascaras duplicadas del mismo objeto"""
    if len(instances) <= 1: return instances
    instances = sorted(instances, key=lambda x: x["area"], reverse=True)  # mas grandes primero
    keep = []
    for i, inst in enumerate(instances):
        suppressed = False
        for j in keep:
            inter = np.logical_and(inst["mask"], instances[j]["mask"]).sum()
            union = np.logical_or(inst["mask"], instances[j]["mask"]).sum()
            if union > 0 and inter / union > iou_threshold:  # solapan >50%
                suppressed = True; break
        if not suppressed: keep.append(i)
    return [instances[i] for i in keep]


def segmentar_clase(image_rgb, class_name):
    """SAM3 con prompt de texto: devuelve instancias individuales (con NMS)"""
    inputs = sam_processor(images=image_rgb, text=class_name, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = sam_model(**inputs)
    pred_masks = outputs.pred_masks
    if pred_masks.dim() == 5: pred_masks = pred_masks.squeeze(1)
    h, w = image_rgb.shape[:2]
    pred_resized = torch.nn.functional.interpolate(pred_masks.float(), size=(h, w), mode="bilinear", align_corners=False)
    probs = torch.sigmoid(pred_resized).squeeze(0).cpu().numpy()  # [200, H, W]

    instances = []
    for i in range(probs.shape[0]):
        mask_i = probs[i] > 0.5
        area = mask_i.sum()
        if area < 30: continue          # filtrar ruido (mascaras muy pequenas)
        ys, xs = np.where(mask_i)
        if len(xs) == 0: continue
        instances.append({
            "mask": mask_i,
            "bbox": (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
            "centroid": (float(xs.mean()), float(ys.mean())),
            "area": int(area),
        })
    return nms(instances)  # eliminar duplicados


def segmentar_clase_combined(image_rgb, class_name):
    """Version combinada: fusiona todas las instancias en una sola mascara"""
    instances = segmentar_clase(image_rgb, class_name)
    if not instances: return np.zeros(image_rgb.shape[:2], dtype=bool)
    combined = np.zeros(image_rgb.shape[:2], dtype=bool)
    for inst in instances: combined |= inst["mask"]
    return combined


def get_dino_embedding(image_bgr):
    """Huella digital visual de 768 dimensiones para un robot"""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    inputs = dino_processor(images=Image.fromarray(image_rgb), return_tensors="pt").to(device)
    with torch.no_grad(): outputs = dino_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten()


def similitud_robots(d1, d2):
    """Similitud coseno 0-100 entre dos embeddings"""
    d1, d2 = np.asarray(d1, np.float32), np.asarray(d2, np.float32)
    n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
    if n1 == 0 or n2 == 0: return 0.0
    return (np.dot(d1, d2) / (n1 * n2) + 1.0) * 50.0


def obtener_vertices_cancha(mask):
    """Poligono de cancha: DBSCAN sobre mascara reducida + convexHull"""
    h, w = mask.shape; scale = 0.25
    small = cv2.resize(mask.astype(np.uint8) * 255, (int(w * scale), int(h * scale)))
    closed = cv2.morphologyEx(small, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    points = cv2.findNonZero(closed)
    if points is None: return None
    pts = (points.reshape(-1, 2) / scale).astype(np.float32)
    if len(pts) > 5000: pts = pts[np.random.choice(len(pts), 5000, replace=False)]
    clustering = DBSCAN(eps=25, min_samples=20).fit(pts)
    labels = clustering.labels_
    valid = labels[labels != -1]
    if len(valid) == 0: return None
    best = np.unique(valid, return_counts=True)[0][np.argmax(np.unique(valid, return_counts=True)[1])]
    filtered = pts[labels == best]
    hull = cv2.convexHull(filtered.astype(np.int32))
    approx = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)
    return approx.reshape(-1, 2)


def agrupar_por_similitud(mat):
    """Agrupa robots en equipos por similitud de embeddings"""
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
    """Frame 0: separa robots en 2 equipos por DINOv2"""
    if len(robots) < 2: return np.zeros(len(robots), int)
    robots = sorted(robots, key=lambda r: r[0])
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
vertices_cancha_global, mascara_cancha_global = None, None
last_field_frame = -10  # actualizar cancha cada 10 frames


def procesar_frame(frame_bgr, _):
    """Callback por frame: SAM3 + DINOv2 + tracking + dibujo"""
    global frame_actual, robots_info, next_robot_id
    global vertices_cancha_global, mascara_cancha_global, last_field_frame
    h, w = frame_bgr.shape[:2]
    image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    overlay = frame_bgr.copy()

    # === 1. Segmentar las 5 clases con el modelo ===
    t0 = time.time()
    mask_limites = segmentar_clase_combined(image_rgb, "limites")
    robot_instances = segmentar_clase(image_rgb, "robot")          # instancias individuales con NMS
    mask_pelota = segmentar_clase_combined(image_rgb, "pelota")
    mask_pa = segmentar_clase_combined(image_rgb, "porteria amarilla")
    mask_pz = segmentar_clase_combined(image_rgb, "porteria azul")
    t_sam = (time.time() - t0) * 1000

    # === 2. Cancha desde "limites" (cada 10 frames para rendimiento) ===
    if frame_actual - last_field_frame >= 10:
        mask_clean = mask_limites.copy()
        mask_clean = cv2.morphologyEx(mask_clean.astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
        contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mascara_cancha_global = np.zeros((h, w), np.uint8)
        if contours:
            cv2.drawContours(mascara_cancha_global, [max(contours, key=cv2.contourArea)], -1, 255, cv2.FILLED)
            vertices_cancha_global = obtener_vertices_cancha(mask_limites)
        last_field_frame = frame_actual

    # === 3. Robots: extraer centroide + DINOv2 de cada instancia ===
    robots_frame = []
    mask_robot_combined = np.zeros((h, w), bool)
    for inst in robot_instances:
        cx, cy = inst["centroid"]; x1, y1, x2, y2 = inst["bbox"]
        mask_robot_combined |= inst["mask"]
        # Filtrar fuera de cancha
        if vertices_cancha_global is not None and \
           cv2.pointPolygonTest(vertices_cancha_global.astype(np.float32), (float(cx), float(cy)), False) < 0:
            continue
        roi = frame_bgr[y1:y2, x1:x2]
        if roi.size == 0: continue
        robots_frame.append((cx, cy, get_dino_embedding(roi)))

    # === 4. Tracking: asignar IDs por distancia + embedding ===
    asignados = set()
    for rid in robots_info:
        if robots_info[rid]["activo"]: robots_info[rid]["missing_frames"] += 1
    for cx, cy, desc in robots_frame:
        mejor_id, mejor_score = None, 1e9
        for rid, info in robots_info.items():
            if not info["activo"] or rid in asignados: continue
            dist = np.linalg.norm(np.array([cx, cy]) - np.array(info["centro"]))
            sim = similitud_robots(desc, info["embedding"]) / 100.0
            score = 0.7 * dist + 30.0 * (1.0 - sim)
            if score < mejor_score and dist < THRESHOLD_DIST: mejor_score, mejor_id = score, rid
        if mejor_id is not None:
            robots_info[mejor_id].update({"centro": (cx, cy), "embedding": desc, "missing_frames": 0, "activo": True})
            asignados.add(mejor_id)
        elif next_robot_id < MAX_ROBOTS:
            robots_info[next_robot_id] = {"centro": (cx, cy), "embedding": desc, "equipo": -1, "activo": True, "missing_frames": 0}
            asignados.add(next_robot_id); next_robot_id += 1

    # === 5. Clasificar equipos en frame 0 ===
    if frame_actual == 0 and len(robots_frame) >= 2:
        etiquetas = clasificar_robots(robots_frame)
        for i, rid in enumerate(sorted(robots_info.keys())):
            if i < len(etiquetas): robots_info[rid]["equipo"] = int(etiquetas[i])

    # === 6. Balon: guardar posicion ===
    ball_contours, _ = cv2.findContours(mask_pelota.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in ball_contours:
        if cv2.contourArea(cnt) > 5:
            M = cv2.moments(cnt)
            if M["m00"] > 0: datos_csv.append([frame_actual, "ball", -1, M["m10"]/M["m00"], M["m01"]/M["m00"]])

    # === 7. Dibujar visualizacion ===
    if mascara_cancha_global is not None: overlay[mascara_cancha_global > 0] = (0, 255, 255)  # cancha cyan
    if vertices_cancha_global is not None:
        cv2.polylines(overlay, [vertices_cancha_global.astype(np.int32)], True, (255, 0, 255), 2)  # poligono magenta
        for j, (x, y) in enumerate(vertices_cancha_global): datos_vertices.append([frame_actual, j, int(x), int(y)])

    # Mascaras coloreadas de cada clase
    overlay[mask_robot_combined] = (0, 0, 255)      # rojo
    overlay[mask_pelota] = (0, 255, 0)               # verde
    overlay[mask_pa] = (0, 255, 255)                 # amarillo
    overlay[mask_pz] = (255, 0, 0)                   # azul
    overlay[mask_limites] = (255, 255, 0)            # cyan

    # Etiquetas de clase en el centro de cada mascara
    for mask, nombre, color in [(mask_limites, "limites", (255, 255, 0)), (mask_pelota, "pelota", (0, 255, 0)),
                                  (mask_pa, "porteria amarilla", (0, 255, 255)), (mask_pz, "porteria azul", (255, 0, 0))]:
        if mask.any():
            ys, xs = np.where(mask); cx, cy = int(xs.mean()), int(ys.mean())
            cv2.putText(overlay, nombre, (cx - 40, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Cajas individuales de cada robot
    for inst in robot_instances:
        x1, y1, x2, y2 = inst["bbox"]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 1)

    # IDs de robot con color por equipo
    for rid, info in robots_info.items():
        if not info["activo"]: continue
        cx, cy = info["centro"]; eq = info.get("equipo", -1)
        color = (0, 0, 255) if eq == 0 else (255, 0, 0) if eq == 1 else (255, 255, 255)
        cv2.circle(overlay, (int(cx), int(cy)), 8, color, -1)
        cv2.putText(overlay, f"R{rid}", (int(cx) + 10, int(cy) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        datos_csv.append([frame_actual, f"robot_{rid}", info.get("equipo", -1), cx, cy])

    # Marcar inactivos tras MAX_MISSED frames sin ver
    for rid in list(robots_info.keys()):
        if robots_info[rid]["missing_frames"] > MAX_MISSED: robots_info[rid]["activo"] = False

    result = cv2.addWeighted(overlay, 0.5, frame_bgr, 0.5, 0)  # 50% opacidad
    print(f"Frame {frame_actual}: {len(robots_frame)} robots, SAM3: {t_sam:.0f}ms")
    frame_actual += 1
    return result


def main():
    video_path = sys.argv[1] if len(sys.argv) > 1 else "IMG_9796.MOV"
    if not os.path.exists(video_path): print(f"No encontrado: {video_path}"); return
    out_v = f"analisis_2_{os.path.basename(video_path)}"
    sv.process_video(source_path=video_path, target_path=out_v, callback=procesar_frame)
    base = os.path.splitext(os.path.basename(video_path))[0]
    # Guardar CSVs
    with open(f"centroides_2_{base}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["frame", "robot_id", "equipo", "cx", "cy"]); w.writerows(datos_csv)
    with open(f"vertices_2_{base}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["frame", "vertice", "x", "y"]); w.writerows(datos_vertices)
    print(f"Completado: {out_v}")


if __name__ == "__main__":
    main()
