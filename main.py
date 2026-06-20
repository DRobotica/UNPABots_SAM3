import cv2
import numpy as np
import torch
import csv
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from ultralytics import YOLO, SAM
from helpers import similitud_robots, agrupar_robots_por_similitud

# =====================================================================
# CONFIGURACIÓN GLOBAL
# =====================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "facebook/dinov2-base"
processor = AutoImageProcessor.from_pretrained(model_name)
dino_model = AutoModel.from_pretrained(model_name).to(device)
dino_model.eval()
# =====================================================================

def get_dino_embedding(image_bgr):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image_rgb)
    inputs = processor(images=pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = dino_model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1)
    return embedding.cpu().numpy().flatten()

def clasificar_robots_inicial(robots_detectados):
    if len(robots_detectados) == 0:
        return np.array([])
    robots_detectados = sorted(robots_detectados, key=lambda r: r[0])
    X = np.array([d for _, _, d, _ in robots_detectados])
    n = len(X)
    matriz_similitud = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                matriz_similitud[i, j] = similitud_robots(X[i], X[j])
    matriz_similitud = (matriz_similitud + matriz_similitud.T) / 2
    equipos = agrupar_robots_por_similitud(matriz_similitud)
    etiquetas = np.zeros(n, dtype=int)
    for id_equipo, group in enumerate(equipos):
        for idx in group:
            etiquetas[idx] = id_equipo
    return etiquetas

def main():
    print(f"Corriendo localmente en dispositivo: {device.upper()}")

    yolo_model = YOLO("bestv8s.pt").to(device)
    sam_model = SAM("sam3.pt").to(device)
    
    class_names = ['cancha', 'limites', 'pelota', 'porteria', 'robot']
    THRESHOLD_DIST = 70
    MAX_MISSED = 8
    
    robots_info = {}
    next_robot_id = 0
    
    # --- MEDIDAS REALES CONFIGURADAS A 2 PX/CM ---
    ancho_cm = 158
    largo_cm = 219
    escala = 2
    
    ancho_cenital = largo_cm * escala  # 438 píxeles
    alto_cenital = ancho_cm * escala   # 316 píxeles

    color_palette = {
        "robot": (255, 0, 165),      
        "pelota": (0, 255, 255),     
        "porteria": (255, 120, 0),   
        "limites": (0, 0, 255)       
    }

    video_path = "video1.mp4"
    cap = cv2.VideoCapture(video_path)
    w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out_persp = cv2.VideoWriter("resultado_perspectiva.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (w_orig, h_orig))
    out_cenital = cv2.VideoWriter("resultado_cenital.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (ancho_cenital, alto_cenital))

    f_csv = open("telemetria_avanzada.csv", mode='w', newline='', encoding='utf-8')
    csv_writer = csv.writer(f_csv)
    csv_writer.writerow(['frame', 'id_objeto', 'equipo', 'pixel_x', 'pixel_y', 'cenital_x', 'cenital_y'])

    # --- CONTROL DE CALIBRACIÓN ÚNICA ---
    H = None 
    puntos_origen_fijos = None
    frame_actual = 0

    print("Iniciando fase de calibración automática inicial...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        yolo_results = yolo_model(frame, verbose=False)[0]
        bboxes = yolo_results.boxes.xyxy.cpu().numpy()
        class_ids = yolo_results.boxes.cls.cpu().numpy()

        mask_cancha = np.zeros((h_orig, w_orig), dtype=np.uint8)
        mask_porterias = np.zeros((h_orig, w_orig), dtype=np.uint8)
        robots_frame = [] 
        balon_pos = None

        if len(bboxes) > 0:
            sam_results = sam_model(frame, bboxes=bboxes, verbose=False)[0]
            if sam_results.masks is not None:
                masks = sam_results.masks.data.cpu().numpy()

                for i, mask in enumerate(masks):
                    cls_name = class_names[int(class_ids[i])]
                    binary_mask = (mask > 0.5).astype(np.uint8)

                    # --- SEGMENTACIÓN VISUAL EN PERSPECTIVA ---
                    if cls_name in color_palette:
                        color = color_palette[cls_name]
                        frame[binary_mask > 0] = (frame[binary_mask > 0] * 0.7 + np.array(color, dtype=np.uint8) * 0.3).astype(np.uint8)

                    if cls_name == 'cancha':
                        mask_cancha = cv2.bitwise_or(mask_cancha, binary_mask)
                    elif cls_name == 'porteria':
                        mask_porterias = cv2.bitwise_or(mask_porterias, binary_mask)
                        M = cv2.moments(binary_mask)
                        if M["m00"] != 0:
                            px, py = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                            cv2.putText(frame, "Porteria", (px - 20, py), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    elif cls_name == 'robot':
                        M = cv2.moments(binary_mask)
                        cx = int(M["m10"]/M["m00"]) if M["m00"] != 0 else int((bboxes[i][0]+bboxes[i][2])/2)
                        cy = int(M["m01"]/M["m00"]) if M["m00"] != 0 else int((bboxes[i][1]+bboxes[i][3])/2)
                        
                        x1, y1, x2, y2 = bboxes[i].astype(int)
                        roi = frame[max(0,y1):min(h_orig,y2), max(0,x1):min(w_orig,x2)]
                        if roi.size > 0:
                            emb = get_dino_embedding(roi)
                            robots_frame.append((cx, cy, emb, bboxes[i]))
                    elif cls_name == 'pelota':
                        M = cv2.moments(binary_mask)
                        bx = int(M["m10"]/M["m00"]) if M["m00"] != 0 else int((bboxes[i][0]+bboxes[i][2])/2)
                        by = int(M["m01"]/M["m00"]) if M["m00"] != 0 else int((bboxes[i][1]+bboxes[i][3])/2)
                        balon_pos = (bx, by)
                        cv2.putText(frame, "Pelota", (bx + 10, by), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # --- ETAPA DE CALIBRACIÓN: EJECUTADA UNA SOLA VEZ AL PRINCIPIO ---
        # --- ETAPA DE CALIBRACIÓN: CORREGIDA CONTRA EFECTO ESPEJO ---
        if H is None:
            contornos, _ = cv2.findContours(mask_cancha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contornos) > 0:
                cancha_principal = max(contornos, key=cv2.contourArea)
                if cv2.contourArea(cancha_principal) > 10000:
                    hull = cv2.convexHull(cancha_principal).reshape(-1, 2)
                    
                    # 1. Extraer extremos geométricos de la masa verde
                    suma = hull.sum(axis=1)
                    diff = np.diff(hull, axis=1).flatten()
                    
                    pt_ul = hull[np.argmin(suma)]  # Top-Left original
                    pt_br = hull[np.argmax(suma)]  # Bottom-Right original
                    pt_ur = hull[np.argmin(diff)]  # Top-Right original
                    pt_bl = hull[np.argmax(diff)]  # Bottom-Left original
                    
                    # 2. Forzar el ordenamiento correcto en sentido horario (Clockwise)
                    # Evita que la homografía cruce o invierta los ejes cartesianos
                    puntos_origen_fijos = np.array([pt_ul, pt_ur, pt_br, pt_bl], dtype=np.float32)
                    
                    # Destino por defecto: Cancha acostada (Largo en X, Ancho en Y)
                    puntos_destino = np.array([
                        [0, 0], 
                        [ancho_cenital, 0], 
                        [ancho_cenital, alto_cenital], 
                        [0, alto_cenital]
                    ], dtype=np.float32)
                    
                    # Medir distancias para evaluar si la cámara ve la cancha parada o acostada
                    dist_horizontal = np.linalg.norm(pt_ul - pt_ur)
                    dist_vertical = np.linalg.norm(pt_ul - pt_bl)
                    
                    if dist_horizontal < dist_vertical:
                        # Si está parada, reordenamos el destino manteniendo la coherencia horaria
                        # para evitar el efecto espejo (Top-Left pasa a mapearse con la orientación real)
                        puntos_destino = np.array([
                            [0, alto_cenital], 
                            [0, 0], 
                            [ancho_cenital, 0], 
                            [ancho_cenital, alto_cenital]
                        ], dtype=np.float32)

                    H = cv2.findHomography(puntos_origen_fijos, puntos_destino, cv2.RANSAC, 5.0)[0]
                    print(f"¡Calibración exitosa y corregida contra espejeado en el frame {frame_actual}!")

        # Dibujar de forma rígida los puntos de calibración fijados al inicio
        if puntos_origen_fijos is not None:
            for pt in puntos_origen_fijos:
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 8, (0, 0, 255), -1)

        # Generar lienzo cenital usando la matriz fija inmutable
        vista_cenital = np.zeros((alto_cenital, ancho_cenital, 3), dtype=np.uint8)
        if H is not None:
            vista_cenital = cv2.warpPerspective(frame, H, (ancho_cenital, alto_cenital))

        # --- SECCIÓN DE TRACKING HISTÓRICO DINOv2 ---
        for r_id in robots_info:
            if robots_info[r_id]["activo"]:
                robots_info[r_id]["missing_frames"] += 1

        if frame_actual == 0 and len(robots_frame) > 0:
            etiquetas = clasificar_robots_inicial(robots_frame)
            for idx, (cx, cy, emb, _) in enumerate(robots_frame):
                if next_robot_id < 4: 
                    robots_info[next_robot_id] = {
                        "centro": (cx, cy), "embedding": emb, "equipo": etiquetas[idx],
                        "activo": True, "missing_frames": 0
                    }
                    next_robot_id += 1
        else:
            for cx, cy, emb, _ in robots_frame:
                mejor_id = None
                mejor_score = 1e9
                for r_id, info in robots_info.items():
                    if not info["activo"]:
                        continue
                    dist = np.linalg.norm(np.array([cx, cy]) - np.array(info["centro"]))
                    sim_emb = similitud_robots(emb, info["embedding"]) / 100.0
                    score = (0.7 * dist) + (30.0 * (1.0 - sim_emb))

                    if score < mejor_score and dist < THRESHOLD_DIST:
                        mejor_score = score
                        mejor_id = r_id

                if mejor_id is not None:
                    robots_info[mejor_id]["centro"] = (cx, cy)
                    robots_info[mejor_id]["embedding"] = emb
                    robots_info[mejor_id]["missing_frames"] = 0
                    robots_info[mejor_id]["activo"] = True
                else:
                    mejor_id_desaparecido = None
                    max_sim = -1
                    for r_id, info in robots_info.items():
                        if not info["activo"]: 
                            sim = similitud_robots(emb, info["embedding"])
                            if sim > max_sim and sim > 75.0: 
                                max_sim = sim
                                mejor_id_desaparecido = r_id
                    
                    if mejor_id_desaparecido is not None:
                        robots_info[mejor_id_desaparecido]["centro"] = (cx, cy)
                        robots_info[mejor_id_desaparecido]["embedding"] = emb
                        robots_info[mejor_id_desaparecido]["missing_frames"] = 0
                        robots_info[mejor_id_desaparecido]["activo"] = True

        for r_id in list(robots_info.keys()):
            if robots_info[r_id]["missing_frames"] > MAX_MISSED:
                robots_info[r_id]["activo"] = False

        # --- REGISTRO Y MAPEO CENITAL ---
        if balon_pos is not None:
            bx, by = balon_pos
            bcx, bcy = "", ""
            if H is not None:
                pt_p = cv2.perspectiveTransform(np.array([[[bx, by]]], dtype=np.float32), H)[0][0]
                bcx, bcy = int(pt_p[0]), int(pt_p[1])
                if 0 <= bcx < ancho_cenital and 0 <= bcy < alto_cenital:
                    cv2.circle(vista_cenital, (bcx, bcy), 6, (0, 255, 0), -1)
            csv_writer.writerow([frame_actual, "ball", -1, bx, by, bcx, bcy])

        for r_id, info in robots_info.items():
            if not info["activo"]:
                continue
            cx, cy = info["centro"]
            eq = info["equipo"]
            color = (0, 0, 255) if eq == 1 else (255, 0, 0) 

            cv2.circle(frame, (int(cx), int(cy)), 8, color, -1)
            cv2.putText(frame, f"R{r_id}", (int(cx)+10, int(cy)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            rcx, rcy = "", ""
            if H is not None:
                pt_p = cv2.perspectiveTransform(np.array([[[cx, cy]]], dtype=np.float32), H)[0][0]
                rcx, rcy = int(pt_p[0]), int(pt_p[1])
                if 0 <= rcx < ancho_cenital and 0 <= rcy < alto_cenital:
                    cv2.circle(vista_cenital, (rcx, rcy), 7, color, -1)
                    cv2.circle(vista_cenital, (rcx, rcy), 8, (255, 255, 255), 1)
                    cv2.putText(vista_cenital, f"R{r_id}", (rcx+8, rcy-8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

            csv_writer.writerow([frame_actual, f"robot_{r_id}", eq, int(cx), int(cy), rcx, rcy])

        out_persp.write(frame)
        out_cenital.write(vista_cenital)
        frame_actual += 1

    cap.release()
    out_persp.release()
    out_cenital.release()
    f_csv.close()
    print("¡Procesamiento finalizado! La homografía se mantuvo perfectamente rígida durante todo el video.")

if __name__ == "__main__":
    main()