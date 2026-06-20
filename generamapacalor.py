import supervision as sv 
from ultralytics import YOLO 
import cv2 
import numpy as np 
import urllib.request 
from pathlib import Path 
import matplotlib.pyplot as plt # 1. Importa Matplotlib
import matplotlib.patches as mpatches
import pandas as pd
import matplotlib as mpl
from matplotlib.animation import FuncAnimation

cap = cv2.VideoCapture("video2.mp4") 
ret, primer_frame = cap.read() 
cap.release() 

# 2. Convierte el color porque OpenCV lee en BGR y Matplotlib usa RGB
primer_frame_rgb = cv2.cvtColor(primer_frame, cv2.COLOR_BGR2RGB)


# Esquinas del campo detectadas con OpenCV (orden: TL, TR, BR, BL)
SOURCE_POINTS = np.float32([
      [38, 201],   # 1 — esquina superior-izquierda
    [548, 147],   # 2 — esquina superior-derecha
    [657, 1060],   # 3 — esquina inferior-derecha
    [170, 1141],   # 4 — esquina inferior-izquierda
])
# Campo canónico: 364 × 486 px  →  RCJ Soccer Field 2023 (182 × 243 cm) a 2 px/cm
# Arriba = portería amarilla | Abajo = portería azul
CAMPO_W, CAMPO_H = 316, 438
PORAMA_LX, PORAMA_LY = 78, 0
PORAMA_RX, PORAMA_RY = 238, 0
PORAZUL_LX, POAZUL_RX = 78, 438
PORAZUL_RX, PORTAZUL_RY = 238,438

ESCALA_PX_CM = 2.0  # 1 cm real = 2 px en el campo canónico
TARGET_POINTS = np.float32([
    [       0,        0],   # 1 → TL canónico
    [CAMPO_W ,         0],   # 2 → TR canónico
    [CAMPO_W , CAMPO_H],   # 3 → BR canónico
    [       0, CAMPO_H ],   # 4 → BL canónico
])
# Verificación visual: pinta los 4 puntos numerados sobre la imagen
PUNTO_COLORS = [(255, 220, 0), (0, 180, 255), (255, 60, 60), (60, 200, 60)]
vis = primer_frame_rgb.copy()
for idx, (pt, color) in enumerate(zip(SOURCE_POINTS, PUNTO_COLORS)):
    x, y = int(pt[0]), int(pt[1])
    cv2.circle(vis, (x, y), 22, color, -1)
    cv2.putText(vis, str(idx + 1), (x - 8, y + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)



# Calcular la homografía a partir de los 4 pares de puntos
H = cv2.getPerspectiveTransform(SOURCE_POINTS, TARGET_POINTS)

# H_vis: misma homografía pero con MARGEN de contexto fuera del campo
# Los robots cerca del borde no se cortan en la visualización
MARGEN   = 20                            # px de buffer alrededor de los corners
CANVAS_W = CAMPO_W + 2 * MARGEN         # 404 px
CANVAS_H = CAMPO_H + 2 * MARGEN         # 526 px
H_vis    = cv2.getPerspectiveTransform(SOURCE_POINTS, TARGET_POINTS + MARGEN)


# Aplicar la homografía a toda la imagen
warped = cv2.warpPerspective(primer_frame_rgb, H_vis, (CANVAS_W, CANVAS_H))
warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)

campo_w = 316          # Reemplaza con tu CAMPO_W real
campo_h = 468          # Reemplaza con tu CAMPO_H real
ESCALA_PX_CM = 2        # Reemplaza con tu ESCALA_PX_CM real

# ==========================================
# 2. CARGA DEL ARCHIVO CSV Y ADAPTACIÓN (OPCIÓN B)
# ==========================================
csv_path = "telemetria_avanzada.csv" 
df = pd.read_csv(csv_path)

# 1. Limpiamos espacios en blanco invisibles
df.columns = df.columns.str.strip()

# 2. Renombramos para que tus multiplicaciones con H sigan funcionando igual
df = df.rename(columns={
    'id_objeto': 'robot_id',
    'pixel_x': 'cx',
    'pixel_y': 'cy'
})


# Limpiamos espacios en blanco invisibles en los nombres de las columnas
df.columns = df.columns.str.strip()

# Filtramos para ignorar el balón
robots_clean = df[df["robot_id"] != "ball"].copy()

# ==========================================
# 3. PROYECCIÓN HOMOGRÁFICA DE TODO EL CSV
# ==========================================
puntos_camara = np.stack([robots_clean['cx'].values, robots_clean['cy'].values], axis=1)
puntos_camara = np.expand_dims(puntos_camara, axis=1).astype(np.float32)

puntos_proyectados = cv2.perspectiveTransform(puntos_camara, H)

robots_clean['x_canon'] = puntos_proyectados[:, 0, 0]
robots_clean['y_canon'] = puntos_proyectados[:, 0, 1]

# ==========================================
# 4. CREACIÓN DEL LIENZO VECTORIAL DE LA CANCHA
# ==========================================
fig, ax = plt.subplots(figsize=(7, 9.3))
ax.set_facecolor("#1b4332")
ax.set_xlim(0, campo_w)
ax.set_ylim(campo_h, 0)   

ax.add_patch(mpatches.Rectangle((0, 0), campo_w, campo_h, lw=2, edgecolor="#74c69d", facecolor="none"))
ax.axhline(y=campo_h / 2, color="#74c69d", lw=1.5)
ax.add_patch(plt.Circle((campo_w / 2, campo_h / 2), radius=int(30 * ESCALA_PX_CM), color="#74c69d", fill=False, lw=1.5))
    
pen_w, pen_h = int(80 * ESCALA_PX_CM), int(40 * ESCALA_PX_CM)
pen_x = (campo_w - pen_w) / 2
for y_pen in (0, campo_h - pen_h):
    ax.add_patch(mpatches.Rectangle((pen_x, y_pen), pen_w, pen_h, lw=1, edgecolor="#74c69d", facecolor="none"))
        
# --- CORRECCIÓN DE PARÉNTESIS AQUÍ ---
goal_w = int(60 * ESCALA_PX_CM)
goal_x = (campo_w - goal_w) / 2

# El paréntesis de Rectangle ahora cierra correctamente al final de la configuración estética
ax.add_patch(mpatches.Rectangle((goal_x, 0), goal_w, int(8 * ESCALA_PX_CM), lw=2, edgecolor="#ffd60a", facecolor="#ffd60a33"))
ax.add_patch(mpatches.Rectangle((goal_x, campo_h - int(8 * ESCALA_PX_CM)), goal_w, int(8 * ESCALA_PX_CM), lw=2, edgecolor="#00b4d8", facecolor="#00b4d833"))

ax.text(campo_w / 2, 22, "AMARILLO", color="#ffd60a", fontsize=8, ha="center")
ax.text(campo_w / 2, campo_h - 22, "AZUL", color="#00b4d8", fontsize=8, ha="center", va="bottom")

# ==========================================
# 5. MAPA DE CALOR CON LAS NUEVAS COORDENADAS
# ==========================================
# Desempaquetamos los 4 valores; 'im' guardará el objeto gráfico real
counts, xedges, yedges, im = ax.hist2d(
    robots_clean["x_canon"],
    robots_clean["y_canon"],
    bins=40,
    cmap="hot",
    alpha=0.55, 
    range=[[0, campo_w], [0, campo_h]] 
)

# Ahora 'im' contiene exactamente lo que la barra de colores necesita
fig.colorbar(im, ax=ax, label="Frecuencia de posición")
plt.title("Mapa de calor proyectado (Homografía H)")
plt.axis("off") 
plt.show()

