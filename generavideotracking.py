import supervision as sv 
from ultralytics import YOLO 
import cv2 
import numpy as np 
import urllib.request 
from pathlib import Path 
import matplotlib.pyplot as plt # 1. Importa Matplotlib
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
import pandas as pd
import matplotlib as mpl

# Configura captura de video
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



# Le dice a Matplotlib exactamente dónde buscar el ejecutable de FFmpeg
mpl.rcParams['animation.ffmpeg_path'] = r"c:\Users\QT2\anaconda3\envs\supervision\Library\bin\ffmpeg.exe"

# --- 1. CONFIGURACIÓN Y CONSTANTES ---
       # Archivo con tus coordenadas
VIDEO_ORIGINAL_PATH = "video2.mp4" # El video original de la cámara
OUTPUT_VIDEO_PATH = "tracking.mp4" # Video táctico resultante

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

# Configuración del plano sagital
ESCALA_PX_CM = 2.0  
CAMPO_W = int(180 * ESCALA_PX_CM)  
CAMPO_H = int(240 * ESCALA_PX_CM)  

# Matriz de Homografía H (Coloca tu matriz real calculada)
# H = np.array([...])

# Diccionario de equipos (0: Azul, 1: Rojo)
COLORES_EQUIPOS = {
    0: "#00b4d8",
    1: "#ef233c"
}

# --- 2. EXTRAER FPS AUTOMÁTICAMENTE DE LA CÁMARA ---
def obtener_fps_video(video_path: str) -> float:
    """Abre el video original para leer los FPS nativos configurados en la cámara."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Advertencia: No se pudo abrir {video_path}. Se usarán 30 FPS por defecto.")
        return 30.0
    
    fps = cap.get(cv2.CAP_PROP_FPS) # Obtener FPS reales del archivo
    cap.release()
    
    # Validación por si el codec del video no reporta la métrica correctamente
    if fps <= 0:
        return 30.0
    return fps

FPS_NATURAL = obtener_fps_video(VIDEO_ORIGINAL_PATH)
print(f"-> Detectados {FPS_NATURAL:.2f} fotogramas por segundo (FPS) en el video original.")

# --- 3. FUNCIÓN DE PROYECCIÓN DE HOMOGRAFÍA ---
def project_point(pt_cam: np.ndarray, H: np.ndarray) -> tuple:
    """Proyecta un punto de coordenadas de cámara al campo canónico."""
    projected = cv2.perspectiveTransform(pt_cam.reshape(1, 1, 2).astype(np.float32), H)
    return (int(projected[0][0][0]), int(projected[0][0][1]))


# --- 4. CARGA COMPLETA DE DATOS Y FILTRO DE SUAVIZADO ---
#df = pd.read_csv(csv_path)

# CONFIGURACIÓN DEL FILTRO: 
# Una ventana de 3 o 5 frames es ideal para suavizar sin perder la posición real.
TAMANO_VENTANA = 5  

print("Aplicando filtro de media móvil para suavizar las trayectorias...")

# Ordenar los datos por frame para asegurar que el promedio temporal sea correcto
df = df.sort_values(by=['robot_id', 'frame']).reset_index(drop=True)

# Aplicar el filtro de suavizado de forma independiente para cada robot/balón
df['cx'] = df.groupby('robot_id')['cx'].transform(
    lambda x: x.rolling(window=TAMANO_VENTANA, min_periods=1, center=True).mean()
)
df['cy'] = df.groupby('robot_id')['cy'].transform(
    lambda x: x.rolling(window=TAMANO_VENTANA, min_periods=1, center=True).mean()
)

# Obtener la lista de frames únicos ya suavizados
lista_frames = sorted(df['frame'].unique())

# Diccionario para almacenar el historial de posiciones de cada objeto
HISTORIAL_TRAYECTORIAS = {}


# Diccionario para almacenar el historial de posiciones de cada objeto
# Formato: {"R1": [(x1, y1), (x2, y2), ...], "ball": [...]}
HISTORIAL_TRAYECTORIAS = {}

# --- 5. CONFIGURACIÓN DEL BUFFER DE OPENCV Y RENDERIZADO ---
plt.ioff()
fig, ax = plt.subplots(figsize=(7, 9.3), dpi=100)

def draw_frame_to_buffer(frame_id):
    """Dibuja el campo, trayectorias e iconos en la figura, regresando una matriz de píxeles."""
    ax.clear()
    ax.set_facecolor("#1b4332")
    ax.set_xlim(0, CAMPO_W)
    ax.set_ylim(CAMPO_H, 0) # Eje Y invertido (0 arriba)

    # --- Dibujo de líneas estáticas del campo ---
    ax.add_patch(mpatches.Rectangle((0, 0), CAMPO_W, CAMPO_H, lw=2, edgecolor="#74c69d", facecolor="none"))
    ax.axhline(y=CAMPO_H / 2, color="#74c69d", lw=1.5)
    ax.add_patch(plt.Circle((CAMPO_W / 2, CAMPO_H / 2), radius=int(30 * ESCALA_PX_CM), color="#74c69d", fill=False, lw=1.5))
    
    pen_w, pen_h = int(80 * ESCALA_PX_CM), int(40 * ESCALA_PX_CM)
    pen_x = (CAMPO_W - pen_w) / 2
    for y_pen in (0, CAMPO_H - pen_h):
        ax.add_patch(mpatches.Rectangle((pen_x, y_pen), pen_w, pen_h, lw=1, edgecolor="#74c69d", facecolor="none"))
        
    goal_w = int(60 * ESCALA_PX_CM)
    goal_x = (CAMPO_W - goal_w) / 2
    ax.add_patch(mpatches.Rectangle((goal_x, 0), goal_w, int(8 * ESCALA_PX_CM), lw=2, edgecolor="#ffd60a", facecolor="#ffd60a33"))
    ax.add_patch(mpatches.Rectangle((goal_x, CAMPO_H - int(8 * ESCALA_PX_CM)), goal_w, int(8 * ESCALA_PX_CM), lw=2, edgecolor="#00b4d8", facecolor="#00b4d833"))
    ax.text(CAMPO_W / 2, 22, "AMARILLO", color="#ffd60a", fontsize=8, ha="center")
    ax.text(CAMPO_W / 2, CAMPO_H - 22, "AZUL", color="#00b4d8", fontsize=8, ha="center", va="bottom")
    
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    ax.tick_params(colors="#555")
    ax.set_title(f"Mapa Táctico — Frame {frame_id}", fontsize=13, color="white", pad=10)
    
    # --- Procesar datos del frame actual ---
    df_frame = df[df['frame'] == frame_id]
    
    # Listas temporales para dibujar los elementos actuales AL FINAL (para que queden por encima de las líneas)
    robots_actuales = []
    balon_actual = None

    for _, row in df_frame.iterrows():
        obj_id = str(row['robot_id']).strip()
        equipo = int(row['equipo'])
        x_cam = float(row['cx'])
        y_cam = float(row['cy'])
        
        # Mapeo de homografía
        pos_cam = np.float32([[x_cam, y_cam]])
        x_canon, y_canon = project_point(pos_cam, H)
        
         # Inicializar la lista histórica del objeto si es la primera vez que aparece
        if obj_id not in HISTORIAL_TRAYECTORIAS:
            HISTORIAL_TRAYECTORIAS[obj_id] = []

        # Guardar la posición actual en su historial
        HISTORIAL_TRAYECTORIAS[obj_id].append((x_canon, y_canon))
        
        # Separar elementos para el renderizado por capas
        if "ball" in obj_id or equipo == -1:
            balon_actual = (x_canon, y_canon)
        else:
            color_asignado = COLORES_EQUIPOS.get(equipo, "#ffffff")
            robots_actuales.append((x_canon, y_canon, obj_id, color_asignado))

    # --- Capa 1: Dibujar las líneas de rastro (Trayectorias pasadas) ---
    for obj_id, puntos in HISTORIAL_TRAYECTORIAS.items():
        if len(puntos) > 1:
            pts_array = np.array(puntos)
            
            # Definir color de la línea de rastro
            if "ball" in obj_id:
                color_linea = "#ff9500" # Rastro naranja para el balón
                estilo_linea = ":"      # Línea punteada para el balón
                lw_linea = 1.5
            else:
                # Buscar el color del equipo del robot usando los datos del dataframe
                equipo_sample = df[df['robot_id'] == obj_id]['equipo'].iloc[0]
                color_linea = COLORES_EQUIPOS.get(int(equipo_sample), "#ffffff")
                estilo_linea = "-"      # Línea sólida para robots
                lw_linea = 2.0
                
            # Graficar la línea histórica completa hasta el momento actual
            ax.plot(pts_array[:, 0], pts_array[:, 1], color=color_linea, 
                    linestyle=estilo_linea, linewidth=lw_linea, alpha=0.6, zorder=3)

    # --- Capa 2: Dibujar los robots en su posición actual ---
    for x, y, name, col in robots_actuales:
        ax.scatter(x, y, s=350, c=col, zorder=5, edgecolors="white", linewidths=1.5)
        numero_id = name.split('_')[-1] if '_' in name else name[-1]
        ax.text(x, y, numero_id, ha="center", va="center", fontsize=10, fontweight="bold", color="white", zorder=6)
        ax.text(x, y + 22, name, ha="center", color=col, fontsize=8, zorder=6)

    # --- Capa 3: Dibujar el balón en su posición actual ---
    if balon_actual is not None:
        bx, by = balon_actual
        ax.scatter(bx, by, s=180, c="#ff9500", zorder=5, edgecolors="white", linewidths=1.5, marker="o")
        ax.text(bx + 14, by - 14, "⚽", fontsize=11, zorder=6)

    plt.tight_layout()
    
    # Convertir a imagen BGR para OpenCV
    fig.canvas.draw()
    img_bgr = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
    return img_bgr

# --- 6. CONFIGURAR ESCRITOR DE VIDEO DE OPENCV ---
print("Inicializando escritor de video nativo con OpenCV...")
frame_muestra = draw_frame_to_buffer(lista_frames[0])
alto_v, ancho_v, _ = frame_muestra.shape

# Reiniciar el historial después de la muestra para el renderizado real
HISTORIAL_TRAYECTORIAS = {}

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, FPS_NATURAL, (ancho_v, alto_v))

# --- 7. BUCLE SECUENCIAL DE GUARDADO ---
print(f"Procesando {len(lista_frames)} frames con rastro de trayectorias...")

for idx, frame_id in enumerate(lista_frames):
    foto_frame = draw_frame_to_buffer(frame_id)
    video_writer.write(foto_frame)
    
    if idx % 20 == 0:
        print(f"Progreso: {idx}/{len(lista_frames)} frames procesados...")

video_writer.release()
plt.close(fig)
print(f"¡Éxito absoluto! Video táctico con rastro guardado en: {OUTPUT_VIDEO_PATH}")

