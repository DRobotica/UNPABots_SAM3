# UNPABots_SAM3
Convocatoria Copa FutBotMX, Capítulo Visión por Computadora.

# Analítica de Video y Visualización - Copa FutbotMX Capitulo Visión por Computadora
Este proyecto implementa un pipeline de visión por computadora de dos etapas (YOLO + SAM 3) acoplado a un extractor de características DINOv2 para el seguimiento, clasificación por equipos y proyección cenital (Bird's-Eye View) de partidos de la Copa FutBotMX.
El sistema está optimizado para ejecutarse localmente aprovechando la aceleración por hardware (CUDA) en GPUs de alto rendimiento (como la NVIDIA GeForce RTX 5070 Ti).

# Características Clave
* Pipeline de detección y segmentación eficiente: Utiliza un modelo YOLO personalizado para detectar los elementos del juego y delega los bounding boxes a SAM 3 para obtener máscaras de segmentación ultra-precisas.
  
* Clasificación y tracking con DINOv2: Extrae embeddings visuales de las ROIs de los robots y aplica métricas de similitud coseno junto con DBSCAN para agruparlos automáticamente en 2 equipos y mantener sus IDs de forma persistente (soporta hasta 4 IDs simultáneos en juego).
  
* Homografía rígida automatizada: Corrige la perspectiva de la cámara para generar un plano táctico cenital basado en las dimensiones reales de la cancha (219 × 158 cm) con una resolución escalada a 2 píxeles por centímetro.

* Estabilización geométrica avanzada: Incorpora un filtro de historial de frames (promedio móvil) combinado con puntos de control basados en el centroide de las porterías para mitigar vibraciones y oclusiones en los vértices.
  
* Exportación de elemetría:** Genera de manera incondicional un archivo `.csv` continuo con las coordenadas de píxeles y cenitales de los robots y la pelota frame a frame.
  
* Visualización de estadísticas: Con los datos del archivo `.csv` genera un video del tracking de los robots y la pelota y un mapa de calor del partido.


# Arquitectura del Software

El proyecto está modularizado en dos componentes principales para facilitar su mantenimiento:
1. `helpers.py`: Módulo que aloja las funciones matemáticas y algorítmicas secundarias:
    * Ordenamiento de vértices en sentido horario para homografía.
    * Extracción robusta de envolvente convexa (*Convex Hull*).
    * Cálculo de similitud coseno y agrupamiento de vectores DINOv2.
2. `main.py`: Script principal encargado de coordinar el flujo de datos:
    * Inicialización y alojamiento de modelos en la GPU (`cuda`).
    * Lectura y procesamiento secuencial del video del partido.
    * Renderizado de máscaras translúcidas y etiquetas visuales.
    * Escritura síncrona de los archivos de video resultantes y base de datos de telemetría.
      
Para la parte de visualización se tienen dos componentes:
1. `generavideotracking.py`: Script donde se genera un video a partir de los datos del archivo `.csv` con la vista cenital de la cancha de forma animada donde se muestra el tracking de los robots y la pelota.
2. `generamapacalor.py`: Script donde se genera un mapa de calor a partir de los datos del archivo `.csv` donde se muestran las partes del campo donde estuvieron los robot y la pelota el mayor tiempo.

# Requisitos e Instalación

## Especificaciones del equipo

El proyecto se desarrollo y ejecutó localmente en un equipo con las siguientes especificaciones:
  - Windows 11
  - Procesador: Intel(R) Core(TM) Ultra 9 275HX (2.70 GHz)
  - RAM: 32.0 GB
  - GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU (12 GB)
  - Python 3.11 (via Anaconda)
  - 
# Instalación del entorno
1 Instalar Anaconda
    Descarga: https://www.anaconda.com/download
    Seguir el instalador con opciones por defecto.

2 Crear entorno conda con Python 3.11 y activarlo:
    conda create -n supervision python=3.11 -y
    conda activate supervision

3 Instalar PyTorch con soporte CUDA 13.0 (RTX 5070 Ti):
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

4 Instalar librerias de vision por computadora:
    pip install supervision ultralytics trackers

5 Instalar transformers (HuggingFace) y dependencias:
    pip install transformers

6 Instalar dependencias adicionales:
    pip install scikit-learn pycocotools opencv-python tqdm timm numpy

7 Instalar Jupyter (opcional):
    pip install jupyter

Preparación del dataset (roboflow)

1 Captura de datos
    - Se utilizaron videos de partidos de la Copa FutBotMX
    - Los videos utilizados se encuentran en la carpeta DataForRoboflow (disponible en el link de drive)
    - Se subieron a Roboflow (roboflow.com) para anotación

2 Extracción de frames
    - Se extrajo 1 frame cada 2 segundos de cada video
    - Total de imágenes obtenidas: 1,537
    - Se realizó la limpieza de datos duplicados/borrosos quedando 1,500 imágenes
    - Las 150 imágenes se dividieron en: 120 train / 30 valid

3 Segmentación por detección de objetos en Roboflow
    - Del total de imágenes, se seleccionó el 10% (150 imágenes) para anotación
    - Se segmentaron manualmente 5 clases:
        1. limites          (líneas blancas de la cancha)
        2. pelota           (balón de futbol)
        3. porteria         (ambas porterías)
        4. cancha           (cancha verde)
        5. robot            (todos los robots en general)

4 Division del dataset
    - 120 imágenes para entrenamiento (train)
    - 30 imágenes para validación (valid)
    - Formato de exportación: YOLOv8
    - Ubicación: FineTuningYOLOv8/dataset/train/ y FineTuningYOLOv8/dataset/valid/


Estructura del proyecto

UNPABots_SAM3/
|
|-- FineTuning/                     # Entrenamiento y pruebas básicas de Fine-tuning de SAM3 (Primer proyecto descartado)
|   
|-- FineTuningYOLOv8/               # Entrenamiento de Fine-tuning de YOLOv8
|   |-- dataset/                    # Dataset YOLOv8 para entrenamiento
|       |-- train/                  # Imágenes de entrenamiento
|       |-- valid/                  # Imágenes de validación
|       |-- data.yaml               # Clases segmentadas
|   |-- train_yolo.py               # Script de entrenamiento
|   |-- yolo26n.pt                  # 
|   |-- yolov8s.pt                  # 
|
|-- bestv8s.pt                      # Modelo de YOLOv8 personalizado (fine-tuning)
|-- generarmapacalor.py             # Script para generar el mapa de calor
|-- generarvideotracking.py         # Script para generar un video animado del tracking de los robots y la pelota
|-- helpers.py                      # Módulo que aloja las funciones matemáticas y algorítmicas secundarias
|-- main.py                         # Script principal encargado de coordinar el flujo de datos
|-- mapacalor_1.png                 # Imagen del mapa de calor generado de video1.mp4
|-- mapacalor_2.png                 # Imagen del mapa de calor generado de video2.mp4
|-- resultado_cenital_1.mp4         # Video generado con la vista cenital de video1.mp4
|-- resultado_cenital_2.mp4         # Video generado con la vista cenital de video2.mp4
|-- resultado_perspectiva_1.mp4     # Video generado con la segmentación de video1.mp4
|-- resultado_perspectiva_2.mp4     # Video generado con la segmentación de video2.mp4
|-- sam3.pt                         # Modelo de SAM3 (disponible en el link de drive)
|-- telemetria_avanzada_1.csv       # Archivo con los datos obtenidos del análisis de video1.mp4
|-- telemetria_avanzada_2.csv       # Archivo con los datos obtenidos del análisis de video2.mp4
|-- tracking_1.mp4                  # Video animado generado con el tracking de los robots y pelota de video1.mp4
|-- tracking_2.mp4                  # Video animado generado con el tracking de los robots y pelota de video2.mp4
|-- video1.mp4                      # Video de prueba 1
|-- video2.mp4                      # Video de prueba 2
|
|-- LICENSE.txt


Reel de Instagram

Se coloca un reel de Instagram en donde, de forma resumida, se presentan los resultados del proyecto:

https://www.instagram.com/reel/DZy-KmtijtO/?igsh=MWR3NWZlYWs1eWZkdQ==


Link a drive

Se coloca un link a drive con videos de los resultados, video explicativo y archivos necesarios para correr el proyecto:

https://drive.google.com/drive/folders/1jxJx3a7FrNykwqK2Q1bZ2tcAKyDSzxre?usp=sharing
