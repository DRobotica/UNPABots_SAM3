# Fine-Tuning SAM3 para FutBotMX
## Copa FutBotMX - Segmentación de robots, pelota, porterías y límites

# Especificaciones de la máquina
* Sistema Operativo: Windows 11
* Procesador: 13th Gen Intel(R) Core(TM) i7-13650HX (2.60 GHz)
* Memoria RAM: 16.0 GB
* GPU: NVIDIA GeForce RTX 5050 Laptop GPU (8 GB VRAM)
* Entorno: Python 3.11 (vía Anaconda)

# Instalación del entorno: 

### 1.1 Instalar Anaconda
Descarga: https://www.anaconda.com/download

Seguir el instalador con opciones por defecto.

### 1.2 Crear entorno conda con Python 3.11 y activarlo:
```bash
conda create -n supervision python=3.11 -y
conda activate supervision
```

### 1.3 Instalar PyTorch con soporte CUDA 13.0 (RTX 5050):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

### 1.4 Instalar librerias de visión por computadora:
```bash
pip install supervision ultralytics trackers
```

### 1.5Instalar transformers (HuggingFace) y dependencias:
```bash
 pip install transformers
```

### 1.6 Instalar dependencias adicionales:
```bash
pip install scikit-learn pycocotools opencv-python tqdm
```

### 1.7 Instalar Jupyter (opcional):
```bash
 pip install jupyter
```
# Preparación del dataset (Roboflow)

### 2.1 Captura de datos
* Se utilizaron videos de partidos de la Copa FutBolMX
* Videos originales en: `DataForRoboflow/` los cuales se encuentran en el drive
* Se subieron a Roboflow (roboflow.com) para anotación

### 2.2 Extracción de frames
* Se extrajo 1 frame cada 2 segundos de cada video
* Total de imágenes obtenidas: 1,537
* Después de limpieza de datos duplicados/borrosos: 1,500 imágenes
* De esas 1,500, se tomó solo el 10% (150 imágenes) para anotación
<img width="1912" height="1026" alt="image" src="https://github.com/user-attachments/assets/ace54ddd-39ea-4fbe-b7da-f26f1492d3b4" />



### 2.3 Segmentación semántica manual en Roboflow
* Se seleccionaron 150 imágenes del total para anotación
* Se segmentaron manualmente 5 clases:
    * limites (líneas de la cancha)
    *  pelota (balón de fútbol)
    *  porteria amarilla
    *  porteria azul
    *  robot (todos los robots en general)

### 2.4 División del dataset
* 120 imágenes para entrenamiento (`train`)
* 30 imágenes para validación (`valid`)
* Formato de exportación: COCO Segmentation
* Ubicación: `dataset/train/` y `dataset/valid/`

<img width="1917" height="1027" alt="image" src="https://github.com/user-attachments/assets/a20b4e53-fff6-4531-9dba-7e144c7de5db" />

# Estructura del proyecto

```text
El proyecto completo ha sido agregado al link de drive para mayor claridad: https://drive.google.com/drive/folders/1jxJx3a7FrNykwqK2Q1bZ2tcAKyDSzxre?usp=sharing
FineTuning/
|
|-- sam3_training/                  # Entrenamiento y pruebas básicas de SAM3
|   |-- train_sam3.py               # Fine-tuning de SAM3 con HuggingFace
|   |-- test_inference.py           # Prueba de segmentación en imágenes
|   |-- prueba_video.py             # Prueba de segmentación en video
|   |-- outputs/                    # Videos/imágenes generados por las pruebas
|
|-- analisis_partido/               # Análisis avanzado de partido (SAM3 + DINOv2)
|   |-- analisis_partido.py         # Con SAM3 base (sam3.pt, vía ultralytics)
|   |-- analisis_partido_2.py       # Con modelo fine-tuning (soccer_sam3_final)
|   |-- outputs_tuned/              # Resultados del modelo fine-tuning
|       |-- analisis_2_*.mp4        # Videos procesados
|       |-- centroides_2_*.csv      # Posiciones de robots (frame, id, equipo, x, y)
|       |-- vertices_2_*.csv        # Vértices de la cancha (frame, vertice, x, y)
|
|-- dataset/                        # Dataset COCO para entrenamiento
|   |-- train/                      # 120 imágenes + _annotations.coco.json
|   |-- valid/                      # 30 imágenes + _annotations.coco.json
|
|-- soccer_sam3_final/              # Modelo SAM3 fine-tuning (formato HuggingFace)
|   |-- config.json                 # Configuración del modelo
|   |-- model.safetensors           # Pesos del modelo entrenado
|   |-- processor_config.json       # Configuración del procesador
|   |-- tokenizer.json              # Tokenizador de texto
|   |-- tokenizer_config.json
|
|-- checkpoints/                    # Checkpoints guardados durante el entrenamiento
|   |-- sam3_soccer_decoder_epoch_*.pth
|   |-- sam3_soccer_decoder_best.pth
|
|-- DataForRoboflow/                # Videos originales de partidos (.MOV)
|-- sam3.pt                         # Modelo SAM3 base de Meta (~3.4 GB)
|-- IMG_9796.MOV                    # Video de prueba
|-- recortado.mp4                   # Video recortado de prueba
|-- LICENSE.txt
```
# Entrenamiento del modelo SAM3

## 4.1 Script: sam3_training/train_sam3.py

* Objetivo: Entrenar el mask_decoder de SAM3 para que aprenda a segmentar
    las 5 clases del dataset usando prompts de texto.

* Arquitectura:
  - Modelo base: facebook/sam3 (840M parametros)
  - Solo se entrena el mask_decoder (~32M parametros, 3.9% del total)
  - El resto (backbone ViT, encoder de texto, decoder DETR) se congela
  - Loss: Binary Cross Entropy (BCE) + Dice Loss
  - Optimizador: AdamW con learning rate 1e-5

  * Dataset:
   - Cada imagen genera una muestra por cada clase presente
   - ~120 imagenes x ~4 clases = ~480 muestras de entrenamiento
   - Batch size: 4

* Duracion del entrenamiento:
   - Aproximadamente 12 horas para 10 epocas

* Comando:
  ```bash
      python sam3_training/train_sam3.py
   ```
* Salida:
  - Checkpoints en checkpoints/ (uno por epoca + el mejor)
  - Modelo final en soccer_sam3_final/ (listo para inferencia)

# Prueba del modelo en imágenes

## 5.1 Script: sam3_training/test_inference.py

* Objetivo: Cargar el modelo fine-tuning y probarlo en imagenes
    individuales del dataset de validacion.

* Flujo:
  * Carga el modelo desde soccer_sam3_final/
    
  * Para cada imagen, ejecuta 5 inferencias (1 por clase)
    
  * Combina las mascaras en una imagen con colores:
    - limites           -> cyan
    - pelota            -> azul
    - porteria amarilla -> amarillo
    - porteria azul     -> naranja
    - robot             -> magenta
    
  * Guarda el resultado en /outputs/inferencia/
<img width="1920" height="1080" alt="seg_IMG_9783_MOV-0000_jpg rf XwbzNYbAQiwDR4bbJRDN" src="https://github.com/user-attachments/assets/366b7767-69c9-4ec9-acb5-f3d0d0eec824" />

* Comando:
 ```bash
      python sam3_training/test_inference.py
```

# Prueba del modelo en video

## 6.1 Script: sam3_training/prueba_video.py

* Objetivo: Procesar un video completo frame por frame con SAM3 fine-tuning, mostrando las mascaras de las 5 clases.

* Modos de uso:
  - Con video:   python sam3_training/prueba_video.py "video.mov"
  - Simulacion:  python sam3_training/prueba_video.py (usa los frames del dataset como video)

* Duracion:
  - Videos de 8-16 segundos: aproximadamente 2-3 horas
  - SAM3 procesa ~5 inferencias por frame (~200ms cada una)

* Salida:
  - Video con mascaras coloreadas superpuestas
    
<img width="1918" height="1084" alt="image" src="https://github.com/user-attachments/assets/56c6e286-6224-4265-8a9d-4e8bf8ffb23e" />


 # Analisis avanzado de partido (SAM3 + DINOv2 + Tracking)
 
## 7.1 Script: analisis_partido/analisis_partido.py

* Usa SAM3 base (sam3.pt via ultralytics) con prompts:
 - "white lines", "robot", "ball"

* Pipeline por frame:
  - SAM3 segmenta los 3 conceptos en UNA llamada
  - Lineas blancas -> DBSCAN -> poligono de la cancha
  - Cada robot -> DINOv2 embedding (huella digital visual)
  - Tracking: asigna IDs consistentes (R0-R3) por embedding + distancia
  -  Clasifica robots en 2 equipos por similitud visual (frame 0)
  -  Exporta CSV con posiciones y vertices de cancha

  * Comando:
 ```bash
      python analisis_partido/analisis_partido.py "video.mov"
 ```

7.2 Script: analisis_partido/analisis_partido_2.py

    Usa el modelo SAM3 fine-tuning (soccer_sam3_final/) con tus 5 clases.
    Incluye Non-Maximum Suppression (NMS) para evitar mascaras duplicadas.

    Pipeline por frame:
      1. SAM3 segmenta cada clase por separado (5 llamadas, ~1s total)
      2. NMS filtra mascaras duplicadas del mismo objeto
      3. "limites" -> DBSCAN -> poligono de cancha (cada 10 frames)
      4. Cada instancia de "robot" -> DINOv2 embedding individual
      5. Tracking identico a analisis_partido.py
      6. Dibuja mascaras individuales, cajas, etiquetas y nombres de clase

    Comando:
      python analisis_partido/analisis_partido_2.py "video.mov"

    Salida:
      - analisis_2_<video>.mp4  (video con tracking y mascaras)
      - centroides_2_<video>.csv (frame, robot_id, equipo, cx, cy, ball)
      - vertices_2_<video>.csv   (frame, vertice, x, y del poligono)


====================================================================
  8. NOTAS IMPORTANTES
====================================================================

8.1 Archivo sam3.pt
    - Modelo base de Meta (~3.4 GB)
    - Solo lo usa analisis_partido.py 
    - Requiere acceso a HuggingFace: https://huggingface.co/facebook/sam3

8.2 Modelo fine-tuning (soccer_sam3_final/)
    - Formato HuggingFace, listo para cargar con Sam3Model.from_pretrained()
    - Lo usan: test_inference.py, prueba_video.py, analisis_partido_2.py
    - NO es compatible con el codigo de analisis_partido.py (que usa ultralytics)

8.3 Compatibilidad
    - Todo el proyecto fue desarrollado y probado en Windows 11
    - Entorno conda: supervision (Python 3.11)
    - GPU: NVIDIA RTX 5050 Laptop (8 GB VRAM, CUDA 13.0)
