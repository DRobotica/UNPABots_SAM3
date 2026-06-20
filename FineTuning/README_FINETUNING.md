# Fine-Tuning SAM3 para FutBotMX
## Copa FutBotMX - Segmentación de robots, pelota, porterías y límites

# Especificaciones de la máquina
* Sistema Operativo: Windows 11
* Procesador: 13th Gen Intel(R) Core(TM) i7-13650HX (2.60 GHz)
* Memoria RAM: 16.0 GB
* GPU: NVIDIA GeForce RTX 5050 Laptop GPU (8 GB VRAM)
* Entorno: Python 3.11 (vía Anaconda)

#Instalación del entorno: 

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
* Se consideraron 5 clases: limites, pelota, porteria azul, porteria amarilla y robot
* Las 150 imágenes se dividieron en: 120 train / 30 valid

<img width="1912" height="1026" alt="image" src="https://github.com/user-attachments/assets/ace54ddd-39ea-4fbe-b7da-f26f1492d3b4" />

<img width="1917" height="1027" alt="image" src="https://github.com/user-attachments/assets/a20b4e53-fff6-4531-9dba-7e144c7de5db" />


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
