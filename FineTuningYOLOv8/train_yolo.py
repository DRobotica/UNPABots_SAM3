# ============================================================
# train_yolo.py
# Fine-tuning de YOLOv8 para deteccion de objetos en la Copa FutBotMX
# Clases: limites, pelota, porteria amarilla, porteria azul, robot
# ============================================================

from ultralytics import YOLO
import torch


def main():
    # Usar GPU si esta disponible, si no CPU
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {'GPU' if device == 0 else 'CPU'}")

    # Cargar modelo base YOLOv8s (small) pre-entrenado en COCO
    # Opciones: yolov8n (nano), yolov8s (small), yolov8m (medium), yolov8l (large)
    model = YOLO("yolov8s.pt")

    results = model.train(
        # Ruta al archivo YAML con la configuracion del dataset
        data="./dataset/data.yaml",

        # Numero de epocas de entrenamiento
        epochs=50,

        # Tamano de imagen de entrada (640x640 es el estandar)
        imgsz=640,

        # Batch size: reduce si te quedas sin memoria GPU, aumenta si sobra
        batch=16,

        # Dispositivo: 0 para GPU, "cpu" para CPU
        device=device,

        # Early stopping: detiene si no mejora en 10 epocas
        patience=10,

        # Guardar checkpoints durante el entrenamiento
        save=True,

        # Nombre del experimento (los pesos se guardan en runs/detect/futbolmx_yolo/)
        name="train",

        # Sobrescribir si ya existe un experimento con ese nombre
        exist_ok=True,

        # Learning rate inicial
        lr0=0.01,

        # Factor de LR final (lr_final = lr0 * lrf)
        lrf=0.01,

        # Momento del optimizador SGD
        momentum=0.937,

        # Weight decay para regularizacion
        weight_decay=0.0005,

        # Epocas de calentamiento del learning rate
        warmup_epochs=3,

        # Usar cosine learning rate scheduler
        cos_lr=True,

        # Data augmentation (mosaic, flip, hsv, etc.)
        augment=True,

        # Semilla para reproducibilidad
        seed=42,
    )

    print(f"\nEntrenamiento completado. Mejor modelo: {results.save_dir}/weights/best.pt")

    # Evaluar el modelo entrenado en el conjunto de validacion
    metrics = model.val()
    print(f"\nResultados validacion:")
    print(f"  mAP50:    {metrics.box.map50:.4f}")     # mAP con IoU=0.5
    print(f"  mAP50-95: {metrics.box.map:.4f}")       # mAP promedio con IoU 0.5-0.95

    # Exportar a ONNX para despliegue/inferencia mas rapida
    model.export(format="onnx")
    print("Modelo exportado a ONNX.")


if __name__ == "__main__":
    main()
