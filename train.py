from ultralytics import YOLO # type: ignore

model = YOLO('runs/detect/train2/weights/best.pt') 

# 2. Iniciar el entrenamiento
model.train(
    data='data/dataset_futbol/data.yaml', 
    epochs=20, 
    imgsz=1280, 
    device="mps",  # <--- IMPORTANTE: 'mps' en lugar de 0 o 'cpu'
    batch=12        # Ajusta según tu RAM (si da error, baja a 8)
)