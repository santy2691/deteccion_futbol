from ultralytics import YOLO 
import os
from dotenv import load_dotenv

if __name__ == "__main__":
    load_dotenv()
    CONFIG_MODEL = os.getenv('CONFIG_MODEL')
    CONFIG_DEVICE = os.getenv('CONFIG_DEVICE')
    DATA_SET_PATH = os.getenv('URL_DATASET')

    model = YOLO(CONFIG_MODEL) 

    # 2. Iniciar el entrenamiento
    model.train(
        data=DATA_SET_PATH, 
        epochs=50, 
        imgsz=1280, 
        device=CONFIG_DEVICE,  # <--- IMPORTANTE: 'mps' en lugar de 0 o 'cpu'
        batch=16        # Ajusta según tu RAM (si da error, baja a 8)
    )
