import supervision as sv 
from tqdm import tqdm

def detect_crops(path_video, model, device):
  frame_generator = sv.get_video_frames_generator(source_path=path_video)

  crops = []
  for i in tqdm(frame_generator, desc='obteniendo crops'):
    result = model(i, device=device, verbose=False, imgsz=1280)[0]
    detection = sv.Detections.from_ultralytics(result)