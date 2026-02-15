import argparse
import supervision as sv
import cv2
from ultralytics import YOLO  # type: ignore
import numpy as np


#model = YOLO('runs/detect/train4/weights/best.pt')
model = YOLO('yolo26n.pt')
box_annotator = sv.BoxAnnotator(
    color= sv.ColorPalette.from_hex(['#FF8C00','#00BFFF',"#FF1493",'#FFD700']),
    thickness=2
)

def main(path_video):
    print(f"prosesando video: {path_video}")
    frame_generator = sv.get_video_frames_generator(source_path=path_video)

    for i, frame in enumerate(frame_generator):

        result = model(frame, device="mps", verbose=False, imgsz=1280)[0]
        detections = sv.Detections.from_ultralytics(result)

        annotated_frame = frame.copy()

        annotated_frame = box_annotator.annotate(
            scene=annotated_frame,
            detections=detections 
        )

        cv2.imshow("proccess video",annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_video")
    args = parser.parse_args()

    main(args.path_video)