import argparse
import supervision as sv
import cv2
from ultralytics import YOLO  # type: ignore
import numpy as np
import os
from dotenv import load_dotenv



load_dotenv()
CONFIG_MODEL = os.getenv('CONFIG_MODEL')
CONFIG_DEVICE = os.getenv('CONFIG_DEVICE')
BALL_ID = 0

model = YOLO(CONFIG_MODEL)
ellipse_annotator = sv.EllipseAnnotator(
    color= sv.ColorPalette.from_hex(['#FF8C00','#00BFFF',"#FF1493",'#FFD700']),
    thickness=2
)

triangle_annotator = sv.TriangleAnnotator(
    color= sv.Color.from_hex("#00BFFF"),
    base=20,
    height=17
)  

traker = sv.ByteTrack()
traker.reset()

laber_annotator = sv.LabelAnnotator(
    color= sv.ColorPalette.from_hex(['#FF8C00','#00BFFF',"#FF1493",'#FFD700']),
    text_color=  sv.Color.from_hex('#000000'),
    text_scale=0.5,
    text_thickness=1,
    text_position= sv.Position.BOTTOM_CENTER
)


def main(path_video):
    print(f"prosesando video: {path_video}")
    frame_generator = sv.get_video_frames_generator(source_path=path_video)

    for i, frame in enumerate(frame_generator):

        result = model(frame, device=CONFIG_DEVICE, verbose=False, imgsz=1280)[0]
        detections = sv.Detections.from_ultralytics(result)

        ball_detections = detections[detections.class_id == BALL_ID]
        ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)
        
        
        all_detections = detections[detections.class_id != BALL_ID]
        all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)
        all_detections.class_id = all_detections.class_id - 1
        all_detections = traker.update_with_detections(detections=all_detections)

        labels = [
            f"#: {track_id}" for track_id in all_detections.tracker_id
        ]

        annotated_frame = frame.copy()

        annotated_frame = ellipse_annotator.annotate(
            scene=annotated_frame,
            detections=all_detections
        )

        annotated_frame = triangle_annotator.annotate(
            scene=annotated_frame,
            detections=ball_detections
        ) 

        annotated_frame = laber_annotator.annotate(
            scene=annotated_frame,
            detections=all_detections,
            labels=labels
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