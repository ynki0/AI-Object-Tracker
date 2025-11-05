import cv2
from ultralytics import YOLO
from src.tracker import ObjectTracker

model = YOLO('yolov8n.pt')
tracker = ObjectTracker()

cap = cv2.VideoCapture('data/input_video.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    detections = results[0].boxes.xyxy.cpu().numpy()

    tracked_objects = tracker.update(detections)

    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = obj
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f'ID {int(obj_id)}', (int(x1), int(y1)-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow('AI Object Tracker', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
