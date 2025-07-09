import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os
import json

# Configuration
DISPATCH_AREA = (870, 50, 1850, 600)  # x1, y1, x2, y2
DETECTION_CLASSES = ['dish', 'tray']  
CLASSIFICATION_CLASSES = ['empty', 'kakigori', 'not_empty']
YOLO_MODEL_PATH = 'models/detection/best.pt'
CLASSIFIER_MODEL_PATH = 'models/classification/classification_model.h5'
INPUT_VIDEO = "test.mp4"
OUTPUT_VIDEO = "output/monitoring_output.mp4"
TRACKING_JSON = "videos/output/tracking.json"
FRAME_OUTPUT_DIR = "videos/frames"


def load_models():
    '''Load YOLO and classifier models for object detection and classification.'''
    print("[INFO] Loading models...")
    yolo = YOLO(YOLO_MODEL_PATH)
    classifier = load_model(CLASSIFIER_MODEL_PATH)

    tracker = DeepSort(
        max_age=50,
        n_init=3,
        max_cosine_distance=0.4,
        nn_budget=None
    )
    return yolo, classifier, tracker

def preprocess_for_classifier(crop):
    '''Preprocess the cropped image for classification.'''
    if crop.size == 0:
        return None
    img = cv2.resize(crop, (224, 224))
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

def process_frame(frame, yolo, classifier, tracker, frame_id, tracking_records):
    '''Process video frame for object detection and classification.'''
    # Define the dispatch area and visualize it
    x1, y1, x2, y2 = DISPATCH_AREA
    vis_frame = frame.copy()

    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
    dispatch_crop = frame[y1:y2, x1:x2]

    detections = []
    results = yolo(dispatch_crop, conf=0.5, verbose=False)
    
    for box in results[0].boxes:
        cx1, cy1, cx2, cy2 = map(int, box.xyxy[0].tolist())   # Get coordinates of the bounding box
        conf = float(box.conf[0])
        class_id = int(box.cls[0])
        det_type = DETECTION_CLASSES[class_id]
        # Calculate absolute coordinates relative to the dispatch area
        abs_x1, abs_y1 = x1 + cx1, y1 + cy1
        abs_x2, abs_y2 = x1 + cx2, y1 + cy2
        w, h = abs_x2 - abs_x1, abs_y2 - abs_y1

        detections.append(([abs_x1, abs_y1, w, h], conf, det_type))

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        l, t, r, b = map(int, track.to_ltrb())

        status = "unknown"
        item_crop = frame[t:b, l:r]
        processed = preprocess_for_classifier(item_crop)
        if processed is not None:
            probs = classifier.predict(processed, verbose=0)[0]
            status = CLASSIFICATION_CLASSES[np.argmax(probs)]

        det_type = track.det_class if hasattr(track, 'det_class') else "unknown"
        label = f"ID:{track.track_id} | {det_type} | {status}"
        color = (0, 255, 0) if status == "empty" else (0, 0, 255)
        cv2.rectangle(vis_frame, (l, t), (r, b), color, 2)
        cv2.putText(vis_frame, label, (l, t - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        tracking_records.append({
            "frame": int(frame_id),
            "id": int(track.track_id),
            "bbox": [l, t, r, b],
            "type": det_type,
            "label": status
        })

    return vis_frame

def run_monitoring(input_path, output_path):
    '''
    Run the monitoring process on the input video.
    
    '''
    yolo, classifier, tracker = load_models()

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    tracking_data = []
    frame_id = 0

    os.makedirs(FRAME_OUTPUT_DIR, exist_ok=True)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = os.path.join(FRAME_OUTPUT_DIR, f"frame_{frame_id}.jpg")
            cv2.imwrite(frame_path, frame)
            processed_frame = process_frame(frame, yolo, classifier, tracker, frame_id, tracking_data)
            out.write(processed_frame)
            frame_id += 1
    finally:
        cap.release()
        out.release()

        os.makedirs(os.path.dirname(TRACKING_JSON), exist_ok=True)
        with open(TRACKING_JSON, "w") as f:
            json.dump(tracking_data, f, indent=2)

if __name__ == "__main__":
    run_monitoring()