from app import celery
from models import FocusSessions, FocusLabels, db
import cv2
import numpy as np
import base64
import uuid
import joblib
from ultralytics import YOLO
import logging

logger = logging.getLogger(__name__)

UPPER_BODY_LANDMARKS = [0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12]
JOINT_PAIRS = [
    (5, 7, 9), (6, 8, 10), (11, 5, 6), (12, 5, 6),
    (5, 11, 12), (6, 11, 12), (0, 5, 11), (0, 6, 12)
]

def calculate_angle(a, b, c):
    a1 = np.array(a)
    b1 = np.array(b)
    c1 = np.array(c)
    ba = a1 - b1
    bc = c1 - b1
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    try:
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)
    except Exception as e:
        logger.error(f"Error calculating angle: {str(e)}")
        return 0.0

def normalize_keypoints(keypoints, confs):
    features = {}
    if 5 in keypoints and 6 in keypoints and confs.get(5, 0) > 0.3 and confs.get(6, 0) > 0.3:
        center = np.mean([keypoints[5], keypoints[6]], axis=0)
        shoulder_dist = np.linalg.norm(np.array(keypoints[5]) - np.array(keypoints[6]))
        norm_factor = shoulder_dist if shoulder_dist != 0 else 1.0
    else:
        center = np.array([0, 0], dtype=np.float32)
        norm_factor = 1.0

    rel_coords = {}
    norm_coords = {}
    for i in UPPER_BODY_LANDMARKS:
        if i in keypoints and confs.get(i, 0) > 0.3:
            rel = np.array(keypoints[i], dtype=np.float64) - center
            norm = rel / norm_factor
            rel_coords[i] = rel
            norm_coords[i] = norm
            features[f'rel_x_{i}'] = round(float(rel[0]), 3)
            features[f'rel_y_{i}'] = round(float(rel[1]), 3)
            features[f'norm_x_{i}'] = round(float(norm[0]), 3)
            features[f'norm_y_{i}'] = round(float(norm[1]), 3)
        else:
            features[f'rel_x_{i}'] = 0.0
            features[f'rel_y_{i}'] = 0.0
            features[f'norm_x_{i}'] = 0.0
            features[f'norm_y_{i}'] = 0.0

    for a, b, c in JOINT_PAIRS:
        if a in rel_coords and b in rel_coords and c in rel_coords:
            angle_rel = calculate_angle(rel_coords[a], rel_coords[b], rel_coords[c])
            angle_norm = calculate_angle(norm_coords[a], norm_coords[b], norm_coords[c])
        else:
            angle_rel = angle_norm = 0.0
        features[f'angle_{a}_{b}_{c}_rel'] = round(float(angle_rel), 3)
        features[f'angle_{a}_{b}_{c}_norm'] = round(float(angle_norm), 3)

    return features

@celery.task
def process_focus_frame(attendance_id, frame_base64, student_id=None):
    try:
        img_data = base64.b64decode(frame_base64.split(",")[1])
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image")

        model = YOLO("app/models/yolov8n-pose.pt")
        results = model.predict(img, conf=0.5, verbose=False)[0]
        keypoints = {i: kp[:2] for i, kp in enumerate(results.keypoints.xy.cpu().numpy()[0]) if i in UPPER_BODY_LANDMARKS}
        confs = {i: conf for i, conf in enumerate(results.keypoints.conf.cpu().numpy()[0]) if i in UPPER_BODY_LANDMARKS}

        if not keypoints:
            label = "unfocused"
        else:
            features = normalize_keypoints(keypoints, confs)
            feature_columns = [
                f'{t}_{c}_{i}' for i in UPPER_BODY_LANDMARKS for t in ['rel', 'norm'] for c in ['x', 'y']
            ] + [f'angle_{a}_{b}_{c}_{t}' for a, b, c in JOINT_PAIRS for t in ['rel', 'norm']]
            feature_vector = [features.get(col, 0.0) for col in feature_columns]

            lr_model = joblib.load("app/models/logistic_model.pkl")
            label = "focused" if lr_model.predict([feature_vector])[0] == 1 else "unfocused"

        focus_label = FocusLabels(
            attendance_id=attendance_id,
            student_id=student_id,
            frame_id=str(uuid.uuid4()),
            label=label
        )
        db.session.add(focus_label)
        db.session.commit()
        logger.info(f"Processed frame for attendance_id {attendance_id}: {label}")
    except Exception as e:
        logger.error(f"Error processing frame for attendance_id {attendance_id}: {str(e)}")
        db.session.rollback()

@celery.task
def calculate_focus_index(attendance_id):
    try:
        labels = FocusLabels.query.filter_by(attendance_id=attendance_id).all()
        if not labels:
            logger.warning(f"No labels found for attendance_id {attendance_id}")
            return

        focused_count = sum(1 for label in labels if label.label == "focused")
        total_count = len(labels)
        focus_index = (focused_count / total_count * 100) if total_count > 0 else 0.0

        session = FocusSessions(
            attendance_id=attendance_id,
            student_id=None,
            focus_index=focus_index,
            closed=True
        )
        db.session.add(session)
        db.session.commit()
        logger.info(f"Calculated focus index for attendance_id {attendance_id}: {focus_index}%")
    except Exception as e:
        logger.error(f"Error calculating focus index for attendance_id {attendance_id}: {str(e)}")
        db.session.rollback()