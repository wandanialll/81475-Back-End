from flask import Blueprint, request, jsonify
import numpy as np
import cv2
import base64
from ultralytics import YOLO
import joblib
import pandas as pd
from app.models import Attendance, Lecturer, db, PoseData, PoseFocusIndex
# from app.recognition.attendance_recognition import authenticate_lecturer, decode_image
from app.routes.attendance_recognition import authenticate_lecturer, decode_image
import logging
from datetime import datetime

pose_focus_recognition_bp = Blueprint("pose_focus_recognition", __name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize YOLO and pose model
yolo_model = YOLO("app/models/yolov8n-pose.pt")
pose_model = joblib.load("app/models/logistic_focus_model_4.pkl")

UPPER_BODY_LANDMARKS = [0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12]
JOINT_PAIRS = [
    (5, 7, 9), (6, 8, 10), (11, 5, 7), (12, 6, 8),
    (5, 11, 12), (6, 12, 11), (0, 5, 11), (0, 6, 12)
]

def calculate_angle(a, b, c):
    """Calculate angle between three points"""
    a = np.array(a[:2], dtype=np.float64)
    b = np.array(b[:2], dtype=np.float64)
    c = np.array(c[:2], dtype=np.float64)
    angle = np.degrees(np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0]))
    angle = np.abs(angle)
    if angle > 180:
        angle = 360 - angle
    return float(angle)

def calculate_normalized_angle(a, b, c):
    """Calculate normalized angle"""
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    c = np.array(c, dtype=np.float64)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return float(np.degrees(angle))

def process_pose_data(img, session_id):
    """Process image for pose detection and store results"""
    results = yolo_model.predict(img, conf=0.5, verbose=False)
    result = results[0]
    
    if result.keypoints is None or not hasattr(result.keypoints, 'conf'):
        return False
    
    keypoints = result.keypoints.xy.cpu().numpy()
    confs = result.keypoints.conf.cpu().numpy()
    num_people = keypoints.shape[0]
    
    if num_people == 0:
        return False
    
    for person_id in range(num_people):
        person_kps = keypoints[person_id]
        person_confs = confs[person_id]
        keypoints_dict = {
            i: [float(person_kps[i][0]), float(person_kps[i][1]), float(person_confs[i])]
            for i in UPPER_BODY_LANDMARKS
        }
        
        angles = []
        for a, b, c in JOINT_PAIRS:
            if person_confs[a] > 0.3 and person_confs[b] > 0.3 and person_confs[c] > 0.3:
                angle = calculate_angle(person_kps[a], person_kps[b], person_kps[c])
            else:
                angle = 0.0
            angles.append(float(angle))
        
        center = np.mean([person_kps[5], person_kps[6]], axis=0) if 5 in keypoints_dict and 6 in keypoints_dict else np.array([0.0, 0.0], dtype=np.float64)
        shoulder_dist = np.linalg.norm(np.array(person_kps[5]) - np.array(person_kps[6])) if 5 in keypoints_dict and 6 in keypoints_dict else 1.0
        norm_factor = float(shoulder_dist) if shoulder_dist != 0 else 1.0
        
        rel_coords = {i: (np.array(keypoints_dict[i][:2], dtype=np.float64) - center).tolist() for i in UPPER_BODY_LANDMARKS}
        norm_coords = {i: [float(coord / norm_factor) for coord in rel_coords[i]] for i in UPPER_BODY_LANDMARKS}
        rel_angles = [
            calculate_normalized_angle(rel_coords[a], rel_coords[b], rel_coords[c])
            if a in rel_coords and b in rel_coords and c in rel_coords else 0.0
            for a, b, c in JOINT_PAIRS
        ]
        norm_angles = [
            calculate_normalized_angle(norm_coords[a], norm_coords[b], norm_coords[c])
            if a in norm_coords and b in norm_coords and c in norm_coords else 0.0
            for a, b, c in JOINT_PAIRS
        ]
        
        pose_data = PoseData(
            session_id=session_id,
            person_id=person_id,
            keypoints={f"kp_{i}": keypoints_dict[i] for i in UPPER_BODY_LANDMARKS},
            angles={f"angle_{a}_{b}_{c}": float(angles[idx]) for idx, (a, b, c) in enumerate(JOINT_PAIRS)},
            rel_coords={f"kp_{i}_rel": rel_coords[i] for i in UPPER_BODY_LANDMARKS},
            norm_coords={f"kp_{i}_norm": norm_coords[i] for i in UPPER_BODY_LANDMARKS},
            rel_angles={f"angle_{a}_{b}_{c}_rel": float(rel_angles[idx]) for idx, (a, b, c) in enumerate(JOINT_PAIRS)},
            norm_angles={f"angle_{a}_{b}_{c}_norm": float(norm_angles[idx]) for idx, (a, b, c) in enumerate(JOINT_PAIRS)},
            timestamp=datetime.utcnow()
        )
        db.session.add(pose_data)
    
    db.session.commit()
    return True

def run_inference_and_calculate_focus(session_id):
    """Run inference on stored pose data and calculate focus index"""
    pose_data_records = PoseData.query.filter_by(session_id=session_id).all()
    if not pose_data_records:
        return 0.0
    
    data = []
    for record in pose_data_records:
        row = {}
        for i in UPPER_BODY_LANDMARKS:
            kp = record.keypoints[f"kp_{i}"]
            row[f"kp_{i}_rel_x"] = float(record.rel_coords[f"kp_{i}_rel"][0])
            row[f"kp_{i}_rel_y"] = float(record.rel_coords[f"kp_{i}_rel"][1])
            row[f"kp_{i}_norm_x"] = float(record.norm_coords[f"kp_{i}_norm"][0])
            row[f"kp_{i}_norm_y"] = float(record.norm_coords[f"kp_{i}_norm"][1])
        for idx, (a, b, c) in enumerate(JOINT_PAIRS):
            row[f"angle_{a}_{b}_{c}_rel"] = float(record.rel_angles[f"angle_{a}_{b}_{c}_rel"])
            row[f"angle_{a}_{b}_{c}_norm"] = float(record.norm_angles[f"angle_{a}_{b}_{c}_norm"])
        data.append(row)
    
    df = pd.DataFrame(data)
    try:
        X = df[pose_model.feature_names_in_]
        y_pred = pose_model.predict(X)
        focus_count = (y_pred == 1).sum()
        total_count = len(y_pred)
        focus_index = focus_count / total_count if total_count > 0 else 0.0
    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        focus_index = 0.0
    
    focus_record = PoseFocusIndex(
        session_id=session_id,
        focus_index=float(focus_index),
        timestamp=datetime.utcnow()
    )
    db.session.add(focus_record)
    db.session.commit()
    return float(focus_index)

@pose_focus_recognition_bp.route("/api/attendance/calculate-focus-index", methods=["POST", "OPTIONS"])
def calculate_focus_index():
    if request.method == "OPTIONS":
        response = jsonify({"message": "OK"})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type, Authorization")
        response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        return response

    try:
        id_token = request.headers.get("Authorization")
        if not id_token or not id_token.startswith("Bearer "):
            return jsonify({"error": "Authorization token required"}), 401
        id_token = id_token.split("Bearer ")[1]

        lecturer = authenticate_lecturer(id_token)
        if isinstance(lecturer, tuple):
            return jsonify(lecturer[0]), lecturer[1]

        if not request.json:
            return jsonify({"error": "No JSON data provided"}), 400

        data = request.json
        session_id = data.get("session_id")
        reset = data.get("reset", False)
        image_data = data.get("image")

        if not session_id:
            return jsonify({"error": "session_id is required"}), 400

        session = Attendance.query.filter_by(session_id=session_id).first()
        if not session:
            return jsonify({"error": "Invalid session_id"}), 404

        if session.closed:
            return jsonify({"error": "Session is already closed"}), 400

        if reset:
            try:
                PoseData.query.filter_by(session_id=session_id).delete()
                PoseFocusIndex.query.filter_by(session_id=session_id).delete()
                db.session.commit()
                logger.info(f"Reset pose data for session {session_id}")
                return jsonify({"status": "reset", "session_id": session_id}), 200
            except Exception as e:
                logger.error(f"Error resetting pose data: {str(e)}")
                db.session.rollback()
                return jsonify({"error": "Failed to reset pose data"}), 400

        if not image_data:
            return jsonify({"error": "image data is required"}), 400

        img = decode_image(image_data)
        success = process_pose_data(img, session_id)

        if not success:
            return jsonify({"status": "no_people_detected", "session_id": session_id}), 200

        # Optionally calculate current focus index without closing
        focus_index = run_inference_and_calculate_focus(session_id)
        return jsonify({"status": "pose_data_stored", "session_id": session_id, "focus_index": focus_index}), 200

    except Exception as e:
        logger.error(f"Unexpected error in calculate_focus_index: {str(e)}")
        db.session.rollback()
        return jsonify({"error": "Internal server error"}), 500