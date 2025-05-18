from flask import Blueprint, request, jsonify
import numpy as np
import cv2
import base64
from sklearn.metrics.pairwise import cosine_similarity
from app.models import Attendance, db
from app.recognition.insightface_loader import face_app, student_embeddings

attendance_recognition_bp = Blueprint("attendance_recognition", __name__)

THRESHOLD = 0.5  # You can adjust this as needed

def decode_image(base64_str):
    img_data = base64.b64decode(base64_str.split(",")[1])
    nparr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def find_best_match(embedding):
    best_id = None
    best_score = -1
    for sid, ref_emb in student_embeddings.items():
        score = cosine_similarity([embedding], [ref_emb])[0][0]
        if score > best_score:
            best_id = sid
            best_score = score
    return best_id, best_score

def mark_attendance(student_id, session_id):
    existing = Attendance.query.filter_by(student_id=student_id, session_id=session_id).first()
    if not existing:
        record = Attendance(
            student_id=student_id,
            session_id=session_id,
            present=True
        )
        db.session.add(record)
    else:
        existing.present = True
    db.session.commit()

@attendance_recognition_bp.route("/api/attendance/mark-by-face", methods=["POST", "OPTIONS"])
def mark_by_face():
    if request.method == "OPTIONS":
        response = jsonify({"message": "OK"})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type, Authorization")
        response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        return response
    
    data = request.json
    session_id = data.get("session_id")
    img = decode_image(data.get("image"))

    faces = face_app.get(img)
    if not faces:
        return jsonify({"status": "no_faces_detected", "allAccounted": False}), 200

    results = []

    for face in faces:
        emb = face.embedding
        matched_id, score = find_best_match(emb)
        if score >= THRESHOLD:
            mark_attendance(matched_id, session_id)
            results.append({"student_id": matched_id, "score": float(score)})
    
    total_students = (
        db.session.query(Attendance).filter_by(session_id=session_id)
        .distinct(Attendance.student_id).count()
    )

    total_present = (
        db.session.query(Attendance).filter_by(session_id=session_id, present=True)
        .distinct(Attendance.student_id).count()
    )

    all_accounted = total_present == total_students

    return jsonify({"status": "ok", "matches": results, "allAccounted": all_accounted}), 200
