from flask import Blueprint, request, jsonify
import numpy as np
import cv2
import base64
from sklearn.metrics.pairwise import cosine_similarity
from app.models import Attendance, Lecturer, db
from app.recognition.insightface_loader import face_app, student_embeddings
import logging
from firebase_admin import auth as firebase_auth

attendance_recognition_bp = Blueprint("attendance_recognition", __name__)
THRESHOLD = 0.5
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def authenticate_lecturer(id_token):
    try:
        decoded_token = firebase_auth.verify_id_token(id_token, clock_skew_seconds=60)
        lecturer_email = decoded_token.get("email")
        
        if not lecturer_email:
            return {"error": "Unauthorized", "details": "Email not found in token"}, 401
            
        lecturer = Lecturer.query.filter_by(email=lecturer_email).first()
        if not lecturer:
            return {"error": "Lecturer not found"}, 404
            
        return lecturer
    
    except firebase_auth.ExpiredIdTokenError:
        return {"error": "Unauthorized", "details": "Token has expired"}, 401
    except firebase_auth.InvalidIdTokenError:
        return {"error": "Unauthorized", "details": "Invalid token"}, 401
    except Exception as e:
        return {"error": "Unauthorized", "details": str(e)}, 401
    

def decode_image(base64_str):
    """Decode base64 image string to OpenCV format"""
    try:
        img_data = base64.b64decode(base64_str.split(",")[1])
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image")
        return img
    except Exception as e:
        logging.error(f"Image decoding error: {str(e)}")
        raise

def find_best_match(embedding, session_id):
    """Find best matching student embedding, excluding already marked students"""
    try:
        # Get already accounted student IDs
        accounted_ids = set(
            a.student_id for a in Attendance.query.filter_by(session_id=session_id, present=True).all()
        )
        
        if not student_embeddings:
            logging.warning("No student embeddings available")
            return None, 0.0, False
        
        best_id = None
        best_score = -1
        is_first_recognition = False
        
        for sid, ref_emb in student_embeddings.items():
            if sid in accounted_ids:
                continue  # Skip already marked students
            
            try:
                score = cosine_similarity([embedding], [ref_emb])[0][0]
                if score > best_score:
                    best_id = sid
                    best_score = score
                    is_first_recognition = True  # This will be the first time we see this student
            except Exception as e:
                logging.error(f"Error computing similarity for student {sid}: {str(e)}")
                continue
        
        return best_id, best_score, is_first_recognition
    except Exception as e:
        logging.error(f"Error in find_best_match: {str(e)}")
        return None, 0.0, False

def mark_attendance(student_id, session_id):
    """Mark attendance for a student in a session"""
    try:
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
        return True
    except Exception as e:
        logging.error(f"Error marking attendance for student {student_id}: {str(e)}")
        db.session.rollback()
        return False

def extract_face_image(img, bbox):
    """Extract and encode face crop from image"""
    try:
        x1, y1, x2, y2 = [int(i) for i in bbox]
        
        # Validate bbox coordinates
        h, w = img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Invalid bounding box coordinates")
        
        face_crop = img[y1:y2, x1:x2]
        success, buffer = cv2.imencode(".jpg", face_crop)
        
        if not success:
            raise ValueError("Failed to encode face image")
        
        face_base64 = base64.b64encode(buffer).decode("utf-8")
        return f"data:image/jpeg;base64,{face_base64}"
    except Exception as e:
        logging.error(f"Error extracting face image: {str(e)}")
        return None

@attendance_recognition_bp.route("/api/attendance/mark-by-face", methods=["POST", "OPTIONS"])
def mark_by_face():
    """Main endpoint for facial recognition attendance marking"""
    
    # Handle CORS preflight
    if request.method == "OPTIONS":
        response = jsonify({"message": "OK"})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type, Authorization")
        response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        return response
    
    try:
        # Validate request data
        if not request.json:
            return jsonify({"error": "No JSON data provided"}), 400
        
        data = request.json
        session_id = data.get("session_id")
        reset = data.get("reset", False)
        image_data = data.get("image")
        
        if not session_id:
            return jsonify({"error": "session_id is required"}), 400
        
        if not image_data:
            return jsonify({"error": "image data is required"}), 400
        
        # Handle reset functionality
        if reset:
            try:
                Attendance.query.filter_by(session_id=session_id).update({Attendance.present: False})
                db.session.commit()
                logging.info(f"Reset attendance for session {session_id}")
            except Exception as e:
                logging.error(f"Error resetting attendance: {str(e)}")
                db.session.rollback()
                return jsonify({"error": "Failed to reset attendance"}), 500
        
        # Decode and process image
        try:
            img = decode_image(image_data)
        except Exception as e:
            return jsonify({"error": "Invalid image data"}), 400
        
        # Detect faces
        try:
            faces = face_app.get(img)
        except Exception as e:
            logging.error(f"Face detection error: {str(e)}")
            return jsonify({"error": "Face detection failed"}), 500
        
        if not faces:
            return jsonify({"status": "no_faces_detected", "allAccounted": False}), 200
        
        # Process detected faces
        results = []
        for i, face in enumerate(faces):
            try:
                emb = face.embedding
                matched_id, score, is_first_recognition = find_best_match(emb, session_id)
                
                # Only process if this is a new recognition above threshold
                if matched_id and score >= THRESHOLD and is_first_recognition:
                    result = {
                        "student_id": matched_id,
                        "score": float(score)
                    }
                    
                    # Extract face image for first recognition
                    face_data_url = extract_face_image(img, face.bbox)
                    if face_data_url:
                        result["face"] = face_data_url
                    
                    # Mark attendance
                    if mark_attendance(matched_id, session_id):
                        results.append(result)
                        logging.info(f"First recognition - Marked attendance for student {matched_id} with score {score:.3f}")
                    else:
                        logging.error(f"Failed to mark attendance for student {matched_id}")
                elif matched_id and score >= THRESHOLD:
                    # Student already recognized, skip silently
                    logging.debug(f"Student {matched_id} already recognized, skipping (score: {score:.3f})")
                
            except Exception as e:
                logging.error(f"Error processing face {i}: {str(e)}")
                continue
        
        # Calculate attendance statistics
        try:
            total_students = (
                db.session.query(Attendance).filter_by(session_id=session_id)
                .distinct(Attendance.student_id).count()
            )
            total_present = (
                db.session.query(Attendance).filter_by(session_id=session_id, present=True)
                .distinct(Attendance.student_id).count()
            )
            all_accounted = total_present == total_students and total_students > 0
        except Exception as e:
            logging.error(f"Error calculating attendance stats: {str(e)}")
            total_students = total_present = 0
            all_accounted = False
        
        return jsonify({
            "status": "ok",
            "matches": results,
            "allAccounted": all_accounted,
            "stats": {
                "total_students": total_students,
                "total_present": total_present
            }
        }), 200
        
    except Exception as e:
        logging.error(f"Unexpected error in mark_by_face: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500
    
# @attendance_recognition_bp.route("/api/attendance/close-sheet/<session_id>", methods=["POST", "OPTIONS"])
# def close_sheet(session_id):
#     if request.method == "OPTIONS":
#         return jsonify({"message": "OK"}), 200, {
#             "Access-Control-Allow-Origin": "*",
#             "Access-Control-Allow-Headers": "Content-Type, Authorization",
#             "Access-Control-Allow-Methods": "POST, OPTIONS"
#         }

#     try:
#         id_token = request.headers.get('Authorization', '').replace('Bearer ', '')
#         lecturer = authenticate_lecturer(id_token)
#         if isinstance(lecturer, tuple):
#             return jsonify(lecturer[0]), lecturer[1]

#         session = Attendance.query.filter_by(session_id=session_id).first()
#         if not session:
#             return jsonify({"error": "Session not found"}), 404
#         session.closed = True
#         session.session_ended_at = db.func.now()
#         db.session.commit()

#         calculate_focus_index.delay(session.attendance_id)
#         logger.info(f"Closed session {session_id}, queued focus index calculation")
#         return jsonify({"message": "Session closed, focus index calculation queued"}), 200
#     except Exception as e:
#         logger.error(f"Error in close_sheet: {str(e)}")
#         return jsonify({"error": "Internal server error"}), 500