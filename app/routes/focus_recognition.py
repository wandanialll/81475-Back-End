from flask import Blueprint, request, jsonify
from models import Attendance, FocusSessions, FocusLabels, db
from tasks import process_focus_frame, calculate_focus_index
import logging
from attendance_recognition import authenticate_lecturer

focus_bp = Blueprint("focus", __name__)
logger = logging.getLogger(__name__)



@focus_bp.route("/api/focus/process-frame", methods=["POST", "OPTIONS"])
def process_frame():
    if request.method == "OPTIONS":
        return jsonify({"message": "OK"}), 200, {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
            "Access-Control-Allow-Methods": "POST, OPTIONS"
        }

    try:
        data = request.json
        session_id = data.get("session_id")
        image_data = data.get("image")
        student_id = data.get("student_id")  # Optional

        if not session_id or not image_data:
            return jsonify({"error": "session_id and image are required"}), 400

        session = Attendance.query.filter_by(session_id=session_id, closed=False).first()
        if not session:
            return jsonify({"error": "Invalid or closed session"}), 404

        process_focus_frame.delay(session.attendance_id, image_data, student_id)
        return jsonify({"status": "queued"}), 202
    except Exception as e:
        logger.error(f"Error in process_frame: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@focus_bp.route("/api/focus/session/<session_id>", methods=["GET", "OPTIONS"])
def get_focus_index(session_id):
    if request.method == "OPTIONS":
        return jsonify({"message": "OK"}), 200, {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
            "Access-Control-Allow-Methods": "GET, OPTIONS"
        }

    try:
        id_token = request.headers.get('Authorization', '').replace('Bearer ', '')
        lecturer = authenticate_lecturer(id_token)
        if isinstance(lecturer, tuple):
            return jsonify(lecturer[0]), lecturer[1]

        session = Attendance.query.filter_by(session_id=session_id).first()
        if not session:
            return jsonify({"error": "Session not found"}), 404

        focus_session = FocusSessions.query.filter_by(attendance_id=session.attendance_id).first()
        if not focus_session or not focus_session.closed:
            return jsonify({"status": "pending"}), 200

        return jsonify({
            "status": "complete",
            "session_id": session.session_id,
            "focus_index": focus_session.focus_index,
            "timestamp": focus_session.timestamp.isoformat()
        }), 200
    except Exception as e:
        logger.error(f"Error in get_focus_index: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500