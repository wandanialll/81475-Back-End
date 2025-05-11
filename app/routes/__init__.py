from flask import Blueprint, jsonify, request
from app.models import Course, Enrollment, Student, Lecturer
from app.db import db
import firebase_admin
from firebase_admin import auth as firebase_auth, credentials

# Initialize Firebase only once
cred = credentials.Certificate("C:\\Users\\dania\\Desktop\\siBijak\\wanmuhammaddanial81475-firebase-adminsdk-fbsvc-c09ba41e29.json")
firebase_admin.initialize_app(cred)

api = Blueprint('api', __name__)

# Handle the OPTIONS request manually for preflight
@api.route('/api/lecturer/courses', methods=["OPTIONS"])
def options_lecturer_courses():
    response = jsonify({"message": "OK"})
    response.headers.add('Access-Control-Allow-Origin', '*')  # Allow all origins (change to specific origin in production)
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
    return response

@api.route("/api/lecturer/courses", methods=["GET"])
def get_lecturer_courses():
    id_token = request.headers.get("Authorization", "").replace("Bearer ", "")
    print("Received Token:", id_token)
    try:
        decoded_token = firebase_auth.verify_id_token(id_token, clock_skew_seconds=60)
        print("Decoded Token:", decoded_token)
        lecturer_email = decoded_token["email"]
    except Exception as e:
        print(f"Error decoding token: {e}")
        return jsonify({"error": "Unauthorized", "details": str(e)}), 401

    # Query the lecturer by email
    lecturer = Lecturer.query.filter_by(email=lecturer_email).first()

    if not lecturer:
        return jsonify([])

    # Debugging: Print out courses for lecturer
    print("Courses for lecturer:", [course.name for course in lecturer.courses])

    # Ensure we are only returning courses associated with the lecturer
    courses = [
        {"course_id": c.course_id, "name": c.name}
        for c in lecturer.courses
    ]

    # Explicitly add CORS headers in the response
    response = jsonify(courses)
    response.headers.add('Access-Control-Allow-Origin', '*')  # Allow all origins (change to specific origin in production)
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')

    return response
