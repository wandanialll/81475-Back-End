import os
import uuid
from flask import Blueprint, jsonify, request, Response
import requests
from werkzeug.utils import secure_filename
from app.models import Course, Enrollment, Student, Lecturer, Attendance, StudentPhoto
from app.db import db
import firebase_admin
from firebase_admin import auth as firebase_auth, credentials
from google.generativeai import types
from flask_cors import CORS
from sqlalchemy import func
from sqlalchemy.sql import case
from sqlalchemy.orm import joinedload
import json
import pickle
import numpy as np
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from insightface.app import FaceAnalysis
import base64
import cv2
from app.recognition.insightface_loader import face_app, student_embeddings
import google.generativeai as genai
from typing import Union, Tuple

# credentials certificate using pythondotenv
from dotenv import load_dotenv
load_dotenv()

credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

# Initialize the model
model = genai.GenerativeModel('gemini-1.5-flash')


# Initialize Firebase only once
cred = credentials.Certificate(credentials_path)
firebase_admin.initialize_app(cred)

api = Blueprint('api', __name__)

  # Allow all origins (change to specific origin in production)
#CORS(api, resources={r"/api/*": {"origins": "*"}})



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

# api for main lecturer dashboard
@api.route("/api/lecturer/dashboard", methods=["GET", "OPTIONS"])
def get_lecturer_dashboard():
    if request.method == "OPTIONS":
        response = jsonify({"message": "OK"})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET, OPTIONS')
        return response

    id_token = request.headers.get("Authorization", "").replace("Bearer ", "")
    try:
        decoded_token = firebase_auth.verify_id_token(id_token, clock_skew_seconds=60)
        lecturer_email = decoded_token["email"]
    except Exception as e:
        return jsonify({"error": "Unauthorized", "details": str(e)}), 401

    lecturer = Lecturer.query.filter_by(email=lecturer_email).first()
    if not lecturer:
        return jsonify({"error": "Lecturer not found"}), 404

    # Track unique students across all courses
    unique_student_ids = set()
    overall_total_present = 0
    overall_total_absents = 0
    overall_total_alerts = 0

    course_data = []

    for course in lecturer.courses:
        enrolled_students = course.enrollments
        course_total_students = len(enrolled_students)
        present_count = 0
        alerts = 0

        student_info = []
        for enrollment in enrolled_students:
            student = enrollment.student
            unique_student_ids.add(student.student_id)  # Count student only once

            attendances = [a for a in student.attendance_records if a.course_id == course.course_id]

            was_present = any(a.present for a in attendances)
            absence_count = sum(1 for a in attendances if not a.present)

            present_status = "present" if was_present else "absent"
            last_seen = max([a.timestamp.strftime("%Y-%m-%d %H:%M") for a in attendances], default="N/A")

            if present_status == "present":
                present_count += 1

            if absence_count >= 3:  # Customize alert rule here
                alerts += 1

            student_info.append({
                "studentId": student.student_id,
                "name": student.name,
                "status": present_status,
                "lastSeen": last_seen,
                "absences": absence_count
            })

        absents = course_total_students - present_count
        overall_total_present += present_count
        overall_total_absents += absents
        overall_total_alerts += alerts

        course_data.append({
            "course_id": course.course_id,
            "name": course.name,
            "totalStudents": course_total_students,
            "totalPresent": present_count,
            "totalAbsents": absents,
            "alerts": alerts,
            "students": student_info
        })

    return jsonify({
        "totalUniqueStudents": len(unique_student_ids),
        "totalPresent": overall_total_present,
        "totalAbsents": overall_total_absents,
        "totalAlerts": overall_total_alerts,
        "courses": course_data
    })

# gemini  interaction
def query_gemini(query):
    import re
    api_key = os.getenv("GEMINI_API_KEY")
    gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": query}]
            }
        ],
        "generationConfig": {
            "maxOutputTokens": 100,
            "responseMimeType": "text/plain"
        },
        "systemInstruction": {
            "parts": [{
                "text": (
                    "Route all requests to either: \ncommands: (\n"
                    "\"create attendance\": {\"type\": \"navigation\", \"label\": \"Create Attendance Page\", \"route\": \"/create-attendance\"},\n"
                    "\"view attendance\": {\"type\": \"navigation\", \"label\": \"View Attendance\", \"route\": \"/attendance/view\"},\n"
                    "\"add new student\": {\"type\": \"navigation\", \"label\": \"Add New Student\", \"route\": \"/add-student\"},\n"
                    "\"view students\": {\"type\": \"navigation\", \"label\": \"View Students\", \"route\": \"/students\"},\n"
                    "\"view courses\": {\"type\": \"navigation\", \"label\": \"View Courses\", \"route\": \"/courses\"}\n"
                    ")\n\n"
                    "Return response in two-part JSON:\n{\n  \"uiRespond\": (short simple message confirming),\n  \"backendRespond\": (command title)\n}\n\n"
                    "If user request does not fit any of the commands, return a simple message and code [NO_VALID_REQUEST].\n"
                    "IMPORTANT: Do NOT format output as markdown. Return raw JSON only.\n"
                    "IMPORTANT: Only respond in English."
                )
            }]
        }
    }

    headers = {
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(gemini_url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()

        text = data["candidates"][0]["content"]["parts"][0]["text"]

        # Strip markdown code blocks if wrapped in ```json
        match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()

        try:
            parsed = json.loads(text)
            return {
                "type": "llm-response",
                "uiRespond": parsed.get("uiRespond", ""),
                "backendRespond": parsed.get("backendRespond", "")
            }
        except json.JSONDecodeError:
            return {
                "type": "llm-response",
                "uiRespond": text,
                "backendRespond": ""
            }

    except Exception as e:
        return {
            "type": "llm-error",
            "label": f"LLM Error: {str(e)}"
        }


@api.route("/api/search", methods=["GET", "OPTIONS"])
def search():
    if request.method == "OPTIONS":
        response = jsonify({"message": "OK"})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        response.headers.add("Access-Control-Allow-Methods", "GET, OPTIONS")
        return response

    id_token = request.headers.get("Authorization", "").replace("Bearer ", "")
    try:
        decoded_token = firebase_auth.verify_id_token(id_token, clock_skew_seconds=60)
        lecturer_email = decoded_token["email"]
    except Exception as e:
        return jsonify({"error": "Unauthorized", "details": str(e)}), 401

    lecturer = Lecturer.query.filter_by(email=lecturer_email).first()
    if not lecturer:
        print("Lecturer not found:", lecturer_email)
        return jsonify([])

    query = request.args.get("q", "").strip().lower()
    print("Query received:", query)

    if not query:
        return jsonify([])

    results = []

    command_map = {
        "create attendance": {"type": "navigation", "label": "Create Attendance Page", "route": "/create-attendance"},
        "view attendance": {"type": "navigation", "label": "View Attendance", "route": "/attendance/view"},
        "add new student": {"type": "navigation", "label": "Add New Student", "route": "/add-student"},
        "view students": {"type": "navigation", "label": "View Students", "route": "/students"},
        "view courses": {"type": "navigation", "label": "View Courses", "route": "/courses"},
    }

    # Match commands
    for cmd, val in command_map.items():
        if query in cmd.lower():
            results.append(val)

    # Match course names
    courses = [course for course in lecturer.courses if query in course.name.lower()]
    results.extend([
        {
            "type": "course",
            "course_id": course.course_id,
            "label": course.name,
            "route": f"/course/{course.course_id}/dashboard"
        }
        for course in courses
    ])

    # Fallback to LLM
    if not results:
        llm_result = query_gemini(query)
        if llm_result["type"] == "llm-response":
            backend_cmd = llm_result.get("backendRespond", "").lower()
            if backend_cmd in command_map:
                results.append(command_map[backend_cmd])
            else:
                results.append({
                    "type": "llm-response",
                    "label": llm_result.get("uiRespond", "No UI response"),
                    "llm_backend": backend_cmd
                })
        else:
            return jsonify(results if isinstance(results, list) else [results])

    print("Search results:", results)

    response = jsonify(results)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
    response.headers.add("Access-Control-Allow-Methods", "GET, OPTIONS")
    return response

@api.route("/api/course/<int:course_id>/dashboard", methods=["GET"])
def get_course_dashboard(course_id):
    id_token = request.headers.get("Authorization", "").replace("Bearer ", "")
    try:
        decoded_token = firebase_auth.verify_id_token(id_token, clock_skew_seconds=60)
        lecturer_email = decoded_token["email"]
    except Exception as e:
        return jsonify({"error": "Unauthorized", "details": str(e)}), 401

    lecturer = Lecturer.query.filter_by(email=lecturer_email).first()
    if not lecturer:
        return jsonify({"error": "Lecturer not found"}), 404

    course = Course.query.filter_by(course_id=course_id, lecturer_id=lecturer.lecturer_id).first()
    if not course:
        return jsonify({"error": "Course not found"}), 404

    enrolled_students = [
        {
            "studentId": e.student.student_id,
            "name": e.student.name,
            "status": "present" if any(
                a.present for a in e.student.attendance_records if a.course_id == course.course_id
            ) else "absent",
            "lastSeen": max([a.timestamp.strftime("%Y-%m-%d %H:%M") for a in e.student.attendance_records if a.course_id == course.course_id], default="N/A")
        }
        for e in course.enrollments
    ]

    total_students = len(enrolled_students)
    total_present = sum(1 for s in enrolled_students if s["status"] == "present")
    total_absents = total_students - total_present

    # You can define your own logic for alerts; here we just simulate it
    alerts = sum(1 for s in enrolled_students if s["status"] == "absent")

    return jsonify({
        "name": course.name,
        "totalStudents": total_students,
        "totalPresent": total_present,
        "totalAbsents": total_absents,
        "alerts": alerts,
        "students": enrolled_students
    })

@api.route("/api/attendance/create-sheet", methods=["POST", "OPTIONS"])
def create_attendance_sheet():
    if request.method == "OPTIONS":
        response = jsonify({"message": "OK"})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        return response
     
    id_token = request.headers.get("Authorization", "").replace("Bearer ", "")
    try:
        decoded_token = firebase_auth.verify_id_token(id_token, clock_skew_seconds=60)
        lecturer_email = decoded_token["email"]
    except Exception as e:
        return jsonify({"error": "Unauthorized", "details": str(e)}), 401

    data = request.get_json()
    course_id = data.get("course_id")
    session_id = str(uuid.uuid4())

    lecturer = Lecturer.query.filter_by(email=lecturer_email).first()
    if not lecturer:
        return jsonify({"error": "Lecturer not found"}), 404

    course = Course.query.filter_by(course_id=course_id, lecturer_id=lecturer.lecturer_id).first()
    if not course:
        return jsonify({"error": "Course not found or unauthorized"}), 403

    students = [e.student for e in course.enrollments]

    for student in students:
        attendance = Attendance(
            student_id=student.student_id,
            course_id=course.course_id,
            session_id=session_id,
            present=False,  # initially marked absent
            closed=False  # attendance sheet is not closed yet
        )
        db.session.add(attendance)

    db.session.commit()
    return jsonify({"message": f"Attendance sheet created for {len(students)} students."}), 201

@api.route("/api/course/<int:course_id>/attendance/open-sheets", methods=["GET", "OPTIONS"])
def get_open_sheets(course_id):
    if request.method == "OPTIONS":
        response = jsonify({"message": "OK"})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        return response
    
    id_token = request.headers.get("Authorization", "").replace("Bearer ", "")
    try:
        decoded_token = firebase_auth.verify_id_token(id_token, clock_skew_seconds=60)
        lecturer_email = decoded_token["email"]
    except Exception as e:
        return jsonify({"error": "Unauthorized", "details": str(e)}), 401

    lecturer = Lecturer.query.filter_by(email=lecturer_email).first()
    if not lecturer:
        return jsonify({"error": "Lecturer not found"}), 404
    
    sheets = db.session.query(Attendance.session_id, db.func.min(Attendance.timestamp).label("started_at")) \
        .filter_by(course_id=course_id, closed=False) \
        .group_by(Attendance.session_id) \
        .all()

    return jsonify([
        {"session_id": s.session_id, "started_at": s.started_at.isoformat()}
        for s in sheets
    ])

# get attendance sheet by session_id
@api.route("/api/attendance/sheet/<session_id>", methods=["GET", "OPTIONS"])
def get_attendance_sheet(session_id):
    if request.method == "OPTIONS":
        response = jsonify({"message": "OK"})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET, OPTIONS')
        return response

    id_token = request.headers.get("Authorization", "").replace("Bearer ", "")
    try:
        decoded_token = firebase_auth.verify_id_token(id_token, clock_skew_seconds=60)
        lecturer_email = decoded_token["email"]
    except Exception as e:
        return jsonify({"error": "Unauthorized", "details": str(e)}), 401

    lecturer = Lecturer.query.filter_by(email=lecturer_email).first()
    if not lecturer:
        return jsonify({"error": "Lecturer not found"}), 404

    attendance_records = Attendance.query.filter_by(session_id=session_id).all()
    if not attendance_records:
        return jsonify({"error": "No attendance records found for this session"}), 404

    records = [
        {
            "student_id": record.student_id,
            "present": record.present,
            "timestamp": record.timestamp.isoformat()
        }
        for record in attendance_records
    ]

    return jsonify(records)
    
@api.route("/api/attendance/close-sheet", methods=["POST", "OPTIONS"])
def close_attendance_sheet():
    if request.method == "OPTIONS":
        response = jsonify({"message": "OK"})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        return response

    id_token = request.headers.get("Authorization", "").replace("Bearer ", "")
    try:
        decoded_token = firebase_auth.verify_id_token(id_token, clock_skew_seconds=60)
        lecturer_email = decoded_token["email"]
    except Exception as e:
        return jsonify({"error": "Unauthorized", "details": str(e)}), 401

    lecturer = Lecturer.query.filter_by(email=lecturer_email).first()
    if not lecturer:
        return jsonify({"error": "Lecturer not found"}), 404
    
    data = request.get_json()
    session_id = data.get("session_id")

    records = Attendance.query.filter_by(session_id=session_id, closed=False).all()
    for record in records:
        record.closed = True

    db.session.commit()
    return jsonify({"message": f"Closed sheet {session_id}"}), 200

@api.route("/api/enroll", methods=["POST", "OPTIONS"])
def enroll_student():
    if request.method == "OPTIONS":
        response = jsonify({"message": "OK"})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type, Authorization")
        response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        return response

    name = request.form.get("name")
    photos = request.files.getlist("photo")

    if not name:
        return jsonify({"error": "Name is required"}), 400
    if len(photos) != 3:
        return jsonify({"error": "Exactly 3 photos are required"}), 400

    # Create student
    new_student = Student(name=name)
    db.session.add(new_student)
    db.session.commit()  # Must commit here to get student_id

    # Save photos to DB
    for idx, photo in enumerate(photos, start=1):
        filename = secure_filename(f"{name}_{idx}.jpg")
        new_photo = StudentPhoto(
            student_id=new_student.student_id,
            image_data=photo.read(),
            filename=filename,
            mimetype=photo.mimetype
        )
        db.session.add(new_photo)

    db.session.commit()

    return jsonify({
        "success": True,
        "message": "Enrollment successful",
        "student_id": new_student.student_id,
        "photos_saved": 3
    }), 200

@api.route("/api/photo/<int:photo_id>")
def serve_photo(photo_id):
    photo = StudentPhoto.query.get_or_404(photo_id)
    return Response(photo.image_data, mimetype=photo.mimetype)

# overall attendance for lecturer
@api.route('/api/lecturer/attendance/overall-sessions', methods=['GET', 'OPTIONS'])
def lecturer_attendance_performance():
    if request.method == "OPTIONS":
        response = jsonify({"message": "OK"})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type, Authorization")
        response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        return response

    id_token = request.headers.get("Authorization", "").replace("Bearer ", "")
    try:
        decoded_token = firebase_auth.verify_id_token(id_token, clock_skew_seconds=60)
        lecturer_email = decoded_token["email"]
    except Exception as e:
        return jsonify({"error": "Unauthorized", "details": str(e)}), 401

    lecturer = Lecturer.query.filter_by(email=lecturer_email).first()
    if not lecturer:
        return jsonify({"error": "Lecturer not found"}), 404

    course_ids = [c.course_id for c in lecturer.courses]

    if not course_ids:
        return jsonify({
            "lecturer": lecturer.name,
            "sessions": []
        })

    present_count_expr = func.sum(
    case((Attendance.present, 1), else_=0)
    )

    session_stats = (
        db.session.query(
            Attendance.session_id,
            present_count_expr.label("present_count"),
            func.count(Attendance.attendance_id).label("total_count")
        )
        .filter(Attendance.course_id.in_(course_ids))
        .group_by(Attendance.session_id, Attendance.session_created_at)
        .order_by(Attendance.session_created_at.asc())
        .all()
    )

    sessions = []
    for session in session_stats:
        sessions.append({
            "sessionId": session.session_id,
            "present": session.present_count,
            "absent": session.total_count - session.present_count,
        })

    return jsonify({
        "lecturer": lecturer.name,
        "sessions": sessions,
        "sessionId": session.session_id,
    })

@api.route("/api/chat", methods=["POST", "OPTIONS"])
def gemini_chat():
    # Initialize response
    response = None
    
    if request.method == "OPTIONS":
        response = jsonify({"message": "OK"})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type, Authorization")
        response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        return response
    
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            response = jsonify({"error": "No data provided"})
            response.headers.add("Access-Control-Allow-Origin", "*")
            return response, 400
        
        message = data.get("message")
        if not message:
            response = jsonify({"error": "Message is required"})
            response.headers.add("Access-Control-Allow-Origin", "*")
            return response, 400
        
        # Optional: Get conversation history for context
        history = data.get("history", [])
        
        # Prepare the prompt with history if provided
        if history:
            context_messages = []
            for msg in history[-10:]:  # Last 10 messages for context
                role = "Human" if msg.get("role") == "user" else "Assistant" 
                context_messages.append(f"{role}: {msg.get('content', '')}")
            
            full_prompt = "\n".join(context_messages) + f"\nHuman: {message}\nAssistant:"
        else:
            full_prompt = message
        
        # Generate response from Gemini
        gemini_response = model.generate_content(full_prompt)
        
        if not gemini_response.text:
            response = jsonify({"error": "No response generated"})
            response.headers.add("Access-Control-Allow-Origin", "*")
            return response, 500
        
        response = jsonify({
            "success": True,
            "response": gemini_response.text,
            "message": "Response generated successfully"
        })
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response, 200
        
    except Exception as e:
        response = jsonify({
            "error": "Failed to generate response",
            "details": str(e)
        })
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response, 500


@api.route("/api/student-details/<int:student_id>", methods=["GET", "OPTIONS"])
def get_student_details(student_id: int):
    # Initialize response
    response = None
    
    if request.method == "OPTIONS":
        response = jsonify({"message": "OK"})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type, Authorization")
        response.headers.add("Access-Control-Allow-Methods", "GET, OPTIONS")
        return response

    # Validate student_id
    if student_id <= 0:
        response = jsonify({"error": "Invalid student ID"})
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response, 400

    # Authenticate lecturer
    id_token = request.headers.get("Authorization", "").replace("Bearer ", "")
    try:
        decoded_token = firebase_auth.verify_id_token(id_token, clock_skew_seconds=60)
        lecturer_email = decoded_token.get("email")
        if not lecturer_email:
            response = jsonify({"error": "Unauthorized", "details": "Email not found in token"})
            response.headers.add("Access-Control-Allow-Origin", "*")
            return response, 401
    except firebase_auth.ExpiredIdTokenError:
        response = jsonify({"error": "Unauthorized", "details": "Token has expired"})
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response, 401
    except firebase_auth.InvalidIdTokenError:
        response = jsonify({"error": "Unauthorized", "details": "Invalid token"})
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response, 401
    except Exception as e:
        response = jsonify({"error": "Unauthorized", "details": str(e)})
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response, 401

    # Fetch lecturer
    lecturer = Lecturer.query.filter_by(email=lecturer_email).first()
    if not lecturer:
        response = jsonify({"error": "Lecturer not found"})
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response, 404

    # Fetch student with eager loading
    try:
        student = Student.query.options(
            joinedload(Student.enrollments).joinedload(Enrollment.course),
            joinedload(Student.photos)
        ).get(student_id)
        if not student:
            response = jsonify({"error": "Student not found"})
            response.headers.add("Access-Control-Allow-Origin", "*")
            return response, 404
    except Exception as e:
        response = jsonify({"error": "Database error", "details": str(e)})
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response, 500

    # Authorization check: Ensure student is enrolled in a course taught by the lecturer
    lecturer_course_ids = {course.course_id for course in lecturer.courses}
    student_course_ids = {enrollment.course.course_id for enrollment in student.enrollments if enrollment.course}
    if not lecturer_course_ids & student_course_ids:  # No intersection
        response = jsonify({"error": "Forbidden", "details": "No access to this student’s data"})
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response, 403

    # Fetch attendance records
    attendance_records = Attendance.query.filter_by(student_id=student.student_id).all()

    # Format attendance records
    records = [
        {
            "courseId": record.course_id,
            "sessionId": record.session_id,
            "present": record.present,
            "timestamp": record.timestamp.isoformat()
        }
        for record in attendance_records
    ]

    # Format student photos
    photos = [
        {
            "photoId": photo.photo_id,
            "filename": photo.filename,
            "mimetype": photo.mimetype,
            "capturedAt": photo.captured_at.isoformat(),
            "imageData": photo.image_data.hex()  # Convert binary to hex string for JSON
        }
        for photo in student.photos
    ]

    response = jsonify({
        "studentId": student.student_id,
        "name": student.name,
        "courses": [
            {
                "courseId": enrollment.course.course_id,
                "name": enrollment.course.name
            }
            for enrollment in student.enrollments
            if enrollment.course
        ],
        "attendanceRecords": records,
        "photos": photos
    })
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response, 200

# @api.route("/api/student-details/<int:student_id>", methods=["GET", "OPTIONS"])
# def get_student_details(student_id: int) -> Union[Tuple[dict, int], dict]:
#     if request.method == "OPTIONS":
#         response = jsonify({"message": "OK"})
#         response.headers.add("Access-Control-Allow-Origin", "*")
#         response.headers.add("Access-Control-Allow-Headers", "Content-Type, Authorization")
#         response.headers.add("Access-Control-Allow-Methods", "GET, OPTIONS")
#         return response

#     # Validate student_id
#     if student_id <= 0:
#         return jsonify({"error": "Invalid student ID"}), 400

#     # Authenticate lecturer
#     id_token = request.headers.get("Authorization", "").replace("Bearer ", "")
#     try:
#         decoded_token = firebase_auth.verify_id_token(id_token, clock_skew_seconds=60)
#         lecturer_email = decoded_token.get("email")
#         if not lecturer_email:
#             return jsonify({"error": "Unauthorized", "details": "Email not found in token"}), 401
#     except firebase_auth.ExpiredIdTokenError:
#         return jsonify({"error": "Unauthorized", "details": "Token has expired"}), 401
#     except firebase_auth.InvalidIdTokenError:
#         return jsonify({"error": "Unauthorized", "details": "Invalid token"}), 401
#     except Exception as e:
#         return jsonify({"error": "Unauthorized", "details": str(e)}), 401

#     # Fetch lecturer
#     lecturer = Lecturer.query.filter_by(email=lecturer_email).first()
#     if not lecturer:
#         return jsonify({"error": "Lecturer not found"}), 404

#     # Fetch student with eager loading
#     try:
#         student = Student.query.options(
#             joinedload(Student.enrollments).joinedload(Enrollment.course),
#             joinedload(Student.photos)
#         ).get(student_id)
#         if not student:
#             return jsonify({"error": "Student not found"}), 404
#     except Exception as e:
#         return jsonify({"error": "Database error", "details": str(e)}), 500

#     # Authorization check: Ensure student is enrolled in a course taught by the lecturer
#     lecturer_course_ids = {course.course_id for course in lecturer.courses}
#     student_course_ids = {enrollment.course.course_id for enrollment in student.enrollments if enrollment.course}
#     if not lecturer_course_ids & student_course_ids:  # No intersection
#         return jsonify({"error": "Forbidden", "details": "No access to this student’s data"}), 403

#     # Fetch attendance records
#     attendance_records = Attendance.query.filter_by(student_id=student.student_id).all()

#     # Format attendance records
#     records = [
#         {
#             "courseId": record.course_id,
#             "sessionId": record.session_id,
#             "present": record.present,
#             "timestamp": record.timestamp.isoformat()
#         }
#         for record in attendance_records
#     ]

#     # Format student photos
#     photos = [
#         {
#             "photoId": photo.photo_id,
#             "filename": photo.filename,
#             "mimetype": photo.mimetype,
#             "capturedAt": photo.captured_at.isoformat(),
#             "imageData": photo.image_data.hex()  # Convert binary to hex string for JSON
#         }
#         for photo in student.photos
#     ]

#     return jsonify({
#         "studentId": student.student_id,
#         "name": student.name,
#         "courses": [
#             {
#                 "courseId": enrollment.course.course_id,
#                 "name": enrollment.course.name
#             }
#             for enrollment in student.enrollments
#             if enrollment.course
#         ],
#         "attendanceRecords": records,
#         "photos": photos
#     }), 200