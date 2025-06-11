import os
import uuid
import json
import pickle
import base64
import numpy as np
from datetime import datetime
from typing import Union, Tuple, Dict, List, Any

import cv2
import requests
from flask import Blueprint, jsonify, request, Response
from werkzeug.utils import secure_filename
from sqlalchemy import func
from sqlalchemy.sql import case
from sqlalchemy.orm import joinedload
from sklearn.metrics.pairwise import cosine_similarity
from insightface.app import FaceAnalysis
import google.generativeai as genai

# Local imports
from app.models import Course, Enrollment, Student, Lecturer, Attendance, StudentPhoto
from app.db import db
from app.recognition.insightface_loader import face_app, student_embeddings

# Firebase imports
import firebase_admin
from firebase_admin import auth as firebase_auth, credentials

# Environment setup
from dotenv import load_dotenv
load_dotenv()

# Configuration
class Config:
    GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

# Initialize services
def initialize_services():
    """Initialize Firebase and Gemini services"""
    # Firebase
    if not firebase_admin._apps:
        cred = credentials.Certificate(Config.GOOGLE_APPLICATION_CREDENTIALS)
        firebase_admin.initialize_app(cred)
    
    # Gemini
    genai.configure(api_key=Config.GEMINI_API_KEY)
    return genai.GenerativeModel('gemini-1.5-flash')

# Initialize
model = initialize_services()
api = Blueprint('api', __name__)

# Utility functions
def add_cors_headers(response):
    """Add CORS headers to response"""
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
    return response

def handle_preflight():
    """Handle OPTIONS preflight requests"""
    response = jsonify({"message": "OK"})
    return add_cors_headers(response)

def authenticate_lecturer(id_token: str) -> Union[Lecturer, Tuple[Dict, int]]:
    """Authenticate lecturer and return lecturer object or error response"""
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

def get_auth_token(request) -> str:
    """Extract auth token from request headers"""
    return request.headers.get("Authorization", "").replace("Bearer ", "")

# Gemini AI utilities
class GeminiService:
    """Service class for Gemini AI interactions"""
    
    COMMAND_MAP = {
        "create attendance": {"type": "navigation", "label": "Create Attendance Page", "route": "/create-attendance"},
        "view attendance": {"type": "navigation", "label": "View Attendance", "route": "/attendance/view"},
        "add new student": {"type": "navigation", "label": "Add New Student", "route": "/add-student"},
        "view students": {"type": "navigation", "label": "View Students", "route": "/students"},
        "view courses": {"type": "navigation", "label": "View Courses", "route": "/courses"},
    }
    
    SYSTEM_INSTRUCTION = (
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
    
    @staticmethod
    def query_gemini(query: str) -> Dict[str, Any]:
        """Query Gemini API for command routing"""
        import re
        
        payload = {
            "contents": [{"role": "user", "parts": [{"text": query}]}],
            "generationConfig": {
                "maxOutputTokens": 100,
                "responseMimeType": "text/plain"
            },
            "systemInstruction": {
                "parts": [{"text": GeminiService.SYSTEM_INSTRUCTION}]
            }
        }
        
        headers = {"Content-Type": "application/json"}
        url = f"{Config.GEMINI_API_URL}?key={Config.GEMINI_API_KEY}"
        
        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            
            # Strip markdown code blocks
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
    
    @staticmethod
    def generate_chat_response(message: str, history: List[Dict] = None) -> str:
        """Generate chat response using Gemini"""
        if history:
            context_messages = []
            for msg in history[-10:]:
                role = "Human" if msg.get("role") == "user" else "Assistant"
                context_messages.append(f"{role}: {msg.get('content', '')}")
            
            full_prompt = "\n".join(context_messages) + f"\nHuman: {message}\nAssistant:"
        else:
            full_prompt = message
        
        response = model.generate_content(full_prompt)
        return response.text

# Data processing utilities
class AttendanceProcessor:
    """Utility class for attendance-related operations"""
    
    @staticmethod
    def calculate_student_status(student: Student, course_id: int) -> Dict[str, Any]:
        """Calculate student attendance status for a course"""
        attendances = [a for a in student.attendance_records if a.course_id == course_id]
        
        was_present = any(a.present for a in attendances)
        absence_count = sum(1 for a in attendances if not a.present)
        last_seen = max([a.timestamp.strftime("%Y-%m-%d %H:%M") for a in attendances], default="N/A")
        
        return {
            "studentId": student.student_id,
            "name": student.name,
            "status": "present" if was_present else "absent",
            "lastSeen": last_seen,
            "absences": absence_count
        }
    
    @staticmethod
    def get_course_statistics(course: Course) -> Dict[str, int]:
        """Get attendance statistics for a course"""
        enrolled_students = course.enrollments
        total_students = len(enrolled_students)
        present_count = 0
        alerts = 0
        
        for enrollment in enrolled_students:
            student_info = AttendanceProcessor.calculate_student_status(
                enrollment.student, course.course_id
            )
            
            if student_info["status"] == "present":
                present_count += 1
            
            if student_info["absences"] >= 3:
                alerts += 1
        
        return {
            "totalStudents": total_students,
            "totalPresent": present_count,
            "totalAbsents": total_students - present_count,
            "alerts": alerts
        }

# API Routes
@api.route('/api/lecturer/courses', methods=["GET", "OPTIONS"])
def get_lecturer_courses():
    if request.method == "OPTIONS":
        return handle_preflight()
    
    id_token = get_auth_token(request)
    lecturer = authenticate_lecturer(id_token)
    
    if isinstance(lecturer, tuple):  # Error response
        response = jsonify(lecturer[0])
        return add_cors_headers(response), lecturer[1]
    
    courses = [
        {"course_id": c.course_id, "name": c.name}
        for c in lecturer.courses
    ]
    
    response = jsonify(courses)
    return add_cors_headers(response)

@api.route("/api/lecturer/dashboard", methods=["GET", "OPTIONS"])
def get_lecturer_dashboard():
    if request.method == "OPTIONS":
        return handle_preflight()
    
    id_token = get_auth_token(request)
    lecturer = authenticate_lecturer(id_token)
    
    if isinstance(lecturer, tuple):
        response = jsonify(lecturer[0])
        return add_cors_headers(response), lecturer[1]
    
    unique_student_ids = set()
    overall_stats = {"present": 0, "absents": 0, "alerts": 0}
    course_data = []
    
    for course in lecturer.courses:
        course_stats = AttendanceProcessor.get_course_statistics(course)
        student_info = []
        
        for enrollment in course.enrollments:
            student = enrollment.student
            unique_student_ids.add(student.student_id)
            
            student_data = AttendanceProcessor.calculate_student_status(student, course.course_id)
            student_info.append(student_data)
        
        overall_stats["present"] += course_stats["totalPresent"]
        overall_stats["absents"] += course_stats["totalAbsents"]
        overall_stats["alerts"] += course_stats["alerts"]
        
        course_data.append({
            "course_id": course.course_id,
            "name": course.name,
            **course_stats,
            "students": student_info
        })
    
    response_data = {
        "totalUniqueStudents": len(unique_student_ids),
        "totalPresent": overall_stats["present"],
        "totalAbsents": overall_stats["absents"],
        "totalAlerts": overall_stats["alerts"],
        "courses": course_data
    }
    
    return jsonify(response_data)

@api.route("/api/search", methods=["GET", "OPTIONS"])
def search():
    if request.method == "OPTIONS":
        return handle_preflight()
    
    id_token = get_auth_token(request)
    lecturer = authenticate_lecturer(id_token)
    
    if isinstance(lecturer, tuple):
        response = jsonify([])
        return add_cors_headers(response)
    
    query = request.args.get("q", "").strip().lower()
    if not query:
        return jsonify([])
    
    results = []
    
    # Match direct commands
    for cmd, val in GeminiService.COMMAND_MAP.items():
        if query in cmd.lower():
            results.append(val)
    
    # Match course names
    matching_courses = [course for course in lecturer.courses if query in course.name.lower()]
    results.extend([
        {
            "type": "course",
            "course_id": course.course_id,
            "label": course.name,
            "route": f"/course/{course.course_id}/dashboard"
        }
        for course in matching_courses
    ])
    
    # Fallback to LLM
    if not results:
        llm_result = GeminiService.query_gemini(query)
        if llm_result["type"] == "llm-response":
            backend_cmd = llm_result.get("backendRespond", "").lower()
            if backend_cmd in GeminiService.COMMAND_MAP:
                results.append(GeminiService.COMMAND_MAP[backend_cmd])
            else:
                results.append({
                    "type": "llm-response",
                    "label": llm_result.get("uiRespond", "No UI response"),
                    "llm_backend": backend_cmd
                })
    
    response = jsonify(results)
    return add_cors_headers(response)

@api.route("/api/course/<int:course_id>/dashboard", methods=["GET"])
def get_course_dashboard(course_id: int):
    id_token = get_auth_token(request)
    lecturer = authenticate_lecturer(id_token)
    
    if isinstance(lecturer, tuple):
        return jsonify(lecturer[0]), lecturer[1]
    
    course = Course.query.filter_by(course_id=course_id, lecturer_id=lecturer.lecturer_id).first()
    if not course:
        return jsonify({"error": "Course not found"}), 404
    
    course_stats = AttendanceProcessor.get_course_statistics(course)
    enrolled_students = [
        AttendanceProcessor.calculate_student_status(e.student, course.course_id)
        for e in course.enrollments
    ]
    
    response_data = {
        "name": course.name,
        **course_stats,
        "students": enrolled_students
    }
    
    return jsonify(response_data)

@api.route("/api/attendance/create-sheet", methods=["POST", "OPTIONS"])
def create_attendance_sheet():
    if request.method == "OPTIONS":
        return handle_preflight()
    
    id_token = get_auth_token(request)
    lecturer = authenticate_lecturer(id_token)
    
    if isinstance(lecturer, tuple):
        return jsonify(lecturer[0]), lecturer[1]
    
    data = request.get_json()
    course_id = data.get("course_id")
    session_id = str(uuid.uuid4())
    
    course = Course.query.filter_by(course_id=course_id, lecturer_id=lecturer.lecturer_id).first()
    if not course:
        return jsonify({"error": "Course not found or unauthorized"}), 403
    
    students = [e.student for e in course.enrollments]
    
    for student in students:
        attendance = Attendance(
            student_id=student.student_id,
            course_id=course.course_id,
            session_id=session_id,
            present=False,
            closed=False
        )
        db.session.add(attendance)
    
    db.session.commit()
    return jsonify({"message": f"Attendance sheet created for {len(students)} students."}), 201

@api.route("/api/course/<int:course_id>/attendance/open-sheets", methods=["GET", "OPTIONS"])
def get_open_sheets(course_id: int):
    if request.method == "OPTIONS":
        return handle_preflight()
    
    id_token = get_auth_token(request)
    lecturer = authenticate_lecturer(id_token)
    
    if isinstance(lecturer, tuple):
        return jsonify(lecturer[0]), lecturer[1]
    
    sheets = db.session.query(
        Attendance.session_id, 
        db.func.min(Attendance.timestamp).label("started_at")
    ).filter_by(course_id=course_id, closed=False).group_by(Attendance.session_id).all()
    
    return jsonify([
        {"session_id": s.session_id, "started_at": s.started_at.isoformat()}
        for s in sheets
    ])

@api.route("/api/attendance/sheet/<session_id>", methods=["GET", "OPTIONS"])
def get_attendance_sheet(session_id: str):
    if request.method == "OPTIONS":
        return handle_preflight()
    
    id_token = get_auth_token(request)
    lecturer = authenticate_lecturer(id_token)
    
    if isinstance(lecturer, tuple):
        return jsonify(lecturer[0]), lecturer[1]
    
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
        return handle_preflight()
    
    id_token = get_auth_token(request)
    lecturer = authenticate_lecturer(id_token)
    
    if isinstance(lecturer, tuple):
        return jsonify(lecturer[0]), lecturer[1]
    
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
        return handle_preflight()
    
    name = request.form.get("name")
    photos = request.files.getlist("photo")
    
    if not name:
        return jsonify({"error": "Name is required"}), 400
    if len(photos) != 3:
        return jsonify({"error": "Exactly 3 photos are required"}), 400
    
    # Create student
    new_student = Student(name=name)
    db.session.add(new_student)
    db.session.commit()
    
    # Save photos
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
def serve_photo(photo_id: int):
    photo = StudentPhoto.query.get_or_404(photo_id)
    return Response(photo.image_data, mimetype=photo.mimetype)

@api.route('/api/lecturer/attendance/overall-sessions', methods=['GET', 'OPTIONS'])
def lecturer_attendance_performance():
    if request.method == "OPTIONS":
        return handle_preflight()
    
    id_token = get_auth_token(request)
    lecturer = authenticate_lecturer(id_token)
    
    if isinstance(lecturer, tuple):
        return jsonify(lecturer[0]), lecturer[1]
    
    course_ids = [c.course_id for c in lecturer.courses]
    if not course_ids:
        return jsonify({"lecturer": lecturer.name, "sessions": []})
    
    present_count_expr = func.sum(case((Attendance.present, 1), else_=0))
    
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
    
    sessions = [
        {
            "sessionId": session.session_id,
            "present": session.present_count,
            "absent": session.total_count - session.present_count,
        }
        for session in session_stats
    ]
    
    return jsonify({
        "lecturer": lecturer.name,
        "sessions": sessions,
    })

@api.route("/api/chat", methods=["POST", "OPTIONS"])
def gemini_chat():
    if request.method == "OPTIONS":
        return handle_preflight()
    
    try:
        data = request.get_json()
        if not data:
            response = jsonify({"error": "No data provided"})
            return add_cors_headers(response), 400
        
        message = data.get("message")
        if not message:
            response = jsonify({"error": "Message is required"})
            return add_cors_headers(response), 400
        
        history = data.get("history", [])
        gemini_response = GeminiService.generate_chat_response(message, history)
        
        if not gemini_response:
            response = jsonify({"error": "No response generated"})
            return add_cors_headers(response), 500
        
        response = jsonify({
            "success": True,
            "response": gemini_response,
            "message": "Response generated successfully"
        })
        return add_cors_headers(response), 200
        
    except Exception as e:
        response = jsonify({
            "error": "Failed to generate response",
            "details": str(e)
        })
        return add_cors_headers(response), 500

@api.route("/api/student-details/<int:student_id>", methods=["GET", "OPTIONS"])
def get_student_details(student_id: int):
    if request.method == "OPTIONS":
        return handle_preflight()
    
    if student_id <= 0:
        response = jsonify({"error": "Invalid student ID"})
        return add_cors_headers(response), 400
    
    id_token = get_auth_token(request)
    lecturer = authenticate_lecturer(id_token)
    
    if isinstance(lecturer, tuple):
        response = jsonify(lecturer[0])
        return add_cors_headers(response), lecturer[1]
    
    try:
        student = Student.query.options(
            joinedload(Student.enrollments).joinedload(Enrollment.course),
            joinedload(Student.photos)
        ).get(student_id)
        
        if not student:
            response = jsonify({"error": "Student not found"})
            return add_cors_headers(response), 404
            
    except Exception as e:
        response = jsonify({"error": "Database error", "details": str(e)})
        return add_cors_headers(response), 500
    
    # Authorization check
    lecturer_course_ids = {course.course_id for course in lecturer.courses}
    student_course_ids = {enrollment.course.course_id for enrollment in student.enrollments if enrollment.course}
    
    if not lecturer_course_ids & student_course_ids:
        response = jsonify({"error": "Forbidden", "details": "No access to this student's data"})
        return add_cors_headers(response), 403
    
    # Fetch attendance records
    attendance_records = Attendance.query.filter_by(student_id=student.student_id).all()
    
    records = [
        {
            "courseId": record.course_id,
            "sessionId": record.session_id,
            "present": record.present,
            "timestamp": record.timestamp.isoformat()
        }
        for record in attendance_records
    ]
    
    photos = [
        {
            "photoId": photo.photo_id,
            "filename": photo.filename,
            "mimetype": photo.mimetype,
            "capturedAt": photo.captured_at.isoformat(),
            "imageData": photo.image_data.hex()
        }
        for photo in student.photos
    ]
    
    response_data = {
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
    }
    
    response = jsonify(response_data)
    return add_cors_headers(response), 200