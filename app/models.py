from .db import db
from datetime import datetime

class Student(db.Model):
    __tablename__ = 'students'
    student_id = db.Column(db.Integer, primary_key=True, autoincrement=False)
    name = db.Column(db.String, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    enrollments = db.relationship('Enrollment', backref='student')
    pose_snapshots = db.relationship('PoseSnapshot', backref='student')
    focus_indexes = db.relationship('FocusIndex', backref='student')
    chatbot_interactions = db.relationship('ChatbotInteraction', backref='student')
    attendance_records = db.relationship('Attendance', backref='student')
    photos = db.relationship('StudentPhoto', backref='student')

class StudentPhoto(db.Model):
    __tablename__ = 'student_photos'
    photo_id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('students.student_id'), nullable=False)
    image_data = db.Column(db.LargeBinary, nullable=False)
    filename = db.Column(db.String, nullable=False)
    mimetype = db.Column(db.String, nullable=False)
    captured_at = db.Column(db.DateTime, default=datetime.utcnow)

class Course(db.Model):
    __tablename__ = 'courses'
    course_id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String, nullable=False)
    lecturer_id = db.Column(db.Integer, db.ForeignKey('lecturers.lecturer_id'))

    enrollments = db.relationship('Enrollment', backref='course')
    attendance_records = db.relationship('Attendance', backref='course')

class Enrollment(db.Model):
    __tablename__ = 'enrollments'
    enrollment_id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('students.student_id'))
    course_id = db.Column(db.Integer, db.ForeignKey('courses.course_id'))

class PoseSnapshot(db.Model):
    __tablename__ = 'pose_snapshots'
    pose_id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('students.student_id'))
    captured_at = db.Column(db.DateTime, default=datetime.utcnow)
    data_path = db.Column(db.String, nullable=False)

class FocusIndex(db.Model):
    __tablename__ = 'focus_indexes'
    focus_id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('students.student_id'))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    score = db.Column(db.Float, nullable=False)

class ChatbotInteraction(db.Model):
    __tablename__ = 'chatbot_interactions'
    interaction_id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('students.student_id'))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    message = db.Column(db.Text, nullable=False)
    sender = db.Column(db.String, nullable=False)

class Attendance(db.Model):
    __tablename__ = 'attendance'
    attendance_id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('students.student_id'))
    course_id = db.Column(db.Integer, db.ForeignKey('courses.course_id'))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    present = db.Column(db.Boolean, default=True)
    session_id = db.Column(db.String, nullable=False)
    session_created_at = db.Column(db.DateTime, default=datetime.utcnow)
    session_ended_at = db.Column(db.DateTime)
    closed = db.Column(db.Boolean, default=False)

class Lecturer(db.Model):
    __tablename__ = 'lecturers'
    lecturer_id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String, nullable=False)
    email = db.Column(db.String, nullable=False, unique=True)

    courses = db.relationship('Course', backref='lecturer')

class FocusSessions(db.Model):
    __tablename__ = 'focus_sessions'
    id = db.Column(db.Integer, primary_key=True)
    attendance_id = db.Column(db.Integer, db.ForeignKey('attendance.attendance_id', ondelete='CASCADE'), nullable=False)
    student_id = db.Column(db.String, nullable=True)
    focus_index = db.Column(db.Float, nullable=True)
    timestamp = db.Column(db.DateTime, server_default=db.func.now())
    closed = db.Column(db.Boolean, default=False)

class FocusLabels(db.Model):
    __tablename__ = 'focus_labels'
    id = db.Column(db.Integer, primary_key=True)
    attendance_id = db.Column(db.Integer, db.ForeignKey('attendance.attendance_id', ondelete='CASCADE'), nullable=False)
    student_id = db.Column(db.String, nullable=True)
    frame_id = db.Column(db.String, nullable=True)
    label = db.Column(db.String, nullable=False)
    timestamp = db.Column(db.DateTime, server_default=db.func.now())