from .db import db
from datetime import datetime

class Student(db.Model):
    __tablename__ = 'students'
    student_id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    enrollments = db.relationship('Enrollment', backref='student')
    pose_snapshots = db.relationship('PoseSnapshot', backref='student')
    focus_indexes = db.relationship('FocusIndex', backref='student')
    chatbot_interactions = db.relationship('ChatbotInteraction', backref='student')
    attendance_records = db.relationship('Attendance', backref='student')


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
    data_path = db.Column(db.String, nullable=False)  # Link to file (CSV/JSON/etc)


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
    sender = db.Column(db.String, nullable=False)  # 'student' or 'bot'


class Attendance(db.Model):
    __tablename__ = 'attendance'
    attendance_id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('students.student_id'))
    course_id = db.Column(db.Integer, db.ForeignKey('courses.course_id'))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    present = db.Column(db.Boolean, default=True)
    session_id = db.Column(db.String, nullable=False)  # UUID or timestamp string
    closed = db.Column(db.Boolean, default=False)

class Lecturer(db.Model):
    __tablename__ = 'lecturers'
    lecturer_id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String, nullable=False)
    email = db.Column(db.String, nullable=False, unique=True)

    courses = db.relationship('Course', backref='lecturer')
