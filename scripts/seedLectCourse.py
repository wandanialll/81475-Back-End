

import os
import sys

# Get the absolute path to the directory containing the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Get the absolute path to the project root directory (one level up from the script directory)
project_root = os.path.dirname(script_dir)

# Add the project root to the Python path
sys.path.insert(0, project_root)

from app import create_app
from app.db import db
from app.models import Lecturer, Course
app = create_app()

with app.app_context():
    print("seeding lecturer and courses...")
    # Replace with your Firebase user email
    lecturer_email = "sysadtestlogin@gmail.com"

    # Check if already exists
    lecturer = Lecturer.query.filter_by(email=lecturer_email).first()
    if not lecturer:
        lecturer = Lecturer(name="Lecturer A", email=lecturer_email)
        db.session.add(lecturer)
        db.session.commit()

    # Create courses
    course1 = Course(name="Basics in Human Nature", lecturer_id=lecturer.lecturer_id)
    course2 = Course(name="Amaran Hidup dan Taufan", lecturer_id=lecturer.lecturer_id)

    db.session.add_all([course1, course2])
    db.session.commit()

    print("Lecturer and courses inserted.")
