from flask import Flask
from flask_cors import CORS  # Importing the CORS module
from .db import db
from dotenv import load_dotenv
import os
from .routes import api
from .routes.attendance_recognition import attendance_recognition_bp
from .routes.pose_focus_recognition import pose_focus_recognition_bp
from flask import jsonify
from celery import Celery
from .celery_utils import make_celery

celery = None

def create_app():
    load_dotenv()
    app = Flask(__name__)
    app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
    app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'
    
    global celery
    celery = make_celery(app)

    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv("DATABASE_URL")
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.register_blueprint(api)
    app.register_blueprint(attendance_recognition_bp)
    app.register_blueprint(pose_focus_recognition_bp)
    print(f"Registered blueprints: {[bp.name for bp in app.blueprints.values()]}")
    print("=== REGISTERED ROUTES ===")
    for rule in app.url_map.iter_rules():
        methods = ','.join(rule.methods)
        print(f"{rule.rule} -> {rule.endpoint} [{methods}]")
    print("========================")
    @app.route("/debug/routes")
    def debug_routes():
        return jsonify([str(rule) for rule in app.url_map.iter_rules()])
    


    db.init_app(app)

    # CORS Setup for all routes, allow any origin
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    with app.app_context():
        db.create_all()

    

    return app


