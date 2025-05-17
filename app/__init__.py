from flask import Flask
from flask_cors import CORS  # Importing the CORS module
from .db import db
from dotenv import load_dotenv
import os
from .routes import api

def create_app():
    load_dotenv()
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv("DATABASE_URL")
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.register_blueprint(api)

    db.init_app(app)

    # CORS Setup for all routes, allow any origin
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    with app.app_context():
        db.create_all()

    return app
