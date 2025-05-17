from flask import Flask
from app import create_app
from flask_cors import CORS
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

app = create_app()

# Apply CORS globally to all routes
CORS(app, resources={r"/api/*": {"origins": "https://fyp.wandanial.com"}},
     supports_credentials=True,
     allow_headers=["Content-Type", "Authorization", "Access-Control-Allow-Origin", "Access-Control-Allow-Credentials", "X-Requested-With"],
     allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])

if __name__ == "__main__":
    app.run(debug=False)
