from flask import Flask
from app import create_app
from flask_cors import CORS
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

app = create_app()
CORS(app)

if __name__ == "__main__":
    app.run(debug=False)
