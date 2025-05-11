from flask import Flask
from app import create_app
from flask_cors import CORS

app = create_app()

# Apply CORS globally to all routes
CORS(app)

if __name__ == "__main__":
    app.run(debug=False)
