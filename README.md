#81475 backend

**ACCESS THE FULL SITE AT:**

[FULL SYSTEM ACCESS HERE](https://fyp.wandanial.com)

OR

https://fyp.wandanial.com


## 1. Frontend
- Developed using **React** and **TypeScript**.
- Provides UIs for:
  - Students
  - Staff
  - Administrators
- **Features**:
  - Enrollment Module
  - Attendance Dashboard
  - Analytics Dashboard (Focus Index, Reports)
- Communicates with backend via **RESTful APIs**.

## 2. Backend
- Built with **Python** using **Flask** (or FastAPI).
- Hosted on a **DigitalOcean Droplet**.
- **Responsibilities**:
  - API handling
  - Facial Recognition (FaceNet, ArcFace)
  - Body Pose Detection (MediaPipe)
  - Preprocessing (OpenCV for face detection, landmark alignment, normalization)

## 3. Asynchronous Processing
- Managed by **Celery** integrated with Flask.
- Handles:
  - Body pose analysis
  - Batch facial recognition
- Uses **Redis** or **RabbitMQ** as message broker.

## 4. Machine Learning Models
- **Facial Recognition**:
  - Pre-trained models (FaceNet, ArcFace, Dlib)
  - Converts faces into high-dimensional embeddings
- **Body Pose Analysis**:
  - MediaPipe for pose landmarks
  - ML model to assess attention/focus
- Hosted on backend for real-time or async inference.

## 5. Storage & Database
- **DigitalOcean PostgreSQL** (Managed DB):
  - Stores:
    - User data (students, staff, admins)
    - Attendance records
    - Focus index and analysis data
    - Processed images and pose/embedding data

## 6. Hosting & Deployment
- **DigitalOcean Droplet**:
  - Hosts backend server
  - Handles model inference and API responses
- **Celery Workers**:
  - Deployed to handle non-blocking, async tasks.

## 7. Scalability
- Modular architecture supports:
  - New features (analytics, notifications)
  - Third-party integrations via API

## 8. User Interaction
- Accessible via web on **desktop and mobile**.
- **Chatbot Integration**:
  - Generate attendance sheets
  - Retrieve student/attendance data
- **Live Attendance Validation**:
  - Via camera feed + facial recognition
