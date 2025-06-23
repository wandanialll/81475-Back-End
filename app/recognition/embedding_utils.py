from app.models import Student, FaceEmbedding, db
from insightface.app import FaceAnalysis
import numpy as np
from PIL import Image
from io import BytesIO

face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=0)

def generate_student_embedding(student_id):
    student = Student.query.get(student_id)
    if not student or not student.photos:
        return None

    embeddings = []

    for photo in student.photos:
        try:
            img = Image.open(BytesIO(photo.image_data)).convert("RGB")
            img_np = np.array(img)
            faces = face_app.get(img_np)
            if not faces:
                continue
            embeddings.append(faces[0]['embedding'])
        except Exception as e:
            print(f"[WARN] Failed to process image: {photo.filename}: {e}")

    if not embeddings:
        return None

    avg_embedding = np.mean(embeddings, axis=0)

    # Save or update in DB
    existing = FaceEmbedding.query.filter_by(student_id=student_id).first()
    if existing:
        existing.embedding = avg_embedding
    else:
        new_embedding = FaceEmbedding(student_id=student_id, embedding=avg_embedding)
        db.session.add(new_embedding)

    db.session.commit()
    return avg_embedding
