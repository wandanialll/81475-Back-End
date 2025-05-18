import pickle
from insightface.app import FaceAnalysis
import numpy as np
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

# Load the InsightFace model once
face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0)

embeedings_path = os.path.join(current_dir, "insightface_embeddings.pkl")

with open(embeedings_path, "rb") as f:
    student_embeddings = pickle.load(f)



# Normalization (optional if you trained that way)
def normalize(vec):
    return vec / (np.linalg.norm(vec) + 1e-10)

# Pre-normalize embeddings for fast similarity
for k in student_embeddings:
    student_embeddings[k] = normalize(student_embeddings[k])
