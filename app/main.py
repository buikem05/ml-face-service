import numpy as np
import cv2
import os
import urllib.request
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from insightface.app import FaceAnalysis
import mediapipe as mp

app_api = FastAPI(title="ML Face Service")

MODEL_DIR = "/app/models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Auto-download MediaPipe model if missing ──────────
LANDMARKER_PATH = f"{MODEL_DIR}/face_landmarker.task"
if not os.path.exists(LANDMARKER_PATH):
    print("Downloading face_landmarker.task...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        LANDMARKER_PATH
    )
    print("face_landmarker.task downloaded!")

# ── Load InsightFace (auto-downloads buffalo_l if missing) ────
print("Loading InsightFace buffalo_l...")
face_app = FaceAnalysis(name="buffalo_l", root=MODEL_DIR)
face_app.prepare(ctx_id=-1, det_size=(640, 640))  # CPU mode
print("InsightFace ready!")

# ── MediaPipe setup ───────────────────────────────────
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

landmarker_options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=LANDMARKER_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

SIMILARITY_THRESHOLD = 0.63

# ── Helpers ───────────────────────────────────────────

def get_embedding(img, face):
    emb = face.embedding
    return (emb / np.linalg.norm(emb)).tolist()

def eye_aspect_ratio(landmarks, eye_indices):
    pts = [(landmarks[i].x, landmarks[i].y) for i in eye_indices]
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    return (A + B) / (2.0 * C)

def detect_blink(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    with FaceLandmarker.create_from_options(landmarker_options) as detector:
        result = detector.detect(mp_image)
    if not result.face_landmarks:
        return None, None
    lm = result.face_landmarks[0]
    left_ear  = eye_aspect_ratio(lm, LEFT_EYE)
    right_ear = eye_aspect_ratio(lm, RIGHT_EYE)
    avg_ear = (left_ear + right_ear) / 2
    return round(avg_ear, 3), avg_ear < 0.20

# ── Endpoints ─────────────────────────────────────────

@app_api.get("/")
async def root():
    return {
        "status": "running",
        "endpoints": ["/ml/detect", "/ml/embedding", "/ml/liveness", "/ml/verify"]
    }

@app_api.get("/health")
async def health():
    return {"status": "healthy"}

@app_api.post("/ml/detect")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "invalid image"}, status_code=400)
    faces = face_app.get(img)
    results = [
        {"bbox": f.bbox.tolist(), "score": float(f.det_score), "landmarks": f.kps.tolist()}
        for f in faces
    ]
    return JSONResponse({"faces": results, "count": len(faces)})

@app_api.post("/ml/embedding")
async def embedding(file: UploadFile = File(...)):
    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "invalid image"}, status_code=400)
    faces = face_app.get(img)
    if not faces:
        return JSONResponse({"error": "no face detected"}, status_code=400)
    emb = get_embedding(img, faces[0])
    return JSONResponse({"embedding": emb, "dim": 512})

@app_api.post("/ml/liveness")
async def liveness(file: UploadFile = File(...)):
    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "invalid image"}, status_code=400)
    ear, is_blink = detect_blink(img)
    if ear is None:
        return JSONResponse({"status": "no_face", "is_live": False})
    return JSONResponse({
        "ear": ear,
        "is_live": not is_blink,
        "status": "live" if not is_blink else "pending"
    })

@app_api.post("/ml/verify")
async def verify(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    c1 = await file1.read()
    c2 = await file2.read()
    img1 = cv2.imdecode(np.frombuffer(c1, np.uint8), cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(np.frombuffer(c2, np.uint8), cv2.IMREAD_COLOR)
    if img1 is None or img2 is None:
        return JSONResponse({"error": "invalid image"}, status_code=400)
    faces1 = face_app.get(img1)
    faces2 = face_app.get(img2)
    if not faces1 or not faces2:
        return JSONResponse({"error": "face not detected in one or both images"}, status_code=400)
    emb1 = np.array(get_embedding(img1, faces1[0]))
    emb2 = np.array(get_embedding(img2, faces2[0]))
    score = float(np.dot(emb1, emb2))
    return JSONResponse({
        "match": score >= SIMILARITY_THRESHOLD,
        "score": round(score, 4),
        "threshold": SIMILARITY_THRESHOLD
    })
