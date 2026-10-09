# FaceMatch — Real-Time Face Detection and Recognition

Streamlit application with continuous webcam frames using WebRTC, Ultralytics YOLO face detection, uploaded-image detection, SQLite profile enrollment, and OpenCV LBPH matching.

## Files

- `app.py` — full Streamlit application
- `requirements.txt` — dependencies
- `.gitignore` — excludes local database, secrets, and model weights

## Important: YOLO face model

The app expects a YOLO model trained specifically for face detection. The default model path is `yolov8n-face.pt`. This is not an Ultralytics official COCO model name guaranteed to download automatically. Provide a compatible face-trained weights file, or set `YOLO_FACE_MODEL` to a valid model path. A missing/unloadable YOLO model triggers an OpenCV Haar Cascade fallback.

Do not use a generic COCO `yolov8n.pt` model expecting dedicated face boxes; it is trained for general object classes, including people, not individual faces.

## Run locally

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The browser asks for camera permission. Click **START** in the Live Webcam tab. Webcam streaming normally requires localhost or HTTPS and a browser that supports WebRTC.

## Secrets

Create `.streamlit/secrets.toml` locally (do not commit it):

```toml
ADMIN_PASSWORD = "choose-a-long-unique-password"
YOLO_FACE_MODEL = "yolov8n-face.pt"
```

Alternatively set environment variables `ADMIN_PASSWORD` and `YOLO_FACE_MODEL`.

## Deploy to Streamlit Community Cloud

1. Create a GitHub repository and upload `app.py`, `requirements.txt`, `README.md`, and `.gitignore`.
2. Make the face-trained model file accessible to the deployed app. For large weights, consider a permitted model-hosting URL and download it during startup; do not commit large files unless appropriate.
3. Create a Streamlit Community Cloud app pointing to `app.py`.
4. Add `ADMIN_PASSWORD` and `YOLO_FACE_MODEL` under **App settings → Secrets**.
5. Deploy and test camera permissions, model loading, profile enrollment, and recognition.

WebRTC may require STUN/TURN network access depending on the host and network. Some corporate or restricted networks block peer connections. Local SQLite storage on cloud hosting may not persist across restarts/redeploys, so use durable storage before relying on saved profiles.

## Privacy

Only enroll people with their informed permission. Face crops are biometric data. The app provides deletion controls and stores cropped grayscale face images rather than uploaded originals, but SQLite itself is not encrypted. This prototype is not suitable for surveillance, access control, or consequential identity decisions.
