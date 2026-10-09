import hashlib
import hmac
import io
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

import av
import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

APP_TITLE = "FaceMatch — Real-Time Face Detection"
DB_PATH = Path(os.getenv("FACEMATCH_DB_PATH", "faces.db"))
MAX_UPLOAD_MB = 8
FACE_SIZE = (160, 160)
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# Set YOLO_FACE_MODEL in Streamlit Secrets or environment to a face-trained model path.
# Example: YOLO_FACE_MODEL = "models/yolov8n-face.pt"
try:
    _secret_model = str(st.secrets.get("YOLO_FACE_MODEL", "")).strip()
except Exception:
    _secret_model = ""
YOLO_FACE_MODEL = _secret_model or os.getenv("YOLO_FACE_MODEL", "yolov8n-face.pt")

st.set_page_config(page_title=APP_TITLE, page_icon="🔎", layout="wide")


# ------------------------------ Database ------------------------------------
def connect_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=10)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS people (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL COLLATE NOCASE UNIQUE,
            face_crop BLOB NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def list_people():
    with connect_db() as conn:
        return conn.execute(
            "SELECT id, name, created_at FROM people ORDER BY name COLLATE NOCASE"
        ).fetchall()


def get_training_rows():
    with connect_db() as conn:
        return conn.execute("SELECT id, name, face_crop FROM people ORDER BY id").fetchall()


def insert_person(name, face_crop):
    with connect_db() as conn:
        conn.execute(
            "INSERT INTO people(name, face_crop, created_at) VALUES (?, ?, ?)",
            (name.strip(), sqlite3.Binary(face_crop), datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()


def delete_person(person_id):
    with connect_db() as conn:
        conn.execute("DELETE FROM people WHERE id = ?", (int(person_id),))
        conn.commit()


def delete_all_people():
    with connect_db() as conn:
        conn.execute("DELETE FROM people")
        conn.commit()


# --------------------------- Image processing -------------------------------
def load_image(uploaded_file):
    raw = uploaded_file.getvalue()
    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        raise ValueError(f"Image exceeds the {MAX_UPLOAD_MB} MB upload limit.")
    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
        if img.width * img.height > 20_000_000:
            raise ValueError("Image dimensions exceed 20 megapixels.")
        return np.array(img)
    except (UnidentifiedImageError, OSError):
        raise ValueError("The uploaded file is not a readable image.")


@st.cache_resource(show_spinner="Loading face detection model…")
def load_yolo_model(model_path):
    if YOLO is None:
        return None, "Ultralytics could not be imported."
    try:
        model = YOLO(model_path)
        return model, None
    except Exception as exc:
        return None, f"Could not load YOLO model '{model_path}': {exc}"


def detect_faces(rgb_image, detector="YOLO", conf=0.25):
    """Return (boxes, detector_name, warning). Boxes are x,y,w,h."""
    if detector == "YOLO":
        model, error = load_yolo_model(YOLO_FACE_MODEL)
        if model is not None:
            try:
                result = model.predict(source=rgb_image, conf=conf, verbose=False)[0]
                boxes = []
                if result.boxes is not None:
                    for coords in result.boxes.xyxy.cpu().numpy():
                        x1, y1, x2, y2 = [int(v) for v in coords[:4]]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(rgb_image.shape[1], x2), min(rgb_image.shape[0], y2)
                        if x2 > x1 and y2 > y1:
                            boxes.append((x1, y1, x2 - x1, y2 - y1))
                return sorted(boxes, key=lambda b: (b[0], b[1])), "Ultralytics YOLO", None
            except Exception as exc:
                error = f"YOLO inference failed: {exc}"
        # Fall back for usability; it is still a face detector, but less robust.
        boxes = detect_faces_haar(rgb_image)
        return boxes, "OpenCV Haar Cascade fallback", error
    return detect_faces_haar(rgb_image), "OpenCV Haar Cascade", None


def detect_faces_haar(rgb_image):
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    detector = cv2.CascadeClassifier(CASCADE_PATH)
    if detector.empty():
        raise RuntimeError("Could not load OpenCV Haar Cascade detector.")
    boxes = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    return sorted([(int(x), int(y), int(w), int(h)) for x, y, w, h in boxes], key=lambda b: (b[0], b[1]))


def crop_face(rgb_image, box):
    x, y, w, h = box
    H, W = rgb_image.shape[:2]
    px, py = int(w * 0.18), int(h * 0.18)
    x1, y1 = max(0, x - px), max(0, y - py)
    x2, y2 = min(W, x + w + px), min(H, y + h + py)
    crop = rgb_image[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, FACE_SIZE, interpolation=cv2.INTER_AREA)
    return cv2.equalizeHist(gray)


def encode_crop(gray):
    ok, encoded = cv2.imencode(".png", gray)
    if not ok:
        raise RuntimeError("Could not encode face crop.")
    return encoded.tobytes()


def decode_crop(blob):
    arr = np.frombuffer(blob, dtype=np.uint8)
    gray = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    return None if gray is None else cv2.resize(gray, FACE_SIZE)


def build_recognizer():
    rows = get_training_rows()
    if not rows:
        return None, {}
    if not hasattr(cv2, "face"):
        raise RuntimeError("OpenCV LBPH is unavailable. Install opencv-contrib-python-headless.")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    images, labels, label_to_name = [], [], {}
    for person_id, name, blob in rows:
        crop = decode_crop(blob)
        if crop is not None:
            images.append(crop)
            labels.append(int(person_id))
            label_to_name[int(person_id)] = name
    if not images:
        return None, {}
    recognizer.train(images, np.asarray(labels, dtype=np.int32))
    return recognizer, label_to_name


def recognize_boxes(rgb, boxes, recognizer, label_to_name, threshold):
    labels = []
    for box in boxes:
        if recognizer is None:
            labels.append("Face detected")
            continue
        crop = crop_face(rgb, box)
        label, distance = recognizer.predict(crop)
        if label in label_to_name and distance <= threshold:
            labels.append(f"{label_to_name[label]} ({distance:.0f})")
        else:
            labels.append(f"Unknown ({distance:.0f})")
    return labels


def annotate(rgb, boxes, labels=None):
    out = rgb.copy()
    for i, (x, y, w, h) in enumerate(boxes):
        label = labels[i] if labels and i < len(labels) else f"Face {i + 1}"
        color = (0, 190, 90) if not label.startswith("Unknown") else (230, 70, 50)
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        cv2.putText(out, label[:42], (x, max(24, y - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color, 2, cv2.LINE_AA)
    return out


# ------------------------------ Admin ---------------------------------------
def get_admin_password():
    try:
        secret = str(st.secrets.get("ADMIN_PASSWORD", "")).strip()
    except Exception:
        secret = ""
    return secret or os.getenv("ADMIN_PASSWORD", "").strip()


def admin_unlocked():
    expected = get_admin_password()
    if not expected:
        st.sidebar.warning("Set ADMIN_PASSWORD in Streamlit Secrets to enable registration/deletion.")
        return False
    supplied = st.sidebar.text_input("Administrator password", type="password")
    if supplied:
        good = hashlib.sha256(supplied.encode()).digest()
        wanted = hashlib.sha256(expected.encode()).digest()
        if hmac.compare_digest(good, wanted):
            st.sidebar.success("Administrator access enabled for this session.")
            return True
        st.sidebar.error("Incorrect administrator password.")
    return False


# -------------------------- Real-time processor -----------------------------
class FaceVideoProcessor(VideoProcessorBase):
    def __init__(self, detector, recognize, threshold, conf):
        self.detector = detector
        self.recognize = recognize
        self.threshold = threshold
        self.conf = conf
        self.recognizer, self.label_to_name = None, {}
        self.lock = threading.Lock()
        self.last_warning = None
        if recognize:
            try:
                self.recognizer, self.label_to_name = build_recognizer()
            except Exception as exc:
                self.last_warning = str(exc)

    def recv(self, frame):
        bgr = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        try:
            # Protect model inference in case the video backend invokes callbacks concurrently.
            with self.lock:
                boxes, _, warning = detect_faces(rgb, self.detector, self.conf)
                if warning:
                    self.last_warning = warning
                labels = recognize_boxes(rgb, boxes, self.recognizer, self.label_to_name,
                                         self.threshold) if self.recognize else None
                output = annotate(rgb, boxes, labels)
                cv2.putText(output, f"Faces: {len(boxes)}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (255, 220, 0), 2, cv2.LINE_AA)
            out_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            return av.VideoFrame.from_ndarray(out_bgr, format="bgr24")
        except Exception as exc:
            self.last_warning = str(exc)
            return frame


# ---------------------------------- UI ---------------------------------------
st.title("🔎 FaceMatch")
st.caption("Real-time face detection and enrolled-face matching with YOLO, WebRTC, and SQLite")
st.info(
    "Privacy: only enroll or process people with permission. Face crops are biometric data. "
    "This educational prototype is not suitable for surveillance, access control, or consequential identity decisions."
)

with st.sidebar:
    st.header("Detection settings")
    detector_choice = st.selectbox("Detector", ["YOLO", "OpenCV Haar Cascade"])
    confidence = st.slider("YOLO confidence", 0.10, 0.90, 0.25, 0.05)
    threshold = st.slider("LBPH distance threshold", 25, 100, 55, 5,
                          help="Lower is stricter. Distance is not a probability.")
    st.caption(f"Model path: `{YOLO_FACE_MODEL}`")
    st.caption("Use a YOLO model trained specifically for faces, not a generic COCO model.")
    st.divider()
    admin_ok = admin_unlocked()

live_tab, upload_tab, register_tab, manage_tab = st.tabs(
    ["🔴 Live Webcam", "🖼️ Uploaded Image", "➕ Register Face", "🗃️ Manage Database"]
)

with live_tab:
    st.subheader("Real-time webcam detection")
    st.write("Allow camera access in your browser, then click **START**. Bounding boxes update as video frames arrive.")
    live_recognize = st.checkbox("Also recognize registered faces", value=True)
    if live_recognize and not get_training_rows():
        st.caption("No enrolled profiles found yet. The stream will show detected faces until profiles are registered.")
    webrtc_ctx = webrtc_streamer(
        key="facematch-live-webcam",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=lambda: FaceVideoProcessor(
            detector_choice, live_recognize, threshold, confidence
        ),
        async_processing=True,
    )
    if webrtc_ctx.video_processor and webrtc_ctx.video_processor.last_warning:
        st.warning(webrtc_ctx.video_processor.last_warning)
    st.caption("If the camera does not connect on a hosted app, check HTTPS, browser permissions, firewall, and WebRTC network access.")

with upload_tab:
    st.subheader("Detect and optionally recognize uploaded photographs")
    upload = st.file_uploader("Upload JPG or PNG", type=["jpg", "jpeg", "png"], key="query_image")
    do_recognize = st.checkbox("Match against registered profiles", value=True, key="upload_recognize")
    if upload:
        try:
            rgb = load_image(upload)
            boxes, used, warning = detect_faces(rgb, detector_choice, confidence)
            if warning:
                st.warning(warning)
            recognizer, label_to_name = (build_recognizer() if do_recognize else (None, {}))
            labels = recognize_boxes(rgb, boxes, recognizer, label_to_name, threshold) if do_recognize else None
            c1, c2 = st.columns(2)
            with c1:
                st.image(rgb, caption="Original image", use_container_width=True)
            with c2:
                st.image(annotate(rgb, boxes, labels), caption=f"Results — {used}", use_container_width=True)
            st.success(f"Detected {len(boxes)} face(s).") if boxes else st.warning("No faces detected.")
            if labels:
                for i, label in enumerate(labels, start=1):
                    st.write(f"**Face {i}:** {label}")
        except Exception as exc:
            st.error(str(exc))

with register_tab:
    st.subheader("Register a person with consent")
    st.warning("Only a normalized grayscale face crop is stored; the original photo is not saved.")
    with st.form("register_face_form", clear_on_submit=True):
        person_name = st.text_input("Display name", max_chars=80)
        source = st.radio("Reference photo source", ["Upload image", "Capture webcam photo"], horizontal=True)
        if source == "Upload image":
            ref_file = st.file_uploader("Reference photo (one face only)", type=["jpg", "jpeg", "png"], key="register_upload")
        else:
            ref_file = st.camera_input("Take a reference photo", key="register_camera")
        consent = st.checkbox("I confirm that the person has given permission for enrollment and storage of their face crop.")
        submit = st.form_submit_button("Register face", disabled=not admin_ok)
    if submit:
        if not admin_ok:
            st.error("Administrator access is required.")
        elif not person_name.strip():
            st.error("Enter a display name.")
        elif not consent:
            st.error("Consent confirmation is required.")
        elif ref_file is None:
            st.error("Choose or capture a reference photo.")
        else:
            try:
                rgb = load_image(ref_file)
                boxes, used, warning = detect_faces(rgb, detector_choice, confidence)
                if warning:
                    st.warning(warning)
                if len(boxes) != 1:
                    st.error(f"Exactly one face is required; detected {len(boxes)} using {used}.")
                else:
                    insert_person(person_name.strip(), encode_crop(crop_face(rgb, boxes[0])))
                    st.success(f"Registered '{person_name.strip()}'.")
            except sqlite3.IntegrityError:
                st.error("That name already exists. Choose a unique name.")
            except Exception as exc:
                st.error(str(exc))

with manage_tab:
    st.subheader("Enrolled profiles")
    profiles = list_people()
    if profiles:
        st.dataframe([{"ID": p[0], "Name": p[1], "Created (UTC)": p[2]} for p in profiles],
                     use_container_width=True, hide_index=True)
        choices = {f"{p[1]} (ID {p[0]})": p[0] for p in profiles}
        selected = st.selectbox("Select a profile to delete", list(choices.keys()))
        confirm_one = st.checkbox("Confirm deletion of selected biometric profile")
        if st.button("Delete selected profile", disabled=not (admin_ok and confirm_one)):
            delete_person(choices[selected])
            st.success("Profile deleted.")
            st.rerun()
    else:
        st.info("No profiles are registered yet.")
    confirm_all = st.checkbox("Confirm permanent deletion of ALL biometric profiles")
    if st.button("Delete ALL profiles", type="primary", disabled=not (admin_ok and confirm_all)):
        delete_all_people()
        st.success("All profiles deleted.")
        st.rerun()

st.divider()
st.caption("Prototype limitations: LBPH is a basic demo recognizer, not security-grade. Hosted local SQLite may be ephemeral. Test only with consented images.")
