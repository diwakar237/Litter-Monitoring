"""
Litter Monitoring Dashboard
----------------------------
A Gradio web app that detects littering events in uploaded video, logs them
anonymously (no face ID, no fines), and shows a hotspot dashboard.

Runs on Hugging Face Spaces free CPU tier. Degrades gracefully if optional
pieces (YOLO weights, Supabase credentials, MoveNet) aren't configured yet,
so you can test the UI and pipeline locally before wiring everything up.
"""

import os
import time
import json
import cv2
import numpy as np
import pandas as pd
import gradio as gr

# ---------------------------------------------------------------------------
# 1. Litter detector (YOLOv8) — falls back to a stub if ultralytics/weights
#    aren't available yet, so the app still runs for local testing.
# ---------------------------------------------------------------------------
MODEL_PATH = os.environ.get("LITTER_MODEL_PATH", "best.pt")
_yolo_model = None
_yolo_error = None

try:
    from ultralytics import YOLO
    if os.path.exists(MODEL_PATH):
        _yolo_model = YOLO(MODEL_PATH)
    else:
        _yolo_error = f"Weights file '{MODEL_PATH}' not found. Add best.pt to the Space or set LITTER_MODEL_PATH."
except Exception as e:
    _yolo_error = f"ultralytics not available: {e}"


def detect_litter(frame_bgr):
    """Returns a list of {label, conf, box:[x1,y1,x2,y2]} for one frame."""
    if _yolo_model is None:
        return []
    results = _yolo_model.predict(frame_bgr, conf=0.45, verbose=False)[0]
    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = _yolo_model.names[cls_id]
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()
        detections.append({"label": label, "conf": conf, "box": xyxy})
    return detections


# ---------------------------------------------------------------------------
# 2. Optional hand-release check (MoveNet). Off by default (SIMPLE_MODE=True)
#    since it needs an object tracker to be reliable; a detection alone is
#    used to flag an event in the meantime.
# ---------------------------------------------------------------------------
SIMPLE_MODE = os.environ.get("SIMPLE_MODE", "true").lower() == "true"
_movenet = None

if not SIMPLE_MODE:
    try:
        import tensorflow as tf
        import tensorflow_hub as hub
        _movenet = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
    except Exception as e:
        print(f"[warn] MoveNet unavailable, falling back to SIMPLE_MODE: {e}")
        SIMPLE_MODE = True


def check_hand_release(frame_bgr):
    """Placeholder heuristic. Returns True (treat every detection as an
    event) in SIMPLE_MODE. Replace with real tracking logic for production."""
    if SIMPLE_MODE or _movenet is None:
        return True
    # Real implementation would keep a rolling wrist-position history and
    # check that a detected object separated from a wrist over N frames.
    return True


# ---------------------------------------------------------------------------
# 3. Face-blur pass for evidence images (privacy protection for bystanders,
#    NOT identification).
# ---------------------------------------------------------------------------
_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def blur_faces(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        roi = frame_bgr[y:y + h, x:x + w]
        if roi.size:
            frame_bgr[y:y + h, x:x + w] = cv2.GaussianBlur(roi, (35, 35), 30)
    return frame_bgr


# ---------------------------------------------------------------------------
# 4. Event storage — Supabase if configured, otherwise a local JSONL file so
#    the app is fully usable without any external service.
# ---------------------------------------------------------------------------
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
_supabase = None

if SUPABASE_URL and SUPABASE_KEY:
    try:
        from supabase import create_client
        _supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        print(f"[warn] Supabase client init failed, using local storage: {e}")

LOCAL_EVENTS_PATH = os.environ.get("LOCAL_EVENTS_PATH", "events.jsonl")
LOCAL_EVIDENCE_DIR = os.environ.get("LOCAL_EVIDENCE_DIR", "evidence")
os.makedirs(LOCAL_EVIDENCE_DIR, exist_ok=True)


def log_event(annotated_frame_bgr, location_tag, detection):
    """Saves one event + evidence snapshot. Uses Supabase if configured,
    otherwise writes to local disk (fine for demo/testing, not for a real
    always-on Space since local disk is ephemeral there)."""
    safe_frame = blur_faces(annotated_frame_bgr.copy())
    ts = time.time()
    filename = f"{location_tag}_{int(ts)}.jpg"

    if _supabase is not None:
        ok, buf = cv2.imencode(".jpg", safe_frame)
        try:
            _supabase.storage.from_("evidence").upload(f"events/{filename}", buf.tobytes())
            image_url = _supabase.storage.from_("evidence").get_public_url(f"events/{filename}")
        except Exception as e:
            print(f"[warn] Supabase upload failed, saving locally instead: {e}")
            image_url = _save_local_evidence(safe_frame, filename)
        try:
            _supabase.table("litter_events").insert({
                "location_tag": location_tag,
                "object_label": detection["label"],
                "confidence": detection["conf"],
                "evidence_image_url": image_url,
            }).execute()
        except Exception as e:
            print(f"[warn] Supabase insert failed: {e}")
    else:
        image_url = _save_local_evidence(safe_frame, filename)
        _append_local_event({
            "detected_at": pd.Timestamp.now().isoformat(),
            "location_tag": location_tag,
            "object_label": detection["label"],
            "confidence": detection["conf"],
            "evidence_image_url": image_url,
        })

    return {"location": location_tag, "label": detection["label"], "url": image_url}


def _save_local_evidence(frame_bgr, filename):
    path = os.path.join(LOCAL_EVIDENCE_DIR, filename)
    cv2.imwrite(path, frame_bgr)
    return path


def _append_local_event(event: dict):
    with open(LOCAL_EVENTS_PATH, "a") as f:
        f.write(json.dumps(event) + "\n")


def _load_all_events() -> pd.DataFrame:
    if _supabase is not None:
        try:
            rows = _supabase.table("litter_events").select("*").execute().data
            return pd.DataFrame(rows)
        except Exception as e:
            print(f"[warn] Supabase read failed: {e}")
            return pd.DataFrame()
    if not os.path.exists(LOCAL_EVENTS_PATH):
        return pd.DataFrame()
    rows = [json.loads(line) for line in open(LOCAL_EVENTS_PATH) if line.strip()]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. Main video processing loop
# ---------------------------------------------------------------------------
FRAME_SKIP = int(os.environ.get("FRAME_SKIP", "3"))
RESIZE_WIDTH = int(os.environ.get("RESIZE_WIDTH", "640"))


def process_video(video_path, location_tag, progress=gr.Progress()):
    if video_path is None:
        return None, pd.DataFrame(), "Upload a video first."
    if _yolo_model is None:
        return None, pd.DataFrame(), f"⚠️ Detector not loaded: {_yolo_error}"
    if not location_tag:
        location_tag = "Camera-01"

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    frame_idx = 0
    run_events = []
    last_annotated = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        progress(frame_idx / total_frames, desc="Scanning frames...")
        if frame_idx % FRAME_SKIP != 0:
            continue

        h, w = frame.shape[:2]
        scale = RESIZE_WIDTH / w
        frame_small = cv2.resize(frame, (RESIZE_WIDTH, int(h * scale)))

        detections = detect_litter(frame_small)
        annotated = frame_small.copy()
        for d in detections:
            x1, y1, x2, y2 = map(int, d["box"])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(annotated, f'{d["label"]} {d["conf"]:.2f}',
                        (x1, max(y1 - 8, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        if detections and check_hand_release(frame_small):
            for d in detections:
                event = log_event(annotated, location_tag, d)
                run_events.append(event)

        last_annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    cap.release()

    status = f"Processed {frame_idx} frames, logged {len(run_events)} event(s)."
    table = pd.DataFrame(run_events) if run_events else pd.DataFrame(columns=["location", "label", "url"])
    return last_annotated, table, status


def load_dashboard():
    df = _load_all_events()
    if df.empty:
        return None, pd.DataFrame(), "No events logged yet — run Live Monitoring first."
    by_location = df.groupby("location_tag").size().reset_index(name="count")
    display_cols = [c for c in ["detected_at", "location_tag", "object_label", "confidence"] if c in df.columns]
    return by_location, df[display_cols].sort_values(display_cols[0], ascending=False), f"{len(df)} total event(s)."


# ---------------------------------------------------------------------------
# 6. Gradio UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="Litter Monitoring Dashboard") as demo:
    gr.Markdown("# 🚯 Litter Monitoring Dashboard")
    gr.Markdown(
        "Upload a video (simulated CCTV feed). Detected littering events are "
        "logged anonymously — location, timestamp, object type, and a snapshot "
        "with any bystander faces blurred. No identity matching, no fines."
    )
    if _yolo_error:
        gr.Markdown(f"⚠️ **{_yolo_error}**")
    if _supabase is None:
        gr.Markdown("ℹ️ Supabase not configured — events are being saved locally in `events.jsonl`.")

    with gr.Tab("Live Monitoring"):
        with gr.Row():
            video_in = gr.Video(label="Upload video")
            with gr.Column():
                location_in = gr.Textbox(label="Location tag", value="Camera-01")
                run_btn = gr.Button("Run detection", variant="primary")
                status_out = gr.Markdown()
        video_out = gr.Image(label="Last annotated frame")
        log_out = gr.Dataframe(label="Events logged this run")
        run_btn.click(process_video, [video_in, location_in], [video_out, log_out, status_out])

    with gr.Tab("Event Log & Hotspots"):
        refresh_btn = gr.Button("Refresh")
        summary_out = gr.Markdown()
        chart_out = gr.BarPlot(x="location_tag", y="count", title="Events by location")
        table_out = gr.Dataframe(label="All events")
        refresh_btn.click(load_dashboard, None, [chart_out, table_out, summary_out])

if __name__ == "__main__":
    demo.launch()
