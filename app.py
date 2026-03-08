import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import time
from ultralytics import YOLO

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HelmetGuard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Styles ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

:root {
    --bg:      #f7f7f5;
    --surface: #ffffff;
    --border:  #e0e0dc;
    --text:    #111;
    --muted:   #888;
}

html, body, [class*="css"], [data-testid="stAppViewContainer"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background: var(--bg) !important;
    color: var(--text);
}
[data-testid="stAppViewContainer"] > .main { background: var(--bg) !important; }
section.main > div { padding-top: 1.5rem; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { max-width: 1200px; padding: 2rem 3rem; }

.hg-header {
    display: flex; align-items: baseline; gap: .75rem;
    border-bottom: 1.5px solid var(--text);
    padding-bottom: .75rem; margin-bottom: 2rem;
}
.hg-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.1rem; font-weight: 500; letter-spacing: .06em;
}
.hg-sub { color: var(--muted); font-size: .75rem; letter-spacing: .04em; }

.stTabs [data-baseweb="tab-list"] {
    gap: 0; background: transparent !important;
    border-bottom: 1.5px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: .7rem !important; letter-spacing: .1em !important;
    padding: .5rem 1.2rem !important; color: var(--muted) !important;
    background: transparent !important; border: none !important;
    border-bottom: 2px solid transparent !important;
    margin-bottom: -1.5px !important;
}
.stTabs [aria-selected="true"] {
    color: var(--text) !important;
    border-bottom: 2px solid var(--text) !important;
}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] { display: none !important; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.5rem !important; }

[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
}
[data-testid="stExpander"] summary {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: .7rem !important; letter-spacing: .1em !important;
    color: var(--muted) !important; text-transform: uppercase !important;
}

.stSlider label {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: .7rem !important; color: var(--muted) !important;
    letter-spacing: .05em !important;
}

[data-testid="metric-container"] {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 4px; padding: .9rem 1.1rem;
}
[data-testid="metric-container"] label {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: .62rem !important; letter-spacing: .1em !important;
    color: var(--muted) !important; text-transform: uppercase !important;
}
[data-testid="metric-container"] [data-testid="metric-value"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 1.7rem !important; font-weight: 500 !important;
    color: var(--text) !important;
}

.stButton > button {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: .72rem !important; letter-spacing: .08em !important;
    background: var(--text) !important; color: var(--bg) !important;
    border: none !important; border-radius: 3px !important;
    padding: .45rem 1.3rem !important;
}
.stButton > button:hover { opacity: .8 !important; }
.stButton > button:disabled { opacity: .3 !important; cursor: default !important; }

[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 4px !important;
}

.stImage img { border: 1px solid var(--border); border-radius: 4px; }

.img-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: .62rem; letter-spacing: .1em;
    color: var(--muted); text-transform: uppercase; margin-bottom: .3rem;
}

.badge {
    display: inline-block; padding: 2px 10px; border-radius: 20px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: .65rem; letter-spacing: .08em; text-transform: uppercase;
}
.badge-ok   { background:#dcfce7; color:#166534; border:1px solid #bbf7d0; }
.badge-err  { background:#fee2e2; color:#991b1b; border:1px solid #fecaca; }
.badge-live { background:#dcfce7; color:#166534; border:1px solid #bbf7d0; }
.badge-stop { background:#f3f4f6; color:#6b7280; border:1px solid #e5e7eb; }

hr { border: none; border-top: 1px solid var(--border); margin: 1.2rem 0; }
</style>
""", unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hg-header">
    <span class="hg-title">HELMETGUARD</span>
    <span class="hg-sub">helmet detection · YOLOv11s</span>
</div>
""", unsafe_allow_html=True)

# ─── Constants ────────────────────────────────────────────────────────────────
CLASS_NAMES = ['helmet', 'two_wheeler', 'without_helmet']
MODEL_PATH  = r"runs\detect\train\weights\best.pt"

# ─── Model ────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        if os.path.exists(MODEL_PATH):
            return YOLO(MODEL_PATH), None
        for fb in ['best.pt', 'model.pt']:
            if os.path.exists(fb):
                return YOLO(fb), None
        return None, f"Model not found at: {MODEL_PATH}"
    except Exception as e:
        return None, str(e)

model, model_error = load_model()

if model_error:
    st.markdown(f'<span class="badge badge-err">⚠ Model unavailable — {model_error}</span>',
                unsafe_allow_html=True)
else:
    st.markdown('<span class="badge badge-ok">● Model loaded</span>', unsafe_allow_html=True)

st.markdown("<div style='height:.75rem'></div>", unsafe_allow_html=True)

# ─── Settings ─────────────────────────────────────────────────────────────────
with st.expander("⚙  SETTINGS", expanded=False):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        helmet_conf   = st.slider("Helmet confidence",    0.0, 1.0, 0.40, 0.05)
    with c2:
        nohelmet_conf = st.slider("No-helmet confidence", 0.0, 1.0, 0.25, 0.05)
    with c3:
        box_thickness = st.slider("Box thickness",        1,   6,   2)
    with c4:
        label_scale   = st.slider("Label size",           0.3, 1.5, 0.6, 0.05)

# ─── Helpers ──────────────────────────────────────────────────────────────────
def compute_iou(b1, b2):
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter   = max(0, x2-x1) * max(0, y2-y1)
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    return inter / (a1 + a2 - inter + 1e-6)


def _draw(img, label, conf, x1, y1, x2, y2, color):
    cv2.rectangle(img, (x1,y1), (x2,y2), color, box_thickness)
    text = f"{label}  {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, label_scale, 1)
    cv2.rectangle(img, (x1, y1-th-8), (x1+tw+6, y1), color, -1)
    cv2.putText(img, text, (x1+3, y1-5),
                cv2.FONT_HERSHEY_SIMPLEX, label_scale, (255,255,255), 1, cv2.LINE_AA)


def run_inference(frame):
    """High-quality dual-pass inference (images & video)."""
    counts = {"helmet": 0, "without_helmet": 0, "two_wheeler": 0}
    if model is None:
        return frame, counts

    res1 = model(frame,              conf=0.15, iou=0.45, imgsz=1536, max_det=300, verbose=False)[0]
    res2 = model(cv2.flip(frame, 1), conf=0.15, iou=0.45, imgsz=1536, max_det=300, verbose=False)[0]

    img = frame.copy()
    w   = frame.shape[1]

    flip_boxes = []
    for box in res2.boxes:
        cls = int(box.cls[0])
        if cls not in [0, 2]: continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        flip_boxes.append((cls, w-x2, y1, w-x1, y2))

    for box in res1.boxes:
        cls   = int(box.cls[0])
        if cls not in [0, 2]: continue
        conf  = float(box.conf[0])
        label = CLASS_NAMES[cls]
        if label == "helmet"         and conf < helmet_conf:   continue
        if label == "without_helmet" and conf < nohelmet_conf: continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        agreed = any(
            fcls == cls and compute_iou((x1,y1,x2,y2),(fx1,fy1,fx2,fy2)) > 0.4
            for fcls,fx1,fy1,fx2,fy2 in flip_boxes
        )
        if not agreed: continue

        counts[label] += 1
        color = (34, 197, 94) if label == "helmet" else (68, 68, 239)
        _draw(img, label, conf, x1, y1, x2, y2, color)

    return img, counts


def run_inference_fast(frame):
    """Single-pass, smaller imgsz for real-time webcam speed."""
    counts = {"helmet": 0, "without_helmet": 0, "two_wheeler": 0}
    if model is None:
        return frame, counts

    res = model(frame, conf=0.25, iou=0.45, imgsz=640, max_det=100, verbose=False)[0]
    img = frame.copy()

    for box in res.boxes:
        cls   = int(box.cls[0])
        if cls not in [0, 2]: continue
        conf  = float(box.conf[0])
        label = CLASS_NAMES[cls]
        if label == "helmet"         and conf < helmet_conf:   continue
        if label == "without_helmet" and conf < nohelmet_conf: continue

        counts[label] += 1
        color = (34, 197, 94) if label == "helmet" else (68, 68, 239)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        _draw(img, label, conf, x1, y1, x2, y2, color)

    return img, counts


def show_stats(counts):
    c1, c2, c3 = st.columns(3)
    c1.metric("With helmet",    counts.get("helmet", 0))
    c2.metric("Without helmet", counts.get("without_helmet", 0))
    c3.metric("Total riders",   counts.get("helmet", 0) + counts.get("without_helmet", 0))


# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab_photo, tab_video, tab_webcam = st.tabs(["IMAGE", "VIDEO", "WEBCAM"])

# ══ IMAGE ════════════════════════════════════════════════════════════════════
with tab_photo:
    up = st.file_uploader("", type=['jpg','jpeg','png','bmp','webp'],
                           label_visibility='collapsed')
    if up:
        img_pil  = Image.open(up).convert('RGB')
        frame_np = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        with st.spinner("Running inference…"):
            annotated, counts = run_inference(frame_np.copy())

        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<p class="img-label">Original</p>', unsafe_allow_html=True)
            st.image(img_pil, use_container_width=True)
        with col2:
            st.markdown('<p class="img-label">Detected</p>', unsafe_allow_html=True)
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

        st.markdown("<hr>", unsafe_allow_html=True)
        show_stats(counts)

# ══ VIDEO ════════════════════════════════════════════════════════════════════
with tab_video:
    up_vid = st.file_uploader("", type=['mp4','avi','mov','mkv'],
                               label_visibility='collapsed', key='vid')
    if up_vid:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(up_vid.read()); tfile.flush(); tfile.close()

        if st.button("▶  Process video"):
            cap   = cv2.VideoCapture(tfile.name)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps   = cap.get(cv2.CAP_PROP_FPS) or 30

            frame_ph = st.empty()
            prog_ph  = st.progress(0)
            stats_ph = st.empty()
            idx = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                annotated, counts = run_inference(frame)
                frame_ph.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                               use_container_width=True)
                idx += 1
                if total > 0:
                    prog_ph.progress(min(idx / total, 1.0))
                with stats_ph.container():
                    show_stats(counts)
                time.sleep(1.0 / fps)

            cap.release()
            prog_ph.progress(1.0)
            st.markdown('<p style="color:#888;font-size:.75rem;font-family:monospace">✓ Done.</p>',
                        unsafe_allow_html=True)

# ══ WEBCAM ═══════════════════════════════════════════════════════════════════
with tab_webcam:
    if "cam_on" not in st.session_state:
        st.session_state.cam_on = False

    col_a, col_b, _ = st.columns([1, 1, 4])
    with col_a:
        if st.button("▶  Start", key="cam_start", disabled=st.session_state.cam_on):
            st.session_state.cam_on = True
            st.rerun()
    with col_b:
        if st.button("■  Stop", key="cam_stop", disabled=not st.session_state.cam_on):
            st.session_state.cam_on = False
            st.rerun()

    if st.session_state.cam_on:
        st.markdown('<span class="badge badge-live">● live</span>', unsafe_allow_html=True)
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("Cannot open webcam.")
            st.session_state.cam_on = False
            st.rerun()
        else:
            frame_ph = st.empty()
            stats_ph = st.empty()

            while st.session_state.cam_on:
                ret, frame = cap.read()
                if not ret:
                    break
                annotated, counts = run_inference_fast(frame)
                frame_ph.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                               use_container_width=True)
                with stats_ph.container():
                    show_stats(counts)

            cap.release()
    else:
        st.markdown('<span class="badge badge-stop">● stopped</span>', unsafe_allow_html=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<hr>
<div style="display:flex;justify-content:space-between">
    <span style="font-family:'IBM Plex Mono',monospace;font-size:.62rem;color:#aaa;letter-spacing:.1em">HELMETGUARD · YOLOv11s</span>
    <span style="font-family:'IBM Plex Mono',monospace;font-size:.62rem;color:#aaa">helmet · two_wheeler · without_helmet</span>
</div>
""", unsafe_allow_html=True)