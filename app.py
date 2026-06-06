from ultralytics import YOLO
import cv2
import streamlit as st
import tempfile
import os
import time
import numpy as np
from collections import OrderedDict
import torch


# Page config and dark theme CSS
st.set_page_config(page_title="dtector and tracker", layout="wide")
st.markdown(
    """
    <style>
    .stApp { background-color: #0e1117; color: #e6edf3; }
    .stSidebar { background-color: #0b0f14; }
    .css-1d391kg { background-color: #0b0f14; }
    .stButton>button { background-color: #1f6feb; color: white; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title
st.title("Detector Tracker")


def load_model(path):
    try:
        with st.spinner(f"Loading model from {path} ..."):
            model = YOLO(path)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


class CentroidTracker:
    def __init__(self, maxDisappeared=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared

    def register(self, centroid, bbox):
        self.objects[self.nextObjectID] = (centroid, bbox)
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        # rects: list of bboxes [x1,y1,x2,y2]
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        inputCentroids = []
        for (x1, y1, x2, y2) in rects:
            cX = int((x1 + x2) / 2.0)
            cY = int((y1 + y2) / 2.0)
            inputCentroids.append((cX, cY))

        if len(self.objects) == 0:
            for i, centroid in enumerate(inputCentroids):
                self.register(centroid, rects[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = [c for (c, _) in self.objects.values()]

            D = np.linalg.norm(np.array(objectCentroids)[:, None] - np.array(inputCentroids)[None, :], axis=2)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                objectID = objectIDs[row]
                self.objects[objectID] = (inputCentroids[col], rects[col])
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])) - usedRows
            unusedCols = set(range(0, D.shape[1])) - usedCols

            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col], rects[col])

        return self.objects


# Sidebar options
st.sidebar.header("Settings")
model_path = st.sidebar.text_input("Model path", value="yolo26n.pt")
tracker_cfg = st.sidebar.selectbox("Tracker config", options=["bytetrack.yaml", "sort.yaml"], index=0)
show_boxes = st.sidebar.checkbox("Show boxes", value=True)
 # (counting removed)
start_button = st.sidebar.button("Start processing")
frame_skip = st.sidebar.slider("Frame skip (process every Nth frame)", 1, 10, 2)
resize_width = st.sidebar.slider("Resize processing width", 320, 1280, 640)
show_fps = st.sidebar.checkbox("Show processing FPS", value=True)
conf_threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.25)
max_disappeared = st.sidebar.slider("Max disappeared frames", 1, 200, 50)
# Device selection (auto chooses CUDA if available)
device_option = st.sidebar.selectbox("Device", options=["auto", "cpu", "cuda:0"], index=0)
use_gpu = False
if device_option == "cpu":
    device = "cpu"
elif device_option == "cuda:0":
    device = "cuda:0"
else:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"):
        use_gpu = True

video_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

# Debugging options
show_detection_info = st.sidebar.checkbox("Show detection info", value=True)


if "stop_processing" not in st.session_state:
    st.session_state.stop_processing = False

if st.sidebar.button("Stop"):
    st.session_state.stop_processing = True

if start_button and video_file is not None:

    # Save uploaded file to a temp file
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    try:
        tmp.write(video_file.read())
        tmp.flush()
        tmp.close()

        # Load model
        model = load_model(model_path)
        if model is None:
            raise RuntimeError("Model load failed")

        # Try to move model to selected device
        try:
            model.to(device)
        except Exception:
            st.warning(f"Could not move model to {device}; continuing on CPU")

        # Open uploaded video file
        cap = cv2.VideoCapture(tmp.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25

        stframe = st.empty()
        progress_bar = st.progress(0)
        # counting removed
        last_results = None
        last_proc_time = None
        ct = CentroidTracker(maxDisappeared=max_disappeared)
        det_info_holder = st.sidebar.empty()

        frame_idx = 0

        while cap.isOpened():
            if st.session_state.stop_processing:
                st.warning("Processing stopped by user")
                break

            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            # Resize frame for faster processing
            if resize_width and frame.shape[1] != resize_width:
                scale = resize_width / frame.shape[1]
                new_h = max(1, int(frame.shape[0] * scale))
                frame_proc = cv2.resize(frame, (resize_width, new_h))
            else:
                frame_proc = frame


            # Run detection every Nth frame and update centroid tracker
            rects = []
            if frame_idx % frame_skip == 0:
                t_start = time.time()
                results = model(frame_proc)
                last_proc_time = time.time() - t_start

                boxes = getattr(results[0], "boxes", None)
                det_summary = []
                if boxes is not None:
                    for box in boxes:
                        try:
                            conf = float(box.conf[0])
                        except Exception:
                            conf = 1.0
                        cls = int(box.cls[0]) if getattr(box, "cls", None) is not None else -1
                        name = model.names.get(cls, str(cls)) if hasattr(model, 'names') else str(cls)
                        if conf < conf_threshold:
                            continue
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        rects.append((x1, y1, x2, y2))
                        det_summary.append((name, conf))
                        # overlay confidence near box
                        if show_detection_info:
                            cv2.putText(frame_proc, f"{name} {conf:.2f}", (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 100), 1)

                # update sidebar detection info
                if show_detection_info:
                    if len(det_summary) == 0:
                        det_info_holder.text("Detections: 0")
                    else:
                        labels = ", ".join([f"{n}:{c:.2f}" for (n, c) in det_summary])
                        det_info_holder.text(f"Detections: {len(det_summary)} — {labels}")
            else:
                # If skipping, keep last rects (centroid tracker will handle disappearance)
                rects = [bbox for (_, bbox) in ct.objects.values()]

            objects = ct.update(rects)

                # Draw tracked objects
            for objectID, (centroid, bbox) in objects.items():
                x1, y1, x2, y2 = bbox
                cx, cy = centroid
                if show_boxes:
                    cv2.rectangle(frame_proc, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame_proc, f"ID:{objectID}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 2)
                    cv2.circle(frame_proc, (cx, cy), 5, (255, 0, 0), -1)
                # counting removed

            # Optionally show processing FPS
            if show_fps and (last_proc_time is not None):
                fps_disp = 1.0 / max(last_proc_time, 1e-6)
                cv2.putText(frame_proc, f"FPS: {fps_disp:.1f}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

            # Convert and display
            frame_disp = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2RGB)
            stframe.image(frame_disp, channels="RGB", width=frame_proc.shape[1])

            # Update progress
            if total_frames > 0:
                progress_bar.progress(min(frame_idx / total_frames, 1.0))

            # Let Streamlit breathe
            time.sleep(max(1.0 / fps, 0.01))

        cap.release()
        st.success("Processing completed")

    finally:
        try:
            os.remove(tmp.name)
        except Exception:
            pass

    # Reset stop flag for next run
    st.session_state.stop_processing = False

elif start_button and video_file is None:
    st.warning("Please upload a video before starting processing.")