# dtector and tracker

Lightweight object detection and tracking dashboard built with Streamlit and Ultralytics YOLO.

## Features

- Object detection using Ultralytics YOLO model (load custom weights)
- Simple centroid-based tracking with stable IDs
- Dark themed Streamlit UI and sidebar controls
- Frame-skip, resize and FPS display options to reduce lag

## Requirements

- Python 3.8+
- See `requirements.txt` (streamlit, ultralytics, opencv-python, numpy)

## Quick install

```bash
python -m pip install -r requirements.txt
```

## Run the app

```bash
python -m streamlit run app.py
```

If you get `st: The term 'st' is not recognized`, use the `python -m streamlit run app.py` form shown above.

## How it works

- Upload a video in the app UI.
- The app resizes frames (configurable) and runs YOLO detection on every Nth frame (configurable `Frame skip`).
- Detected bounding boxes are filtered by a configurable `Confidence threshold`.
- A simple `CentroidTracker` matches detections between frames and assigns stable IDs.

## Sidebar controls (what to tune)

- **Model path**: path to your YOLO weights (default: `yolo26n.pt`). Place the file in the project root or provide absolute path.
- **Confidence threshold**: lower to detect weaker predictions, raise to reduce false positives.
- **Frame skip**: process every Nth frame (increase to reduce CPU usage / lag).
- **Resize processing width**: reduce to lower CPU/GPU load; smaller resolution = faster inference.
- **Show processing FPS**: overlays the measured processing FPS.
- **Max disappeared frames**: how long the tracker holds objects before deregistering them.
- **Start / Stop**: control processing run.

## Tuning recommendations

- If the app lags: increase `Frame skip` (e.g., 3-5) and reduce `Resize processing width` (320-640).
- If detections are missing: lower `Confidence threshold` (try 0.2) or increase processing width.
- For best performance, run on a machine with a CUDA GPU and install a CUDA-enabled `torch` build.

## Using a GPU (optional)

1. Install PyTorch with CUDA support: see https://pytorch.org for commands matching your CUDA version.
2. In `app.py`, after loading the model in `load_model()`, move it to the GPU (example):

```py
model = YOLO(path)
model.to('cuda:0')
```

3. Restart the Streamlit app.

## Common issues & troubleshooting

- Streamlit not found / `st` command error: use `python -m streamlit run app.py`.
- Model load error: ensure `model_path` is correct and the weight file is compatible with Ultralytics YOLO.
- No detections: ensure the model supports the target classes; try lowering the confidence threshold.
- Video not uploaded/played: make sure the file is a supported format (mp4/avi/mov).

## Notes

- The app currently uses a simple centroid tracker for stable IDs; this is lightweight and works well in many scenarios but is not as robust as specialized re-identification trackers for very long-term identity persistence.
- The sidebar still shows a `Tracker config` selector for backward compatibility, but the app uses the internal centroid tracker by default.

## Contributing

- Feel free to open PRs to add features: webcam input, GPU autodetection, model download helper, or a better tracking algorithm (DeepSORT / ByteTrack integration).

## License & contact

- This project is provided as-is for experimentation. Add your preferred license and contact info.
