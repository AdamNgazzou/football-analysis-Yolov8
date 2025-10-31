# Football Match Spatial Analysis ⚽🚀

A modular Computer Vision pipeline that extracts high-level tactical metrics (player/ball trajectories, team possession, speeds and distances) from raw broadcast-style video. The system uses a fine-tuned YOLOv8 detector combined with tracking, camera-motion compensation and a view transformer to produce real-world spatial data (meters, m/s) suitable for tactical analysis and visualization.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-red.svg)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-18b68f)](https://ultralytics.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub last commit](https://img.shields.io/github/last-commit/AdamNgazzou/football-analysis-Yolov8)](https://github.com/AdamNgazzou/football-analysis-Yolov8)

## Visuals

![Demo GIF / Screenshot placeholder](assets/demo.gif)

Replace `assets/demo.gif` with a short GIF or screenshot showing annotated output frames.

## Key Features

- Object detection with YOLOv8 fine‑tuned on custom football data 🎯
- Multi-object tracking (players, referees, ball) with configurable tracker parameters ⚙️
- Camera Movement Estimator to compensate pan/tilt/zoom and remove camera-induced motion 🧭
- View Transformer that maps pixel coordinates to real-world field coordinates (meters) — enables true speed and distance metrics 📐
- Ball position interpolation and track smoothing for robust analytics ✨
- Team color assignment and team-level possession tracking 🟦🟥
- Save / load intermediate stubs (pickled track data) for fast re-run and offline editing 💾

## Technical Pipeline

1. Read frames from an input video in `input_videos/`.
2. Detect objects per-frame using YOLOv8 (`models/best.pt`).
3. Track detections across frames using ByteTrack / SORT → produce `tracks` stubs in `stubs/`.
4. Add pixel positions per-object (foot position for players, center for ball).
5. Estimate and compensate camera movement per-frame (Camera Movement Estimator).
6. Transform compensated pixel coordinates to world coordinates via the View Transformer (meters).
7. Interpolate missing ball positions and smooth player trajectories.
8. Compute speeds (m/s) and cumulative distances (m) in `speed_and_distance_estimator/`.
9. Assign teams (color clustering) and compute possession timelines in `team_assigner/`.
10. Produce annotated output video in `output_videos/` and persist stubs for later use.

## Tech Stack

- Python 3.10+ 🐍
- YOLOv8 (Ultralytics) for detection 🔍
- OpenCV for image processing & perspective transforms 🖼️
- supervision / ByteTrack or SORT for tracking 🧭
- scikit-learn (KMeans) for team color clustering 🎨
- numpy, pandas for data processing

## Repository Overview (modules)

- `trackers/` — detection + tracking logic and annotation drawing
- `team_assigner/` — color clustering and team assignment
- `camera_movement_estimator/` — camera motion estimation and compensation
- `view_transformer/` — pixel → world coordinate transform (meters)
- `speed_and_distance_estimator/` — compute speeds, distances and derived metrics
- `stubs/` — cached pickled tracks & camera movement for fast replay
- `input_videos/`, `models/`, `output_videos/`, `runs/detect/`

## Setup & Installation (PowerShell)

Prerequisites:

- Python 3.10+
- (Optional) CUDA and GPU drivers for faster YOLO inference

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Place YOLO weights into `models/` (e.g. `models/best.pt`) and input videos into `input_videos/`.

## Usage

Process a video with defaults:

```powershell
python main.py
```

Notes:

- `main.py` uses `input_videos/08fd33_4.mp4` by default and writes annotated output to `output_videos/`.
- To reuse cached detection/tracking results, enable `read_from_stub=True` and set `stub_path` inside `main.py`.
- For raw detection tests you can run `yolo_inference.py`.

## Recommended Tracker Parameters (football)

Tune these for crowded scenes and short occlusions:

```python
self.tracker = sv.ByteTrack(
    track_activation_threshold=0.3,    # detect true objects while limiting false starts
    lost_track_buffer=20,              # frames to keep a lost track before deletion
    minimum_matching_threshold=0.7,    # stricter matching reduces ID switches
    frame_rate=24                      # match your video FPS
)
```

Adjust thresholds per video resolution and camera behavior.

## Results & Metrics

Outputs produced by the pipeline:

- Annotated video with overlays → `output_videos/*.avi` 🖼️
- Pickled stubs with structured tracks → `stubs/track_stubs*.pkl` (players, referees, ball) 💾
- Camera movement stubs → `stubs/camera_movement_*.pkl` 🧭
- Per-player time-series: pixel positions, adjusted positions, transformed world positions (meters), instantaneous speed (m/s), cumulative distance (m) 📈
- Team possession / ball-control timeline (array per-frame)

These artifacts can be exported to CSV/JSON for downstream analysis or visualization.

## Customization & Extension

- Replace or fine-tune `models/best.pt` to improve detection for your data.
- Adjust tracker thresholds / distance functions in `trackers/tracker.py` to reduce ID switches.
- Add appearance descriptors (jersey embeddings) for re-identification to improve consistency.
- Calibrate the `view_transformer` source/target points to your camera/pitch for accurate meter-scale results.

## Troubleshooting

- Frequent ID switches: increase `minimum_matching_threshold` or add appearance-based matching.
- KMeans errors (empty crops): ensure bounding boxes are clamped to frame bounds before color clustering.
- Use `stubs/` to speed development: generate once with full detection and reuse for algorithm tuning.

## License & Contribution

Add a license file as needed. Contributions welcome — open an issue or submit a PR with improvements.

---

Want a demo GIF and a CSV export of player speeds? Provide a short sample clip and I can add example scripts and a notebook for visualization. 🚀
