# Multi-Camera Real-Time Video Processing System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![PySide6](https://img.shields.io/badge/PySide6-Qt6-41CD52.svg)](https://wiki.qt.io/Qt_for_Python)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8-yellowgreen.svg)](https://github.com/ultralytics/ultralytics)

A real-time, fault-tolerant multi-camera video processing and analytics service in Python. Ingests concurrent video streams (RTSP, webcams, and local video files), runs YOLOv8 object detection & tracking (person & vehicle), performs event analytics (bidirectional line crossing counter & restricted zone intrusion alert), saves events and violation snapshots to SQLite, restreams annotated feeds via FFmpeg/RTSP, and provides both a REST API (FastAPI) and an on-demand Qt6 desktop monitoring dashboard.

---

## 🏛️ System Architecture

```text
[ Video Sources ] (RTSP / Webcam / Local Video Files)
       │
       ▼
 [ CaptureService ] ───── (Threaded, exponential backoff auto-reconnect, zero-latency buffer)
       │
       ▼
 [ DetectorService ] ──── (YOLOv8 object detection & persistent multi-object tracking)
       │
       ▼
[ AnalyticsService ] ────┬──> [ StorageService ] ──> SQLite DB & Snapshots (`resources/database/`)
                         └──> [ LoggingService ] ──> Centralized file log (`logs/app.log`)
       │
       ▼
 [ StreamerService ] ────> FFmpeg RTSP Re-streaming (Auto-restart on exit)
       │
       ├─────────────────────────────────┐
       ▼                                 ▼
[ FastAPI Server ] (Port 8000)    [ PySide6 UI Monitor ] (On-demand with `--ui`)
├─ Web Swagger UI (`/docs`)       └─ Live grid view with real-time counters & alerts
├─ Direct MJPEG live stream
└─ REST endpoints (F10)
```

---

## 📁 Repository Structure

```text
multi_camera_process_system/
├── config.yaml                    # Centralized system YAML configuration (F09)
├── main.py                        # Entrypoint (Service mode / UI mode with --ui)
├── requirements.txt               # Dependencies
├── README.md                      # Documentation
├── .gitignore                     # Git ignore rules
│
├── main_app/
│   ├── controllers/
│   │   ├── camera_controller.py   # Multi-camera pipeline lifecycle manager
│   │   └── api_controller.py      # FastAPI REST API controller & MJPEG streamer (F10)
│   ├── services/
│   │   ├── config_service.py      # Pydantic schema validation for config.yaml (F09)
│   │   ├── capture_service.py     # Threaded capture, auto-reconnect & drop-frame (F01, F02)
│   │   ├── detector_service.py    # YOLOv8 object detection & MOT tracking (F03, F04)
│   │   ├── analytics_service.py   # Line crossing & zone intrusion analyzers (F05, F06)
│   │   ├── storage_service.py     # SQLite persistence & snapshots manager (F07)
│   │   ├── streamer_service.py    # FFmpeg RTSP streamer with auto-recovery (F08)
│   │   └── logging_service.py     # Centralized logging into logs/app.log
│   └── ui/
│       └── main_window.py         # PySide6 desktop monitoring dashboard
│
├── resources/
│   ├── database/                  # SQLite database (events.db) & violation snapshots
│   ├── models/                    # YOLO model weights (yolov8n.pt)
│   └── videos/                    # Sample test video feeds (.gitkeep preserved)
│
├── logs/                          # Centralized runtime application logs (app.log)
└── tests/
    └── test_analytics.py          # Pytest unit tests for line counter & zone intrusion
```

---

## ⚡ Getting Started

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/multi-camera-processing-system.git
cd multi-camera-processing-system

# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Unit Tests

```bash
pytest -v
```

### 3. Launch the Application

#### Option A: Run as a Backend Service (Standard)
```bash
python main.py
```
- **API & Swagger UI:** Open [http://localhost:8000/docs](http://localhost:8000/docs)
- **Watch Live Camera directly in Browser (MJPEG):** Open [http://localhost:8000/cameras/cam_1/live](http://localhost:8000/cameras/cam_1/live)

#### Option B: Run Service with PySide6 (Qt6) Monitoring GUI
```bash
python main.py --ui
```
This starts the backend API service and simultaneously opens the desktop monitoring window showing real-time camera feeds, bounding boxes, counting lines, ROI polygons, and event tables.

---

## 📡 REST API Endpoints (F10)

- `GET /health` : Health check and camera count.
- `GET /cameras` : Live status, FPS, latency, and reconnect count for all cameras.
- `POST /cameras/{camera_id}/start` : Start a camera pipeline dynamically.
- `POST /cameras/{camera_id}/stop` : Stop a camera pipeline dynamically.
- `GET /events` : Query violation events (filters: `camera_id`, `start_time`, `end_time`, `limit`).
- `GET /cameras/{camera_id}/live` : Live video stream viewable directly in any browser.
- `GET /snapshots/{filename}` : Access captured violation snapshot images.

---

## 📄 License
This project is licensed under the MIT License.
