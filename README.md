# Multi-Camera Real-Time Video Processing System

> **Status: Work in Progress (WIP)**  
> An intelligent, scalable multi-camera processing and analytics pipeline built in Python, designed to handle concurrent video streams with low latency, robust auto-reconnection, and edge AI vision tasks.

---

## 📌 Project Overview

This project is a high-performance Python application designed to ingest and process multiple video sources simultaneously (RTSP camera streams, webcams, and pre-recorded video files). Built for video surveillance and intelligent analytics, the system isolates failures per camera stream, guarantees real-time frame processing by eliminating buffer lag, and provides an interactive monitoring GUI.

### Core Processing Pipeline

```
[ Camera / RTSP / Video ]
           │
           ▼
    [ Video Capture ] ──── (Auto-reconnect & Drop-frame buffer)
           │
           ▼
  [ Detect & Track ] ──── (YOLOv8 + Object Tracking)
           │
           ▼
  [ Event Processing ] ──┬──> [ Event Database (SQLite) & Snapshots ]
           │             └──> [ Rotating File Logger ]
           ▼
    [ Visual Overlay ]
           │
           ▼
 [ FFmpeg / RTSP Out ] ───> Live Stream (VLC / Client)
```

---

## 🚀 Key Features & Roadmap

The project is built against strict real-time multi-camera processing specifications:

| Code | Feature Description | Status |
| :---: | :--- | :---: |
| **F01** | Multi-source ingestion (RTSP, webcam, files) with automatic reconnect and fault isolation (one failed stream will not crash others) | **Implemented** |
| **F02** | Zero-latency buffer strategy: always process the freshest frame, dropping queued frames to prevent lag accumulation | **Implemented** |
| **F03** | Person & Vehicle detection using YOLO (v8+) with dynamic thresholds and input size | *In Progress* |
| **F04** | Real-time multi-object tracking (MOT) | *In Progress* |
| **F05** | Bidirectional line crossing counter (inbound / outbound separated) | *Planned* |
| **F06** | Restricted area / ROI intrusion detection (arbitrary polygons) | *Planned* |
| **F07** | Deduplicated event recording (Camera, Track ID, Event Type, Timestamp, Snapshot image) into SQLite | *Planned* |
| **F08** | Frame annotations (bounding boxes, IDs, counters, polygons) and FFmpeg RTSP restreaming with auto-recovery | *Planned* |
| **F09** | Configuration management via YAML/JSON with startup validation | **Implemented** |
| **F10** | FastAPI management service: stream status, FPS metrics, live stream toggle, and event query endpoints | *Planned* |

---

## 🛠️ Architecture & Tech Stack

- **Language:** Python 3.10+
- **GUI Framework:** PySide6 (Qt for Python) with dark theme and dynamic grid layout (up to 16 cameras).
- **Computer Vision:** OpenCV (cv2) with TCP-forced RTSP transport for packet drop resistance.
- **AI & Analytics (Targeted):** Ultralytics YOLOv8 / ByteTrack / BoT-SORT.
- **Streaming & Transcoding:** FFmpeg for RTSP rebroadcasting.
- **Backend & Storage:** FastAPI, SQLite, Pydantic.
- **Testing & Tooling:** Pytest, Python `logging` with rotating file handlers.

---

## 📁 Repository Structure

```text
multi_camera_process_system/
├── main_app/
│   ├── controllers/
│   │   └── camera_controller.py   # Multi-camera orchestration & lifecycle manager
│   ├── services/
│   │   ├── capture.py             # Threaded frame reader with exponential backoff reconnect
│   │   └── logging_setup.py       # Rotating log handlers and exception hooks
│   ├── ui/
│   │   ├── cameras.json           # Active camera stream configuration
│   │   ├── ui_config.json         # Interface layout configuration
│   │   └── main_window.py         # PySide6 desktop monitoring dashboard
│   └── logs/                      # Application runtime logs
├── resources/
│   └── videos/                    # Sample video feeds (.gitkeep preserved)
├── tests/                         # Unit tests (line crossing, ROI intrusion)
├── Description.pdf                # Detailed project requirements & assignment brief
├── main.py                        # Application entrypoint
├── requirements.txt               # Project dependencies
└── README.md
```

---

## ⚡ Getting Started

### 1. Prerequisites

- Python 3.10 or higher
- Git
- (Optional) FFmpeg installed and added to your system PATH

### 2. Installation

Clone the repository:
```bash
git clone https://github.com/<your-username>/multi_camera_process_system.git
cd multi_camera_process_system
```

Create and activate a virtual environment:
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Configuration

Camera sources are defined in [main_app/ui/cameras.json](file:///main_app/ui/cameras.json):
```json
[
  {"name": "CAM 1", "source": "resources/videos/C001.mp4"},
  {"name": "CAM 2", "source": "rtsp://admin:password@192.168.1.100:554/stream"},
  {"name": "CAM 3", "source": 0}
]
```
- **Webcam:** Set `source` to an integer index (e.g. `0`).
- **RTSP:** Set `source` to the RTSP URL (e.g. `rtsp://...`).
- **Video File:** Set `source` to the local file path (files automatically loop when finished).

### 4. Running the Application

Launch the desktop monitoring interface:
```bash
python main.py
```

---

## 🎯 Non-Functional Requirements & Targets

- **Performance:** Maintain $\ge 10\text{ FPS}$ output per camera across 4 concurrent streams with end-to-end latency $< 1\text{ second}$.
- **Clean Teardown:** Graceful shutdown of threads, subprocesses, and connections upon `Ctrl+C` or application exit.
- **Production Logging:** Mask sensitive credentials in RTSP URLs and record reconnect metrics, FPS, and exceptions to rotating logs (`logs/app.log`).
- **Containerization:** Complete Docker / Docker Compose setup for deployment.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE) (or academic project license).
