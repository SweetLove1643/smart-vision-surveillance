# Multi-Camera Video Processing System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![PySide6](https://img.shields.io/badge/PySide6-Qt6-41CD52.svg)](https://wiki.qt.io/Qt_for_Python)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8-yellowgreen.svg)](https://github.com/ultralytics/ultralytics)

A multi-camera video processing and analytics service in Python. Ingests concurrent video streams (RTSP, webcams, and local video files), runs YOLO object detection with per-camera tracking, performs event analytics (bidirectional line crossing counter and restricted-zone intrusion alerts), saves events and violation snapshots to SQLite, restreams annotated feeds via FFmpeg/RTSP, and provides a REST API (FastAPI) and an on-demand Qt6 desktop monitoring dashboard. The measured four-camera CPU run is not real-time; see the performance limitations below.

---

## 🏛️ System Architecture

```text
[ Video Sources ] (RTSP / Webcam / Local Video Files)
       │
       ▼
 [ CaptureService ] ───── (Threaded, exponential backoff auto-reconnect, zero-latency buffer)
       │
       ▼
 [ Shared YOLO Inference API :8001 ] ── (one model, serialized bounded requests)
       │
       ▼
 [ Per-Camera ByteTrack + Analytics ] ──┬──> [ StorageService ] ──> SQLite DB & Snapshots (`resources/database/`)
                         └──> [ LoggingService ] ──> Centralized file log (`logs/app.log`)
       │
       ▼
 [ FFmpeg Publisher ] ──> [ MediaMTX RTSP Server ]
       │
       ├─────────────────────────────────┐
       ▼                                 ▼
[ FastAPI Server ] (Port 8000)    [ PySide6 UI Monitor ] (On-demand with `--ui`)
├─ Web Swagger UI (`/docs`)       └─ Live grid view with annotated feeds & counters
├─ Direct MJPEG live stream
└─ REST endpoints (F10)
```

All camera pipelines send JPEG frames over HTTP to a dedicated YOLO inference
API (`127.0.0.1:8001` by default). The API loads one model and serializes work
through a bounded queue. It runs as a separate API server in the application
process (not a separately managed OS process). Each pipeline keeps its own
ByteTrack tracker and analytics state. Both the desktop UI and MJPEG endpoint
read the published RTSP output, matching the feed VLC receives.
Video publishing is paced independently from inference: each camera keeps only
its newest pending inference frame and displays its latest detections between
inference results. This keeps the video feed moving but does not increase the
model's inference rate; low inference FPS can still make tracking/counting
inaccurate.

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
│   │   ├── inference_api.py       # Dedicated shared YOLO HTTP API
│   │   ├── inference_client.py    # Camera-to-inference API client
│   │   ├── analytics_service.py   # Line crossing & zone intrusion analyzers (F05, F06)
│   │   ├── storage_service.py     # SQLite persistence & snapshots manager (F07)
│   │   ├── streamer_service.py    # FFmpeg RTSP streamer with auto-recovery (F08)
│   │   └── logging_service.py     # Centralized logging into logs/app.log
│   └── ui/
│       └── main_window.py         # PySide6 desktop monitoring dashboard
│
├── resources/
│   ├── database/                  # SQLite database (events.db) & violation snapshots
│   ├── models/                    # YOLO source weights and OpenVINO exports
│   └── videos/                    # Sample test video feeds (.gitkeep preserved)
│
├── logs/                          # Centralized runtime application logs (app.log)
└── tests/                         # Pytest coverage for analytics, inference API,
                                   # camera pipeline, config, and RTSP streamer
```

---

## Model, performance, and counting

- `model.weights` may point to an OpenVINO model directory or a `.pt`/`.onnx` source.
  Source models are exported to OpenVINO FP16 once at startup and then used on CPU. A
  matching `<model>_openvino_model` export is reused when its input size matches
  `model.imgsz`. If export fails, a warning is logged before PyTorch CPU fallback.
- `model.conf` filters detections before they reach ByteTrack. High values (for
  example `0.7`) can remove low-confidence detections ByteTrack would otherwise
  associate across occlusion; the application warns when this value is above
  `0.25`. Tune it using representative footage and verify counting results.
- `model.inference_api_host` and `model.inference_api_port` configure the private
  YOLO service listener (default `127.0.0.1:8001`). Its interactive API docs are
  available at `http://127.0.0.1:8001/docs` while the application is running.
  `POST /infer/{camera_id}` accepts a JPEG image body and returns raw detections;
  ByteTrack remains downstream and isolated per camera.
- `model.inference_timeout_seconds` bounds how long a frame may wait for shared
  inference (default 120 seconds). Queue saturation is returned as HTTP 429 and
  logged as a dropped frame, not as an unhandled server error. Failed inference
  does not feed an empty detection set into ByteTrack.
- The shared API processes inference requests serially. A shared model reduces
  duplicate model memory but does not provide parallel CPU inference. On this
  machine, the four-camera logs from 2026-10-24 around 16:59 showed approximately
  `0.04 inference FPS` per camera, `22.5–25.6 s` inference latency, and
  `3.2–7.9 output FPS`. These are observed samples, not a guaranteed benchmark,
  and are far below real-time detection/counting requirements. Video output FPS
  must not be interpreted as detection FPS. Do not rely on multi-camera counts
  for operational decisions until throughput and accuracy are validated against
  labeled footage.
- Per-camera `input_fps`, `output_fps`, `inference_fps`, average inference latency, and
  processing latency are logged every `model.metrics_log_interval_seconds` to
  `logs/app.log`. `target_fps` is a maximum, not a guaranteed throughput.
- RTSP uses wall-clock frame timestamps, a finer timestamp time base, and
  variable-frame-rate pacing so variable output cadence does not force video to play
  faster than it was produced. Encoded keyframes are emitted regularly so RTSP
  readers can join a live feed without waiting for a long GOP.
- `model.skip_frames` controls the inference sampling cadence relative to the
  output loop. When inference is busy, pending frames are replaced with the
  newest captured frame rather than queued stale frames. Tracking and
  line/zone analytics update only when inference returns; they do not run on
  every output frame. Line-crossing totals are stored per camera in SQLite and
  survive restarts.
- Line and restricted-zone coordinates are configured separately for each camera
  in `config.yaml` and use source-frame pixel coordinates (the included clips are
  1920x1080). The current polygons are provisional overlays approved for initial
  testing; review them against installed camera views before relying on alerts.

## RTSP output and VLC

The application publishes annotated feeds with FFmpeg; it does **not** run an
RTSP server itself. For native Windows CLI operation, the repository includes
`mediamtx.yml` for the separately installed `mediamtx.exe`. Stop any existing
MediaMTX instance first, then start it from the project root with the explicit
configuration:

```powershell
.\mediamtx.exe .\mediamtx.yml
```

The configuration listens on TCP port 8554, accepts publisher/read clients on
the camera paths, and disables unused MediaMTX protocols to avoid unrelated
port conflicts. Keep the server window open, then start the application with
`python main.py --config config.yaml` (do not add `--no-rtsp`). With the sample
configuration, the VLC URL for the first camera is
`rtsp://<server-ip>:8554/live/cam_1` (use `cam_2`, `cam_3`, or `cam_4` for the
other feeds). The configured `localhost` destination only works when the RTSP
server is on the same machine as this application; for a separate server, update
each camera's `rtsp_out` to that server's reachable IP/hostname. `main_window.py`
and `/cameras/{camera_id}/live` consume the same configured RTSP output, rather
than displaying a private in-memory pre-publish frame.

Verify that `Get-NetTCPConnection -LocalPort 8554 -State Listen` reports the
MediaMTX process before starting the application. If FFmpeg logs `400 Bad
Request`, MediaMTX was likely started without this publisher-path configuration.
The native MediaMTX configuration was validated with MediaMTX 1.21.2; the
application's FFmpeg publisher and an RTSP reader were also smoke-tested against
that configured server. A server started with its default configuration (no
publisher path) rejected FFmpeg with `400 Bad Request`, even though port 8554
was open.

For CLI operation without an RTSP server or FFmpeg, use
`python main.py --no-rtsp --camera cam_1`. This keeps capture, shared YOLO
inference, ByteTrack, counting, and the REST API enabled while disabling RTSP
publishing and RTSP readback. The browser live-feed and desktop UI views require
RTSP output and are not available in this mode. `--camera` may be repeated to
select several enabled cameras; omitting it starts all enabled cameras, which
may overload CPU inference. Without `--no-rtsp`, the configured RTSP server must
already be listening before the application starts.

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

#### Option A: Run the CLI/API without an RTSP server
```bash
python main.py --no-rtsp --camera cam_1
```
- **API & Swagger UI:** Open [http://localhost:8000/docs](http://localhost:8000/docs)
- Use `GET /cameras`, `GET /inference`, and `GET /counts` to inspect runtime status.
- Add `--camera cam_2` to select additional cameras; by default all enabled cameras start.

#### Option B: Run as a Backend Service with RTSP output
```bash
python main.py
```
- **API & Swagger UI:** Open [http://localhost:8000/docs](http://localhost:8000/docs)
- **Watch Live Camera directly in Browser (MJPEG):** Open [http://localhost:8000/cameras/cam_1/live](http://localhost:8000/cameras/cam_1/live)

#### Option C: Run Service with PySide6 (Qt6) Monitoring GUI
```bash
python main.py --ui
```
This starts the backend API service and simultaneously opens the desktop monitoring window showing camera feeds, bounding boxes, counting lines, ROI polygons, and event tables. Detection overlays and counters update at inference speed, which may be much slower than the video output.

---

## 📡 REST API Endpoints (F10)

- `GET /health` : API/inference-worker health and camera count; not a full health check of camera pipelines, RTSP publishing, or storage.
- `GET /inference` : Shared model/API status, queue depth, request count, and latency.
- `GET /cameras` : Live status, FPS, latency, and reconnect count for all cameras.
- `GET /counts` : Persisted line-crossing totals by camera.
- `POST /cameras/{camera_id}/start` : Start a camera pipeline dynamically.
- `POST /cameras/{camera_id}/stop` : Stop a camera pipeline dynamically.
- `GET /events` : Query violation events (filters: `camera_id`, `start_time`, `end_time`, `limit`).
- `GET /cameras/{camera_id}/live` : Live video stream viewable directly in any browser.
- `GET /snapshots/{filename}` : Access captured violation snapshot images.

## Validation status (2026-10-24)

- Automated suite: **32 tests passed**; Python compile check and
  `git diff --check` passed. The test run emits one Starlette/httpx deprecation
  warning.
- RTSP: MediaMTX 1.21.2 configuration validation passed. A local integration
  smoke test published H.264 with the application's FFmpeg streamer and
  received a 320×240 frame through an RTSP reader. This verifies the transport
  path, not the four-camera application's sustained operation.
- CPU baseline: the latest recorded four-camera samples were approximately
  0.04 inference FPS per camera with 22.5–25.6 seconds average inference
  latency. The system can publish video independently, but detection/tracking/
  counting is not real-time under this load.
- Counting precision/recall and count error remain **unmeasured** because there
  is no labeled ground-truth dataset. The provisional line/zone coordinates
  still need review against each installed camera view.
- The latest application log contains FFmpeg broken-pipe errors from a run
  against an RTSP server that rejected publishing. Confirm the server was
  started with `.\mediamtx.exe .\mediamtx.yml` and verify the active output
  paths before relying on VLC.

---

## 📄 License
This project is licensed under the MIT License.
