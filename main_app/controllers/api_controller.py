import time
from typing import Optional
import cv2
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from main_app.controllers.camera_controller import CameraController


def create_api_app(controller: CameraController) -> FastAPI:
    """Khởi tạo FastAPI ứng dụng để quản trị toàn bộ hệ thống (F10)."""
    app = FastAPI(
        title="Multi-Camera Real-Time Vision Analytics API",
        description="REST API quản lý trạng thái camera, xem trực tiếp trên web và tra cứu sự kiện vi phạm.",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Phục vụ xem ảnh snapshot vi phạm: http://localhost:8000/snapshots/<tên_ảnh>.jpg
    snapshot_dir = Path(controller.config.database.snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/snapshots", StaticFiles(directory=str(snapshot_dir)), name="snapshots")

    @app.get("/health", tags=["Hệ thống"])
    def health_check():
        return {"status": "ok", "total_cameras": len(controller.pipelines)}

    @app.get("/cameras", tags=["Quản lý Camera"])
    def get_cameras():
        """Xem trạng thái, FPS và số liệu chi tiết của từng camera (F10)."""
        return controller.get_all_status()

    @app.get("/counts", tags=["Thống kê"])
    def get_counts():
        """Trả tổng lượt cắt vạch đã lưu bền vững theo camera."""
        return controller.db.get_counts()

    @app.post("/cameras/{camera_id}/start", tags=["Quản lý Camera"])
    def start_camera(camera_id: str):
        """Bật một camera khi hệ thống đang chạy (F10)."""
        if not controller.start_camera(camera_id):
            raise HTTPException(status_code=404, detail=f"Không tìm thấy camera: {camera_id}")
        return {"message": f"Camera '{camera_id}' đã bật."}

    @app.post("/cameras/{camera_id}/stop", tags=["Quản lý Camera"])
    def stop_camera(camera_id: str):
        """Tắt một camera khi hệ thống đang chạy (F10)."""
        if not controller.stop_camera(camera_id):
            raise HTTPException(status_code=404, detail=f"Không tìm thấy camera: {camera_id}")
        return {"message": f"Camera '{camera_id}' đã tắt."}

    @app.get("/events", tags=["Sự kiện vi phạm"])
    def get_events(
        camera_id: Optional[str] = Query(None, description="Lọc theo mã camera (ví dụ: cam_1)"),
        start_time: Optional[str] = Query(None, description="Từ thời gian (YYYY-MM-DD HH:MM:SS)"),
        end_time: Optional[str] = Query(None, description="Đến thời gian (YYYY-MM-DD HH:MM:SS)"),
        limit: int = Query(50, ge=1, le=500, description="Số lượng sự kiện"),
    ):
        """Tra cứu lịch sử sự kiện đếm xe & xâm nhập vùng cấm lưu trong SQLite (F10)."""
        events = controller.db.get_events(
            camera_id=camera_id,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
        )
        return {"total": len(events), "events": events}

    @app.get("/cameras/{camera_id}/live", tags=["Xem Trực Tiếp"])
    def stream_live_mjpeg(camera_id: str):
        """Xem trực tiếp luồng video trên trình duyệt Web (Chrome/Edge) qua giao thức MJPEG."""
        if camera_id not in controller.pipelines:
            raise HTTPException(status_code=404, detail=f"Không tìm thấy camera: {camera_id}")

        def frame_generator():
            while True:
                frame = controller.get_frame_by_id(camera_id)
                if frame is not None:
                    ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    if ret:
                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
                        )
                time.sleep(0.04)  # ~25 FPS

        return StreamingResponse(
            frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame"
        )

    return app
