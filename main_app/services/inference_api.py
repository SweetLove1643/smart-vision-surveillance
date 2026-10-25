import asyncio
import logging
import threading
import time
from dataclasses import asdict
from typing import Optional

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request

from main_app.services.detector_service import (
    InferenceQueueFullError,
    InferenceUnavailableError,
    YoloInferenceService,
)
logger = logging.getLogger("services.inference_api")
MAX_FRAME_BYTES = 16 * 1024 * 1024


def create_inference_app(inference_service: YoloInferenceService) -> FastAPI:
    app = FastAPI(
        title="Shared YOLO Inference API",
        description="Shared model inference endpoint for camera pipelines.",
        version="1.0.0",
    )

    @app.get("/health")
    def health_check():
        status = inference_service.get_status()
        if not status["worker_running"]:
            raise HTTPException(status_code=503, detail="Inference worker is stopped.")
        return {"status": "ok", **status}

    @app.get("/metrics")
    def get_metrics():
        return inference_service.get_status()

    @app.post("/infer/{camera_id}")
    async def infer(camera_id: str, request: Request):
        if request.headers.get("content-type", "").split(";")[0] != "image/jpeg":
            raise HTTPException(
                status_code=415, detail="Send a JPEG frame as the request body."
            )
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > MAX_FRAME_BYTES:
                    raise HTTPException(
                        status_code=413, detail="Frame exceeds the size limit."
                    )
            except ValueError as exc:
                raise HTTPException(
                    status_code=400, detail="Invalid Content-Length header."
                ) from exc
        frame_bytes = await request.body()
        if not frame_bytes:
            raise HTTPException(status_code=400, detail="Frame body is empty.")
        if len(frame_bytes) > MAX_FRAME_BYTES:
            raise HTTPException(status_code=413, detail="Frame exceeds the size limit.")

        frame = cv2.imdecode(
            np.frombuffer(frame_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if frame is None:
            raise HTTPException(status_code=400, detail="Could not decode JPEG frame.")

        try:
            result = inference_service.submit(frame, camera_id)
        except InferenceQueueFullError as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        except InferenceUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Inference API failed for camera %s.", camera_id)
            raise HTTPException(
                status_code=500, detail="Inference request failed."
            ) from exc

        try:
            detections = await asyncio.wait_for(
                asyncio.wrap_future(result),
                timeout=inference_service.request_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            result.cancel()
            raise HTTPException(
                status_code=504,
                detail=f"Inference của camera {camera_id} quá thời gian chờ.",
            ) from exc
        except InferenceUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Inference API failed for camera %s.", camera_id)
            raise HTTPException(
                status_code=500, detail="Inference request failed."
            ) from exc

        return {"camera_id": camera_id, "detections": [asdict(item) for item in detections]}

    return app


class YoloInferenceAPIServer:
    """Runs the dedicated local HTTP API around the shared inference worker."""

    def __init__(
        self,
        inference_service: YoloInferenceService,
        host: str,
        port: int,
    ) -> None:
        self.inference_service = inference_service
        self.host = host
        self.port = port
        self._server: Optional[uvicorn.Server] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def is_running(self) -> bool:
        return (
            self._thread is not None
            and self._thread.is_alive()
            and self._server is not None
            and self._server.started
        )

    def start(self, timeout: float = 15.0) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self.inference_service.start()
        self._server = uvicorn.Server(
            uvicorn.Config(
                create_inference_app(self.inference_service),
                host=self.host,
                port=self.port,
                log_level="warning",
                lifespan="off",
                timeout_graceful_shutdown=max(
                    10, int(self.inference_service.request_timeout_seconds + 5)
                ),
            )
        )
        self._thread = threading.Thread(
            target=self._server.run,
            name="yolo-inference-api",
            daemon=True,
        )
        self._thread.start()

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._server.started:
                logger.info(
                    "Shared YOLO API is listening at http://%s:%d",
                    self.host,
                    self.port,
                )
                return
            if not self._thread.is_alive():
                self.inference_service.stop()
                raise RuntimeError("Shared YOLO API server stopped during startup.")
            time.sleep(0.05)
        self.stop()
        raise TimeoutError(
            f"Shared YOLO API did not start at {self.host}:{self.port}."
        )

    def stop(self, timeout: float = 15.0) -> None:
        thread = self._thread
        if thread is not None and thread.is_alive():
            if self._server is not None:
                self._server.should_exit = True
            thread.join(timeout=timeout)
            if thread.is_alive():
                raise TimeoutError("Shared YOLO API server did not stop in time.")
        self._thread = None
        self._server = None
        self.inference_service.stop(timeout=timeout)
