import json
from typing import List
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import cv2
import numpy as np

from main_app.services.detector_service import (
    InferenceQueueFullError,
    InferenceUnavailableError,
    RawDetection,
)


class YoloInferenceClient:
    """Sends the newest camera frame to the dedicated local inference API."""

    def __init__(
        self,
        base_url: str,
        camera_id: str,
        timeout_seconds: float = 125.0,
        jpeg_quality: int = 85,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.camera_id = camera_id
        self.timeout_seconds = timeout_seconds
        self.jpeg_quality = jpeg_quality

    def infer(self, frame: np.ndarray) -> List[RawDetection]:
        success, encoded = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
        )
        if not success:
            raise RuntimeError(
                f"Could not JPEG encode frame for camera {self.camera_id}."
            )

        request = Request(
            f"{self.base_url}/infer/{quote(self.camera_id, safe='')}",
            data=encoded.tobytes(),
            headers={"Content-Type": "image/jpeg"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read())
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if exc.code == 429:
                raise InferenceQueueFullError(
                    f"YOLO API queue is full for camera {self.camera_id}: {detail}"
                ) from exc
            if exc.code == 503:
                raise InferenceUnavailableError(
                    f"YOLO API is unavailable for camera {self.camera_id}: {detail}"
                ) from exc
            if exc.code == 504:
                raise TimeoutError(
                    f"YOLO inference timed out for camera {self.camera_id}: {detail}"
                ) from exc
            raise RuntimeError(
                f"YOLO API returned HTTP {exc.code} for camera "
                f"{self.camera_id}: {detail}"
            ) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(
                f"Could not reach YOLO API at {self.base_url} for camera "
                f"{self.camera_id}: {exc}"
            ) from exc

        if payload.get("camera_id") != self.camera_id:
            raise RuntimeError(
                f"YOLO API returned a response for unexpected camera "
                f"{payload.get('camera_id')!r}."
            )
        try:
            detections = []
            for item in payload["detections"]:
                box = tuple(int(value) for value in item["box"])
                if len(box) != 4:
                    raise ValueError("Detection box must contain four coordinates.")
                detections.append(
                    RawDetection(
                        class_id=int(item["class_id"]),
                        class_name=str(item["class_name"]),
                        conf=float(item["conf"]),
                        box=(box[0], box[1], box[2], box[3]),
                    )
                )
            return detections
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("YOLO API response has an invalid schema.") from exc
