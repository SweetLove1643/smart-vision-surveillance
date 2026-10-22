import logging
import logging.handlers
import os
import re
import sys
import threading
from pathlib import Path

# <thư mục gốc project>/logs, không phụ thuộc thư mục đang chạy lệnh.
LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
LOG_FILE = LOG_DIR / "app.log"

FORMAT = "%(asctime)s | %(levelname)-7s | %(threadName)-14s | %(name)-12s | %(message)s"
MAX_BYTES = 5 * 1024 * 1024  # 5 MB / file
BACKUP_COUNT = 3             # giữ app.log + app.log.1 .. app.log.3

_configured = False


def mask_source(source) -> str:
    """Che user:pass trong URL (rtsp://user:pass@host -> rtsp://***@host) trước khi ghi log."""
    return re.sub(r"(://)[^/@\s]+@", r"\1***@", str(source))


def setup_logging(console_level: int = logging.INFO, file_level: int = logging.DEBUG) -> None:
    """Gọi MỘT lần ở đầu main(). File logs/app.log ghi chi tiết (DEBUG), console chỉ hiện INFO trở lên."""
    global _configured
    if _configured:
        return
    _configured = True

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(FORMAT)

    file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8"
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # mức lọc thật nằm ở từng handler
    root.addHandler(file_handler)
    root.addHandler(console_handler)

    logging.captureWarnings(True)
    _install_exception_hooks()


def _install_exception_hooks() -> None:
    """Lỗi chưa bắt (ở luồng chính hoặc thread con) cũng được ghi vào file, kèm traceback."""
    log = logging.getLogger("uncaught")

    def main_hook(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        log.critical("Lỗi chưa bắt", exc_info=(exc_type, exc, tb))

    def thread_hook(args):
        log.critical(
            "Lỗi chưa bắt trong thread %s", args.thread.name if args.thread else "?",
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    sys.excepthook = main_hook
    threading.excepthook = thread_hook


def log_startup_banner() -> None:
    logging.getLogger("main").info("===== Khởi động (pid=%d, python %s) =====", os.getpid(), sys.version.split()[0])