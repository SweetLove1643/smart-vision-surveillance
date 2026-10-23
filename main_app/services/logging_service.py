import logging
import logging.handlers
import os
from pathlib import Path
import re
import sys
import threading

# Ghi tập trung vào <thư mục gốc>/logs/app.log
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "app.log"

FORMAT = "%(asctime)s | %(levelname)-7s | %(threadName)-16s | %(name)-14s | %(message)s"
MAX_BYTES = 5 * 1024 * 1024  # 5 MB / file
BACKUP_COUNT = 3

_configured = False


def mask_source(source) -> str:
    """Che mật khẩu trong URL RTSP trước khi ghi log."""
    return re.sub(r"(://)[^/@\s]+@", r"\1***@", str(source))


def setup_logging(console_level: int = logging.INFO, file_level: int = logging.DEBUG) -> None:
    """Khởi tạo logging tập trung cho toàn bộ hệ thống."""
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
    root.setLevel(logging.DEBUG)
    root.addHandler(file_handler)
    root.addHandler(console_handler)

    logging.captureWarnings(True)

    # Bắt lỗi chưa xử lý trong main thread và sub-thread
    def uncaught_handler(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logging.getLogger("uncaught").critical("Lỗi chưa bắt:", exc_info=(exc_type, exc_value, exc_traceback))

    def thread_uncaught_handler(args):
        logging.getLogger("uncaught").critical(
            "Lỗi chưa bắt trong thread %s:",
            args.thread.name if args.thread else "?",
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    sys.excepthook = uncaught_handler
    threading.excepthook = thread_uncaught_handler


def log_startup_banner() -> None:
    logging.getLogger("main").info(
        "===== Khởi động Hệ thống Xử lý Video (pid=%d, Python %s) =====",
        os.getpid(),
        sys.version.split()[0],
    )
