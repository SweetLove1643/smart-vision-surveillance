import argparse
import logging
import signal
import sys
import threading
import uvicorn
from PySide6.QtWidgets import QApplication

from main_app.controllers.api_controller import create_api_app
from main_app.controllers.camera_controller import CameraController
from main_app.services import load_config, log_startup_banner, setup_logging
from main_app.ui.main_window import MainWindow

logger = logging.getLogger("main")


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Camera Real-Time Vision Analytics System")
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Khởi động giao diện màn hình giám sát PySide6 (Qt6) cùng với hệ thống.",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Đường dẫn tới file cấu hình YAML (mặc định: config.yaml).",
    )
    args = parser.parse_args()

    # 1. Khởi tạo logging tập trung vào logs/app.log
    setup_logging()
    log_startup_banner()

    # 2. Đọc và xác thực cấu hình YAML (F09)
    try:
        config = load_config(args.config)
        logger.info("Nạp cấu hình thành công: %d camera.", len(config.cameras))
    except Exception as exc:
        logger.critical("LỖI CẤU HÌNH KHỞI ĐỘNG (F09): %s", exc)
        sys.exit(1)

    # 3. Khởi tạo CameraController điều phối toàn bộ pipelines
    controller = CameraController(config)

    # 4. Xử lý ngắt Ctrl+C dừng sạch sẽ
    def shutdown_handler(signum, frame):
        logger.info("Nhận tín hiệu dừng (Ctrl+C). Đang giải phóng tài nguyên...")
        controller.stop_all()
        logger.info("Đã dừng an toàn. Tạm biệt!")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    # 5. Khởi động các camera pipelines
    controller.start_all()

    # 6. Tạo FastAPI App (F10)
    api_app = create_api_app(controller)

    if args.ui:
        # Chế độ có UI: Chạy FastAPI trong background thread và mở Qt6 MainWindow
        logger.info("Khởi động FastAPI server nền tại http://%s:%d...", config.server.host, config.server.port)
        api_thread = threading.Thread(
            target=lambda: uvicorn.run(
                api_app, host=config.server.host, port=config.server.port, log_level="warning"
            ),
            daemon=True,
            name="api-server",
        )
        api_thread.start()

        logger.info("Đang mở giao diện quan sát PySide6 (Qt6)...")
        qt_app = QApplication(sys.argv)
        window = MainWindow(controller=controller)
        window.show()
        code = qt_app.exec()
        logger.info("Đóng giao diện Qt6 (exit_code=%s).", code)
        controller.stop_all()
    else:
        # Chế độ Service chuẩn (mặc định): Chạy FastAPI trên main thread
        logger.info(
            "Hệ thống đang chạy chế độ Service tại http://%s:%d (Swagger UI: /docs)",
            config.server.host,
            config.server.port,
        )
        logger.info("Để mở thêm giao diện Qt6 quan sát trực tiếp, dùng lệnh: python main.py --ui")
        try:
            uvicorn.run(
                api_app,
                host=config.server.host,
                port=config.server.port,
                log_level="info",
            )
        finally:
            controller.stop_all()


if __name__ == "__main__":
    main()
