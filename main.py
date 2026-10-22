import sys
import logging

from PySide6.QtWidgets import QApplication

from main_app.services.logging_setup import log_startup_banner, setup_logging
from main_app.ui.main_window import MainWindow


def main() -> None:
    setup_logging()
    log_startup_banner()
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    code = app.exec()
    logging.getLogger("main").info("Thoát ứng dụng (code=%s)", code)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()