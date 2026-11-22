import json
import logging
from pathlib import Path

from logging_config import JsonFormatter, configure_console_logging, configure_file_logging


# Checks that JsonFormatter writes one valid JSON line with the expected fields.
# Why a test: it proves the formatter works without starting the app or writing files.
def test_json_formatter_writes_one_json_object_per_record():
    record = logging.LogRecord(
        name="agents",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="Agent turn %s finished",
        args=("abc",),
        exc_info=None,
    )

    line = JsonFormatter().format(record)
    data = json.loads(line)

    assert data["level"] == "INFO"
    assert data["logger"] == "agents"
    assert data["message"] == "Agent turn abc finished"
    assert "timestamp" in data
    assert "\n" not in line


# Checks that a warning is saved as one JSON line in app.log inside a temp folder.
# tmp_path is a temporary folder pytest creates, so no real logs/ files are touched.
def test_file_logging_writes_json_line(tmp_path: Path) -> None:
    file_handler = configure_file_logging(log_dir=tmp_path, retention_days=7)
    try:
        # Warning is used because the root logger's default level hides INFO.
        logging.getLogger("test").warning("hello %s", "world")
        file_handler.flush()
    finally:
        # Remove the handler so it does not leak into other tests.
        logging.getLogger().removeHandler(file_handler)
        file_handler.close()

    lines = (tmp_path / "app.log").read_text(encoding="utf-8").splitlines()
    data = json.loads(lines[-1])
    assert data["message"] == "hello world"


# Checks that the console setup adds exactly one handler to the root logger.
def test_console_logging_adds_one_handler() -> None:
    root_logger = logging.getLogger()
    original_level = root_logger.level
    handler_count = len(root_logger.handlers)

    console_handler = configure_console_logging()
    try:
        assert len(root_logger.handlers) == handler_count + 1
        assert isinstance(console_handler, logging.StreamHandler)
    finally:
        # Undo the change so other tests see the original logger setup.
        root_logger.removeHandler(console_handler)
        root_logger.setLevel(original_level)
