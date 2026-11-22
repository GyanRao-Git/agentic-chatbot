"""Turn log messages into JSON lines."""

import json
import logging
from datetime import UTC, datetime
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

# Project root is one folder above src/, so logs/ sits at the repository root.
# __file__ is this file's path; resolve() makes it absolute; parents[1] goes up two folders.
DEFAULT_LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
LOG_FILE_NAME = "app.log"


class JsonFormatter(logging.Formatter):
    """Format each log record as one JSON object on one line."""

    # Python calls format() for every log message and prints the string it returns.
    # We subclass logging.Formatter so our JSON output plugs into any log handler.
    # Why not an f-string in each log call? Quotes or newlines in a message would
    # break the JSON. json.dumps() escapes them safely.
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        return json.dumps(entry)


# Sends log lines to the terminal. Adds one console handler to the root logger.
def configure_console_logging(level: int = logging.INFO) -> logging.Handler:
    """Print readable log lines to the terminal."""
    # StreamHandler is a pipe that writes each log line to a stream.
    # With no argument it writes to stderr, which shows in the terminal.
    console_handler = logging.StreamHandler()

    # Formatter sets the text layout, e.g. "2026-11-22 10:15:03 INFO agents: message".
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )

    # The root logger is the parent of every logger, so handlers here catch all logs.
    root_logger = logging.getLogger()

    # setLevel sets the lowest level to keep: INFO hides DEBUG messages.
    root_logger.setLevel(level)

    # addHandler attaches the pipe to the root logger.
    root_logger.addHandler(console_handler)

    # Return the handler so a test can remove it afterwards.
    return console_handler


# Writes logs to logs/app.log as JSON, starting a new file every midnight.
def configure_file_logging(
    log_dir: Path = DEFAULT_LOG_DIR,
    retention_days: int = 7,
) -> TimedRotatingFileHandler:
    """Write JSON log lines to a daily-rotated file and return the handler."""
    # mkdir creates the folder if missing. parents=True also creates parent folders.
    # exist_ok=True means no error if the folder already exists.
    log_dir.mkdir(parents=True, exist_ok=True)

    # Writes to app.log. At each midnight it renames app.log to a dated name
    # and starts a new app.log. backupCount is how many old files to keep.
    # Why not a plain FileHandler? It never rotates, so one file grows forever.
    file_handler = TimedRotatingFileHandler(
        filename=log_dir / LOG_FILE_NAME,
        when="midnight",
        backupCount=retention_days,
        encoding="utf-8",
    )

    # Each record is written with the JsonFormatter from step 1.
    file_handler.setFormatter(JsonFormatter())

    # Attach the handler to the root logger so every module's logs go to the file.
    logging.getLogger().addHandler(file_handler)

    # Return the handler so a test can remove it afterwards.
    return file_handler


# One call that sets up both the terminal output and the daily JSON file.
# The API calls this at startup, so routes and agents never set up logging themselves.
def configure_logging(retention_days: int = 7) -> None:
    """Set up console and daily file logging together."""
    configure_console_logging()
    configure_file_logging(retention_days=retention_days)

    # httpx writes a line for every web request, including each Gemini call.
    # Keep only its warnings and errors so the log stays readable.
    logging.getLogger("httpx").setLevel(logging.WARNING)
