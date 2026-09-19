import json
import logging
from datetime import UTC, datetime
from pathlib import Path


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "event": getattr(record, "event", None),
            "data": getattr(record, "data", None),
        }
        if record.exc_info:
            data["exception"] = self.formatException(record.exc_info)
        return json.dumps(data, allow_nan=False, ensure_ascii=False)


def configure_logging(path: str | Path, level: str = "INFO") -> logging.Handler:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("fractal_detector")
    logger.setLevel(level)
    logger.addHandler(handler)
    return handler
