"""Periodic training image capture for model retraining."""

import json
import logging
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from config import TrainingConfig
from detector import CloudDetector, DetectionResult

logger = logging.getLogger(__name__)


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalar types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class TrainingCapture:
    """Periodically saves images and detection metadata for training."""

    def __init__(self, config: TrainingConfig, detector: CloudDetector):
        self.config = config
        self.detector = detector
        self._last_capture = 0.0

        # Create output directories
        base = Path(config.output_dir)
        self.nighttime_dir = base / "nighttime"
        self.daytime_dir = base / "daytime"
        self.nighttime_dir.mkdir(parents=True, exist_ok=True)
        self.daytime_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Training capture enabled (interval: {config.interval}s, "
            f"output: {base.resolve()})"
        )

    def should_capture(self) -> bool:
        """Check if enough time has elapsed for the next capture."""
        return (time.monotonic() - self._last_capture) >= self.config.interval

    def capture(
        self,
        image_path: str,
        is_daytime: bool,
        sun_altitude: float | None,
        result: DetectionResult | None = None,
    ) -> None:
        """Save a training image and log detection metadata.

        Args:
            image_path: Path to the source image.
            is_daytime: Whether it is currently daytime.
            sun_altitude: Current sun altitude in degrees.
            result: Detection result if already computed (avoids re-running
                    the model). If None, detection will be attempted.
        """
        self._last_capture = time.monotonic()

        target_dir = self.daytime_dir if is_daytime else self.nighttime_dir

        # Timestamped filename
        now = datetime.now(timezone.utc)
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        suffix = Path(image_path).suffix or ".jpg"
        filename = f"{timestamp}{suffix}"
        target_path = target_dir / filename

        # Copy image
        try:
            shutil.copy2(image_path, target_path)
            logger.info(f"Training image saved: {target_path}")
        except Exception as e:
            logger.error(f"Failed to save training image: {e}")
            return

        # Run detection if we don't already have a result
        if result is None:
            try:
                result = self.detector.detect(image_path)
            except Exception as e:
                logger.warning(f"Detection failed for training image: {e}")

        # Build log entry
        entry = {
            "filename": filename,
            "timestamp": now.isoformat(),
            "is_daytime": is_daytime,
            "sun_altitude": round(sun_altitude, 1) if sun_altitude is not None else None,
        }

        if result is not None:
            entry["class_name"] = result.class_name
            entry["confidence"] = round(float(result.confidence) * 100, 1)
            entry["is_cloudy"] = bool(result.is_cloudy)
        else:
            entry["class_name"] = None
            entry["confidence"] = None
            entry["is_cloudy"] = None

        # Append to the per-directory log
        log_path = target_dir / "capture_log.json"
        self._append_to_log(log_path, entry)

    def _append_to_log(self, log_path: Path, entry: dict) -> None:
        """Append an entry to the JSON log file."""
        try:
            if log_path.exists():
                data = json.loads(log_path.read_text())
            else:
                data = []
            data.append(entry)
            log_path.write_text(json.dumps(data, indent=2, cls=_NumpyEncoder) + "\n")
            logger.info(f"Training log updated: {log_path}")
        except Exception as e:
            logger.error(f"Failed to update training log {log_path}: {e}")
