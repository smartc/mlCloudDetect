"""Configuration management for mlCloudDetect."""

import logging
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = """\
# mlCloudDetect Configuration

[observatory]
# Observer location for sun altitude calculations
latitude = 0.0
longitude = 0.0

# Sun altitude threshold (degrees) - skip processing when sun is above this
# -12 = astronomical twilight, -6 = civil twilight, 0 = horizon
daytime_threshold = -12.0

[camera]
# Camera type: "indi-allsky" or "file"
type = "indi-allsky"

# For indi-allsky: camera ID in database
camera_id = 1

# For indi-allsky: path to SQLite database
database_path = "/var/lib/indi-allsky/indi-allsky.sqlite"

# For indi-allsky: base path for images
image_base_path = "/var/www/html/allsky/images"

# For file-based camera: direct path to latest image
image_file = ""

[model]
# Path to ONNX model file (convert from Keras using convert_model.py)
model_path = "model.onnx"

# Path to labels file
labels_path = "labels.txt"

# Image size expected by model
image_size = 224

"""


@dataclass
class ObservatoryConfig:
    latitude: float = 0.0
    longitude: float = 0.0
    daytime_threshold: float = -12.0


@dataclass
class CameraConfig:
    type: str = "indi-allsky"
    camera_id: int = 1
    database_path: str = "/var/lib/indi-allsky/indi-allsky.sqlite"
    image_base_path: str = "/var/www/html/allsky/images"
    image_file: str = ""


@dataclass
class ModelConfig:
    model_path: str = "model.onnx"
    labels_path: str = "labels.txt"
    image_size: int = 224


@dataclass
class Config:
    observatory: ObservatoryConfig = field(default_factory=ObservatoryConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    model: ModelConfig = field(default_factory=ModelConfig)


def load_config(config_path: Path | None = None) -> Config:
    """Load configuration from TOML file.

    If no path is provided, looks for config.toml in the application directory.
    If the file doesn't exist, creates it with default values.
    """
    if config_path is None:
        config_path = Path(__file__).parent / "config.toml"

    if not config_path.exists():
        logger.info(f"Creating default config file: {config_path}")
        config_path.write_text(DEFAULT_CONFIG)

    logger.info(f"Loading config from: {config_path}")

    with open(config_path, "rb") as f:
        data = tomllib.load(f)

    config = Config()

    if "observatory" in data:
        obs = data["observatory"]
        config.observatory = ObservatoryConfig(
            latitude=obs.get("latitude", 0.0),
            longitude=obs.get("longitude", 0.0),
            daytime_threshold=obs.get("daytime_threshold", -12.0),
        )

    if "camera" in data:
        cam = data["camera"]
        config.camera = CameraConfig(
            type=cam.get("type", "indi-allsky"),
            camera_id=cam.get("camera_id", 1),
            database_path=cam.get("database_path", "/var/lib/indi-allsky/indi-allsky.sqlite"),
            image_base_path=cam.get("image_base_path", "/var/www/html/allsky/images"),
            image_file=cam.get("image_file", ""),
        )

    if "model" in data:
        mdl = data["model"]
        config.model = ModelConfig(
            model_path=mdl.get("model_path", "model.onnx"),
            labels_path=mdl.get("labels_path", "labels.txt"),
            image_size=mdl.get("image_size", 224),
        )

    return config
