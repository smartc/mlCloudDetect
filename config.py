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

[mqtt]
# Enable MQTT publishing
enabled = false

# MQTT broker connection
broker = "localhost"
port = 1883
username = ""
password = ""

# MQTT topics
topic = "mlclouddetect/status"

# Home Assistant integration
ha_discovery = true
ha_discovery_prefix = "homeassistant"
device_name = "Cloud Detector"
device_id = "mlclouddetect"

# Thumbnail image publishing
thumbnail_enabled = true
thumbnail_topic = "mlclouddetect/thumbnail"
thumbnail_size = 320
thumbnail_quality = 75

[service]
# Service mode: "single" (run once and exit) or "continuous" (run as daemon)
mode = "single"

# Detection interval in seconds (for continuous mode)
interval = 60

# Number of consecutive readings required before changing state
# Helps prevent rapid state changes due to transient conditions
pending_count = 3
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
class MqttConfig:
    enabled: bool = False
    broker: str = "localhost"
    port: int = 1883
    username: str = ""
    password: str = ""
    topic: str = "mlclouddetect/status"
    ha_discovery: bool = True
    ha_discovery_prefix: str = "homeassistant"
    device_name: str = "Cloud Detector"
    device_id: str = "mlclouddetect"
    thumbnail_enabled: bool = True
    thumbnail_topic: str = "mlclouddetect/thumbnail"
    thumbnail_size: int = 320
    thumbnail_quality: int = 75


@dataclass
class ServiceConfig:
    mode: str = "single"  # "single" or "continuous"
    interval: int = 60  # seconds between detections
    pending_count: int = 3  # consecutive readings to change state


@dataclass
class Config:
    observatory: ObservatoryConfig = field(default_factory=ObservatoryConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    mqtt: MqttConfig = field(default_factory=MqttConfig)
    service: ServiceConfig = field(default_factory=ServiceConfig)


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

    if "mqtt" in data:
        mqtt = data["mqtt"]
        config.mqtt = MqttConfig(
            enabled=mqtt.get("enabled", False),
            broker=mqtt.get("broker", "localhost"),
            port=mqtt.get("port", 1883),
            username=mqtt.get("username", ""),
            password=mqtt.get("password", ""),
            topic=mqtt.get("topic", "mlclouddetect/status"),
            ha_discovery=mqtt.get("ha_discovery", True),
            ha_discovery_prefix=mqtt.get("ha_discovery_prefix", "homeassistant"),
            device_name=mqtt.get("device_name", "Cloud Detector"),
            device_id=mqtt.get("device_id", "mlclouddetect"),
            thumbnail_enabled=mqtt.get("thumbnail_enabled", True),
            thumbnail_topic=mqtt.get("thumbnail_topic", "mlclouddetect/thumbnail"),
            thumbnail_size=mqtt.get("thumbnail_size", 320),
            thumbnail_quality=mqtt.get("thumbnail_quality", 75),
        )

    if "service" in data:
        svc = data["service"]
        config.service = ServiceConfig(
            mode=svc.get("mode", "continuous"),
            interval=svc.get("interval", 60),
            pending_count=svc.get("pending_count", 3),
        )

    return config
