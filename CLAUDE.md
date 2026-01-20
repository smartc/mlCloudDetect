# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

**mlCloudDetect** is a cloud detection system for observatory automation that uses machine learning to analyze allsky camera images and determine sky conditions. It uses ONNX Runtime to run models trained on allsky images, classifying sky conditions as "Clear" or "Cloudy". Results are published via MQTT for Home Assistant integration.

## Architecture

The codebase is designed for Python 3.13 compatibility using ONNX Runtime instead of TensorFlow (which has compatibility issues with Python 3.13).

### Core Components

1. **cloud_detect.py** - Main application entry point
   - Command-line interface for running detection
   - Supports continuous service mode or single-shot detection
   - Uses pysolar to calculate sun altitude and skip processing during daytime
   - Signal handling for graceful shutdown (SIGTERM/SIGINT)
   - `StateTracker` class for hysteresis (prevents rapid state changes)

2. **detector.py** - Cloud detection engine
   - `CloudDetector` class: Loads and executes ONNX models for image classification
   - `ImageSource` class: Retrieves images from INDI-ALLSKY database or file
   - Image preprocessing: resizes to 224x224, normalizes to [-1, 1] range
   - Returns `DetectionResult` dataclass with is_cloudy, class_name, confidence, image_path

3. **config.py** - Configuration management
   - Uses TOML format for configuration (Python 3.11+ built-in tomllib)
   - Dataclass-based configuration with type hints
   - Auto-generates default config.toml if missing
   - Sections: observatory, camera, model, mqtt, service

4. **mqtt.py** - MQTT publishing with Home Assistant integration
   - `MqttPublisher` class: Connects to MQTT broker, publishes results
   - Home Assistant auto-discovery: Creates sensor entities automatically
   - Uses paho-mqtt with MQTTv5 protocol

5. **convert_model.py** - Model conversion utility
   - Converts Keras H5 models to ONNX format
   - Must be run on a machine with TensorFlow installed (Python 3.10-3.12)
   - Only needed once to convert existing models

### Data Flow

```
Image Source (INDI-ALLSKY DB or File)
    |
ImageSource.get_latest_image() -> image path
    |
CloudDetector.detect() -> preprocesses & runs ONNX model
    |
StateTracker.update() -> hysteresis logic
    |
MqttPublisher.publish() -> MQTT broker -> Home Assistant
```

## Development Setup

### Python Environment
- **Python Version**: 3.13+ (designed for modern Python)
- **Key Dependencies**: ONNX Runtime 1.19+, NumPy 2.x, Pillow 11+, paho-mqtt 2.1+

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Required Files (Not in Repo)

These files are excluded via .gitignore and must be provided:
- `model.onnx` - ONNX model file (converted from Keras using convert_model.py)
- `labels.txt` - Class labels (format: "0 Clear\n1 Cloudy\n")
- `config.toml` - Configuration file (auto-generated with defaults if missing)

### Converting Keras Models to ONNX

If you have an existing Keras H5 model (e.g., from Google Teachable Machine), convert it on a machine with TensorFlow:

```bash
# On a machine with Python 3.10-3.12 and TensorFlow
pip install tensorflow tf2onnx onnx

# Convert the model
python convert_model.py keras_model.h5 model.onnx
```

Then copy `model.onnx` to your Python 3.13 machine.

## Running the Application

### Service Mode (default)
```bash
python cloud_detect.py
```
Runs continuously, detecting at configured interval and publishing to MQTT.

### Single Detection
```bash
python cloud_detect.py --single
python cloud_detect.py --image /path/to/image.jpg
```

### Command-Line Options
```
-c, --config PATH   Path to config.toml file
-i, --image PATH    Path to specific image file (forces single mode)
-s, --single        Run single detection and exit
-v, --verbose       Enable verbose logging
-q, --quiet         Suppress output except errors
```

## Configuration (config.toml)

Configuration uses TOML format. Example:

```toml
[observatory]
latitude = 40.0
longitude = -105.0
daytime_threshold = -12.0  # astronomical twilight

[camera]
type = "indi-allsky"  # or "file"
camera_id = 1
database_path = "/var/lib/indi-allsky/indi-allsky.sqlite"
image_base_path = "/var/www/html/allsky/images"
image_file = ""  # for type="file"

[model]
model_path = "model.onnx"
labels_path = "labels.txt"
image_size = 224

[mqtt]
enabled = true
broker = "localhost"
port = 1883
username = ""
password = ""
topic = "mlclouddetect/status"
ha_discovery = true
ha_discovery_prefix = "homeassistant"
device_name = "Cloud Detector"
device_id = "mlclouddetect"

[service]
mode = "continuous"  # or "single"
interval = 60        # seconds between detections
pending_count = 3    # consecutive readings to change state
```

## Code Patterns

### Logging
All modules use Python's logging module:
- Format: `'%(asctime)s - %(name)s - %(levelname)s - %(message)s'`
- Default level: WARNING, configurable via --verbose (INFO) or --quiet (ERROR)

### Type Hints
Code uses modern Python type hints including:
- `str | None` union syntax (Python 3.10+)
- Dataclasses for structured data
- Generic types without imports (Python 3.9+)

### INDI-ALLSKY Integration
When camera type is "indi-allsky":
1. Connects to SQLite database at configured path
2. Queries for latest image by camera ID
3. Constructs full path using image_base_path

### State Hysteresis
The `StateTracker` class prevents rapid state changes:
- Requires `pending_count` consecutive readings before confirming a state change
- Logs pending transitions and confirmations
- Useful for avoiding flapping due to transient cloud conditions

## Systemd Service

A `mlCloudDetect.service` file is provided for running as a systemd service:

```bash
# Install as user service
cp mlCloudDetect.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable mlCloudDetect
systemctl --user start mlCloudDetect

# View logs
journalctl --user -u mlCloudDetect -f
```

## Future Enhancements (Planned)

- ASCOM Alpaca server for Switch and SafetyMonitor devices
- Image sampling for model retraining
