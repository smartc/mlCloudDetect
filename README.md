# mlCloudDetect

![Sample Allsky Image with Cloud Detection](./docs/sample.png)

Cloud detection for observatory automation using machine learning. Analyzes allsky camera images to classify sky conditions as Clear or Cloudy, publishing results via MQTT for Home Assistant integration.

## Features

- **ONNX Runtime** - Lightweight ML inference, Python 3.13 compatible
- **Continuous service mode** - Runs as a daemon with configurable detection interval
- **MQTT publishing** - Real-time status updates to any MQTT broker
- **Home Assistant auto-discovery** - Automatically creates sensors in Home Assistant
- **State hysteresis** - Prevents rapid state changes with configurable pending count
- **Daytime detection** - Automatically skips processing during daylight hours
- **Multiple camera sources** - Supports INDI-ALLSKY database or file-based cameras

## Requirements

- Python 3.13+
- ONNX model file (converted from Keras using `convert_model.py`)
- MQTT broker (optional, for Home Assistant integration)

## Installation

```bash
# Clone the repository
git clone https://github.com/smartc/mlCloudDetect.git
cd mlCloudDetect

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Configuration uses TOML format. On first run, a default `config.toml` is created.

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

## Usage

### Run as Service (default)

```bash
python cloud_detect.py
```

The service will:
1. Connect to MQTT broker (if enabled)
2. Run detection at the configured interval
3. Skip detection during daytime (based on sun altitude)
4. Publish results to MQTT
5. Handle graceful shutdown on SIGTERM/SIGINT

### Single Detection

```bash
# Run once and exit
python cloud_detect.py --single

# Analyze specific image
python cloud_detect.py --image /path/to/image.jpg
```

### Command Line Options

```
-c, --config PATH   Path to config.toml file
-i, --image PATH    Analyze specific image (forces single mode)
-s, --single        Run single detection and exit
-v, --verbose       Enable verbose logging
-q, --quiet         Suppress output except errors
```

## Running as a Systemd Service

1. Copy the service file:
   ```bash
   cp mlCloudDetect.service ~/.config/systemd/user/
   ```

2. Edit paths in the service file to match your installation

3. Enable and start:
   ```bash
   systemctl --user daemon-reload
   systemctl --user enable mlCloudDetect
   systemctl --user start mlCloudDetect
   ```

4. View logs:
   ```bash
   journalctl --user -u mlCloudDetect -f
   ```

## Home Assistant Integration

When MQTT and Home Assistant discovery are enabled, three entities are automatically created:

| Entity | Type | Description |
|--------|------|-------------|
| Sky Condition | Sensor | "Clear" or "Cloudy" |
| Is Cloudy | Binary Sensor | ON when cloudy (device_class: problem) |
| Detection Confidence | Sensor | Percentage confidence |

The binary sensor can be used in automations to control observatory equipment.

## Model Conversion

If you have a Keras H5 model (e.g., from Google Teachable Machine), convert it to ONNX:

```bash
# On a machine with Python 3.10-3.12 and TensorFlow
pip install tensorflow tf2onnx onnx
python convert_model.py keras_model.h5 model.onnx
```

Then copy `model.onnx` to your deployment machine.

## Required Files

These files must be provided (not included in repo):

- `model.onnx` - ONNX model file
- `labels.txt` - Class labels (format: `0 Clear\n1 Cloudy\n`)
- `config.toml` - Configuration (auto-generated with defaults)

## Credits

Based on [Gord Tulloch's mlCloudDetect](https://github.com/gordtulloch/mlCloudDetect).

This fork adds:
- Python 3.13 compatibility via ONNX Runtime
- Continuous service mode with MQTT integration
- Home Assistant auto-discovery
- State hysteresis for stable readings
