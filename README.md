# mlCloudDetect

![Sample Allsky Image with Cloud Detection](./docs/sample.png)

Cloud detection for observatory automation using machine learning. Analyzes allsky camera images to classify sky conditions as Clear or Cloudy, publishing results via MQTT for Home Assistant integration.

## Features

- **ONNX Runtime** - Lightweight ML inference, Python 3.13 compatible
- **Continuous service mode** - Runs as a daemon with configurable detection interval
- **MQTT publishing** - Real-time status updates with automatic reconnection
- **Home Assistant auto-discovery** - Automatically creates sensors and camera entities
- **State hysteresis** - Prevents rapid state changes with configurable pending count
- **Daytime detection** - Automatically skips processing during daylight hours
- **Multiple camera sources** - Supports INDI-ALLSKY database or file-based cameras
- **Training image capture** - Periodically saves images for model retraining
- **PyTorch training pipeline** - Train new models and export directly to ONNX

## Requirements

- Python 3.13+
- ONNX model file (see [Training a New Model](#training-a-new-model) or [Model Conversion](#model-conversion-from-keras))
- MQTT broker (optional, for Home Assistant integration)

## Installation

### Quick Install

```bash
# Clone the repository
git clone https://github.com/smartc/mlCloudDetect.git
cd mlCloudDetect

# Run the installation script
./install.sh
```

The install script will:
- Create a Python virtual environment
- Install dependencies
- Set up the systemd service with correct paths

### Manual Install

```bash
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

# Thumbnail image publishing
thumbnail_enabled = true
thumbnail_topic = "mlclouddetect/thumbnail"
thumbnail_size = 320
thumbnail_quality = 75

# Base URL for linking to full-size images
image_base_url = "https://indi-allsky.local/images"

# Reconnection settings
reconnect_min_delay = 1    # seconds before first reconnect attempt
reconnect_max_delay = 120  # max seconds between reconnect attempts

[service]
mode = "continuous"  # or "single"
interval = 60        # seconds between detections
pending_count = 3    # consecutive readings to change state

[training]
enabled = false
interval = 300       # seconds between captures (5 minutes)
output_dir = "training_data"
```

## Usage

### Single Detection (default)

```bash
# Run once and exit
python cloud_detect.py

# Analyze specific image
python cloud_detect.py --image /path/to/image.jpg
```

### Run as Daemon Service

```bash
python cloud_detect.py --daemon
```

The daemon will:
1. Connect to MQTT broker (if enabled), with automatic reconnection
2. Run detection at the configured interval
3. Skip detection during daytime (based on sun altitude)
4. Publish results to MQTT
5. Capture training images periodically (if enabled)
6. Handle graceful shutdown on SIGTERM/SIGINT

### Command Line Options

```
-c, --config PATH   Path to config.toml file
-i, --image PATH    Analyze specific image
-d, --daemon        Run as continuous service
-v, --verbose       Enable verbose logging
-q, --quiet         Suppress output except errors
```

## Running as a Systemd Service

The install script automatically sets up the systemd service. To start it:

```bash
# Enable and start the service
systemctl --user enable mlCloudDetect
systemctl --user start mlCloudDetect

# View logs
journalctl --user -u mlCloudDetect -f

# Enable service to run after logout (optional)
sudo loginctl enable-linger $USER
```

For manual setup without the install script, copy `mlCloudDetect.service` to `~/.config/systemd/user/` and edit the paths.

## Home Assistant Integration

When MQTT and Home Assistant discovery are enabled, the following entities are automatically created:

| Entity | Type | Description |
|--------|------|-------------|
| Sky Condition | Sensor | "Clear" or "Cloudy" with JSON attributes |
| Is Cloudy | Binary Sensor | ON when cloudy |
| Detection Confidence | Sensor | Percentage confidence |
| Sky Camera | Camera | Thumbnail of latest analyzed image |

The binary sensor can be used in automations to control observatory equipment. Home Assistant discovery messages are re-published on reconnection, so entities survive broker restarts.

## MQTT Reconnection

The MQTT client automatically reconnects if the broker connection is lost. Reconnection uses exponential backoff starting at `reconnect_min_delay` (default 1 second) up to `reconnect_max_delay` (default 120 seconds). On reconnect, Home Assistant discovery is re-published so entities remain registered.

## Training a New Model

mlCloudDetect includes a complete pipeline for collecting training data and training new models using PyTorch with direct ONNX export.

### Step 1: Capture Training Images

Enable training capture in `config.toml`:

```toml
[training]
enabled = true
interval = 300       # capture every 5 minutes
output_dir = "training_data"
```

The service will save images to separate directories based on time of day:

```
training_data/
+-- nighttime/
|   +-- capture_log.json
|   +-- 20260302_031500.jpg
|   +-- ...
+-- daytime/
    +-- capture_log.json
    +-- 20260302_150000.jpg
    +-- ...
```

Each `capture_log.json` contains detection metadata for every captured image:

```json
[
  {
    "filename": "20260302_031500.jpg",
    "timestamp": "2026-03-02T03:15:00+00:00",
    "is_daytime": false,
    "sun_altitude": -25.3,
    "class_name": "Cloudy",
    "confidence": 92.1,
    "is_cloudy": true
  }
]
```

Daytime images are also run through the model to record predictions (the model won't crash on daytime images, though results may be less meaningful).

### Step 2: Sort Images into Class Folders

Copy the `training_data/` directory to your training machine, then use `sort_images.py` to organize images by classification:

```bash
# Sort nighttime images with >=80% confidence (default)
python sort_images.py

# Sort with higher confidence threshold
python sort_images.py --confidence 90

# Include both daytime and nighttime images
python sort_images.py --all

# Include only daytime images
python sort_images.py --daytime

# Preview without copying
python sort_images.py --dry-run

# Custom input/output paths
python sort_images.py --input training_data/nighttime --output ./images
```

This creates an ImageFolder structure ready for training:

```
images/
+-- clear/
|   +-- 20260302_031500.jpg
+-- cloudy/
    +-- 20260302_032000.jpg
```

Review the sorted images and manually correct any misclassifications before training.

### Step 3: Train the Model

On a machine with a GPU (recommended) or capable CPU:

```bash
# Install PyTorch dependencies (not needed on the deployment machine)
pip install torch torchvision onnx

# Train with defaults (15 epochs, MobileNetV3-Small)
python train_model.py --data ./images

# Custom training parameters
python train_model.py --data ./images --epochs 25 --lr 0.0005 --batch-size 16
```

The training script:
- Fine-tunes MobileNetV3-Small (pretrained on ImageNet)
- Uses [-1, 1] normalization matching the detector's preprocessing
- Applies data augmentation (flips, rotation, color jitter) during training
- Tracks the best model by validation accuracy with early stopping
- Exports ONNX with softmax output (probabilities, not raw logits)
- Generates `labels.txt` in the format the detector expects

### Step 4: Deploy

Copy the generated `model.onnx` and `labels.txt` to your deployment machine and restart the service. The detector auto-detects the model's input format (NCHW for PyTorch models, NHWC for TensorFlow/Teachable Machine models), so no configuration changes are needed.

```bash
scp model.onnx labels.txt user@observatory:~/mlCloudDetect/
ssh user@observatory systemctl --user restart mlCloudDetect
```

## Model Conversion from Keras

If you have an existing Keras H5 model (e.g., from Google Teachable Machine), convert it to ONNX:

```bash
# On a machine with Python 3.10-3.12 and TensorFlow
pip install tensorflow tf2onnx onnx
python convert_model.py keras_model.h5 model.onnx
```

Then copy `model.onnx` to your deployment machine. The detector auto-detects the input format, so both Keras-converted and PyTorch-trained models work without configuration changes.

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
- Home Assistant auto-discovery with camera thumbnails
- MQTT automatic reconnection with exponential backoff
- State hysteresis for stable readings
- Training image capture for model retraining
- PyTorch training pipeline with direct ONNX export
