# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

**mlCloudDetect** is a cloud detection system for observatory automation that uses machine learning to analyze allsky camera images and determine sky conditions. It uses ONNX Runtime to run models trained on allsky images, classifying sky conditions as "Clear" or "Cloudy".

## Architecture

The codebase is designed for Python 3.13 compatibility using ONNX Runtime instead of TensorFlow (which has compatibility issues with Python 3.13).

### Core Components

1. **cloud_detect.py** - Main application entry point
   - Command-line interface for running detection
   - Supports single-shot detection or specifying an image directly
   - Uses pysolar to calculate sun altitude and skip processing during daytime
   - Writes roof status to file for observatory automation systems

2. **detector.py** - Cloud detection engine
   - `CloudDetector` class: Loads and executes ONNX models for image classification
   - `ImageSource` class: Retrieves images from INDI-ALLSKY database or file
   - Image preprocessing: resizes to 224x224, normalizes to [-1, 1] range
   - Returns `DetectionResult` dataclass with is_cloudy, class_name, confidence, image_path

3. **config.py** - Configuration management
   - Uses TOML format for configuration (Python 3.11+ built-in tomllib)
   - Dataclass-based configuration with type hints
   - Auto-generates default config.toml if missing

4. **convert_model.py** - Model conversion utility
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
cloud_detect.py -> outputs result
    |
Output: roofStatus.txt + console output
```

## Development Setup

### Python Environment
- **Python Version**: 3.13+ (designed for modern Python)
- **Key Dependencies**: ONNX Runtime 1.19+, NumPy 2.x, Pillow 11+

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

### Basic Usage
```bash
python cloud_detect.py
```

### With Specific Image
```bash
python cloud_detect.py --image /path/to/image.jpg
```

### Command-Line Options
```
-c, --config PATH   Path to config.toml file
-i, --image PATH    Path to specific image file
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

[output]
status_file = "roofStatus.txt"
pending_count = 10
```

## Code Patterns

### Logging
All modules use Python's logging module:
- Format: `'%(asctime)s - %(name)s - %(levelname)s - %(message)s'`
- Default level: INFO, configurable via --verbose/--quiet

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

## Future Enhancements (Planned)

- MQTT integration for Home Assistant
- ASCOM Alpaca server for Switch and SafetyMonitor devices
- Continuous monitoring loop with state hysteresis
- Image sampling for model retraining
