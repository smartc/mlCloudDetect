# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**mlCloudDetect** is a cloud detection system for observatory automation that uses machine learning to analyze allsky camera images and determine whether to open or close an observatory roof. It uses Keras/TensorFlow models trained on allsky images to classify sky conditions as "Clear" or "Cloudy".

## Architecture

The codebase follows a simple object-oriented design with three main components:

### Core Components

1. **mlCloudDetect.py** - Main application entry point
   - Continuous monitoring loop that runs every 60 seconds
   - Implements pending state logic to prevent frequent roof operations
   - Uses pysolar to calculate sun altitude and skip processing during daytime
   - Writes roof status to file for observatory automation systems
   - Publishes status updates to MQTT (if enabled)
   - Maintains cloud/clear counters for state transition hysteresis
   - Includes graceful shutdown handling (KeyboardInterrupt)

2. **mcpClouds.py** - Cloud detection engine (McpClouds class)
   - Loads and executes Keras V2 models for image classification
   - Supports two allsky camera modes: INDI-ALLSKY (database query) and file-based
   - For INDI-ALLSKY: queries SQLite database at `/var/lib/indi-allsky/indi-allsky.sqlite` to get latest image path
   - For file-based cameras: reads image from configured path
   - Image preprocessing: resizes to 224x224, normalizes to [-1, 1] range
   - Optional sampling feature: saves every nth image to training directories for model improvement
   - Returns tuple: (isCloudy: bool, className: str, confidence: float)

3. **mcpConfig.py** - Configuration manager (McpConfig class)
   - Reads/creates `mlCloudDetect.ini` in application directory
   - Auto-generates default config if file doesn't exist
   - Uses Python's configparser for INI file handling

4. **mcpMqtt.py** - MQTT publisher with Home Assistant integration (McpMqtt class)
   - Publishes cloud detection status to MQTT broker
   - Supports Home Assistant AutoDiscovery protocol
   - Creates binary_sensor with device_class "problem" (cloudy=ON, clear=OFF)
   - Handles authentication, reconnection, and graceful disconnection
   - Runs background network loop for reliable MQTT operations

### Data Flow

```
Image Source (INDI-ALLSKY DB or File)
    ↓
McpClouds.isCloudy() → loads image
    ↓
McpClouds.detect() → preprocesses & runs ML model
    ↓
mlCloudDetect.py → applies pending logic with counters
    ↓
Output Destinations:
    ├── roofStatus.txt (file output)
    └── MQTT Broker → Home Assistant (if enabled)
```

### State Machine Logic

The application implements hysteresis to prevent rapid roof open/close cycles:
- Maintains `cloudCount` and `clearCount` counters
- Requires `PENDING` consecutive detections before changing roof state
- Four possible states: "Roof Open", "Roof Closed", "Open Pending", "Close Pending"
- Resets opposite counter when transitioning states

## Development Setup

### Python Environment
- **Python Version**: 3.10 (required for Keras V2 model compatibility)
- **Virtual Environment**: The systemd service uses `.venv` directory

### Install Dependencies
```bash
pip install -r requirements.txt
```

Key dependencies:
- keras==2.15.0 and tensorflow==2.15.0 (V2 models)
- pillow (image processing)
- pysolar (sun altitude calculations)
- paho-mqtt (MQTT support, currently unused in main code)

### Required Files (Not in Repo)

These files are excluded via .gitignore and must be provided:
- `keras_model.h5` - Trained Keras V2 model file
- `labels.txt` - Class labels (typically "0 Clear\n1 Cloudy\n")
- `mlCloudDetect.ini` - Config file (auto-generated with defaults if missing)

Models should be trained using Google Teachable Machine or similar tools that produce Keras V2 format.

## Running the Application

### Manual Run
```bash
python mlCloudDetect.py
```

### Test Mode
```bash
python test.mcpClouds.py
```
Single-shot test that loads one image and reports classification.

### Production Deployment (Linux)
```bash
# Install as systemd service
sudo cp mlCloudDetect.service /etc/systemd/system/
# Edit WorkingDirectory and ExecStart paths in service file
sudo systemctl enable mlCloudDetect.service
sudo systemctl start mlCloudDetect.service
```

### Monitoring
- Logs written to `mlCloudDetect.log` in application directory
- Output file: `roofStatus.txt` (or configured STATUSFILE)
- Service status: `systemctl status mlCloudDetect.service`

## Configuration (mlCloudDetect.ini)

Key configuration parameters:

### Observatory Settings
- `LATITUDE/LONGITUDE` - Observer location for sun calculations
- `ALLSKYCAM` - Camera type: "INDI-ALLSKY", "NONE", or custom (e.g., "TJ")
- `ALLSKYCAMNO` - Camera ID in indi-allsky database (default: "1")
- `ALLSKYFILE` - Path to latest image for non-INDI cameras
- `PENDING` - Minutes of consistent detection before state change (default: 10)
- `DAYTIME` - Sun altitude threshold in degrees (default: -12 for astronomical twilight)

### ML Model Settings
- `KERASMODEL` - Path to Keras model file (default: "keras_model.h5")
- `KERASLABEL` - Path to labels file (default: "labels.txt")

### Image Sampling
- `ALLSKYSAMPLING` - Enable/disable image sampling for training data collection
- `ALLSKYSAMPLERATE` - Save every nth image when sampling enabled
- `ALLSKYSAMPLEDIR` - Directory for saved training samples

### MQTT & Home Assistant Integration
- `MQTT_ENABLED` - Enable/disable MQTT publishing (default: "False")
- `MQTT_BROKER` - MQTT broker hostname or IP address (default: "localhost")
- `MQTT_PORT` - MQTT broker port (default: "1883")
- `MQTT_USERNAME` - MQTT username (empty for anonymous authentication)
- `MQTT_PASSWORD` - MQTT password
- `MQTT_TOPIC` - MQTT topic for publishing cloud status (default: "observatory/clouds")
- `MQTT_HA_DISCOVERY` - Enable Home Assistant AutoDiscovery (default: "True")
- `MQTT_HA_PREFIX` - Home Assistant discovery topic prefix (default: "homeassistant")
- `MQTT_DEVICE_NAME` - Device name in Home Assistant (default: "Observatory Cloud Detector")
- `MQTT_DEVICE_ID` - Unique device identifier (default: "mlclouddetect")

## Code Patterns

### Logging
All modules use Python's logging module with specific loggers:
- Main: logger name is root
- mcpClouds: logger name is "mcpClouds"
- mcpConfig: logger name is "mcpConfig"
- Log format: `'%(asctime)s - %(name)s - %(levelname)s - %(message)s'`

### INDI-ALLSKY Integration
When ALLSKYCAM="INDI-ALLSKY", the system:
1. Connects to SQLite database at `/var/lib/indi-allsky/indi-allsky.sqlite`
2. Queries for latest image: `SELECT image.filename FROM image JOIN camera WHERE camera.id = ALLSKYCAMNO ORDER BY createDate DESC LIMIT 1`
3. Constructs full path: `/var/www/html/allsky/images/{filename}`

### Image Sampling Feature
When enabled, every nth image is copied to subdirectories for retraining:
- Structure: `ALLSKYSAMPLEDIR/Clear/` and `ALLSKYSAMPLEDIR/Cloudy/`
- Filenames: `image_YYYYMMDD_HHMMSS.jpg`
- Counter tracked in McpClouds instance (resets after save)

### MQTT Publishing
The MQTT module publishes JSON payloads with the following structure:
```json
{
  "iscloudy": true,
  "skystate": "cloudy",
  "confidence": 0.9876,
  "roof_status": "Roof Closed",
  "sun_altitude": -15.32,
  "timestamp": "2025-11-15T12:34:56.789012+00:00"
}
```

**Home Assistant Integration:**
- Uses binary_sensor with `device_class: "problem"` (ON=cloudy, OFF=clear)
- AutoDiscovery topic: `homeassistant/binary_sensor/{device_id}/config`
- State topic: Configured via `MQTT_TOPIC` (default: `observatory/clouds`)
- Additional attributes available: skystate, confidence, roof_status, sun_altitude, timestamp
- Publishes with `retain=True` so subscribers get last known state on connection
- Discovery message sent once on startup or reconnection

**MQTT Connection:**
- Background thread handles network loop automatically
- Supports username/password authentication
- Reconnects automatically on network interruption
- Graceful disconnect on program shutdown (Ctrl+C)

## Testing

Currently minimal test coverage. The `test.mcpClouds.py` file demonstrates basic usage but doesn't include assertions or automated testing framework.

**Note:** When testing MQTT functionality, use an MQTT client like `mosquitto_sub` to monitor messages:
```bash
mosquitto_sub -h localhost -t "observatory/clouds" -v
```
