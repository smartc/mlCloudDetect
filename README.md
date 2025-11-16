# mlCloudDetect
![Sample Allsky Image with Cloud Detection](./docs/sample.png)

## About This Fork

This is a fork of Gord Tulloch's original [mlCloudDetect](https://github.com/gordtulloch/mlCloudDetect) project, with added MQTT support and Home Assistant integration.

## Overview

Cloud Detection using AllSky Cameras. This software will examine images from any allsky camera that can produce an image on disk at a known location. For indi-allsky, the software will load the latest image location from the database. Using Machine Learning, the software compares the most recent image with it's training model to produce a file that will tell you whether the sky is cloudy or clear. For observatories that means open the roof or not.

The software also supports pending states where it will delay opening or closing so the observatory isn't constantly opening and closing all night! All messages and configuration is set using an INI file.

## Features

- Machine learning-based cloud detection using Keras/TensorFlow models
- Support for INDI-ALLSKY and file-based allsky cameras
- Pending state logic to prevent rapid roof open/close cycles
- Daytime detection using solar altitude calculations
- File output for observatory automation systems
- **MQTT publishing with Home Assistant AutoDiscovery support**
- Image sampling for training data collection 

Requires Python == 3.10 if not using Windows executables.

## MQTT and Home Assistant Integration

This fork includes MQTT support with automatic Home Assistant discovery:

- Publishes cloud detection status to an MQTT broker
- Automatic discovery in Home Assistant as a binary sensor
- Binary sensor with `device_class: "problem"` (ON=cloudy, OFF=clear)
- Additional attributes: sky state, confidence level, roof status, sun altitude, and timestamp
- Configurable MQTT broker, authentication, and topics via INI file
- Automatic reconnection on network interruption

To enable MQTT, configure the following in `mlCloudDetect.ini`:

```ini
[MQTT]
MQTT_ENABLED = True
MQTT_BROKER = your-broker-hostname
MQTT_PORT = 1883
MQTT_USERNAME = your-username
MQTT_PASSWORD = your-password
MQTT_TOPIC = observatory/clouds
MQTT_HA_DISCOVERY = True
```

## Releases

* Version 1.0.1 returns to using a Keras V2 model and thus requires Python 3.10
* Version 1.0.0+ requires a Keras V3 model and will run in any version of Python. It takes no parameters but uses a config file mlCloudDetect.ini (see below).
* Version 0.9.0 requires Python 3.8 and Keras/Tensorflow 2.11 to support V2 keras model files like those created by Teachable Machine. It requires command line parameters. Run the program without parameters to see usage or see below.

See the Wiki for complete documentation.

## Credits

Original project by [Gord Tulloch](https://github.com/gordtulloch/mlCloudDetect)

MQTT and Home Assistant integration added in this fork.


