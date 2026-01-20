"""MQTT publishing with Home Assistant auto-discovery."""

import json
import logging
from datetime import datetime, timezone

import paho.mqtt.client as mqtt

from config import MqttConfig
from detector import DetectionResult

logger = logging.getLogger(__name__)


class MqttPublisher:
    """Publishes cloud detection results to MQTT with Home Assistant discovery."""

    def __init__(self, config: MqttConfig):
        self.config = config
        self.client: mqtt.Client | None = None
        self._connected = False

    def connect(self) -> bool:
        """Connect to the MQTT broker.

        Returns:
            True if connection successful, False otherwise.
        """
        if not self.config.enabled:
            logger.info("MQTT disabled in configuration")
            return False

        try:
            # Create client with protocol v5
            self.client = mqtt.Client(
                mqtt.CallbackAPIVersion.VERSION2,
                client_id=f"mlclouddetect-{self.config.device_id}",
                protocol=mqtt.MQTTv5,
            )

            # Set up authentication if provided
            if self.config.username:
                self.client.username_pw_set(
                    self.config.username,
                    self.config.password,
                )

            # Set up callbacks
            self.client.on_connect = self._on_connect
            self.client.on_disconnect = self._on_disconnect

            # Connect to broker
            logger.info(f"Connecting to MQTT broker: {self.config.broker}:{self.config.port}")
            self.client.connect(self.config.broker, self.config.port)
            self.client.loop_start()

            return True

        except Exception as e:
            logger.error(f"Failed to connect to MQTT broker: {e}")
            return False

    def disconnect(self) -> None:
        """Disconnect from the MQTT broker."""
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
            self._connected = False
            logger.info("Disconnected from MQTT broker")

    def _on_connect(self, client, userdata, flags, reason_code, properties) -> None:
        """Handle connection to broker."""
        if reason_code == 0:
            logger.info("Connected to MQTT broker")
            self._connected = True
            if self.config.ha_discovery:
                self._publish_ha_discovery()
        else:
            logger.error(f"Failed to connect to MQTT broker: {reason_code}")

    def _on_disconnect(self, client, userdata, flags, reason_code, properties) -> None:
        """Handle disconnection from broker."""
        self._connected = False
        if reason_code != 0:
            logger.warning(f"Unexpected MQTT disconnection: {reason_code}")

    def _publish_ha_discovery(self) -> None:
        """Publish Home Assistant MQTT discovery configuration."""
        device_info = {
            "identifiers": [self.config.device_id],
            "name": self.config.device_name,
            "manufacturer": "mlCloudDetect",
            "model": "Cloud Detector",
            "sw_version": "2.0",
        }

        discovery_prefix = self.config.ha_discovery_prefix
        device_id = self.config.device_id

        # Text sensor showing "Cloudy" or "Clear"
        sky_sensor_config = {
            "name": "Sky Condition",
            "unique_id": f"{device_id}_sky_condition",
            "state_topic": self.config.topic,
            "value_template": "{{ value_json.class_name }}",
            "device": device_info,
            "icon": "mdi:weather-cloudy",
            "json_attributes_topic": self.config.topic,
            "json_attributes_template": "{{ value_json | tojson }}",
        }

        # Binary sensor for automations (is_cloudy true/false)
        binary_sensor_config = {
            "name": "Is Cloudy",
            "unique_id": f"{device_id}_is_cloudy",
            "state_topic": self.config.topic,
            "value_template": "{{ value_json.is_cloudy }}",
            "payload_on": True,
            "payload_off": False,
            "device": device_info,
            "icon": "mdi:cloud-question",
        }

        # Sensor for confidence level
        confidence_sensor_config = {
            "name": "Detection Confidence",
            "unique_id": f"{device_id}_confidence",
            "state_topic": self.config.topic,
            "value_template": "{{ value_json.confidence }}",
            "unit_of_measurement": "%",
            "device": device_info,
            "icon": "mdi:percent",
        }

        # Publish discovery configs
        self.client.publish(
            f"{discovery_prefix}/sensor/{device_id}/sky_condition/config",
            json.dumps(sky_sensor_config),
            retain=True,
        )

        self.client.publish(
            f"{discovery_prefix}/binary_sensor/{device_id}/is_cloudy/config",
            json.dumps(binary_sensor_config),
            retain=True,
        )

        self.client.publish(
            f"{discovery_prefix}/sensor/{device_id}/confidence/config",
            json.dumps(confidence_sensor_config),
            retain=True,
        )

        logger.info("Published Home Assistant discovery configuration")

    def publish(self, result: DetectionResult, sun_altitude: float | None = None) -> bool:
        """Publish detection result to MQTT.

        Args:
            result: The detection result to publish.
            sun_altitude: Optional sun altitude in degrees.

        Returns:
            True if published successfully, False otherwise.
        """
        if not self.config.enabled or not self._connected:
            return False

        payload = {
            "state": "cloudy" if result.is_cloudy else "clear",
            "class_name": result.class_name,
            "confidence": round(result.confidence * 100, 1),
            "is_cloudy": result.is_cloudy,
            "image_path": result.image_path,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if sun_altitude is not None:
            payload["sun_altitude"] = round(sun_altitude, 1)

        try:
            self.client.publish(
                self.config.topic,
                json.dumps(payload),
                retain=True,
            )
            logger.info(f"Published to MQTT: {result.class_name} ({result.confidence:.1%})")
            return True

        except Exception as e:
            logger.error(f"Failed to publish to MQTT: {e}")
            return False
