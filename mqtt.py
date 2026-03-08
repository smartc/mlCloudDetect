"""MQTT publishing with Home Assistant auto-discovery."""

import io
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import paho.mqtt.client as mqtt
from PIL import Image

from config import MqttConfig
from detector import DetectionResult

logger = logging.getLogger(__name__)


class MqttPublisher:
    """Publishes cloud detection results to MQTT with Home Assistant discovery."""

    PAYLOAD_ONLINE = "online"
    PAYLOAD_OFFLINE = "offline"

    def __init__(self, config: MqttConfig):
        self.config = config
        self.client: mqtt.Client | None = None
        self._connected = False
        self._has_connected = False

    @property
    def availability_topic(self) -> str:
        """MQTT topic used for online/offline availability."""
        return f"{self.config.topic}/availability"

    def _setup_client(self) -> None:
        """Create and configure the MQTT client (idempotent)."""
        if self.client is not None:
            return

        self.client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=f"mlclouddetect-{self.config.device_id}",
            protocol=mqtt.MQTTv5,
        )

        if self.config.username:
            self.client.username_pw_set(
                self.config.username,
                self.config.password,
            )

        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect

        self.client.reconnect_delay_set(
            min_delay=self.config.reconnect_min_delay,
            max_delay=self.config.reconnect_max_delay,
        )

        # Set Last Will and Testament so the broker publishes "offline"
        # if we disconnect unexpectedly.
        self.client.will_set(
            self.availability_topic,
            self.PAYLOAD_OFFLINE,
            retain=True,
        )

    def connect(self) -> bool:
        """Connect to the MQTT broker.

        Sets up the client and starts the network loop. If the initial
        TCP connection fails the loop is still started so that paho's
        built-in reconnection logic can keep retrying in the background.

        Returns:
            True if connection successful, False otherwise.
        """
        if not self.config.enabled:
            logger.info("MQTT disabled in configuration")
            return False

        try:
            self._setup_client()

            logger.info(f"Connecting to MQTT broker: {self.config.broker}:{self.config.port}")
            self.client.connect(self.config.broker, self.config.port)
            self.client.loop_start()
            return True

        except Exception as e:
            logger.error(f"Failed to connect to MQTT broker: {e}")
            # Start the network loop anyway so paho's automatic
            # reconnection can keep retrying in the background.
            if self.client is not None:
                try:
                    self.client.loop_start()
                except Exception:
                    pass
            return False

    def reconnect(self) -> bool:
        """Attempt to reconnect if not currently connected.

        Returns:
            True if already connected or reconnection initiated.
        """
        if self._connected:
            return True

        if self.client is None:
            return self.connect()

        try:
            self.client.reconnect()
            return True
        except Exception:
            # reconnect() can fail if the initial connect() never
            # stored the broker address.  Fall back to a fresh connect.
            try:
                self.client.connect(self.config.broker, self.config.port)
                return True
            except Exception as e:
                logger.debug(f"MQTT reconnect attempt failed: {e}")
                return False

    def disconnect(self) -> None:
        """Disconnect from the MQTT broker."""
        if self.client:
            # Publish offline before disconnecting so HA updates immediately
            # rather than waiting for the broker's LWT timeout.
            if self._connected:
                self.client.publish(
                    self.availability_topic,
                    self.PAYLOAD_OFFLINE,
                    retain=True,
                )
            self.client.loop_stop()
            self.client.disconnect()
            self._connected = False
            logger.info("Disconnected from MQTT broker")

    def _on_connect(self, client, userdata, flags, reason_code, properties) -> None:
        """Handle connection to broker."""
        if reason_code == 0:
            if self._has_connected:
                logger.info("Reconnected to MQTT broker")
            else:
                logger.info("Connected to MQTT broker")
                self._has_connected = True
            self._connected = True
            # Publish "online" availability (counterpart to the LWT).
            self.client.publish(
                self.availability_topic,
                self.PAYLOAD_ONLINE,
                retain=True,
            )
            # Re-publish HA discovery on every (re)connect so entities
            # are registered even after broker restarts.
            if self.config.ha_discovery:
                self._publish_ha_discovery()
        else:
            logger.error(f"Failed to connect to MQTT broker: {reason_code}")

    def _on_disconnect(self, client, userdata, flags, reason_code, properties) -> None:
        """Handle disconnection from broker."""
        self._connected = False
        if reason_code != 0:
            logger.warning(
                f"Unexpected MQTT disconnection (reason: {reason_code}). "
                f"Automatic reconnection will be attempted."
            )
        else:
            logger.info("Disconnected from MQTT broker")

    def _publish_ha_discovery(self) -> None:
        """Publish Home Assistant MQTT discovery configuration."""
        device_info = {
            "identifiers": [self.config.device_id],
            "name": self.config.device_name,
            "manufacturer": "mlCloudDetect",
            "model": "Cloud Detector",
            "sw_version": "2.0",
        }

        # Availability config shared by all entities
        availability = {
            "availability_topic": self.availability_topic,
            "payload_available": self.PAYLOAD_ONLINE,
            "payload_not_available": self.PAYLOAD_OFFLINE,
        }

        discovery_prefix = self.config.ha_discovery_prefix
        device_id = self.config.device_id

        # Text sensor showing "Cloudy", "Clear", or "Daytime"
        sky_sensor_config = {
            "name": "Sky Condition",
            "unique_id": f"{device_id}_sky_condition",
            "state_topic": self.config.topic,
            "value_template": "{{ value_json.class_name }}",
            "device": device_info,
            "icon": "mdi:weather-cloudy",
            "json_attributes_topic": self.config.topic,
            "json_attributes_template": "{{ value_json | tojson }}",
            **availability,
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
            **availability,
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
            **availability,
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

        # Camera entity for thumbnail image
        if self.config.thumbnail_enabled:
            camera_config = {
                "name": "Sky Camera",
                "unique_id": f"{device_id}_camera",
                "topic": self.config.thumbnail_topic,
                "device": device_info,
                "icon": "mdi:camera",
                **availability,
            }

            self.client.publish(
                f"{discovery_prefix}/camera/{device_id}/sky_camera/config",
                json.dumps(camera_config),
                retain=True,
            )

        logger.info("Published Home Assistant discovery configuration")

    def _build_image_url(self, image_path: str) -> str | None:
        """Build a URL to the full-size image on the web server.

        Args:
            image_path: Local filesystem path to the image.

        Returns:
            URL to the image, or None if URL cannot be constructed.
        """
        if not self.config.image_base_url:
            return None

        # Extract relative path by finding '/images/' in the path
        # e.g., /var/www/html/allsky/images/ccd/2024/01/20/img.jpg -> ccd/2024/01/20/img.jpg
        images_marker = "/images/"
        if images_marker in image_path:
            relative_path = image_path.split(images_marker, 1)[1]
        else:
            # Fallback: just use the filename
            relative_path = Path(image_path).name

        # Construct URL, ensuring no double slashes
        base_url = self.config.image_base_url.rstrip("/")
        return f"{base_url}/{relative_path}"

    def _create_thumbnail(self, image_path: str) -> bytes | None:
        """Create a thumbnail from the source image.

        Args:
            image_path: Path to the source image.

        Returns:
            JPEG image data as bytes, or None if creation failed.
        """
        try:
            if not Path(image_path).exists():
                logger.warning(f"Image not found for thumbnail: {image_path}")
                return None

            # Open and resize the image
            with Image.open(image_path) as img:
                # Convert to RGB if necessary (e.g., RGBA images)
                if img.mode != "RGB":
                    img = img.convert("RGB")

                # Calculate size maintaining aspect ratio
                size = self.config.thumbnail_size
                img.thumbnail((size, size), Image.Resampling.LANCZOS)

                # Save to bytes buffer as JPEG
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=self.config.thumbnail_quality)
                return buffer.getvalue()

        except Exception as e:
            logger.error(f"Failed to create thumbnail: {e}")
            return None

    def publish_thumbnail(self, image_path: str) -> bool:
        """Publish a thumbnail image to MQTT.

        Args:
            image_path: Path to the source image.

        Returns:
            True if published successfully, False otherwise.
        """
        if not self.config.enabled:
            return False

        if not self._connected:
            logger.warning("MQTT not connected, skipping thumbnail publish (will retry on next publish)")
            return False

        if not self.config.thumbnail_enabled:
            return False

        thumbnail_data = self._create_thumbnail(image_path)
        if thumbnail_data is None:
            return False

        try:
            self.client.publish(
                self.config.thumbnail_topic,
                thumbnail_data,
                retain=True,
            )
            logger.info(f"Published thumbnail ({len(thumbnail_data)} bytes)")
            return True

        except Exception as e:
            logger.error(f"Failed to publish thumbnail: {e}")
            return False

    def publish_daytime(self, sun_altitude: float) -> bool:
        """Publish a daytime status heartbeat to MQTT.

        Keeps HA sensors fresh during daytime when no detection runs.

        Args:
            sun_altitude: Current sun altitude in degrees.

        Returns:
            True if published successfully, False otherwise.
        """
        if not self.config.enabled:
            return False

        if not self._connected:
            self.reconnect()
            if not self._connected:
                return False

        payload = {
            "state": "daytime",
            "class_name": "Daytime",
            "confidence": 0.0,
            "is_cloudy": False,
            "image_path": "",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sun_altitude": round(sun_altitude, 1),
        }

        try:
            self.client.publish(
                self.config.topic,
                json.dumps(payload),
                retain=True,
            )
            logger.info(f"Published daytime status to MQTT (sun: {sun_altitude:.1f}°)")
            return True

        except Exception as e:
            logger.error(f"Failed to publish daytime status: {e}")
            return False

    def publish(self, result: DetectionResult, sun_altitude: float | None = None) -> bool:
        """Publish detection result to MQTT.

        Args:
            result: The detection result to publish.
            sun_altitude: Optional sun altitude in degrees.

        Returns:
            True if published successfully, False otherwise.
        """
        if not self.config.enabled:
            return False

        if not self._connected:
            logger.warning("MQTT not connected, attempting reconnect")
            self.reconnect()
            if not self._connected:
                logger.warning("MQTT reconnect failed, skipping publish")
                return False

        payload = {
            "state": "cloudy" if result.is_cloudy else "clear",
            "class_name": result.class_name,
            "confidence": round(result.confidence * 100, 1),
            "is_cloudy": result.is_cloudy,
            "image_path": result.image_path,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Add image URL if configured
        image_url = self._build_image_url(result.image_path)
        if image_url:
            payload["image_url"] = image_url

        if sun_altitude is not None:
            payload["sun_altitude"] = round(sun_altitude, 1)

        try:
            self.client.publish(
                self.config.topic,
                json.dumps(payload),
                retain=True,
            )
            logger.info(f"Published to MQTT: {result.class_name} ({result.confidence:.1%})")

            # Also publish thumbnail if enabled
            if self.config.thumbnail_enabled:
                self.publish_thumbnail(result.image_path)

            return True

        except Exception as e:
            logger.error(f"Failed to publish to MQTT: {e}")
            return False
