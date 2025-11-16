import json
import paho.mqtt.client as mqtt
from mcpConfig import McpConfig

import logging
logger = logging.getLogger("mcpMqtt")

class McpMqtt(object):
    """MQTT Publisher with Home Assistant AutoDiscovery support"""

    def __init__(self):
        self.config = McpConfig()
        self.client = None
        self.connected = False
        self.discovery_sent = False

        # Only initialize if MQTT is enabled
        if self.config.get("MQTT_ENABLED") == "True":
            self._initialize_client()

    def _initialize_client(self):
        """Initialize MQTT client with callbacks"""
        try:
            self.client = mqtt.Client(client_id=self.config.get("MQTT_DEVICE_ID"))

            # Set up callbacks
            self.client.on_connect = self._on_connect
            self.client.on_disconnect = self._on_disconnect
            self.client.on_publish = self._on_publish

            # Set up authentication if credentials provided
            username = self.config.get("MQTT_USERNAME")
            password = self.config.get("MQTT_PASSWORD")
            if username and username.strip():
                self.client.username_pw_set(username, password)
                logger.info("MQTT authentication configured for user: %s", username)

            # Connect to broker
            broker = self.config.get("MQTT_BROKER")
            port = int(self.config.get("MQTT_PORT"))
            logger.info("Connecting to MQTT broker: %s:%d", broker, port)

            self.client.connect(broker, port, keepalive=60)
            self.client.loop_start()  # Start background thread for network loop

        except Exception as e:
            logger.error("Failed to initialize MQTT client: %s", str(e))
            self.client = None

    def _on_connect(self, client, userdata, flags, rc):
        """Callback when connected to MQTT broker"""
        if rc == 0:
            self.connected = True
            logger.info("Successfully connected to MQTT broker")

            # Send Home Assistant AutoDiscovery message if enabled
            if self.config.get("MQTT_HA_DISCOVERY") == "True":
                self._publish_ha_discovery()
        else:
            self.connected = False
            logger.error("Failed to connect to MQTT broker, return code: %d", rc)

    def _on_disconnect(self, client, userdata, rc):
        """Callback when disconnected from MQTT broker"""
        self.connected = False
        if rc != 0:
            logger.warning("Unexpected disconnection from MQTT broker, return code: %d", rc)
        else:
            logger.info("Disconnected from MQTT broker")

    def _on_publish(self, client, userdata, mid):
        """Callback when message is published"""
        logger.debug("Message published, mid: %d", mid)

    def _publish_ha_discovery(self):
        """Publish Home Assistant AutoDiscovery configuration for binary sensor"""
        try:
            device_id = self.config.get("MQTT_DEVICE_ID")
            device_name = self.config.get("MQTT_DEVICE_NAME")
            ha_prefix = self.config.get("MQTT_HA_PREFIX")
            state_topic = self.config.get("MQTT_TOPIC")

            # Discovery topic format: <discovery_prefix>/binary_sensor/<device_id>/config
            discovery_topic = f"{ha_prefix}/binary_sensor/{device_id}/config"

            # Home Assistant AutoDiscovery payload
            discovery_payload = {
                "name": device_name,
                "unique_id": f"{device_id}_clouds",
                "device_class": "problem",  # ON=problem(cloudy), OFF=no problem(clear)
                "state_topic": state_topic,
                "value_template": "{{ value_json.iscloudy }}",
                "payload_on": True,
                "payload_off": False,
                "json_attributes_topic": state_topic,
                "json_attributes_template": "{{ {'skystate': value_json.skystate, 'confidence': value_json.confidence, 'roof_status': value_json.roof_status, 'sun_altitude': value_json.sun_altitude, 'timestamp': value_json.timestamp} | tojson }}",
                "device": {
                    "identifiers": [device_id],
                    "name": device_name,
                    "model": "mlCloudDetect",
                    "manufacturer": "Gord Tulloch / Community Fork",
                    "sw_version": "1.0.1"
                }
            }

            # Publish discovery message with retain flag
            result = self.client.publish(
                discovery_topic,
                json.dumps(discovery_payload),
                qos=1,
                retain=True
            )

            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                logger.info("Published Home Assistant AutoDiscovery to: %s", discovery_topic)
                self.discovery_sent = True
            else:
                logger.error("Failed to publish AutoDiscovery, error code: %d", result.rc)

        except Exception as e:
            logger.error("Error publishing Home Assistant AutoDiscovery: %s", str(e))

    def publish_cloud_status(self, is_cloudy, sky_state, confidence, roof_status, sun_altitude):
        """
        Publish cloud detection status to MQTT

        Args:
            is_cloudy (bool): True if cloudy, False if clear
            sky_state (str): "cloudy" or "clear"
            confidence (float): ML model confidence score (0-1)
            roof_status (str): Current roof status message
            sun_altitude (float): Current sun altitude in degrees
        """
        # Skip if MQTT not enabled or not connected
        if self.config.get("MQTT_ENABLED") != "True":
            return

        if not self.connected or self.client is None:
            logger.warning("Cannot publish - not connected to MQTT broker")
            return

        try:
            import datetime

            # Build JSON payload
            payload = {
                "iscloudy": is_cloudy,
                "skystate": sky_state.lower(),
                "confidence": round(confidence, 4),
                "roof_status": roof_status,
                "sun_altitude": round(sun_altitude, 2),
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
            }

            topic = self.config.get("MQTT_TOPIC")

            # Publish with retain flag so subscribers get last known state
            result = self.client.publish(
                topic,
                json.dumps(payload),
                qos=1,
                retain=True
            )

            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                logger.info("Published cloud status to MQTT: %s", payload)
            else:
                logger.error("Failed to publish cloud status, error code: %d", result.rc)

        except Exception as e:
            logger.error("Error publishing cloud status: %s", str(e))

    def disconnect(self):
        """Disconnect from MQTT broker and stop loop"""
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
            logger.info("MQTT client disconnected")
