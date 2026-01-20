#!/usr/bin/env python3
"""
mlCloudDetect - Cloud detection for observatory automation.

Uses an ONNX model trained on allsky camera images to classify
sky conditions as Clear or Cloudy. Runs as a service with periodic
MQTT status updates.
"""

import argparse
import logging
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from pysolar.solar import get_altitude

from config import Config, load_config
from detector import CloudDetector, DetectionResult, ImageSource
from mqtt import MqttPublisher

# Configure logging - default to WARNING, use -v for verbose
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stderr),
    ],
)
logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
_shutdown_requested = False


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    global _shutdown_requested
    logger.info(f"Received signal {signum}, initiating shutdown...")
    _shutdown_requested = True


def get_sun_altitude(latitude: float, longitude: float) -> float:
    """Get current sun altitude in degrees."""
    now = datetime.now(timezone.utc)
    return get_altitude(latitude, longitude, now)


def print_result(result: DetectionResult, sun_altitude: float | None = None) -> None:
    """Print detection result to console."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {result.class_name} ({result.confidence:.1%})", end="")
    if sun_altitude is not None:
        print(f" | Sun: {sun_altitude:.1f}°", end="")
    print()


def run_detection(config: Config, image_path: str | None = None) -> DetectionResult:
    """Run a single cloud detection.

    Args:
        config: Application configuration.
        image_path: Optional path to image file. If not provided,
                   will fetch the latest image from configured source.

    Returns:
        DetectionResult with classification details.
    """
    # Initialize detector
    detector = CloudDetector(config.model)

    # Get image path
    if image_path:
        path = image_path
    else:
        source = ImageSource(config.camera)
        path = source.get_latest_image()

    # Run detection
    return detector.detect(path)


class StateTracker:
    """Tracks state with hysteresis to prevent rapid changes."""

    def __init__(self, pending_count: int = 3):
        self.pending_count = pending_count
        self.current_state: bool | None = None  # None = unknown, True = cloudy, False = clear
        self.pending_state: bool | None = None
        self.consecutive_count = 0

    def update(self, is_cloudy: bool) -> tuple[bool, bool]:
        """Update state with new reading.

        Args:
            is_cloudy: Current detection result.

        Returns:
            Tuple of (state_changed, confirmed_state).
            state_changed is True if the state actually changed.
        """
        # First reading - initialize state
        if self.current_state is None:
            self.current_state = is_cloudy
            self.pending_state = None
            self.consecutive_count = 0
            logger.info(f"Initial state: {'Cloudy' if is_cloudy else 'Clear'}")
            return True, is_cloudy

        # Same as current state - reset pending
        if is_cloudy == self.current_state:
            if self.pending_state is not None:
                logger.info(f"Pending state cancelled, returning to {'Cloudy' if self.current_state else 'Clear'}")
            self.pending_state = None
            self.consecutive_count = 0
            return False, self.current_state

        # Different from current state
        if self.pending_state is None:
            # Start pending transition
            self.pending_state = is_cloudy
            self.consecutive_count = 1
            logger.info(f"State change pending: {'Clear' if self.current_state else 'Cloudy'} -> {'Cloudy' if is_cloudy else 'Clear'} ({self.consecutive_count}/{self.pending_count})")
        elif is_cloudy == self.pending_state:
            # Continue pending transition
            self.consecutive_count += 1
            logger.info(f"State change pending: {self.consecutive_count}/{self.pending_count}")

            if self.consecutive_count >= self.pending_count:
                # Transition confirmed
                old_state = self.current_state
                self.current_state = is_cloudy
                self.pending_state = None
                self.consecutive_count = 0
                logger.info(f"State changed: {'Cloudy' if old_state else 'Clear'} -> {'Cloudy' if is_cloudy else 'Clear'}")
                return True, is_cloudy
        else:
            # Pending state changed direction, restart
            self.pending_state = is_cloudy
            self.consecutive_count = 1

        return False, self.current_state


def run_service(config: Config, mqtt_publisher: MqttPublisher | None, quiet: bool) -> int:
    """Run continuous detection service.

    Args:
        config: Application configuration.
        mqtt_publisher: Optional MQTT publisher.
        quiet: Suppress console output.

    Returns:
        Exit code.
    """
    global _shutdown_requested

    # Initialize state tracker
    state_tracker = StateTracker(config.service.pending_count)

    # Initialize detector and image source once
    detector = CloudDetector(config.model)
    source = ImageSource(config.camera)

    logger.info(f"Starting service (interval: {config.service.interval}s, pending_count: {config.service.pending_count})")
    if not quiet:
        print(f"mlCloudDetect service started (interval: {config.service.interval}s)")

    while not _shutdown_requested:
        try:
            # Check sun altitude
            sun_altitude = None
            is_daytime = False
            if config.observatory.latitude != 0.0 or config.observatory.longitude != 0.0:
                sun_altitude = get_sun_altitude(
                    config.observatory.latitude,
                    config.observatory.longitude,
                )
                is_daytime = sun_altitude > config.observatory.daytime_threshold
                logger.info(f"Sun altitude: {sun_altitude:.1f}° (daytime: {is_daytime})")

            if is_daytime:
                if not quiet:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"[{timestamp}] Daytime (Sun: {sun_altitude:.1f}°) - skipping detection")
            else:
                # Get latest image and run detection
                image_path = source.get_latest_image()
                result = detector.detect(image_path)

                # Update state with hysteresis
                state_changed, confirmed_state = state_tracker.update(result.is_cloudy)

                # Output result
                if not quiet:
                    print_result(result, sun_altitude)

                # Publish to MQTT (always publish latest reading)
                if mqtt_publisher:
                    mqtt_publisher.publish(result, sun_altitude)

        except FileNotFoundError as e:
            logger.warning(f"Image not found: {e}")
        except Exception as e:
            logger.error(f"Detection error: {e}")

        # Wait for next interval (check shutdown frequently)
        wait_time = config.service.interval
        while wait_time > 0 and not _shutdown_requested:
            sleep_time = min(wait_time, 1.0)
            time.sleep(sleep_time)
            wait_time -= sleep_time

    logger.info("Service shutdown complete")
    if not quiet:
        print("mlCloudDetect service stopped")

    return 0


def run_single(config: Config, mqtt_publisher: MqttPublisher | None, image_path: str | None, quiet: bool) -> int:
    """Run single detection and exit.

    Args:
        config: Application configuration.
        mqtt_publisher: Optional MQTT publisher.
        image_path: Optional specific image to analyze.
        quiet: Suppress console output.

    Returns:
        Exit code.
    """
    # Check sun altitude (skip during daytime unless image specified)
    sun_altitude = None
    if config.observatory.latitude != 0.0 or config.observatory.longitude != 0.0:
        sun_altitude = get_sun_altitude(
            config.observatory.latitude,
            config.observatory.longitude,
        )
        logger.info(f"Sun altitude: {sun_altitude:.1f} degrees")

        if sun_altitude > config.observatory.daytime_threshold and not image_path:
            logger.info(
                f"Skipping detection: sun altitude ({sun_altitude:.1f}) "
                f"above threshold ({config.observatory.daytime_threshold})"
            )
            print(f"Daytime - sun altitude {sun_altitude:.1f} degrees, skipping detection")
            return 0

    # Run detection
    result = run_detection(config, image_path)

    # Output result
    if not quiet:
        print()
        print("=" * 50)
        print("  mlCloudDetect - Sky Classification Result")
        print("=" * 50)
        print(f"  Image:      {result.image_path}")
        print(f"  Class:      {result.class_name}")
        print(f"  Confidence: {result.confidence:.1%}")
        print(f"  Is Cloudy:  {'Yes' if result.is_cloudy else 'No'}")
        if sun_altitude is not None:
            print(f"  Sun Alt:    {sun_altitude:.1f} degrees")
        print("=" * 50)
        print()

    # Publish to MQTT
    if mqtt_publisher:
        mqtt_publisher.publish(result, sun_altitude)

    return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Cloud detection for observatory automation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-c", "--config",
        type=Path,
        help="Path to config.toml file (default: ./config.toml)",
    )
    parser.add_argument(
        "-i", "--image",
        type=str,
        help="Path to image file (overrides camera configuration, forces single mode)",
    )
    parser.add_argument(
        "-s", "--single",
        action="store_true",
        help="Run single detection and exit (overrides config)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress output except errors",
    )

    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    elif args.quiet:
        logging.getLogger().setLevel(logging.ERROR)

    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    mqtt_publisher = None

    try:
        # Load configuration
        config = load_config(args.config)

        # Initialize MQTT if enabled
        if config.mqtt.enabled:
            mqtt_publisher = MqttPublisher(config.mqtt)
            mqtt_publisher.connect()
            # Give MQTT time to connect
            time.sleep(0.5)

        # Determine run mode
        if args.image or args.single or config.service.mode == "single":
            return run_single(config, mqtt_publisher, args.image, args.quiet)
        else:
            return run_service(config, mqtt_publisher, args.quiet)

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Error: {e}")
        return 1
    finally:
        if mqtt_publisher:
            mqtt_publisher.disconnect()


if __name__ == "__main__":
    sys.exit(main())
