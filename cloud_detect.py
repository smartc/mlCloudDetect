#!/usr/bin/env python3
"""
mlCloudDetect - Cloud detection for observatory automation.

Uses an ONNX model trained on allsky camera images to classify
sky conditions as Clear or Cloudy.
"""

import argparse
import logging
import sys
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


def get_sun_altitude(latitude: float, longitude: float) -> float:
    """Get current sun altitude in degrees."""
    now = datetime.now(timezone.utc)
    return get_altitude(latitude, longitude, now)


def print_result(result: DetectionResult, sun_altitude: float | None = None) -> None:
    """Print detection result to console."""
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
        help="Path to image file (overrides camera configuration)",
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

    mqtt_publisher = None

    try:
        # Load configuration
        config = load_config(args.config)

        # Initialize MQTT if enabled
        if config.mqtt.enabled:
            mqtt_publisher = MqttPublisher(config.mqtt)
            mqtt_publisher.connect()

        # Check sun altitude (skip during daytime unless image specified)
        sun_altitude = None
        if config.observatory.latitude != 0.0 or config.observatory.longitude != 0.0:
            sun_altitude = get_sun_altitude(
                config.observatory.latitude,
                config.observatory.longitude,
            )
            logger.info(f"Sun altitude: {sun_altitude:.1f} degrees")

            if sun_altitude > config.observatory.daytime_threshold and not args.image:
                logger.info(
                    f"Skipping detection: sun altitude ({sun_altitude:.1f}) "
                    f"above threshold ({config.observatory.daytime_threshold})"
                )
                print(f"Daytime - sun altitude {sun_altitude:.1f} degrees, skipping detection")
                return 0

        # Run detection
        result = run_detection(config, args.image)

        # Output result
        if not args.quiet:
            print_result(result, sun_altitude)

        # Publish to MQTT
        if mqtt_publisher:
            mqtt_publisher.publish(result, sun_altitude)

        return 0

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
