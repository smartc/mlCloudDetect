"""Cloud detection using ONNX Runtime."""

import logging
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path

# Suppress ONNX Runtime device discovery warnings
os.environ['ORT_DISABLE_GPU_DEVICE_ENUMERATION'] = '1'

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

# Disable ONNX Runtime's verbose logging
ort.set_default_logger_severity(3)  # 3 = ERROR only

from config import CameraConfig, ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Result of a cloud detection."""

    is_cloudy: bool
    class_name: str
    confidence: float
    image_path: str


class CloudDetector:
    """Detects clouds in allsky camera images using an ONNX model."""

    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.session: ort.InferenceSession | None = None
        self.input_name: str = ""
        self.labels: list[str] = []
        self._load_model()
        self._load_labels()

    def _load_model(self) -> None:
        """Load the ONNX model."""
        model_path = Path(self.model_config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        logger.info(f"Loading ONNX model: {model_path}")

        # Create inference session
        self.session = ort.InferenceSession(
            str(model_path),
            providers=['CPUExecutionProvider']
        )

        # Get input name for inference
        self.input_name = self.session.get_inputs()[0].name
        logger.info(f"Model loaded successfully (input: {self.input_name})")

    def _load_labels(self) -> None:
        """Load class labels from file."""
        labels_path = Path(self.model_config.labels_path)
        if not labels_path.exists():
            logger.warning(f"Labels file not found: {labels_path}, using defaults")
            self.labels = ["Clear", "Cloudy"]
            return

        logger.info(f"Loading labels from: {labels_path}")
        with open(labels_path, "r") as f:
            # Labels format: "0 Clear\n1 Cloudy\n"
            self.labels = []
            for line in f:
                line = line.strip()
                if line:
                    # Remove index prefix if present (e.g., "0 Clear" -> "Clear")
                    parts = line.split(maxsplit=1)
                    if len(parts) == 2 and parts[0].isdigit():
                        self.labels.append(parts[1])
                    else:
                        self.labels.append(line)

        logger.info(f"Loaded labels: {self.labels}")

    def _preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess image for model inference.

        - Resize to model input size (224x224)
        - Normalize to [-1, 1] range
        """
        size = self.model_config.image_size

        # Load and convert to RGB
        image = Image.open(image_path).convert("RGB")

        # Resize with center crop to maintain aspect ratio
        image = ImageOps.fit(image, (size, size), Image.Resampling.LANCZOS)

        # Convert to numpy array
        image_array = np.asarray(image, dtype=np.float32)

        # Normalize to [-1, 1] range (Google Teachable Machine format)
        normalized = (image_array / 127.5) - 1.0

        # Add batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
        return np.expand_dims(normalized, axis=0)

    def detect(self, image_path: str) -> DetectionResult:
        """Run cloud detection on an image.

        Args:
            image_path: Path to the image file.

        Returns:
            DetectionResult with classification details.
        """
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        logger.info(f"Detecting clouds in: {image_path}")

        # Preprocess
        input_data = self._preprocess_image(image_path)

        # Run inference
        outputs = self.session.run(None, {self.input_name: input_data})
        predictions = outputs[0]

        # Get result
        class_index = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][class_index])
        class_name = self.labels[class_index] if class_index < len(self.labels) else f"Unknown_{class_index}"

        is_cloudy = class_name.lower() != "clear"

        logger.info(f"Detection result: {class_name} (confidence: {confidence:.4f})")

        return DetectionResult(
            is_cloudy=is_cloudy,
            class_name=class_name,
            confidence=confidence,
            image_path=image_path,
        )


class ImageSource:
    """Retrieves the latest image from various sources."""

    def __init__(self, camera_config: CameraConfig):
        self.config = camera_config

    def get_latest_image(self) -> str:
        """Get path to the latest image.

        Returns:
            Path to the latest image file.

        Raises:
            RuntimeError: If image cannot be retrieved.
        """
        if self.config.type.lower() == "indi-allsky":
            return self._get_indi_allsky_image()
        elif self.config.type.lower() == "file":
            return self._get_file_image()
        else:
            raise ValueError(f"Unknown camera type: {self.config.type}")

    def _get_indi_allsky_image(self) -> str:
        """Query INDI-ALLSKY database for latest image."""
        db_path = self.config.database_path

        if not Path(db_path).exists():
            raise FileNotFoundError(f"INDI-ALLSKY database not found: {db_path}")

        logger.info(f"Querying INDI-ALLSKY database: {db_path}")

        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            query = """
                SELECT image.filename
                FROM image
                JOIN camera ON camera.id = image.camera_id
                WHERE camera.id = ?
                ORDER BY image.createDate DESC
                LIMIT 1
            """

            cursor.execute(query, (self.config.camera_id,))
            row = cursor.fetchone()
            conn.close()

            if row is None:
                raise RuntimeError(f"No images found for camera ID {self.config.camera_id}")

            filename = row[0]
            image_path = f"{self.config.image_base_path}/{filename}"

            logger.info(f"Latest image: {image_path}")
            return image_path

        except sqlite3.Error as e:
            raise RuntimeError(f"Database error: {e}")

    def _get_file_image(self) -> str:
        """Get image from configured file path."""
        if not self.config.image_file:
            raise ValueError("No image_file configured for file-based camera")

        if not Path(self.config.image_file).exists():
            raise FileNotFoundError(f"Image file not found: {self.config.image_file}")

        logger.info(f"Using image file: {self.config.image_file}")
        return self.config.image_file
