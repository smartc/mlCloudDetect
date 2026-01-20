#!/usr/bin/env python3
"""
Convert Keras H5 model to ONNX format.

This script requires TensorFlow and tf2onnx, which may not work on Python 3.13.
Run this on a machine with Python 3.10-3.12 and TensorFlow installed.

Requirements (for conversion machine only):
    pip install tensorflow tf2onnx onnx

Usage:
    python convert_model.py keras_model.h5 model.onnx
    python convert_model.py keras_model.h5  # outputs keras_model.onnx
"""

import argparse
import sys
from pathlib import Path

# Patch NumPy for compatibility with tf2onnx on newer NumPy versions
# tf2onnx uses deprecated np.object, np.bool, etc. that were removed in NumPy 1.24+
import numpy as np
if not hasattr(np, 'object'):
    np.object = object
if not hasattr(np, 'bool'):
    np.bool = bool
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'complex'):
    np.complex = complex
if not hasattr(np, 'str'):
    np.str = str


def convert_keras_to_onnx(input_path: str, output_path: str) -> None:
    """Convert a Keras H5 model to ONNX format."""
    import tensorflow as tf
    import tf2onnx

    # Custom DepthwiseConv2D that ignores unrecognized arguments
    # Needed for models created with older Keras versions
    class CompatibleDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
        def __init__(self, *args, **kwargs):
            # Remove arguments not recognized by newer Keras
            kwargs.pop('groups', None)
            super().__init__(*args, **kwargs)

    # Custom objects for loading older Keras models
    custom_objects = {
        'DepthwiseConv2D': CompatibleDepthwiseConv2D,
    }

    print(f"Loading Keras model: {input_path}")
    model = tf.keras.models.load_model(input_path, compile=False, custom_objects=custom_objects)

    print("Model summary:")
    model.summary()

    print(f"\nConverting to ONNX format...")

    # Get input spec from model
    input_signature = [tf.TensorSpec(model.input_shape, tf.float32, name='input')]

    # Convert to ONNX
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=input_signature)

    # Save ONNX model
    import onnx
    onnx.save(onnx_model, output_path)

    print(f"ONNX model saved to: {output_path}")

    # Verify the model
    print("\nVerifying ONNX model...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("Model verification passed!")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert Keras H5 model to ONNX format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python convert_model.py keras_model.h5 model.onnx
    python convert_model.py keras_model.h5  # outputs keras_model.onnx

Requirements (install on conversion machine):
    pip install tensorflow tf2onnx onnx
        """,
    )
    parser.add_argument("input", help="Path to Keras H5 model file")
    parser.add_argument(
        "output",
        nargs="?",
        help="Path for output ONNX model (default: same name with .onnx extension)",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1

    if args.output:
        output_path = args.output
    else:
        output_path = str(input_path.with_suffix(".onnx"))

    try:
        convert_keras_to_onnx(str(input_path), output_path)
        return 0
    except ImportError as e:
        print(f"Error: Missing required package: {e}", file=sys.stderr)
        print("\nInstall required packages with:", file=sys.stderr)
        print("    pip install tensorflow tf2onnx onnx", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
