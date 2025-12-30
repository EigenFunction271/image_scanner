"""Tests for image preprocessing module."""

import numpy as np
import pytest
from PIL import Image

from image_screener.preprocessing import ImagePreprocessor


def test_image_preprocessor_initialization():
    """Test ImagePreprocessor initialization."""
    preprocessor = ImagePreprocessor(target_size=512)
    assert preprocessor.target_size == 512
    assert preprocessor.normalize_to_float is True


def test_load_image_grayscale(tmp_path):
    """Test loading and converting image to grayscale."""
    # Create a test RGB image
    img = Image.new("RGB", (100, 100), color=(255, 128, 64))
    img_path = tmp_path / "test.png"
    img.save(img_path)

    preprocessor = ImagePreprocessor()
    loaded = preprocessor.load_image(img_path)

    assert loaded.shape == (100, 100)
    assert loaded.dtype == np.float32
    assert loaded.min() >= 0.0
    assert loaded.max() <= 1.0


def test_load_image_unsupported_format(tmp_path):
    """Test error handling for unsupported formats."""
    preprocessor = ImagePreprocessor()
    invalid_path = tmp_path / "test.txt"
    # Create the file so it exists but has wrong format
    invalid_path.write_text("not an image")

    with pytest.raises(ValueError, match="Unsupported image format"):
        preprocessor.load_image(invalid_path)


def test_load_image_not_found():
    """Test error handling for missing files."""
    preprocessor = ImagePreprocessor()

    with pytest.raises(FileNotFoundError):
        preprocessor.load_image("nonexistent.png")


def test_resize_with_padding():
    """Test image resizing with padding."""
    preprocessor = ImagePreprocessor(target_size=512)

    # Create a rectangular image
    image = np.random.rand(200, 300).astype(np.float32)
    resized, original_shape = preprocessor.resize_with_padding(image, 512)

    assert resized.shape == (512, 512)
    assert original_shape == (200, 300)
    assert resized.dtype == np.float32


def test_resize_square_image():
    """Test resizing a square image."""
    preprocessor = ImagePreprocessor(target_size=256)

    image = np.random.rand(100, 100).astype(np.float32)
    resized, _ = preprocessor.resize_with_padding(image, 256)

    assert resized.shape == (256, 256)


def test_preprocess_pipeline(tmp_path):
    """Test complete preprocessing pipeline."""
    # Create a test image
    img = Image.new("RGB", (150, 200), color=(128, 128, 128))
    img_path = tmp_path / "test.jpg"
    img.save(img_path)

    preprocessor = ImagePreprocessor(target_size=512)
    result = preprocessor.preprocess(img_path)

    assert result.shape == (512, 512)
    assert result.dtype == np.float32
    assert result.min() >= 0.0
    assert result.max() <= 1.0

