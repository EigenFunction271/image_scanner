"""Tests for SpectralPeakDetector main class."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from image_screener.spectral_peak_detector import SpectralPeakDetector


def test_spectral_peak_detector_initialization():
    """Test SpectralPeakDetector initialization."""
    detector = SpectralPeakDetector(
        target_size=1024, sensitivity=1.5, high_freq_threshold=0.4
    )
    assert detector.target_size == 1024
    assert detector.sensitivity == 1.5
    assert detector.high_freq_threshold == 0.4


def test_analyze_image(tmp_path):
    """Test analyzing a single image."""
    # Create a test image
    img = Image.new("RGB", (200, 200), color=(128, 128, 128))
    img_path = tmp_path / "test.png"
    img.save(img_path)

    detector = SpectralPeakDetector(target_size=512)
    result = detector.analyze(img_path)

    assert result.image_path == str(img_path)
    assert 0.0 <= result.artifact_score <= 1.0
    assert result.num_peaks >= 0
    assert isinstance(result.peaks, list)
    assert result.log_magnitude_spectrum.shape == (512, 512)
    assert result.preprocessed_image.shape == (512, 512)


def test_batch_analyze(tmp_path):
    """Test batch analysis of multiple images."""
    # Create multiple test images
    image_paths = []
    for i in range(3):
        img = Image.new("RGB", (100, 100), color=(i * 50, i * 50, i * 50))
        img_path = tmp_path / f"test_{i}.png"
        img.save(img_path)
        image_paths.append(img_path)

    detector = SpectralPeakDetector()
    results = detector.batch_analyze(image_paths)

    assert len(results) == 3
    for result in results:
        assert 0.0 <= result.artifact_score <= 1.0
        assert result.num_peaks >= 0

