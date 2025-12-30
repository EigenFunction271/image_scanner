"""Tests for wavelet-based detector."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from image_screener.feature_extractor import (
    compute_cross_scale_correlation,
    compute_energy_features,
    compute_ggd_features,
    compute_noise_consistency,
    compute_statistical_moments,
    detect_periodic_artifacts,
    extract_all_features,
)
from image_screener.wavelet_detector import WaveletDetector


def test_wavelet_detector_initialization():
    """Test WaveletDetector initialization."""
    detector = WaveletDetector(wavelet='db4', levels=3)
    assert detector.wavelet == 'db4'
    assert detector.levels == 3
    assert detector.classifier is None
    assert not detector.is_fitted


def test_preprocess_image(tmp_path):
    """Test image preprocessing."""
    # Create a test image
    img = Image.new('RGB', (200, 200), color=(128, 128, 128))
    img_path = tmp_path / "test.png"
    img.save(img_path)

    detector = WaveletDetector()
    preprocessed = detector.preprocess_image(img_path)

    assert preprocessed.shape[0] >= 200
    assert preprocessed.shape[1] >= 200
    assert preprocessed.dtype == np.float32
    assert preprocessed.min() >= 0.0
    assert preprocessed.max() <= 1.0


def test_preprocess_image_too_small(tmp_path):
    """Test error handling for images that are too small."""
    img = Image.new('RGB', (32, 32), color=(128, 128, 128))
    img_path = tmp_path / "small.png"
    img.save(img_path)

    detector = WaveletDetector()
    with pytest.raises(ValueError, match="Image too small"):
        detector.preprocess_image(img_path)


def test_decompose():
    """Test wavelet decomposition."""
    detector = WaveletDetector(levels=3)
    image = np.random.rand(256, 256).astype(np.float32)

    subbands = detector.decompose(image)

    # Should have 10 subbands: LL3 + 9 detail subbands
    assert 'LL3' in subbands
    for level in [1, 2, 3]:
        for orientation in ['LH', 'HL', 'HH']:
            assert f'{orientation}{level}' in subbands


def test_extract_features():
    """Test feature extraction."""
    detector = WaveletDetector(levels=3)
    image = np.random.rand(256, 256).astype(np.float32)
    subbands = detector.decompose(image)

    features = detector.extract_features(subbands)

    assert isinstance(features, np.ndarray)
    assert len(features) >= 85  # Should have ~85-90 features
    assert np.all(np.isfinite(features))


def test_compute_energy_features():
    """Test energy feature computation."""
    subbands = {
        'LL3': np.random.rand(32, 32),
        'LH1': np.random.rand(128, 128),
        'HL1': np.random.rand(128, 128),
        'HH1': np.random.rand(128, 128),
        'LH2': np.random.rand(64, 64),
        'HL2': np.random.rand(64, 64),
        'HH2': np.random.rand(64, 64),
        'LH3': np.random.rand(32, 32),
        'HL3': np.random.rand(32, 32),
        'HH3': np.random.rand(32, 32),
    }

    features = compute_energy_features(subbands)

    assert len(features) == 15
    assert np.all(np.isfinite(features))
    assert np.all(features >= 0)


def test_compute_statistical_moments():
    """Test statistical moment computation."""
    subbands = {
        'LH1': np.random.rand(128, 128),
        'HL1': np.random.rand(128, 128),
        'HH1': np.random.rand(128, 128),
        'LH2': np.random.rand(64, 64),
        'HL2': np.random.rand(64, 64),
        'HH2': np.random.rand(64, 64),
        'LH3': np.random.rand(32, 32),
        'HL3': np.random.rand(32, 32),
        'HH3': np.random.rand(32, 32),
    }

    features = compute_statistical_moments(subbands)

    assert len(features) == 36  # 9 subbands × 4 moments
    assert np.all(np.isfinite(features))


def test_compute_ggd_features():
    """Test GGD parameter computation."""
    subbands = {
        'LH1': np.random.rand(128, 128),
        'HL1': np.random.rand(128, 128),
        'HH1': np.random.rand(128, 128),
        'LH2': np.random.rand(64, 64),
        'HL2': np.random.rand(64, 64),
        'HH2': np.random.rand(64, 64),
        'LH3': np.random.rand(32, 32),
        'HL3': np.random.rand(32, 32),
        'HH3': np.random.rand(32, 32),
    }

    features = compute_ggd_features(subbands)

    assert len(features) == 18  # 9 subbands × 2 parameters
    assert np.all(np.isfinite(features))
    assert np.all(features >= 0)  # Alpha and beta should be positive


def test_detect_periodic_artifacts():
    """Test periodic artifact detection."""
    hh1 = np.random.rand(128, 128)
    features = detect_periodic_artifacts(hh1)

    assert len(features) == 6
    assert np.all(np.isfinite(features))
    assert np.all(features >= 0)


def test_compute_noise_consistency():
    """Test noise consistency computation."""
    hh1 = np.random.rand(128, 128)
    features = compute_noise_consistency(hh1)

    assert len(features) == 3
    assert np.all(np.isfinite(features))
    assert features[0] >= 0  # Mean variance should be non-negative


def test_extract_all_features():
    """Test complete feature extraction."""
    subbands = {
        'LL3': np.random.rand(32, 32),
        'LH1': np.random.rand(128, 128),
        'HL1': np.random.rand(128, 128),
        'HH1': np.random.rand(128, 128),
        'LH2': np.random.rand(64, 64),
        'HL2': np.random.rand(64, 64),
        'HH2': np.random.rand(64, 64),
        'LH3': np.random.rand(32, 32),
        'HL3': np.random.rand(32, 32),
        'HH3': np.random.rand(32, 32),
    }

    features = extract_all_features(subbands)

    assert len(features) >= 85
    assert np.all(np.isfinite(features))


def test_save_load_model(tmp_path):
    """Test model saving and loading."""
    detector = WaveletDetector()

    # Create dummy training data
    X_train = np.random.rand(100, 87)
    y_train = np.random.randint(0, 2, 100)

    # Train
    detector.fit(X_train, y_train)

    # Save
    model_path = tmp_path / "test_model.pkl"
    detector.save_model(model_path)

    # Load
    detector2 = WaveletDetector()
    detector2.load_model(model_path)

    assert detector2.is_fitted
    assert detector2.wavelet == detector.wavelet
    assert detector2.levels == detector.levels

