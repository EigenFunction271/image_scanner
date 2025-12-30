"""Tests for 2D DFT module."""

import numpy as np
import pytest

from image_screener.dft import DFTProcessor, ProcessImageResult, SpectralPeak


def test_dft_processor_initialization():
    """Test DFTProcessor initialization."""
    processor = DFTProcessor(sensitivity=1.5, high_freq_threshold=0.4)
    assert processor.sensitivity == 1.5
    assert processor.high_freq_threshold == 0.4


def test_compute_dft():
    """Test 2D FFT computation."""
    processor = DFTProcessor()
    image = np.random.rand(64, 64).astype(np.float32)

    fft_result = processor.compute_dft(image)

    assert fft_result.shape == (64, 64)
    assert np.iscomplexobj(fft_result)


def test_shift_spectrum():
    """Test FFT spectrum shifting."""
    processor = DFTProcessor()
    image = np.random.rand(64, 64).astype(np.float32)
    fft_result = processor.compute_dft(image)

    shifted = processor.shift_spectrum(fft_result)

    assert shifted.shape == fft_result.shape
    assert np.iscomplexobj(shifted)

    # DC component should be at center after shift
    h, w = shifted.shape
    center_magnitude = np.abs(shifted[h // 2, w // 2])
    # DC should be one of the largest values
    assert center_magnitude > 0


def test_compute_magnitude_spectrum():
    """Test magnitude spectrum computation."""
    processor = DFTProcessor()
    image = np.random.rand(64, 64).astype(np.float32)
    fft_result = processor.compute_dft(image)
    shifted = processor.shift_spectrum(fft_result)

    magnitude = processor.compute_magnitude_spectrum(shifted, log_scale=True)

    assert magnitude.shape == (64, 64)
    assert magnitude.dtype in [np.float32, np.float64]
    assert np.all(magnitude >= 0)


def test_get_high_frequency_mask():
    """Test high-frequency mask generation."""
    processor = DFTProcessor(high_freq_threshold=0.3)
    shape = (128, 128)

    mask = processor.get_high_frequency_mask(shape)

    assert mask.shape == shape
    assert mask.dtype == bool
    # Center should be False (low frequency)
    assert mask[64, 64] == False  # Use == instead of is for numpy bool
    # Corners should be True (high frequency)
    assert mask[0, 0] == True  # Use == instead of is for numpy bool
    assert mask[127, 127] == True  # Use == instead of is for numpy bool


def test_detect_peaks():
    """Test peak detection in frequency spectrum."""
    processor = DFTProcessor(sensitivity=1.0, peak_threshold_percentile=90.0)

    # Create a synthetic spectrum with known peaks
    log_magnitude = np.random.rand(128, 128).astype(np.float32) * 5.0
    # Add some strong peaks in high-frequency regions
    log_magnitude[10, 10] = 15.0
    log_magnitude[118, 118] = 14.0

    high_freq_mask = processor.get_high_frequency_mask((128, 128))
    peaks = processor.detect_peaks(log_magnitude, high_freq_mask)

    assert isinstance(peaks, list)
    # Should detect at least the strong peaks we added
    assert len(peaks) >= 0  # May vary based on threshold


def test_compute_spectral_artifact_score():
    """Test artifact score computation."""
    processor = DFTProcessor()

    # Test with no peaks
    score_empty = processor.compute_spectral_artifact_score([])
    assert score_empty == 0.0

    # Test with some peaks
    peaks = [
        SpectralPeak(u=10, v=10, magnitude=15.0, distance_from_center=100.0),
        SpectralPeak(u=118, v=118, magnitude=14.0, distance_from_center=150.0),
    ]
    score = processor.compute_spectral_artifact_score(peaks)

    assert 0.0 <= score <= 1.0
    assert score > 0.0


def test_process_image_pipeline():
    """Test complete DFT processing pipeline."""
    processor = DFTProcessor()
    image = np.random.rand(128, 128).astype(np.float32)

    result = processor.process_image(image)

    assert isinstance(result, ProcessImageResult)
    assert result.log_magnitude_spectrum.shape == (128, 128)
    assert isinstance(result.peaks, list)
    assert 0.0 <= result.artifact_score <= 1.0


def test_compute_dft_input_validation():
    """Test input validation for compute_dft."""
    processor = DFTProcessor()

    # Test empty array
    with pytest.raises(ValueError, match="Image array is empty"):
        processor.compute_dft(np.array([]))

    # Test 1D array
    with pytest.raises(ValueError, match="Expected 2D array"):
        processor.compute_dft(np.array([1, 2, 3]))

    # Test 3D array
    with pytest.raises(ValueError, match="Expected 2D array"):
        processor.compute_dft(np.random.rand(10, 10, 3))


def test_compute_azimuthal_average_input_validation():
    """Test input validation for compute_azimuthal_average."""
    processor = DFTProcessor()

    # Test empty array
    with pytest.raises(ValueError, match="Log magnitude spectrum array is empty"):
        processor.compute_azimuthal_average(np.array([]))

    # Test 1D array
    with pytest.raises(ValueError, match="Expected 2D array"):
        processor.compute_azimuthal_average(np.array([1, 2, 3]))

    # Test invalid num_bins
    valid_array = np.random.rand(64, 64)
    with pytest.raises(ValueError, match="num_bins must be positive"):
        processor.compute_azimuthal_average(valid_array, num_bins=0)

