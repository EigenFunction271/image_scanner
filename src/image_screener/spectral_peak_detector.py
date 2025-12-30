"""Spectral Peak Detector - Main class for Filter 01 implementation."""

import logging
from pathlib import Path
from typing import NamedTuple, Union

import numpy as np
from pydantic import Field
from pydantic.dataclasses import dataclass

from image_screener.dft import DFTProcessor, ProcessImageResult, SpectralPeak
from image_screener.preprocessing import ImagePreprocessor

logger = logging.getLogger(__name__)

__all__ = ['SpectralPeakDetector', 'DetectionResult']


class DetectionResult(NamedTuple):
    """Result of spectral peak detection analysis."""

    image_path: str
    artifact_score: float
    num_peaks: int
    peaks: list[SpectralPeak]
    log_magnitude_spectrum: np.ndarray
    preprocessed_image: np.ndarray
    grid_strength: float  # Spatial autocorrelation grid pattern strength (0.0-1.0)
    grid_interval_u: float  # Dominant horizontal grid interval
    grid_interval_v: float  # Dominant vertical grid interval
    azimuthal_radii: np.ndarray  # Radial distances for azimuthal average
    azimuthal_average: np.ndarray  # Azimuthal average of magnitude spectrum
    grid_consistency: float  # Grid consistency score (power-of-2 alignment) (0.0-1.0)
    nyquist_symmetry: float  # Nyquist folding symmetry score (0.0-1.0)


@dataclass
class SpectralPeakDetector:
    """
    Filter 01: Spectral Peak Detector (FFT).

    Identifies periodic "grid" artifacts caused by transposed convolutions
    in AI-generated images by analyzing the frequency domain.
    """

    target_size: int = Field(default=512, ge=256, le=2048)
    sensitivity: float = Field(default=1.0, gt=0.0)
    high_freq_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    peak_threshold_percentile: float = Field(default=98.0, ge=0.0, le=100.0)

    def __post_init__(self):
        """Initialize sub-processors."""
        self.preprocessor = ImagePreprocessor(
            target_size=self.target_size, normalize_to_float=True
        )
        self.dft_processor = DFTProcessor(
            sensitivity=self.sensitivity,
            high_freq_threshold=self.high_freq_threshold,
            peak_threshold_percentile=self.peak_threshold_percentile,
        )

    def analyze(self, image_path: Union[str, Path]) -> DetectionResult:
        """
        Analyze an image for spectral artifacts.

        Args:
            image_path: Path to the image file (.jpg, .png, .webp)

        Returns:
            DetectionResult with artifact score, peaks, and spectra
        """
        logger.info(f"Analyzing image: {image_path}")

        # Preprocess image
        preprocessed = self.preprocessor.preprocess(image_path)

        # Process with DFT
        process_result = self.dft_processor.process_image(preprocessed)

        result = DetectionResult(
            image_path=str(image_path),
            artifact_score=process_result.artifact_score,
            num_peaks=len(process_result.peaks),
            peaks=process_result.peaks,
            log_magnitude_spectrum=process_result.log_magnitude_spectrum,
            preprocessed_image=preprocessed,
            grid_strength=process_result.grid_strength,
            grid_interval_u=process_result.grid_interval_u,
            grid_interval_v=process_result.grid_interval_v,
            azimuthal_radii=process_result.azimuthal_radii,
            azimuthal_average=process_result.azimuthal_average,
            grid_consistency=process_result.grid_consistency,
            nyquist_symmetry=process_result.nyquist_symmetry,
        )

        logger.info(
            f"Analysis complete: artifact_score={result.artifact_score:.4f}, "
            f"num_peaks={result.num_peaks}, grid_strength={result.grid_strength:.4f}"
        )

        return result

    def batch_analyze(
        self, image_paths: list[Union[str, Path]]
    ) -> list[DetectionResult]:
        """
        Analyze multiple images in batch.

        Args:
            image_paths: List of paths to image files

        Returns:
            List of DetectionResult objects
        """
        logger.info(f"Batch analyzing {len(image_paths)} images")
        results = [self.analyze(path) for path in image_paths]
        return results

