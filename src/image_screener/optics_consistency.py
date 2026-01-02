"""Optics Consistency Detector - Physical optical law validation.

This module implements detectors that check if an image follows physical
optical laws expected from real cameras:
1. Monotonic low-pass Optical Transfer Function (OTF)
2. Blur occurs before detail and noise injection
3. Depth-of-field blur varies continuously with depth
4. Chromatic aberration is small but non-zero and spatially coherent

REFACTORED: Types are now imported from optics_tests.types module.
"""

import logging
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
from pydantic import Field
from pydantic.dataclasses import dataclass as pydantic_dataclass
from scipy import signal
from scipy.interpolate import interp1d

from image_screener.dft import DFTProcessor, get_center_coords
from image_screener.preprocessing import ImagePreprocessor
from image_screener.optics_tests.types import (
    OpticsTestResult,
    OpticsConsistencyResult,
)

logger = logging.getLogger(__name__)

__all__ = [
    'OpticsConsistencyDetector',
    'OpticsConsistencyResult',
    'OpticsTestResult',
    'FrequencyDomainOpticsTest',
    'EdgePSFTest',
    'DepthOfFieldConsistencyTest',
    'ChromaticAberrationTest',
    'SensorNoiseResidualTest',
]

# ============================================================================
# TEST 1: FREQUENCY DOMAIN OPTICS TEST
# ============================================================================


@pydantic_dataclass
class FrequencyDomainOpticsTest:
    """Test 1: Frequency-domain optics test.

    Checks for monotonic OTF decay by analyzing radial power spectrum.
    Real cameras apply a monotonic low-pass OTF, so the power spectrum
    should decay smoothly without mid-frequency bumps or high-frequency suppression.
    Also checks for missing stochastic noise floor (over-clean spectra).
    """
    
    # Threshold constants for OTF analysis
    HIGH_FREQ_THRESHOLD = 0.7  # Radius threshold for high-frequency region
    MIN_RADIUS_THRESHOLD = 0.1  # Minimum radius to consider
    MAX_RADIUS_THRESHOLD = 0.9  # Maximum radius to consider
    # OTF slope threshold: In log-log space, real camera optics typically show
    # slopes of -1.0 to -2.0 (steep decay). Shallow slopes (> -0.5) indicate
    # non-physical frequency response, often from AI upsampling or artificial enhancement.
    # Threshold of -0.5 provides conservative detection while allowing some margin
    # for natural variation. Original threshold of -0.1 was too permissive.
    OTF_SLOPE_THRESHOLD = -0.5  # Minimum acceptable OTF decay slope (refined from -0.1)
    BUMP_RATIO_THRESHOLD = 0.1  # Maximum acceptable bump ratio (10%)
    SUPPRESSION_RATIO_THRESHOLD = 0.15  # Maximum acceptable suppression ratio (15%)
    RESIDUAL_STD_THRESHOLD = 0.5  # High variance threshold for non-smooth decay
    RESIDUAL_STD_BASE = 0.3  # Base value for residual std penalty calculation
    NOISE_FLOOR_RATIO_THRESHOLD = 0.1  # Minimum noise floor ratio (10% of expected)
    BUMP_DETECTION_FACTOR = 1.5  # Factor for bump detection threshold

    def _estimate_high_frequency_variance(
        self, log_magnitude: np.ndarray, radii: np.ndarray, radial_power: np.ndarray
    ) -> float:
        """
        Estimate variance in high-frequency region of the spectrum.

        Args:
            log_magnitude: Log-magnitude spectrum (H, W)
            radii: Radial distances from azimuthal average
            radial_power: Azimuthal average power values

        Returns:
            Estimated variance in high-frequency region
        """
        # Focus on high frequencies
        high_freq_mask = radii > self.HIGH_FREQ_THRESHOLD
        if np.sum(high_freq_mask) < 5:
            # If not enough high-freq data, use top 20% of frequencies
            high_freq_mask = radii > np.percentile(radii, 80)

        if np.sum(high_freq_mask) == 0:
            return 0.0

        # Compute variance of power in high-frequency region
        high_freq_power = radial_power[high_freq_mask]
        variance = np.var(high_freq_power)

        # Also check spatial variance in high-freq region of 2D spectrum
        h, w = log_magnitude.shape
        center_y, center_x = get_center_coords((h, w))

        # Create mask for high-frequency region in 2D
        y_coords, x_coords = np.ogrid[:h, :w]
        distances = np.sqrt((y_coords - center_y) ** 2 + (x_coords - center_x) ** 2)
        max_distance = np.sqrt(center_y ** 2 + center_x ** 2)
        normalized_distances = distances / max_distance

        high_freq_2d_mask = normalized_distances > self.HIGH_FREQ_THRESHOLD
        if np.sum(high_freq_2d_mask) > 0:
            high_freq_2d_values = log_magnitude[high_freq_2d_mask]
            spatial_variance = np.var(high_freq_2d_values)
            # Combine both measures
            variance = (variance + spatial_variance) / 2.0

        return variance

    def _estimate_expected_noise_floor(self, image: np.ndarray) -> float:
        """
        Estimate expected noise floor based on image characteristics.

        Real sensor noise depends on:
        - Image brightness (darker regions have more noise)
        - Sensor characteristics (assume typical consumer camera)
        - Quantization noise

        Args:
            image: Grayscale image (H, W), float32 [0, 1]

        Returns:
            Expected noise floor variance
        """
        # Estimate noise from image statistics
        # Real images have noise that scales with signal level
        mean_intensity = np.mean(image)
        std_intensity = np.std(image)

        # Base noise floor estimate
        # Typical sensor noise: ~0.5-2% of signal for consumer cameras
        # In log space, this translates to variance in high frequencies
        base_noise_floor = 0.01 * mean_intensity

        # Add quantization noise (from 8-bit conversion)
        quantization_noise = 1.0 / (255.0 * 255.0)  # ~1.5e-5

        # Add texture-dependent noise (real images have more variance)
        texture_noise = 0.1 * std_intensity

        # Expected noise floor in log-magnitude space
        # Convert to log space: log(1 + noise) ≈ noise for small values
        expected_variance = base_noise_floor + quantization_noise + texture_noise

        # Scale to log-magnitude space (typical values are 0.01-0.1)
        # Real images typically have high-freq variance > 0.01
        expected_variance = max(0.01, expected_variance * 10.0)

        return expected_variance

    def test(self, image: np.ndarray) -> OpticsTestResult:
        """
        Test frequency-domain optics consistency.

        Args:
            image: Grayscale image (H, W), float32 [0, 1]

        Returns:
            OpticsTestResult with score and violations
        """
        logger.debug("Running frequency-domain optics test")

        # Compute 2D FFT
        dft_processor = DFTProcessor()
        fft_result = dft_processor.compute_dft(image)
        fft_shifted = dft_processor.shift_spectrum(fft_result)
        magnitude = np.abs(fft_shifted)
        log_magnitude = np.log1p(magnitude)

        # Compute radial average (azimuthal average)
        radii, radial_power = dft_processor.compute_azimuthal_average(
            log_magnitude, num_bins=256
        )

        # Fit log-log slope to check for monotonic decay
        # Use only mid-to-high frequencies (avoid DC and very high freq noise)
        valid_mask = (radii > self.MIN_RADIUS_THRESHOLD) & (radii < self.MAX_RADIUS_THRESHOLD)
        if np.sum(valid_mask) < 10:
            return OpticsTestResult(
                score=0.5,
                violations=["Insufficient frequency data for analysis"],
                diagnostic_data={"radii": radii, "radial_power": radial_power},
            )

        log_radii = np.log(radii[valid_mask] + 1e-10)
        log_power = radial_power[valid_mask]

        # Fit linear model: log(power) = a * log(radius) + b
        # For real optics, we expect negative slope (decay)
        try:
            slope, intercept = np.polyfit(log_radii, log_power, deg=1)
        except Exception as e:
            logger.warning(f"Failed to fit log-log slope: {e}")
            return OpticsTestResult(
                score=0.5,
                violations=["Could not fit power spectrum"],
                diagnostic_data={"radii": radii, "radial_power": radial_power},
            )

        # Check for deviations from smooth decay
        # Compute residuals from linear fit
        predicted = slope * log_radii + intercept
        residuals = log_power - predicted

        # Detect mid-frequency bumps (positive residuals)
        bump_threshold = np.std(residuals) * self.BUMP_DETECTION_FACTOR
        bumps = np.sum(residuals > bump_threshold)
        bump_ratio = bumps / len(residuals)

        # Detect high-frequency suppression (negative residuals at high freq)
        high_freq_mask = radii[valid_mask] > self.HIGH_FREQ_THRESHOLD
        if np.sum(high_freq_mask) > 0:
            high_freq_residuals = residuals[high_freq_mask]
            suppression = np.sum(high_freq_residuals < -bump_threshold)
            suppression_ratio = suppression / len(high_freq_residuals)
        else:
            suppression_ratio = 0.0

        # Score: penalize bumps and suppression
        # Good optics: smooth decay, negative slope, low residuals
        violations = []
        score = 1.0

        if slope > self.OTF_SLOPE_THRESHOLD:
            # Slope is too shallow (less negative than threshold)
            # Real cameras: typically -1.0 to -2.0, so slopes > -0.5 are suspicious
            violations.append(
                f"Non-monotonic OTF decay (slope: {slope:.3f} > {self.OTF_SLOPE_THRESHOLD}) - "
                f"shallow decay suggests non-physical frequency response (AI upsampling or artificial enhancement)"
            )
            # Progressive penalty: more shallow = more suspicious
            # Slope of -0.5 (at threshold) gets 0.5 penalty
            # Slope of 0.0 (flat) gets 0.1 penalty
            # Slope > 0.0 (positive, non-monotonic) gets 0.05 penalty
            if slope > 0.0:
                score *= 0.05  # Positive slope (non-monotonic) - very suspicious
            elif slope > -0.2:
                score *= 0.1   # Very shallow negative slope
            elif slope > -0.35:
                score *= 0.3   # Shallow slope
            else:
                score *= 0.5   # At threshold, moderate penalty

        if bump_ratio > self.BUMP_RATIO_THRESHOLD:
            violations.append(f"Mid-frequency bumps detected ({bump_ratio:.1%})")
            score *= max(0.3, 1.0 - bump_ratio * 2)

        if suppression_ratio > self.SUPPRESSION_RATIO_THRESHOLD:
            violations.append(
                f"High-frequency suppression detected ({suppression_ratio:.1%})"
            )
            score *= max(0.3, 1.0 - suppression_ratio * 2)

        # Check smoothness of decay (low variance in residuals)
        residual_std = np.std(residuals)
        if residual_std > self.RESIDUAL_STD_THRESHOLD:
            violations.append("Non-smooth power spectrum decay")
            score *= max(0.5, 1.0 - (residual_std - self.RESIDUAL_STD_BASE) * 0.5)

        # Test for missing stochastic noise floor (over-clean spectra)
        # Real images have sensor noise that appears as variance in high frequencies
        # AI-generated images may be too clean (near-zero variance)
        high_freq_variance = self._estimate_high_frequency_variance(
            log_magnitude, radii, radial_power
        )
        expected_noise_floor = self._estimate_expected_noise_floor(image)
        
        if high_freq_variance < expected_noise_floor * self.NOISE_FLOOR_RATIO_THRESHOLD:
            violations.append(
                f"Missing noise floor detected (variance: {high_freq_variance:.4f}, "
                f"expected: {expected_noise_floor:.4f})"
            )
            # Penalize more severely for very clean spectra
            variance_ratio = high_freq_variance / (expected_noise_floor + 1e-10)
            score *= max(0.2, variance_ratio * 2)  # Scale penalty with how clean it is

        # NEW METRIC 1: High-frequency noise-floor energy
        # AI often has unnaturally low or unnaturally uniform tail
        # Real sensors: noise floor has stochastic variation
        # AI: noise floor is too clean (low energy) or too uniform (low variance)
        high_freq_mask = radii > self.HIGH_FREQ_THRESHOLD
        if np.sum(high_freq_mask) > 5:
            high_freq_power = radial_power[high_freq_mask]
            high_freq_energy = np.mean(high_freq_power)  # Mean energy in tail
            high_freq_uniformity = 1.0 / (np.std(high_freq_power) + 1e-6)  # Inverse of std (uniformity measure)
            
            # Real sensors: energy should be above threshold, but not too uniform
            # Too low energy = over-clean (AI)
            # Too uniform = artificial smoothing (AI)
            if high_freq_energy < expected_noise_floor * 0.5:
                violations.append(
                    f"Unnaturally low high-frequency energy ({high_freq_energy:.4f}) - "
                    f"suggests AI over-clean noise floor"
                )
                score *= max(0.3, high_freq_energy / (expected_noise_floor * 0.5))
            
            # Check for unnaturally uniform tail (low variance in high-freq power)
            # Real sensors: noise floor has natural variation
            # AI: often has very uniform tail
            if high_freq_uniformity > 50.0:  # Very uniform (low std relative to mean)
                violations.append(
                    f"Unnaturally uniform high-frequency tail (uniformity: {high_freq_uniformity:.2f}) - "
                    f"suggests AI artificial smoothing"
                )
                score *= max(0.4, 1.0 - (high_freq_uniformity - 20.0) / 100.0)
        else:
            high_freq_energy = 0.0
            high_freq_uniformity = 0.0

        # NEW METRIC 2: Mid-band bump metric
        # Ratio of mid-band energy to both low and high bands
        # Real optics: smooth decay, mid-band should be between low and high
        # AI: may have artificial bumps in mid-band (upsampling artifacts)
        low_band_mask = (radii >= 0.1) & (radii < 0.3)
        mid_band_mask = (radii >= 0.3) & (radii < 0.7)
        high_band_mask = radii >= 0.7
        
        if np.sum(low_band_mask) > 0 and np.sum(mid_band_mask) > 0 and np.sum(high_band_mask) > 0:
            low_band_energy = np.mean(radial_power[low_band_mask])
            mid_band_energy = np.mean(radial_power[mid_band_mask])
            high_band_energy = np.mean(radial_power[high_band_mask])
            
            # Compute ratios: mid-band relative to low and high
            # Real optics: mid should be between low and high (smooth decay)
            # AI: mid may be elevated relative to expected decay
            if low_band_energy > 1e-6 and high_band_energy > 1e-6:
                # Expected mid-band energy (interpolation between low and high)
                expected_mid_energy = (low_band_energy + high_band_energy) / 2.0
                
                # Bump metric: how much does mid-band exceed expected?
                mid_bump_ratio = mid_band_energy / (expected_mid_energy + 1e-6)
                
                # Also check ratio to low and high bands directly
                mid_to_low_ratio = mid_band_energy / (low_band_energy + 1e-6)
                mid_to_high_ratio = mid_band_energy / (high_band_energy + 1e-6)
                
                # Real optics: mid should be lower than low, higher than high (smooth decay)
                # If mid is too high relative to low, or too high relative to expected, it's a bump
                if mid_bump_ratio > 1.3:  # Mid-band is 30% higher than expected
                    violations.append(
                        f"Mid-band energy bump detected (ratio: {mid_bump_ratio:.2f}) - "
                        f"mid-band energy exceeds expected decay, suggests AI upsampling artifacts"
                    )
                    score *= max(0.4, 1.0 - (mid_bump_ratio - 1.2) * 0.5)
                
                # Check if mid-band is anomalously high relative to high-band
                # (should be decaying, so mid > high is expected, but if ratio is too extreme, suspicious)
                if mid_to_high_ratio > 3.0:  # Mid is 3x higher than high (too extreme)
                    violations.append(
                        f"Extreme mid-to-high band ratio ({mid_to_high_ratio:.2f}) - "
                        f"suggests non-physical spectral structure"
                    )
                    score *= max(0.5, 1.0 - (mid_to_high_ratio - 2.0) * 0.2)
            else:
                mid_bump_ratio = 1.0
                mid_to_low_ratio = 1.0
                mid_to_high_ratio = 1.0
        else:
            mid_bump_ratio = 1.0
            mid_to_low_ratio = 1.0
            mid_to_high_ratio = 1.0

        # NEW METRIC 3: Spatial stationarity of spectrum
        # AI tends to have more uniform spectral shape across the image
        # Real sensors: spectral shape varies spatially (due to content, lighting, etc.)
        # Compute spectral shape at different spatial regions and check variance
        logger.debug("  Computing spatial stationarity of spectrum...")
        h, w = log_magnitude.shape
        center_y, center_x = get_center_coords((h, w))
        
        # OPTIMIZED: Extract quadrant spectra from full FFT instead of recomputing
        # The log_magnitude is already computed from the full FFT, so we can extract
        # quadrants directly without recomputing FFTs
        quad_size_y = h // 3
        quad_size_x = w // 3
        
        quadrant_spectra = []
        
        # Extract radial power spectrum for each quadrant (no FFT recomputation)
        for qy in range(3):
            for qx in range(3):
                y_start = qy * quad_size_y
                y_end = min((qy + 1) * quad_size_y, h)
                x_start = qx * quad_size_x
                x_end = min((qx + 1) * quad_size_x, w)
                
                # Extract quadrant spectrum directly from precomputed log_magnitude
                # No need to recompute FFT - just extract the spatial region
                quadrant_spectrum = log_magnitude[y_start:y_end, x_start:x_end]
                
                # Compute radial average for this quadrant
                try:
                    q_radii, q_radial_power = dft_processor.compute_azimuthal_average(
                        quadrant_spectrum, num_bins=64  # Fewer bins for smaller regions
                    )
                    
                    # Normalize to same length for comparison (interpolate if needed)
                    if len(q_radial_power) > 0:
                        # Interpolate to match main spectrum length
                        if len(q_radial_power) > 1:
                            f_interp = interp1d(
                                q_radii, q_radial_power, 
                                kind='linear', 
                                bounds_error=False, 
                                fill_value='extrapolate'
                            )
                            q_power_interp = f_interp(radii)
                            quadrant_spectra.append(q_power_interp)
                except Exception:
                    continue  # Skip if quadrant too small
        
        # Compute variance in spectral shape across quadrants
        if len(quadrant_spectra) >= 3:  # Need at least 3 quadrants
            quadrant_spectra_array = np.array(quadrant_spectra)  # (n_quadrants, n_radii)
            
            # Compute variance across quadrants for each frequency bin
            spatial_variance_per_freq = np.var(quadrant_spectra_array, axis=0)  # (n_radii,)
            mean_spatial_variance = np.mean(spatial_variance_per_freq)
            
            # Real sensors: spectral shape varies spatially (high variance)
            # AI: spectral shape is uniform across image (low variance)
            if mean_spatial_variance < 0.1:  # Very uniform spectral shape
                violations.append(
                    f"Unnaturally uniform spectral shape across image (spatial variance: {mean_spatial_variance:.4f}) - "
                    f"suggests AI-generated image with uniform spectral structure"
                )
                score *= max(0.3, mean_spatial_variance * 5)  # Penalize low variance
            
            # Also check if variance is too uniform (low std of variance itself)
            # Real sensors: variance varies by frequency (some freqs more variable than others)
            # AI: variance is uniform across all frequencies (artificial uniformity)
            variance_of_variance = np.std(spatial_variance_per_freq)
            if variance_of_variance < 0.02:  # Variance itself is too uniform
                violations.append(
                    f"Uniform spectral variance pattern (variance of variance: {variance_of_variance:.4f}) - "
                    f"suggests AI artificial spectral uniformity"
                )
                score *= max(0.4, variance_of_variance * 20)
        else:
            mean_spatial_variance = 0.0
            variance_of_variance = 0.0

        score = max(0.0, min(1.0, score))

        if not violations:
            violations.append("Passes monotonic OTF test")

        return OpticsTestResult(
            score=score,
            violations=violations,
            diagnostic_data={
                "radii": radii,
                "radial_power": radial_power,
                "slope": slope,
                "intercept": intercept,
                "residuals": residuals,
                "log_radii": log_radii,
                "log_power": log_power,
                "predicted": predicted,
                "high_freq_variance": high_freq_variance,
                "expected_noise_floor": expected_noise_floor,
                "high_freq_energy": high_freq_energy,
                "high_freq_uniformity": high_freq_uniformity,
                "mid_bump_ratio": mid_bump_ratio,
                "mid_to_low_ratio": mid_to_low_ratio,
                "mid_to_high_ratio": mid_to_high_ratio,
                "spatial_variance": mean_spatial_variance,
                "variance_of_variance": variance_of_variance,
            },
        )


# ============================================================================
# TEST 2: EDGE SPREAD FUNCTION / POINT SPREAD FUNCTION TEST
# ============================================================================
# Detects strong edges, extracts Edge Spread Functions (ESF),
# differentiates to get Line Spread Function (LSF), and flags
# ringing, negative lobes, or inconsistent PSF width.
@pydantic_dataclass
class EdgePSFTest:
    """Test 2: Edge Spread Function / Point Spread Function test.

    Detects strong edges, extracts Edge Spread Functions (ESF),
    differentiates to get Line Spread Function (LSF), and flags
    ringing, negative lobes, or inconsistent PSF width.
    """

    edge_threshold: float = Field(default=0.1, gt=0.0)
    min_edge_length: int = Field(default=20, gt=0)

    def _calculate_phase_parity_symmetry(self, lsf_window: np.ndarray) -> Tuple[float, float, float]:
        """
        Method 3: Phase & Parity Symmetry Analysis.
        
        Identifies AI-generated "zero-phase" ringing artifacts by decomposing the signal
        into its Even and Odd components and analyzing spectral phase linearity.
        
        Args:
            lsf_window: LSF signal window (centered on peak, 1D array)
            
        Returns:
            Tuple of (symmetry_ratio, phase_symmetry_index, phase_variance)
            - symmetry_ratio: Symmetry Power Ratio (SR) - ratio of even component power
            - phase_symmetry_index: Phase Symmetry Index (PSI) - phase linearity measure
            - phase_variance: Variance of phase across mid-to-high frequencies (for AI detection)
        """
        if len(lsf_window) < 4:
            return 0.0, 0.0, 1.0  # High phase variance for short signals
        
        # Step 1: Parity Decomposition
        # Generate the reversed signal
        lsf_reversed = lsf_window[::-1]
        
        # Compute Even component: LSF_even = (LSF + LSF_reversed) / 2
        lsf_even = (lsf_window + lsf_reversed) / 2.0
        
        # Compute Odd component: LSF_odd = (LSF - LSF_reversed) / 2
        lsf_odd = (lsf_window - lsf_reversed) / 2.0
        
        # Calculate Symmetry Power Ratio: SR = Σ(LSF_even²) / (Σ(LSF_even²) + Σ(LSF_odd²))
        even_power = np.sum(lsf_even ** 2)
        odd_power = np.sum(lsf_odd ** 2)
        total_power = even_power + odd_power
        
        if total_power > 1e-10:
            symmetry_ratio = even_power / total_power
        else:
            symmetry_ratio = 0.0
        
        # Step 2: Spectral Phase Analysis
        # Perform FFT on the centered LSF window
        fft_result = np.fft.fft(lsf_window)
        magnitude = np.abs(fft_result)
        phase = np.angle(fft_result)
        
        # Calculate Phase Symmetry Index (PSI):
        # PSI = Σ(Magnitude · cos(Phase)) / Σ(Magnitude)
        # This identifies signals where phase is consistently 0 or π (zero-phase filters)
        # For zero-phase: cos(0) = 1, cos(π) = -1, so PSI approaches 1.0 for symmetric signals
        magnitude_cos_phase = magnitude * np.cos(phase)
        sum_magnitude_cos_phase = np.sum(magnitude_cos_phase)
        sum_magnitude = np.sum(magnitude)
        
        if sum_magnitude > 1e-10:
            phase_symmetry_index = sum_magnitude_cos_phase / sum_magnitude
            # Normalize to [0, 1] range (cos ranges from -1 to 1, but we want 0 to 1)
            # Shift and scale: (PSI + 1) / 2 maps [-1, 1] to [0, 1]
            phase_symmetry_index = (phase_symmetry_index + 1.0) / 2.0
        else:
            phase_symmetry_index = 0.0
        
        # Step 3: Calculate Phase Variance across mid-to-high frequencies
        # True AI: Low phase variance (near zero phase) across all mid-to-high frequencies
        # Professional Sharpening: Higher phase variance in high frequencies due to sensor noise interference
        # Focus on mid-to-high frequencies (skip DC and very low frequencies)
        n_freqs = len(phase)
        mid_high_start = max(1, n_freqs // 4)  # Start from 25% of frequencies
        mid_high_end = n_freqs // 2  # Up to Nyquist (half of frequencies)
        
        if mid_high_end > mid_high_start:
            mid_high_phases = phase[mid_high_start:mid_high_end]
            mid_high_magnitudes = magnitude[mid_high_start:mid_high_end]
            
            # Weight phase variance by magnitude (more weight to significant frequencies)
            if np.sum(mid_high_magnitudes) > 1e-10:
                # Normalize magnitudes for weighting
                weights = mid_high_magnitudes / np.sum(mid_high_magnitudes)
                # Calculate weighted phase variance
                # For zero-phase signals, phases should be near 0 or π, so variance should be low
                # For signals with noise, phases will be more random, variance will be higher
                weighted_mean_phase = np.sum(weights * mid_high_phases)
                phase_variance = np.sum(weights * (mid_high_phases - weighted_mean_phase) ** 2)
                # Normalize by π² to get variance in [0, 1] range (phases are in [-π, π])
                phase_variance = phase_variance / (np.pi ** 2)
            else:
                phase_variance = 1.0  # High variance if no significant frequencies
        else:
            phase_variance = 1.0  # High variance if insufficient frequencies
        
        return float(symmetry_ratio), float(phase_symmetry_index), float(phase_variance)

    def _compute_lsf(self, esf: np.ndarray, normalize: bool = True) -> np.ndarray:
        """
        Compute Line Spread Function (LSF) from Edge Spread Function (ESF).

        Uses Savitzky-Golay smoothing followed by central difference for accuracy.

        Args:
            esf: Edge Spread Function (1D array of intensity values)
            normalize: If True, normalize LSF so peak is 1.0 (for relative thresholds)

        Returns:
            Line Spread Function (LSF) - derivative of ESF, optionally normalized
        """
        # Smooth ESF using Savitzky-Golay filter to reduce noise
        # Window length: 5 (must be odd), polynomial order: 2
        # This preserves edges while reducing noise
        if len(esf) >= 5:
            try:
                esf_smooth = signal.savgol_filter(esf, window_length=5, polyorder=2)
            except Exception:
                # Fallback to Gaussian smoothing if Savitzky-Golay fails
                from scipy.ndimage import gaussian_filter1d
                esf_smooth = gaussian_filter1d(esf, sigma=0.5)
        else:
            esf_smooth = esf

        # Use central difference for more accurate derivative
        # Central difference: (f[i+1] - f[i-1]) / 2
        # This is O(h^2) accurate vs O(h) for forward difference
        if len(esf_smooth) < 3:
            # Too short for central difference, use forward difference
            return np.diff(esf_smooth)

        # Vectorized central difference for interior points
        lsf = (esf_smooth[2:] - esf_smooth[:-2]) / 2.0

        # Pad to match original length (use forward/backward difference at edges)
        lsf_padded = np.zeros(len(esf_smooth))
        lsf_padded[1:-1] = lsf
        # Forward difference at start
        lsf_padded[0] = esf_smooth[1] - esf_smooth[0]
        # Backward difference at end
        lsf_padded[-1] = esf_smooth[-1] - esf_smooth[-2]

        # Normalize to peak = 1.0 for relative thresholding
        if normalize:
            max_abs = np.max(np.abs(lsf_padded))
            if max_abs > 1e-10:  # Avoid division by zero
                lsf_padded = lsf_padded / max_abs

        return lsf_padded

    def _estimate_high_frequency_energy(self, image: np.ndarray) -> float:
        """
        Estimate high-frequency energy in the image.
        
        Used to adjust NCC threshold dynamically:
        - High high-frequency energy → require higher NCC (>0.9) for AI flag
        - Low high-frequency energy → lower NCC threshold acceptable
        
        Args:
            image: Grayscale image (H, W), float32 [0, 1]
            
        Returns:
            High-frequency energy estimate (0.0-1.0)
        """
        # Compute 2D FFT to estimate high-frequency content
        dft_processor = DFTProcessor()
        fft_result = dft_processor.compute_dft(image)
        fft_shifted = dft_processor.shift_spectrum(fft_result)
        magnitude = np.abs(fft_shifted)
        
        h, w = magnitude.shape
        center_y, center_x = get_center_coords((h, w))
        
        # Create mask for high-frequency region
        y_coords, x_coords = np.ogrid[:h, :w]
        distances = np.sqrt((y_coords - center_y) ** 2 + (x_coords - center_x) ** 2)
        max_distance = np.sqrt(center_y ** 2 + center_x ** 2)
        normalized_distances = distances / max_distance
        
        # High frequencies: outer 30% of spectrum
        high_freq_mask = normalized_distances > 0.7
        if np.sum(high_freq_mask) > 0:
            high_freq_energy = np.mean(magnitude[high_freq_mask])
            # Normalize by total energy for relative measure
            total_energy = np.mean(magnitude)
            if total_energy > 1e-10:
                return float(high_freq_energy / total_energy)
        
        return 0.0

    def test(
        self, 
        image: np.ndarray, 
        ca_is_radial: Optional[bool] = None
    ) -> OpticsTestResult:
        """
        Test edge spread function consistency.

        Args:
            image: Grayscale image (H, W), float32 [0, 1]
            ca_is_radial: Optional CA test result - if True, CA is radial (reduces AI penalty)

        Returns:
            OpticsTestResult with score and violations
        """
        logger.debug("Running edge PSF test")
        
        # Estimate high-frequency energy for dynamic NCC threshold
        # This is used later to adjust NCC threshold (higher energy → require higher NCC)
        high_freq_energy = self._estimate_high_frequency_energy(image)
        logger.debug(f"  High-frequency energy: {high_freq_energy:.4f}")

        # Convert to uint8 for edge detection
        img_uint8 = (image * 255).astype(np.uint8)

        # Detect edges using Canny
        edges = cv2.Canny(img_uint8, 50, 150)

        if np.sum(edges) == 0:
            return OpticsTestResult(
                score=0.5,
                violations=["No edges detected for PSF analysis"],
                diagnostic_data={},
            )

        # Find edge pixels
        edge_coords = np.column_stack(np.where(edges > 0))

        if len(edge_coords) < self.min_edge_length:
            return OpticsTestResult(
                score=0.5,
                violations=["Insufficient edges for PSF analysis"],
                diagnostic_data={},
            )

        # Extract ESF samples from edges
        # For each edge pixel, extract perpendicular profile
        # Note: Uses bilinear interpolation to avoid phase errors from nearest-neighbor sampling.
        # Future enhancement: Implement slanted-edge binning for 4× or 8× oversampling
        # (see ISO 12233 standard for MTF measurement)
        esf_samples = []
        lsf_samples = []

        # Sample a subset of edges to avoid excessive computation
        sample_indices = np.linspace(0, len(edge_coords) - 1, min(50, len(edge_coords)), dtype=int)

        for idx in sample_indices:
            y, x = edge_coords[idx]

            # Get edge orientation using gradient
            if y > 0 and y < image.shape[0] - 1 and x > 0 and x < image.shape[1] - 1:
                # Compute gradient direction
                gy = image[y + 1, x] - image[y - 1, x]
                gx = image[y, x + 1] - image[y, x - 1]

                if abs(gx) < 1e-6 and abs(gy) < 1e-6:
                    continue

                # Perpendicular direction (rotate 90 degrees)
                perp_x = -gy
                perp_y = gx
                norm = np.sqrt(perp_x**2 + perp_y**2)
                if norm < 1e-6:
                    continue

                perp_x /= norm
                perp_y /= norm

                # Extract profile perpendicular to edge using bilinear interpolation
                # This avoids phase errors from nearest-neighbor sampling
                profile_length = 41  # Odd number for center pixel
                profile = []

                for i in range(profile_length):
                    offset = i - profile_length // 2
                    py = y + offset * perp_y  # Keep as float for interpolation
                    px = x + offset * perp_x

                    # Check bounds (with margin for interpolation)
                    if 0.5 <= py < image.shape[0] - 0.5 and 0.5 <= px < image.shape[1] - 0.5:
                        # Bilinear interpolation
                        y0 = int(np.floor(py))
                        y1 = min(y0 + 1, image.shape[0] - 1)  # Ensure y1 doesn't exceed bounds
                        x0 = int(np.floor(px))
                        x1 = min(x0 + 1, image.shape[1] - 1)  # Ensure x1 doesn't exceed bounds
                        
                        # Interpolation weights (fractional parts)
                        dy = py - y0
                        dx = px - x0
                        
                        # Bilinear interpolation
                        # If y0 == y1 or x0 == x1 (at boundaries), interpolation reduces correctly
                        val = (
                            image[y0, x0] * (1 - dx) * (1 - dy) +
                            image[y0, x1] * dx * (1 - dy) +
                            image[y1, x0] * (1 - dx) * dy +
                            image[y1, x1] * dx * dy
                        )
                        profile.append(val)
                    else:
                        break

                if len(profile) == profile_length:
                    esf_samples.append(np.array(profile))

        if len(esf_samples) < 5:
            return OpticsTestResult(
                score=0.5,
                violations=["Insufficient ESF samples extracted"],
                diagnostic_data={},
            )

        # Analyze ESF samples
        violations = []
        score = 1.0

        # Check for ringing (oscillations in ESF)
        # AI upsampling often produces symmetric ringing (negative lobes on both sides)
        ringing_scores = []
        symmetric_ringing_count = 0

        for esf in esf_samples:
            # Compute LSF using smoothed ESF and central difference (normalized)
            lsf = self._compute_lsf(esf, normalize=True)

            if len(lsf) == 0:
                continue

            # Find peak location
            peak_idx = np.argmax(np.abs(lsf))

            # Check for symmetric ringing: negative lobes on both sides of peak
            # This is a strong indicator of synthetic kernel artifacts
            left_side = lsf[:peak_idx] if peak_idx > 0 else np.array([])
            right_side = lsf[peak_idx + 1 :] if peak_idx < len(lsf) - 1 else np.array([])

            has_left_negative = (
                len(left_side) > 0 and np.any(left_side < -0.05)
            )  # Relative to normalized peak
            has_right_negative = (
                len(right_side) > 0 and np.any(right_side < -0.05)
            )

            # REFINEMENT: Measure symmetry ratio to distinguish AI (symmetric) from ISP (asymmetric)
            # AI ringing is perfectly symmetric on both sides; ISP sharpening is usually asymmetric
            is_symmetric_ringing = False
            if len(left_side) > 0 and len(right_side) > 0:
                # Compute asymmetry: difference between left and right side magnitudes
                left_magnitude = np.mean(np.abs(left_side)) if len(left_side) > 0 else 0.0
                right_magnitude = np.mean(np.abs(right_side)) if len(right_side) > 0 else 0.0
                total_magnitude = left_magnitude + right_magnitude
                
                if total_magnitude > 1e-6:
                    asymmetry_ratio = abs(left_magnitude - right_magnitude) / total_magnitude
                    # Low asymmetry (< 0.3) = symmetric (AI), high asymmetry (> 0.5) = asymmetric (ISP)
                    # Only count as symmetric ringing if BOTH sides have negative lobes AND low asymmetry
                    if has_left_negative and has_right_negative and asymmetry_ratio < 0.3:
                        is_symmetric_ringing = True
            
            # Count symmetric ringing (both sides negative AND symmetric)
            if is_symmetric_ringing:
                symmetric_ringing_count += 1

            # Also check for general oscillations (alternating signs)
            sign_changes = np.sum(np.diff(np.sign(lsf)) != 0)
            ringing_ratio = sign_changes / len(lsf)
            ringing_scores.append(ringing_ratio)

        avg_ringing = np.mean(ringing_scores) if ringing_scores else 0.0
        symmetric_ratio = (
            symmetric_ringing_count / len(esf_samples) if esf_samples else 0.0
        )

        # Penalize symmetric ringing more heavily (strong AI indicator)
        if symmetric_ratio > 0.2:  # More than 20% of edges show symmetric ringing
            violations.append(
                f"Symmetric ringing detected (AI artifact indicator: {symmetric_ratio:.1%})"
            )
            score *= max(0.2, 1.0 - symmetric_ratio * 2)

        # Also check general oscillation ratio
        if avg_ringing > 0.3:  # More than 30% sign changes indicates ringing
            violations.append(
                f"High-frequency ringing detected (oscillation ratio: {avg_ringing:.2f})"
            )
            score *= max(0.3, 1.0 - (avg_ringing - 0.2) * 2)

        # REPLACED: Negative lobes threshold → Normalized Cross-Correlation (NCC) metric
        # Real sharpening halos (ISP) are often one-sided (asymmetric)
        # Diffusion ringing (AI) tends to be more even-symmetric
        # Measure symmetry using NCC to distinguish AI (symmetric) from ISP (asymmetric)
        # NCC formula: R = Σ(L_mirror ⋅ R_crop) / √(Σ L_mirror² ⋅ Σ R_crop²)
        # R ≈ 1.0: Perfect symmetry (Strong AI signal)
        # R < 0.7: Significant asymmetry (Likely ISP/Optical)
        asymmetry_scores = []
        symmetric_ringing_asymmetry = []
        symmetric_ringing_ncc = []  # Store NCC values for symmetric ringing cases
        
        # Method 3: Phase & Parity Symmetry Analysis
        # Identifies AI-generated "zero-phase" ringing artifacts
        phase_parity_sr_scores = []  # Symmetry Power Ratio (SR)
        phase_parity_psi_scores = []  # Phase Symmetry Index (PSI)
        phase_parity_phase_variance_scores = []  # Phase variance (for AI detection)
        phase_parity_ai_indicators = []  # Combined high-confidence AI indicators
        
        for esf in esf_samples:
            lsf = self._compute_lsf(esf, normalize=True)  # Normalized to peak=1.0

            if len(lsf) == 0:
                continue

            # Find peak location
            peak_idx = np.argmax(np.abs(lsf))
            
            # Split LSF into left and right sides relative to peak
            left_side = lsf[:peak_idx] if peak_idx > 0 else np.array([])
            right_side = lsf[peak_idx + 1:] if peak_idx < len(lsf) - 1 else np.array([])
            
            if len(left_side) == 0 or len(right_side) == 0:
                continue
            
            # Mirror one side to compare symmetry using Normalized Cross-Correlation (NCC)
            # For even-symmetric (AI diffusion), left and right should be similar
            # For odd-symmetric/asymmetric (ISP sharpening), they differ
            
            # Take minimum length to ensure fair comparison
            min_len = min(len(left_side), len(right_side))
            if min_len < 2:
                continue
            
            left_mirror = left_side[-min_len:][::-1]  # Reverse left side (mirror it)
            right_crop = right_side[:min_len]
            
            # Compute Normalized Cross-Correlation (NCC) between mirrored left and right
            # NCC formula: R = Σ(L_mirror ⋅ R_crop) / √(Σ L_mirror² ⋅ Σ R_crop²)
            # R ≈ 1.0: Perfect symmetry (Strong AI signal)
            # R < 0.7: Significant asymmetry (Likely ISP/Optical)
            
            # Compute numerator: dot product
            numerator = np.sum(left_mirror * right_crop)
            
            # Compute denominator: sqrt of product of sums of squares
            left_squared_sum = np.sum(left_mirror ** 2)
            right_squared_sum = np.sum(right_crop ** 2)
            denominator = np.sqrt(left_squared_sum * right_squared_sum)
            
            # Compute NCC coefficient
            if denominator > 1e-10:  # Avoid division by zero
                ncc_coefficient = numerator / denominator
                # Clamp to [-1, 1] range (theoretical bounds for correlation)
                ncc_coefficient = max(-1.0, min(1.0, ncc_coefficient))
            else:
                # If denominator is too small, assume no correlation
                ncc_coefficient = 0.0
            
            # Convert NCC to asymmetry metric for consistency with existing code
            # High NCC (≈1.0) = symmetric = low asymmetry
            # Low NCC (<0.7) = asymmetric = high asymmetry
            # Invert: asymmetry = 1.0 - NCC (so high NCC → low asymmetry)
            relative_asymmetry = 1.0 - ncc_coefficient
            asymmetry_scores.append(relative_asymmetry)
            
            # Also check if there are negative lobes on both sides (symmetric ringing indicator)
            # This is a stronger AI signature when combined with high NCC (symmetric)
            has_left_negative = np.any(left_side < -0.05)
            has_right_negative = np.any(right_side < -0.05)
            
            if has_left_negative and has_right_negative:
                # Both sides have negative lobes - check if symmetric using NCC
                # DYNAMIC NCC THRESHOLD: Require higher NCC (>0.9) for high-confidence AI flag
                # if image has high high-frequency energy (professional sharpening can have lower NCC)
                # For images with low high-frequency energy, use lower threshold (0.7)
                dynamic_ncc_threshold = 0.9 if high_freq_energy > 0.15 else 0.7
                
                if ncc_coefficient >= dynamic_ncc_threshold:
                    # Store both the asymmetry (inverted from NCC) and NCC for symmetric ringing detection
                    symmetric_ringing_asymmetry.append(relative_asymmetry)
                    symmetric_ringing_ncc.append(ncc_coefficient)
            
            # Method 3: Phase & Parity Symmetry Analysis
            # Extract LSF window centered on peak for phase/parity analysis
            # Use a window around the peak (extend equally on both sides)
            window_radius = min(peak_idx, len(lsf) - peak_idx - 1, 20)  # Limit window size
            if window_radius >= 2:
                window_start = max(0, peak_idx - window_radius)
                window_end = min(len(lsf), peak_idx + window_radius + 1)
                lsf_window = lsf[window_start:window_end]
                
                # Center the window on peak (shift peak to center)
                peak_in_window = peak_idx - window_start
                # Create centered window: equal samples on both sides of peak
                min_side = min(peak_in_window, len(lsf_window) - peak_in_window - 1)
                if min_side >= 2:
                    centered_window = lsf_window[peak_in_window - min_side:peak_in_window + min_side + 1]
                    
                    # Calculate phase and parity symmetry (now includes phase variance)
                    sr, psi, phase_var = self._calculate_phase_parity_symmetry(centered_window)
                    phase_parity_sr_scores.append(sr)
                    phase_parity_psi_scores.append(psi)
                    phase_parity_phase_variance = phase_var  # Store for later use
                    
                    # High-confidence AI indicator: SR > 0.85 AND PSI > 0.85 AND negative lobes
                    # This identifies zero-phase symmetric ringing (strong AI signature)
                    # NEW: Also check phase variance - True AI has low phase variance
                    if sr > 0.85 and psi > 0.85 and has_left_negative and has_right_negative:
                        # Check phase variance: True AI has low variance (near zero phase)
                        # Professional sharpening has higher variance due to sensor noise
                        if phase_var < 0.1:  # Low phase variance = AI
                            phase_parity_ai_indicators.append(True)
                        else:
                            phase_parity_ai_indicators.append(False)  # Higher variance = professional sharpening
                    else:
                        phase_parity_ai_indicators.append(False)

        avg_asymmetry = np.mean(asymmetry_scores) if asymmetry_scores else 0.0
        avg_symmetric_ringing_asymmetry = (
            np.mean(symmetric_ringing_asymmetry) if symmetric_ringing_asymmetry else 1.0
        )
        
        # Forensic logic using NCC:
        # - High NCC (≥0.7) with negative lobes on both sides → AI diffusion ringing (symmetric)
        # - Low NCC (<0.7) → Real ISP sharpening (asymmetric, less suspicious)
        # - NCC 0.7-0.85 → Moderate symmetry (ambiguous, less penalty)
        #
        # Note: relative_asymmetry = 1.0 - NCC, so:
        # - High NCC (0.7-1.0) → Low asymmetry (0.0-0.3) → AI (symmetric)
        # - Low NCC (<0.7) → High asymmetry (>0.3) → ISP (asymmetric)
        
        # Penalize symmetric ringing (high NCC + negative lobes on both sides)
        # Use dynamic threshold based on high-frequency energy
        dynamic_ncc_threshold = 0.9 if high_freq_energy > 0.15 else 0.7
        ncc_asymmetry_threshold = 1.0 - dynamic_ncc_threshold  # Convert to asymmetry threshold
        
        if len(symmetric_ringing_asymmetry) > 0 and avg_symmetric_ringing_asymmetry < ncc_asymmetry_threshold:
            symmetric_count = len(symmetric_ringing_asymmetry)
            symmetric_ratio = symmetric_count / len(esf_samples) if esf_samples else 0.0
            
            # Calculate average NCC for symmetric ringing cases (stored directly)
            avg_symmetric_ncc = np.mean(symmetric_ringing_ncc) if symmetric_ringing_ncc else dynamic_ncc_threshold
            
            # CA INTEGRATION: If CA is radial, reduce AI penalty by 50%
            # Real lenses can have symmetric PSFs, but they almost never have tangential CA
            ca_penalty_reduction = 0.5 if (ca_is_radial is True) else 1.0
            
            violations.append(
                f"Even-symmetric ringing detected (NCC: {avg_symmetric_ncc:.3f}, "
                f"{symmetric_ratio:.1%} of edges) - suggests AI diffusion artifacts"
            )
            # Strong penalty for symmetric ringing (AI indicator)
            # Penalty scales with NCC: higher NCC (more symmetric) = stronger penalty
            # NCC 0.7 → penalty factor 0.44, NCC 1.0 → penalty factor 0.2
            penalty_factor = max(0.2, 1.0 - avg_symmetric_ncc * 0.8)
            # Apply CA penalty reduction if CA is radial
            penalty_factor = 1.0 - (1.0 - penalty_factor) * ca_penalty_reduction
            score *= penalty_factor
        
        # Method 3: Phase & Parity Symmetry - High-confidence AI detection
        # If SR > 0.85 AND PSI > 0.85 AND negative lobes, apply strong penalty
        # This catches zero-phase symmetric ringing even if Method 1/2 are borderline
        avg_sr = np.mean(phase_parity_sr_scores) if phase_parity_sr_scores else 0.0
        avg_psi = np.mean(phase_parity_psi_scores) if phase_parity_psi_scores else 0.0
        phase_parity_ai_count = sum(phase_parity_ai_indicators) if phase_parity_ai_indicators else 0
        phase_parity_ai_ratio = phase_parity_ai_count / len(esf_samples) if esf_samples else 0.0
        
        # Log phase/parity metrics for debugging
        if phase_parity_sr_scores:
            logger.debug(
                f"Phase & Parity Analysis: avg_SR={avg_sr:.3f}, avg_PSI={avg_psi:.3f}, "
                f"AI_indicators={phase_parity_ai_count}/{len(esf_samples)} ({phase_parity_ai_ratio:.1%})"
            )
        
        if phase_parity_ai_count > 0:
            # High-confidence AI indicator: zero-phase symmetric ringing
            # UPDATED: Only use "Zero-phase synthetic artifact" text when PSI > 0.9
            if avg_psi > 0.9:
                violation_text = (
                    f"Zero-phase synthetic artifact detected (SR: {avg_sr:.3f}, PSI: {avg_psi:.3f}, "
                    f"{phase_parity_ai_ratio:.1%} of edges) - strong AI diffusion artifact indicator"
                )
            else:
                violation_text = (
                    f"Zero-phase symmetric ringing detected (SR: {avg_sr:.3f}, PSI: {avg_psi:.3f}, "
                    f"{phase_parity_ai_ratio:.1%} of edges) - strong AI diffusion artifact indicator"
                )
            violations.append(violation_text)
            
            # Very strong penalty for zero-phase symmetric ringing (physical optics rarely produce this)
            # Combined metric: (SR + PSI) / 2, penalize more when both are high
            combined_metric = (avg_sr + avg_psi) / 2.0
            penalty_factor = max(0.15, 1.0 - combined_metric * 0.85)  # Stronger penalty than NCC alone
            
            # CA INTEGRATION: If CA is radial, reduce AI penalty by 50%
            ca_penalty_reduction = 0.5 if (ca_is_radial is True) else 1.0
            penalty_factor = 1.0 - (1.0 - penalty_factor) * ca_penalty_reduction
            
            score *= penalty_factor
            logger.debug(
                f"Applied zero-phase ringing penalty: combined_metric={combined_metric:.3f}, "
                f"penalty_factor={penalty_factor:.3f}, ca_penalty_reduction={ca_penalty_reduction:.2f}"
            )
        
        # Note: High asymmetry (> 0.5) is actually less suspicious (real ISP sharpening)
        # We don't penalize high asymmetry, as it's consistent with real camera processing

        # Check PSF width consistency
        psf_widths = []
        for esf in esf_samples:
            lsf = self._compute_lsf(esf, normalize=True)  # Normalized for consistent FWHM
            # Find FWHM (full width at half maximum)
            if len(lsf) > 0:
                max_val = np.max(np.abs(lsf))  # Should be 1.0 if normalized
                half_max = max_val / 2
                above_half = np.abs(lsf) > half_max
                if np.sum(above_half) > 0:
                    width = np.sum(above_half)
                    psf_widths.append(width)

        if len(psf_widths) > 5:
            width_std = np.std(psf_widths)
            width_mean = np.mean(psf_widths)
            if width_mean > 0:
                cv_width = width_std / width_mean  # Coefficient of variation
                if cv_width > 0.5:  # High variation indicates inconsistent PSF
                    violations.append(
                        f"Inconsistent PSF width (CV: {cv_width:.2f})"
                    )
                    score *= max(0.5, 1.0 - (cv_width - 0.3) * 0.5)

        score = max(0.0, min(1.0, score))

        if not violations:
            violations.append("Passes PSF consistency test")

        # Store example ESF/LSF for visualization
        example_esf = esf_samples[0] if esf_samples else None
        example_lsf = (
            self._compute_lsf(example_esf) if example_esf is not None else None
        )

        return OpticsTestResult(
            score=score,
            violations=violations,
            diagnostic_data={
                "esf_samples": esf_samples[:5],  # Store first 5 for visualization
                "lsf_samples": [self._compute_lsf(esf) for esf in esf_samples[:5]],
                "example_esf": example_esf,
                "example_lsf": example_lsf,
                "ringing_scores": ringing_scores,
                "psf_widths": psf_widths,
                # Method 2: NCC metrics
                "avg_ncc": 1.0 - avg_asymmetry if asymmetry_scores else 0.0,  # Convert back to NCC
                "avg_symmetric_ncc": np.mean(symmetric_ringing_ncc) if symmetric_ringing_ncc else 0.0,
                # Method 3: Phase & Parity metrics
                "avg_symmetry_ratio": avg_sr if phase_parity_sr_scores else 0.0,
                "avg_phase_symmetry_index": avg_psi if phase_parity_psi_scores else 0.0,
                "phase_parity_ai_ratio": phase_parity_ai_ratio,
            },
        )


# ============================================================================
# TEST 3: DEPTH-OF-FIELD CONSISTENCY TEST
# ============================================================================
# Estimates local blur radius at edges (content-independent) and checks
# spatial smoothness of blur variation. Real DOF follows the thin lens equation.
@pydantic_dataclass
class DepthOfFieldConsistencyTest:
    """Test 3: Depth-of-field consistency test.

    Estimates local blur radius at edges (content-independent) and checks
    spatial smoothness of blur variation.

    PHYSICS: Real DOF follows the thin lens equation (1/f = 1/d_o + 1/d_i).
    Blur radius varies CONTINUOUSLY with object distance. The transition from
    "in-focus" to "out-of-focus" is smooth and gradual.

    FORENSIC SIGNATURE: AI models (especially "segmentation + blur" pipelines)
    often apply uniform Gaussian blur to background regions, creating discrete
    "cliffs" at subject boundaries. A real lens cannot have sudden jumps in
    blur radius unless there's a physical gap of kilometers between objects.

    This test is one of the strongest forensic indicators for AI-generated images.
    """

    blur_window_size: int = Field(default=21, gt=0, ge=5)
    
    # Optimization constants
    DOF_VIOLATION_RATIO_THRESHOLD = 0.1  # Maximum acceptable violation ratio (10% of triplets)
    EDGE_DENSITY_THRESHOLD = 0.1  # Minimum edge density in window to estimate blur
    
    # DOF consistency test thresholds
    MAX_GRADIENT_THRESHOLD = 2.0  # Large jumps indicate discrete blur regions (AI artifact)
    MEAN_GRADIENT_THRESHOLD = 0.5  # Non-smooth blur variation threshold
    THIN_LENS_VIOLATION_PENALTY = 0.4  # Strong penalty for physics violations
    
    # Conditional test thresholds
    TEXTURE_THRESHOLD = 0.15  # Minimum texture variance for usable blur evidence
    DEFOCUS_GRADIENT_THRESHOLD = 0.3  # Minimum defocus gradient magnitude for usable blur evidence
    MIN_BLUR_EVIDENCE_RATIO = 0.1  # Minimum ratio of image with blur evidence to run test

    def _check_thin_lens_consistency(
        self, blur_map: np.ndarray, valid_mask: np.ndarray
    ) -> List[str]:
        """
        Check if blur radii follow thin lens equation: R ∝ |D - D_focus|.

        Detects impossible configurations like:
        - Two disconnected sharp regions (two focus planes)
        - Non-monotonic blur (sharp near, sharp far, blurry mid)
        - Blur that doesn't correlate with estimated distance

        Args:
            blur_map: 2D array of blur radius estimates
            valid_mask: Boolean mask of valid blur estimates

        Returns:
            List of violation messages
        """
        violations = []
        valid_blur = blur_map[valid_mask]

        if len(valid_blur) < 10:
            return []  # Need sufficient data

        # Find focus plane (minimum blur = in-focus region)
        min_blur = np.nanmin(valid_blur)
        focus_threshold = min_blur * 1.2  # Within 20% of minimum = "sharp"

        # Identify sharp regions (in-focus)
        sharp_mask = (blur_map < focus_threshold) & valid_mask

        # Test 1: Check for multiple disconnected sharp regions
        # A single lens can only have ONE focus plane
        from scipy.ndimage import label

        labeled_sharp, num_sharp_regions = label(sharp_mask)

        if num_sharp_regions > 1:
            # Check if sharp regions are spatially separated
            # (not just noise or measurement error)
            region_sizes = [
                np.sum(labeled_sharp == i) for i in range(1, num_sharp_regions + 1)
            ]
            significant_regions = [
                size for size in region_sizes if size >= 3
            ]  # At least 3 grid points

            if len(significant_regions) > 1:
                violations.append(
                    f"Multiple disconnected sharp regions detected ({len(significant_regions)}) - "
                    f"impossible for single lens (requires multiple focus planes)"
                )

        # Test 2: Check monotonicity of blur away from focus plane
        # Blur should increase as we move away from the focus plane
        # Estimate distance from focus: assume blur ∝ distance (inverse relationship)
        # For objects at distances D1, D2, D3 from focus:
        # If R1 < R2 < R3, then |D1 - D_focus| < |D2 - D_focus| < |D3 - D_focus|
        # This means blur should be monotonic in space (increasing from focus)

        # Find focus region center
        sharp_coords = np.column_stack(np.where(sharp_mask))
        if len(sharp_coords) > 0:
            focus_center_y = np.mean(sharp_coords[:, 0])
            focus_center_x = np.mean(sharp_coords[:, 1])

            # Compute distance from focus center for each valid point
            y_coords, x_coords = np.ogrid[:blur_map.shape[0], :blur_map.shape[1]]
            distances_from_focus = np.sqrt(
                (y_coords - focus_center_y) ** 2 + (x_coords - focus_center_x) ** 2
            )

            # Check correlation: blur should increase with distance from focus
            valid_distances = distances_from_focus[valid_mask]
            valid_blur_for_corr = blur_map[valid_mask]

            if len(valid_distances) > 5:
                # Compute correlation coefficient
                correlation = np.corrcoef(valid_distances, valid_blur_for_corr)[0, 1]

                # Real DOF: positive correlation (blur increases with distance)
                # AI/composite: may have negative or zero correlation
                if correlation < 0.3:  # Weak or negative correlation
                    violations.append(
                        f"Weak blur-distance correlation ({correlation:.2f}) - "
                        f"blur should increase with distance from focus plane"
                    )

        # Test 3: Check for impossible blur patterns
        # Example: Object A (near) blurry, Object B (mid) sharp, Object C (far) sharp
        # This violates R ∝ |D - D_focus| (cannot have two disconnected sharp regions)

        # Sort blur values and check for non-monotonic patterns
        blur_sorted = np.sort(valid_blur)
        blur_diff = np.diff(blur_sorted)

        # Check for large "gaps" in blur distribution
        # Real DOF: continuous distribution
        # AI/composite: may have discrete clusters (sharp cluster + blurry cluster)
        if len(blur_diff) > 5:
            # Find large gaps (potential discrete clusters)
            gap_threshold = np.percentile(blur_diff, 90)  # Top 10% of gaps
            large_gaps = np.sum(blur_diff > gap_threshold * 3)

            if large_gaps > 2:  # Multiple large gaps suggest discrete clusters
                violations.append(
                    f"Discrete blur clusters detected ({large_gaps} large gaps) - "
                    f"suggests composite or AI-generated depth separation"
                )

        # Test 4: Spatial consistency check
        # For any three points A, B, C in space:
        # If A is near focus, B is far, C is near focus again
        # Then blur(A) ≈ blur(C) < blur(B)
        # If we find A blurry, B sharp, C blurry (non-monotonic), it's suspicious

        # Sample triplets and check for impossible patterns
        # This is computationally expensive, so sample deterministically
        sample_size = min(20, len(valid_blur))
        # Use evenly spaced samples for deterministic results
        sample_indices = np.linspace(0, len(valid_blur) - 1, sample_size, dtype=int)
        
        # Get coordinates for valid blur estimates
        valid_coords = np.column_stack(np.where(valid_mask))
        if len(valid_coords) < len(sample_indices):
            return violations  # Not enough data
        
        sample_blur = valid_blur[sample_indices]
        sample_coords = valid_coords[sample_indices]

        # OPTIMIZED: Vectorized triplet checking using broadcasting
        # This reduces O(n³) nested loops to vectorized operations
        n_samples = len(sample_coords)
        
        # Precompute pairwise distances (symmetric matrix)
        # Use vectorized operations to compute all distances at once
        coords_array = np.array(sample_coords)
        # Compute distance matrix using broadcasting
        # Shape: (n_samples, n_samples, 2) -> (n_samples, n_samples)
        diff = coords_array[:, None, :] - coords_array[None, :, :]  # (n, n, 2)
        distances = np.linalg.norm(diff, axis=2)  # (n, n)
        
        # Precompute blur differences (symmetric matrix)
        blur_array = np.array(sample_blur)
        blur_diffs = np.abs(blur_array[:, None] - blur_array[None, :])  # (n, n)
        
        # VECTORIZED: Check triplets using broadcasting
        # Optimized to vectorize inner loops while maintaining early termination
        max_possible_triplets = n_samples * (n_samples - 1) * (n_samples - 2) // 6
        threshold_ratio = max_possible_triplets * self.DOF_VIOLATION_RATIO_THRESHOLD
        
        impossible_patterns = 0
        
        # Vectorized approach: check triplets in batches
        # For efficiency, we'll still use early termination but vectorize the inner checks
        max_dist = np.max(distances)
        dist_threshold = max_dist * 0.5
        
        # Iterate over i, but vectorize j and k checks
        for i in range(n_samples - 2):
            # Get all j > i
            j_candidates = np.arange(i + 1, n_samples - 1)
            
            # Filter j candidates: only check if dist_ij is reasonable (vectorized)
            dist_ij_vec = distances[i, j_candidates]
            valid_j_mask = dist_ij_vec <= dist_threshold
            j_valid = j_candidates[valid_j_mask]
            
            if len(j_valid) == 0:
                continue
            
            # For each valid j, vectorize k checks
            for j_idx, j in enumerate(j_valid):
                k_candidates = np.arange(j + 1, n_samples)
                
                if len(k_candidates) == 0:
                    continue
                
                # Vectorized check for all k candidates
                dist_ik_vec = distances[i, k_candidates]
                dist_jk_vec = distances[j, k_candidates]
                dist_ij = dist_ij_vec[valid_j_mask][j_idx]  # Get dist_ij for this j
                
                # Early check: only proceed if i and k are close (vectorized)
                close_mask = (dist_ik_vec < dist_ij) & (dist_ik_vec < dist_jk_vec)
                k_valid = k_candidates[close_mask]
                
                if len(k_valid) == 0:
                    continue
                
                # Vectorized blur difference checks for all valid k
                blur_diff_ij = blur_diffs[i, j]
                blur_diff_ik_vec = blur_diffs[i, k_valid]
                blur_diff_jk_vec = blur_diffs[j, k_valid]
                
                # Vectorized impossible pattern check
                # Condition: blur_diff_ik > max(blur_diff_ij, blur_diff_jk) * 1.5
                # AND min(blur_diff_ij, blur_diff_jk) < blur_diff_ik * 0.5
                max_blur_ij_jk = np.maximum(blur_diff_ij, blur_diff_jk_vec)
                min_blur_ij_jk = np.minimum(blur_diff_ij, blur_diff_jk_vec)
                
                violation_mask = (
                    (blur_diff_ik_vec > max_blur_ij_jk * 1.5) &
                    (min_blur_ij_jk < blur_diff_ik_vec * 0.5)
                )
                
                impossible_patterns += np.sum(violation_mask)
                
                # Early termination if we've found enough violations
                if impossible_patterns > threshold_ratio:
                    break
            
            if impossible_patterns > threshold_ratio:
                break

        violation_threshold = max_possible_triplets * self.DOF_VIOLATION_RATIO_THRESHOLD
        if impossible_patterns > violation_threshold:
            violations.append(
                f"Non-monotonic blur patterns detected ({impossible_patterns} violations) - "
                f"inconsistent with thin lens equation R ∝ |D - D_focus|"
            )

        return violations

    def _check_blur_evidence(self, image: np.ndarray, gradient_cache: Optional[dict] = None) -> Tuple[bool, float]:
        """
        Check if there's usable blur evidence in the image.
        
        Conditions for usable blur evidence:
        1. Presence of textured background (high local variance)
        2. Presence of strong defocus gradients (spatial variation in blur)
        
        Args:
            image: Grayscale image (H, W), float32 [0, 1]
            gradient_cache: Optional dict for caching gradients (keyed by image hash)
            
        Returns:
            Tuple of (has_evidence, evidence_ratio)
            - has_evidence: True if sufficient blur evidence found
            - evidence_ratio: Ratio of image with blur evidence (0.0-1.0)
        """
        h, w = image.shape
        
        # Method 1: Check for textured background
        # OPTIMIZED: Use vectorized sliding window variance computation
        # Variance = E[X²] - E[X]², computed using convolution
        window_size = 15
        half_window = window_size // 2
        from scipy.ndimage import gaussian_filter
        
        # Compute local mean and mean of squares using Gaussian filter (approximates box filter)
        local_mean = gaussian_filter(image, sigma=half_window)
        local_mean_sq = gaussian_filter(image**2, sigma=half_window)
        texture_map = local_mean_sq - local_mean**2  # Variance = E[X²] - E[X]²
        
        # Sample every 5 pixels for efficiency (texture_map is already computed for all pixels)
        # We'll use the full texture_map but sample for threshold comparison
        
        # Method 2: Check for defocus gradients
        # OPTIMIZED: Use cached gradient if available
        image_hash = hash((image.shape, image.dtype, tuple(image.flat[:100]))) if gradient_cache is not None else None
        
        if gradient_cache is not None and image_hash in gradient_cache:
            logger.debug("  Using cached gradient for blur evidence check")
            gy, gx = gradient_cache[image_hash]
        else:
            gy, gx = np.gradient(image)
            if gradient_cache is not None and image_hash is not None:
                gradient_cache[image_hash] = (gy, gx)
        
        gradient_mag = np.sqrt(gx**2 + gy**2)
        
        # Defocus gradients: high spatial variation in gradient magnitude
        # (blurry regions have lower gradients, sharp regions have higher)
        # Compute second-order gradient (gradient of gradient magnitude)
        gy_grad, gx_grad = np.gradient(gradient_mag)
        defocus_gradient_mag = np.sqrt(gy_grad**2 + gx_grad**2)
        
        # Threshold: texture variance > threshold OR defocus gradient > threshold
        texture_threshold = self.TEXTURE_THRESHOLD
        defocus_threshold = self.DEFOCUS_GRADIENT_THRESHOLD
        
        # Create evidence mask
        texture_evidence = texture_map > texture_threshold
        defocus_evidence = defocus_gradient_mag > defocus_threshold
        evidence_mask = texture_evidence | defocus_evidence
        
        # Compute evidence ratio
        evidence_ratio = np.sum(evidence_mask) / evidence_mask.size if evidence_mask.size > 0 else 0.0
        
        # Has evidence if ratio exceeds minimum threshold
        has_evidence = evidence_ratio >= self.MIN_BLUR_EVIDENCE_RATIO
        
        logger.debug(
            f"Blur evidence check: texture_ratio={np.sum(texture_evidence)/texture_evidence.size:.3f}, "
            f"defocus_ratio={np.sum(defocus_evidence)/defocus_evidence.size:.3f}, "
            f"total_evidence_ratio={evidence_ratio:.3f}, has_evidence={has_evidence}"
        )
        
        return has_evidence, evidence_ratio

    def estimate_local_blur(self, image: np.ndarray, y: int, x: int) -> float:
        """
        Estimate local blur radius using frequency attenuation.
        
        REPLACED: Previous gradient-based method was fragile.
        NEW METHOD: Ratio of Laplacian energy to gradient energy across windows.
        
        Physics: Blur attenuates high frequencies (Laplacian) more than low frequencies (gradient).
        - Sharp regions: High Laplacian energy relative to gradient energy
        - Blurry regions: Low Laplacian energy relative to gradient energy
        
        This is more robust than gradient-based methods because it measures frequency
        content directly rather than relying on edge detection.
        
        Args:
            image: Grayscale image (H, W), float32 [0, 1]
            y, x: Coordinates

        Returns:
            Estimated blur radius in pixels, or NaN if insufficient data
        """
        h, w = image.shape
        half_window = self.blur_window_size // 2

        y_min = max(0, y - half_window)
        y_max = min(h, y + half_window + 1)
        x_min = max(0, x - half_window)
        x_max = min(w, x + half_window + 1)

        patch = image[y_min:y_max, x_min:x_max]

        if patch.size == 0 or patch.size < 9:  # Need at least 3×3
            return np.nan

        # Convert to uint8 for OpenCV operations
        patch_uint8 = (patch * 255).astype(np.uint8)
        
        # Compute gradient energy (low-frequency content)
        # Gradient captures first-order derivatives (low-frequency edges)
        gy, gx = np.gradient(patch)
        gradient_mag = np.sqrt(gx**2 + gy**2)
        gradient_energy = np.sum(gradient_mag**2)  # Sum of squared gradients
        
        # Compute Laplacian energy (high-frequency content)
        # Laplacian captures second-order derivatives (high-frequency detail)
        laplacian = cv2.Laplacian(patch_uint8, cv2.CV_64F)
        laplacian_energy = np.sum(laplacian**2)  # Sum of squared Laplacian
        
        # Frequency attenuation ratio: Laplacian / Gradient
        # Sharp: High ratio (Laplacian >> Gradient) → ratio > 1.0
        # Blurry: Low ratio (Laplacian << Gradient) → ratio < 0.5
        if gradient_energy < 1e-6:
            return np.nan  # No gradient (flat region)
        
        frequency_ratio = laplacian_energy / gradient_energy
        
        # Convert frequency ratio to blur radius
        # Sharp: ratio ≈ 2.0-5.0 → blur ≈ 0-1 pixels
        # Medium: ratio ≈ 0.5-2.0 → blur ≈ 1-3 pixels
        # Blurry: ratio ≈ 0.1-0.5 → blur ≈ 3-8 pixels
        # Very blurry: ratio < 0.1 → blur > 8 pixels
        
        # Mapping: blur_radius = f(frequency_ratio)
        # Use inverse relationship: lower ratio = higher blur
        if frequency_ratio > 2.0:
            blur_radius = 0.5  # Very sharp
        elif frequency_ratio > 1.0:
            # Map frequency_ratio from (1.0, 2.0] to blur_radius [2.0, 1.0]
            # When ratio = 2.0: blur = 1.0 + (2.0 - 2.0) * 1.0 = 1.0
            # When ratio = 1.0: blur = 1.0 + (2.0 - 1.0) * 1.0 = 2.0
            blur_radius = 1.0 + (2.0 - frequency_ratio) * 1.0  # Sharp to medium
        elif frequency_ratio > 0.5:
            blur_radius = 3.0 + (1.0 - frequency_ratio) * 4.0  # Medium to blurry
        elif frequency_ratio > 0.1:
            blur_radius = 7.0 + (0.5 - frequency_ratio) * 6.0  # Blurry to very blurry
        else:
            blur_radius = 10.0  # Very blurry (cap)
        
        return min(blur_radius, 10.0)  # Cap at reasonable value

    def test(self, image: np.ndarray) -> OpticsTestResult:
        """
        Test depth-of-field consistency (CONDITIONAL TEST WITH SOFT SCORING).
        
        First checks for usable blur evidence:
        - Presence of textured background
        - Presence of strong defocus gradients
        
        Uses soft scoring based on evidence ratio:
        - Low evidence: Partial credit (neutral score weighted by evidence)
        - Moderate evidence: DOF test score weighted by evidence
        - High evidence: Full DOF test score
        
        This prevents false positives for deep-focus/clean-background images while
        still providing useful information when some blur evidence exists.

        Args:
            image: Grayscale image (H, W), float32 [0, 1]

        Returns:
            OpticsTestResult with score and violations
        """
        logger.debug("Running depth-of-field consistency test (conditional with soft scoring)")

        # Check for usable blur evidence first
        # Note: gradient_cache is not available at test level, but could be passed from detector
        has_evidence, evidence_ratio = self._check_blur_evidence(image, gradient_cache=None)
        
        # SOFT SCORING: Use evidence_ratio to weight the test
        # evidence_ratio ranges from 0.0 (no evidence) to 1.0 (strong evidence)
        
        # Always compute blur map for diagnostics, even with low evidence
        # This provides useful visualization even when test is soft-scored
        h, w = image.shape
        
        # Sample points on a grid
        grid_spacing = max(10, min(h, w) // 20)
        y_coords = np.arange(0, h, grid_spacing)
        x_coords = np.arange(0, w, grid_spacing)
        total_grid_points = len(y_coords) * len(x_coords)
        logger.debug(f"DOF test: Sampling {len(y_coords)}x{len(x_coords)}={total_grid_points} grid points")

        blur_map = np.zeros((len(y_coords), len(x_coords)))
        blur_map[:] = np.nan  # Initialize with NaN

        # Estimate blur at all grid points using frequency attenuation
        logger.debug("  Estimating blur using frequency attenuation (Laplacian/Gradient energy ratio)...")
        
        processed_count = 0
        # OPTIMIZED: Reduce logging frequency in tight loop
        log_interval = max(10, len(y_coords) // 5)  # Log 5 times total
        for i, y in enumerate(y_coords):
            if i % log_interval == 0 and i > 0:
                logger.debug(f"  Processing row {i}/{len(y_coords)} ({processed_count} blur estimates so far)...")
            for j, x in enumerate(x_coords):
                y_int = int(y)
                x_int = int(x)
                
                # Estimate blur using frequency attenuation (works on any patch)
                blur_est = self.estimate_local_blur(image, y_int, x_int)
                if not np.isnan(blur_est):
                    blur_map[i, j] = blur_est
                    processed_count += 1
        
        logger.debug(
            f"  ✓ Blur map computed: {processed_count}/{total_grid_points} valid estimates. "
            f"Blur range: [{np.nanmin(blur_map):.3f}, {np.nanmax(blur_map):.3f}], "
            f"unique values: {len(np.unique(blur_map[~np.isnan(blur_map)])) if processed_count > 0 else 0}"
        )

        if not has_evidence:
            # Low evidence: Give partial credit based on evidence ratio
            # This prevents false positives while still acknowledging limited evidence
            logger.debug(
                f"Low blur evidence (ratio: {evidence_ratio:.3f} < {self.MIN_BLUR_EVIDENCE_RATIO}) - "
                f"using soft scoring with evidence weight"
            )
            
            # Soft score: Interpolate between neutral (1.0) and a conservative penalty
            # As evidence_ratio approaches 0, score approaches 1.0 (neutral)
            # As evidence_ratio approaches MIN_BLUR_EVIDENCE_RATIO, score approaches 0.9 (slight penalty)
            # This gives partial credit for minimal evidence
            evidence_weight = evidence_ratio / self.MIN_BLUR_EVIDENCE_RATIO if self.MIN_BLUR_EVIDENCE_RATIO > 0 else 0.0
            evidence_weight = min(1.0, max(0.0, evidence_weight))  # Clamp to [0, 1]
            
            # Soft score: 1.0 (neutral) when evidence_weight=0, 0.9 (slight penalty) when evidence_weight=1
            soft_score = 1.0 - (0.1 * evidence_weight)
            
            return OpticsTestResult(
                score=soft_score,
                violations=[
                    f"DOF test: Low blur evidence (ratio: {evidence_ratio:.3f}) - "
                    f"using soft scoring (score: {soft_score:.3f}) - "
                    f"this is normal for deep-focus or clean-background images"
                ],
                diagnostic_data={
                    "skipped": False,
                    "soft_scoring": True,
                    "evidence_ratio": evidence_ratio,
                    "evidence_weight": evidence_weight,
                    "reason": "low_blur_evidence_soft_scoring",
                    "blur_map": blur_map,  # Include blur map for diagnostics
                    "y_coords": y_coords,
                    "x_coords": x_coords,
                },
            )

        logger.debug(f"Blur evidence detected (ratio: {evidence_ratio:.3f}) - proceeding with DOF test (soft scoring)")

        # Blur map already computed above (before the has_evidence check)
        # Continue with DOF analysis using the computed blur_map

        # Filter out NaN values for analysis
        valid_blur = blur_map[~np.isnan(blur_map)]

        if len(valid_blur) < 5:
            return OpticsTestResult(
                score=0.5,
                violations=["Insufficient valid blur estimates for DOF analysis"],
                diagnostic_data={"blur_map": blur_map, "y_coords": y_coords, "x_coords": x_coords},
            )

        # Check spatial smoothness
        # Real DOF should vary smoothly, not in discrete jumps
        violations = []
        score = 1.0

        # Create mask for valid blur estimates
        valid_mask = ~np.isnan(blur_map)
        if np.sum(valid_mask) < 5:
            return OpticsTestResult(
                score=0.5,
                violations=["Insufficient valid blur estimates"],
                diagnostic_data={"blur_map": blur_map, "y_coords": y_coords, "x_coords": x_coords},
            )

        # Compute spatial gradient of blur map (only at valid points)
        # Fill NaN with median for gradient computation (preserves structure)
        blur_map_filled = blur_map.copy()
        if np.any(np.isnan(blur_map_filled)):
            median_blur = np.nanmedian(blur_map)
            blur_map_filled[np.isnan(blur_map_filled)] = median_blur

        blur_gy, blur_gx = np.gradient(blur_map_filled)
        blur_gradient_mag = np.sqrt(blur_gx**2 + blur_gy**2)
        
        # Mask out invalid regions for statistics
        blur_gradient_mag_valid = blur_gradient_mag[valid_mask]

        # DISCRETE JUMPS TEST: Strong forensic indicator
        # 
        # PHYSICS: Real DOF follows thin lens equation: 1/f = 1/d_o + 1/d_i
        # Blur radius varies CONTINUOUSLY with object distance. The transition from
        # "in-focus" to "out-of-focus" is smooth and gradual.
        #
        # FORENSIC SIGNATURE: AI models (especially "segmentation + blur" pipelines)
        # often apply uniform Gaussian blur to background regions, creating discrete
        # "cliffs" at subject boundaries. A real lens cannot have sudden jumps in
        # blur radius unless there's a physical gap of kilometers between objects.
        #
        max_gradient = np.max(blur_gradient_mag_valid) if len(blur_gradient_mag_valid) > 0 else 0.0
        mean_gradient = np.mean(blur_gradient_mag_valid) if len(blur_gradient_mag_valid) > 0 else 0.0

        if max_gradient > self.MAX_GRADIENT_THRESHOLD:
            violations.append(
                f"Discrete blur regions detected (max gradient: {max_gradient:.2f} pixels/grid) - "
                f"non-physical transition consistent with AI segmentation+blur pipeline"
            )
            # Strong penalty for discrete jumps (strong AI indicator)
            score *= max(0.3, 1.0 - (max_gradient - 1.5) * 0.15)

        # MEAN GRADIENT TEST: Overall smoothness check
        # Real DOF: mean gradient typically < 0.3 (smooth variation)
        # AI-generated: higher mean gradient from artificial boundaries
        if mean_gradient > self.MEAN_GRADIENT_THRESHOLD:
            violations.append(
                f"Non-smooth blur variation (mean gradient: {mean_gradient:.2f} pixels/grid) - "
                f"inconsistent with continuous DOF from thin lens equation"
            )
            score *= max(0.5, 1.0 - (mean_gradient - 0.3) * 0.4)

        # SEMANTIC BLUR PATTERN TEST: Detect uniform blur regions (AI signature)
        #
        # RISK OF FALSE POSITIVES: Real "Portrait Mode" or macro photos have high CV
        # (sharp foreground, blurry background) - this is EXPECTED in professional photography.
        #
        # THE REAL AI SIGNATURE: Low local entropy in blur regions.
        # - Natural blur: Contains subtle variations due to 3D depth of background
        # - AI blur: Often uses a single σ value for entire background (numerically identical)
        #
        # Detection: Check for large regions with identical blur values and low local entropy
        violations_semantic = []

        # Test 1: Check for numerically identical blur values (AI uses uniform σ)
        # Real optical blur varies even by 0.01% across a surface
        # AI blur is often numerically identical across many pixels
        blur_unique_ratio = len(np.unique(valid_blur)) / len(valid_blur) if len(valid_blur) > 0 else 1.0
        
        # If >80% of blur values are duplicates, suspicious
        if blur_unique_ratio < 0.2:
            violations_semantic.append(
                f"Uniform blur detected ({blur_unique_ratio:.1%} unique values) - "
                f"numerically identical values suggest AI uniform σ blur"
            )
            score *= max(0.3, blur_unique_ratio * 2)

        # Test 2: Local entropy of blur map
        # OPTIMIZED: Use vectorized operations instead of nested loops
        # Real blur: High entropy (subtle variations across space)
        # AI blur: Low entropy (uniform regions with identical values)
        from scipy.stats import entropy as scipy_entropy
        from scipy.ndimage import generic_filter

        # OPTIMIZED: Use generic_filter for vectorized local entropy computation
        # This is much faster than nested loops
        def compute_window_entropy(window_flat):
            """Compute entropy of a 3×3 window (flattened)."""
            window_flat = window_flat[~np.isnan(window_flat)]
            if len(window_flat) < 3:
                return np.nan
            
            # Compute histogram entropy
            hist, _ = np.histogram(
                window_flat, bins=min(10, len(window_flat)), density=True
            )
            hist = hist[hist > 0]  # Remove zeros
            if len(hist) > 0:
                return scipy_entropy(hist)
            return np.nan

        # Compute local entropy using vectorized filter
        # Only compute for valid pixels (where valid_mask is True)
        logger.debug("  Computing local entropy (vectorized)...")
        local_entropy_map = generic_filter(
            blur_map,
            compute_window_entropy,
            size=3,
            mode='constant',
            cval=np.nan
        )
        
        # Extract entropy values only for valid pixels
        local_entropy_values = local_entropy_map[valid_mask]
        local_entropy_values = local_entropy_values[~np.isnan(local_entropy_values)]
        logger.debug(f"  ✓ Local entropy computed for {len(local_entropy_values)} valid pixels")

        if len(local_entropy_values) > 5:
            mean_local_entropy = np.mean(local_entropy_values)
            # Low entropy (< 1.0) indicates uniform blur regions (AI signature)
            # Real blur typically has entropy > 1.5 (subtle variations)
            if mean_local_entropy < 1.0:
                violations_semantic.append(
                    f"Low blur entropy detected ({mean_local_entropy:.2f}) - "
                    f"uniform blur regions suggest AI single-σ background blur"
                )
                score *= max(0.4, mean_local_entropy / 1.5)

        # Test 3: Check for large uniform regions (spatial clustering of identical values)
        # Find connected regions with identical blur values
        blur_map_rounded = np.round(blur_map * 100) / 100  # Round to 0.01 precision
        blur_map_rounded[~valid_mask] = np.nan

        # Check for large regions with same rounded value
        from scipy.ndimage import label

        max_uniform_region_size = 0
        for unique_val in np.unique(blur_map_rounded[valid_mask]):
            if np.isnan(unique_val):
                continue
            mask = blur_map_rounded == unique_val
            labeled, num_features = label(mask)
            if num_features > 0:
                region_sizes = [
                    np.sum(labeled == i) for i in range(1, num_features + 1)
                ]
                max_uniform_region_size = max(
                    max_uniform_region_size, max(region_sizes) if region_sizes else 0
                )

        # If a large region (>20% of valid pixels) has identical blur, suspicious
        total_valid = np.sum(valid_mask)
        if total_valid > 0:
            uniform_ratio = max_uniform_region_size / total_valid
            if uniform_ratio > 0.2:
                violations_semantic.append(
                    f"Large uniform blur region detected ({uniform_ratio:.1%} of image) - "
                    f"suggests AI uniform σ blur rather than natural depth variation"
                )
                score *= max(0.3, 1.0 - (uniform_ratio - 0.15) * 2)

        # REFINEMENT: Test 4 - Noise Variance in Blurry Regions
        # AI-generated blur is "clean" (low noise variance); ISP-applied blur preserves sensor noise
        # This helps distinguish computational photography (real) from generative reconstruction (AI)
        if len(valid_blur) > 10:
            # Identify blurry regions (blur > median)
            median_blur = np.nanmedian(valid_blur)
            blurry_threshold = median_blur * 1.5
            
            # Map blur estimates back to image coordinates
            blurry_regions = []
            sharp_regions = []
            
            for i, y in enumerate(y_coords):
                for j, x in enumerate(x_coords):
                    if valid_mask[i, j]:
                        blur_val = blur_map[i, j]
                        # Extract local patch for noise analysis
                        patch_size = 5
                        y_patch_min = max(0, int(y) - patch_size // 2)
                        y_patch_max = min(h, int(y) + patch_size // 2 + 1)
                        x_patch_min = max(0, int(x) - patch_size // 2)
                        x_patch_max = min(w, int(x) + patch_size // 2 + 1)
                        
                        if y_patch_max > y_patch_min and x_patch_max > x_patch_min:
                            patch = image[y_patch_min:y_patch_max, x_patch_min:x_patch_max]
                            
                            # Compute local noise variance (high-frequency content)
                            # Use Laplacian to detect fine detail/noise
                            patch_uint8 = (patch * 255).astype(np.uint8)
                            laplacian_var = cv2.Laplacian(patch_uint8, cv2.CV_64F).var()
                            
                            if blur_val > blurry_threshold:
                                blurry_regions.append(laplacian_var)
                            else:
                                sharp_regions.append(laplacian_var)
            
            # Compare noise variance: blurry regions should have similar or higher variance than sharp
            # Real ISP blur: preserves sensor noise (variance similar or higher in blurry regions)
            # AI blur: often removes noise (variance lower in blurry regions)
            if len(blurry_regions) > 5 and len(sharp_regions) > 5:
                blurry_noise_var = np.mean(blurry_regions)
                sharp_noise_var = np.mean(sharp_regions)
                
                # Ratio: blurry/sharp noise variance
                # Real ISP: ratio ≈ 0.8-1.2 (noise preserved)
                # AI blur: ratio < 0.5 (noise removed/clean blur)
                noise_ratio = blurry_noise_var / (sharp_noise_var + 1e-10)
                
                if noise_ratio < 0.5:  # Blurry regions have much less noise (AI signature)
                    violations_semantic.append(
                        f"Clean blur detected (noise ratio: {noise_ratio:.2f}) - "
                        f"blurry regions lack sensor noise, suggesting AI-generated blur rather than ISP"
                    )
                    score *= max(0.4, noise_ratio * 2)  # Penalize clean blur

        # Add all semantic violations
        violations.extend(violations_semantic)

        # THIN LENS CONSISTENCY TEST: Depth-Intensity Correlation
        #
        # PHYSICS: In a real lens, blur radius R is related to object distance D by:
        #   R ∝ |D - D_focus|
        #
        # This means:
        # 1. Objects at the same distance from focus plane have similar blur
        # 2. Blur increases monotonically away from focus plane
        # 3. A single lens CANNOT have two disconnected sharp regions (two focus planes)
        #
        # FORENSIC SIGNATURE: AI/composite images may have:
        # - Two disconnected sharp regions (impossible for single lens)
        # - Blur that doesn't follow R ∝ |D - D_focus| relationship
        # - Non-monotonic blur variation (sharp near, sharp far, blurry mid = impossible)
        #
        thin_lens_violations = self._check_thin_lens_consistency(blur_map, valid_mask)
        violations.extend(thin_lens_violations)
        
        # Apply penalty for thin lens violations (very strong AI indicator)
        if thin_lens_violations:
            score *= self.THIN_LENS_VIOLATION_PENALTY

        score = max(0.0, min(1.0, score))

        # SOFT SCORING: Weight the final score by evidence_ratio
        # This gives partial credit when evidence is moderate (between threshold and 1.0)
        # Full credit when evidence_ratio is high (close to 1.0)
        # The evidence_ratio was computed earlier in the function
        if has_evidence:
            # Compute evidence confidence: how much above the minimum threshold
            # evidence_ratio ranges from MIN_BLUR_EVIDENCE_RATIO to 1.0
            # Map to confidence weight: 0.0 (at threshold) to 1.0 (strong evidence)
            evidence_range = 1.0 - self.MIN_BLUR_EVIDENCE_RATIO
            if evidence_range > 0:
                evidence_confidence = (evidence_ratio - self.MIN_BLUR_EVIDENCE_RATIO) / evidence_range
                evidence_confidence = max(0.0, min(1.0, evidence_confidence))  # Clamp to [0, 1]
            else:
                evidence_confidence = 1.0  # If threshold is 1.0, use full confidence
            
            # Soft scoring: Interpolate between neutral (1.0) and computed score
            # When evidence_confidence is low (just above threshold), give partial credit
            # When evidence_confidence is high (close to 1.0), use full computed score
            # Minimum confidence weight: 0.7 (even at threshold, give 70% weight to computed score)
            # This ensures we still use the DOF test results even with moderate evidence
            confidence_weight = 0.7 + (0.3 * evidence_confidence)  # Range: [0.7, 1.0]
            
            # Soft score: Blend neutral score (1.0) with computed score
            # Higher confidence → more weight on computed score
            soft_score = (1.0 - confidence_weight) * 1.0 + confidence_weight * score
            
            logger.debug(
                f"Soft scoring applied: evidence_ratio={evidence_ratio:.3f}, "
                f"evidence_confidence={evidence_confidence:.3f}, "
                f"confidence_weight={confidence_weight:.3f}, "
                f"computed_score={score:.3f}, final_score={soft_score:.3f}"
            )
            
            score = soft_score
            score = max(0.0, min(1.0, score))  # Final clamp

        if not violations:
            violations.append("Passes DOF consistency test")

        return OpticsTestResult(
            score=score,
            violations=violations,
            diagnostic_data={
                "blur_map": blur_map,
                "y_coords": y_coords,
                "x_coords": x_coords,
                "blur_gradient_mag": blur_gradient_mag,
                "evidence_ratio": evidence_ratio if has_evidence else None,
                "soft_scoring": has_evidence,  # Indicate soft scoring was applied
            },
        )


# ============================================================================
# TEST 5: SENSOR NOISE RESIDUAL TEST
# ============================================================================
# Distinguishes real camera sensor data from AI-generated images by analyzing
# the spatial correlation structure of noise residuals.
@pydantic_dataclass
class SensorNoiseResidualTest:
    """Test 5: Sensor Noise Residual Test.

    Distinguishes real camera sensor data from AI-generated images by analyzing
    the spatial correlation structure of noise residuals.

    PHYSICS: Real camera sensors have structural correlation in noise due to:
    - Bayer demosaicing process (inter-pixel dependencies)
    - Physical sensor patterns (readout noise, pixel crosstalk)
    - ISP processing (color interpolation creates correlations)

    FORENSIC SIGNATURE: AI-generated images have decorrelated noise because:
    - Latent space reconstruction destroys inter-pixel phase relationships
    - Generative models produce independent pixel values
    - No physical sensor structure to preserve

    This test is one of the strongest indicators for AI-generated images.
    """

    correlation_threshold: float = Field(default=0.15, ge=0.0, le=1.0)
    
    # Constants for noise residual analysis
    NOISE_SAMPLE_RATE_FACTOR = 100  # Sample ~1/N of pixels for correlation computation
    MIN_RESIDUAL_STD = 1e-6  # Minimum residual std for normalization
    MIN_AUTOCORR_VALUE = 1e-6  # Minimum autocorrelation value for normalization
    CORRELATION_SCALE_FACTOR = 0.3  # Scale factor for correlation score normalization

    def analyze_noise_residual(
        self, image_rgb: np.ndarray
    ) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """
        Analyze noise residual for spatial correlation.

        Args:
            image_rgb: RGB image (H, W, 3), float32 [0, 1]

        Returns:
            Tuple of (noise_consistency_score, decorrelation_factor, noise_residual, autocorr_region)
            - noise_consistency_score: 0.0-1.0 (1.0 = real sensor, 0.0 = AI)
            - decorrelation_factor: 0.0-1.0 (0.0 = correlated, 1.0 = decorrelated)
            - noise_residual: 2D array of noise residual (H, W)
            - autocorr_region: 11×11 autocorrelation region around center (for structure analysis)
        """
        logger.debug("Analyzing noise residual for spatial correlation")

        # Convert to grayscale for noise analysis
        if image_rgb.ndim == 3:
            image_gray = np.mean(image_rgb, axis=2)
        else:
            image_gray = image_rgb.copy()

        h, w = image_gray.shape
        logger.debug(f"Noise residual test: Processing {h}x{w} image")

        # Validate minimum size for neighbor correlation (needs at least 3×3)
        if h < 3 or w < 3:
            logger.warning(f"Image too small for noise analysis: {h}×{w}. Minimum 3×3 required.")
            noise_residual = np.zeros((h, w))
            autocorr_region = np.zeros((11, 11))
            return 0.0, 1.0, noise_residual, autocorr_region

        # REPLACED: Multi-scale Gaussian subtraction for noise residual extraction
        # More robust than median filter - captures noise at multiple scales
        logger.debug("  Extracting noise residual using multi-scale Gaussian subtraction...")
        from scipy.ndimage import gaussian_filter

        # Convert to float for precise subtraction
        residual = image_gray.astype(float)
        
        # Subtract Gaussian-filtered versions at multiple scales
        # This removes signal at different frequencies, leaving noise residual
        for sigma in [0.5, 1.0, 2.0]:
            residual -= gaussian_filter(image_gray, sigma=sigma)
        
        noise_residual = residual
        
        # FLAT-FIELD MASKING: Identify uniform/flat regions using local gradient
        # Flat regions have low local gradient - these are best for noise analysis
        logger.debug("  Computing flat-field mask using local gradient...")
        
        # Compute local gradient magnitude
        # Use Sobel operator for robust gradient computation
        from scipy.ndimage import sobel
        
        grad_y = sobel(image_gray, axis=0)
        grad_x = sobel(image_gray, axis=1)
        local_gradient = np.sqrt(grad_x**2 + grad_y**2)
        
        # Flat regions: low gradient (uniform areas, no edges)
        # Threshold: gradient < median of local gradients
        gradient_threshold = np.median(local_gradient) * 0.5
        flat_mask = local_gradient < gradient_threshold
        
        logger.debug(f"  ✓ Flat-field mask computed: {np.sum(flat_mask)}/{flat_mask.size} flat pixels ({100*np.sum(flat_mask)/flat_mask.size:.1f}%)")
        
        # Apply flat mask to residual: zero out non-flat regions
        # This focuses noise analysis on uniform areas where noise is most visible
        noise_residual = noise_residual * flat_mask
        
        logger.debug(f"  ✓ Flat mask applied to residual (masked {np.sum(~flat_mask)} non-flat pixels)")
        
        # Normalize residual AFTER flat-field masking
        # Use only flat regions for normalization (more robust)
        if np.sum(flat_mask) > 100:  # Need sufficient flat regions
            flat_residual = noise_residual[flat_mask]
            residual_std = np.std(flat_residual)
            
            if residual_std > self.MIN_RESIDUAL_STD:
                noise_residual = noise_residual / residual_std
                logger.debug(f"  ✓ Noise residual normalized using flat-field regions (std: {residual_std:.4f})")
            else:
                logger.debug(f"  ⚠ Flat-field std too low ({residual_std:.4f}), skipping normalization")
        else:
            # Fallback: use global std if insufficient flat regions
            residual_std = np.std(noise_residual)
            if residual_std > self.MIN_RESIDUAL_STD:
                noise_residual = noise_residual / residual_std
                logger.debug(f"  ✓ Noise residual normalized using global std (insufficient flat regions, std: {residual_std:.4f})")
            else:
                logger.debug(f"  ⚠ Global std too low ({residual_std:.4f}), skipping normalization")
        
        logger.debug("  ✓ Noise residual extracted, masked, and normalized")

        # Compute 2D Autocorrelation Function (ACF)
        # IMPORTANT: Autocorrelation is computed only on masked residual (flat regions)
        # Non-flat regions are zeroed out, so they don't contribute to correlation
        # OPTIMIZED: Use FFT-based correlation which is much faster for large images
        # For 512x512, FFT correlation is ~100x faster than direct correlation
        logger.debug("  Computing 2D autocorrelation using FFT on masked residual (faster for large images)...")
        start_time = time.time()
        
        # Use FFT-based correlation: autocorr = IFFT(FFT(x) * conj(FFT(x)))
        # This is O(n log n) vs O(n²) for direct correlation
        # noise_residual is already masked (non-flat regions = 0)
        fft_residual = np.fft.fft2(noise_residual)
        fft_autocorr = fft_residual * np.conj(fft_residual)
        autocorr_2d = np.real(np.fft.ifft2(fft_autocorr))
        
        # Shift to center (FFT output is not centered)
        autocorr_2d = np.fft.fftshift(autocorr_2d)
        
        elapsed = time.time() - start_time
        logger.debug(f"  ✓ FFT autocorrelation computed in {elapsed:.3f}s")
        
        # Normalize by autocorrelation at origin (center)
        center_y, center_x = get_center_coords(autocorr_2d.shape)
        autocorr_at_origin = autocorr_2d[center_y, center_x]
        if abs(autocorr_at_origin) > self.MIN_AUTOCORR_VALUE:
            autocorr_2d = autocorr_2d / autocorr_at_origin
        else:
            autocorr_2d = np.zeros_like(autocorr_2d)
        
        # Extract only the small region we actually use (11×11 around center)
        # This saves memory and makes the return value more efficient
        region_size = 5
        autocorr_region = autocorr_2d[
            center_y - region_size : center_y + region_size + 1,
            center_x - region_size : center_x + region_size + 1,
        ].copy()
        logger.debug("  ✓ Autocorrelation region extracted")

        # FIXED: Compute 8-neighbor correlation using correct Pearson formula
        # IMPORTANT: Sample only flat pixels (PRNU lives in flat regions, not edges/textures)
        # IMPORTANT: Do NOT subtract neighbor means - this removes the covariance we want to measure
        logger.debug("  Computing 8-neighbor correlations on flat pixels only...")
        start_time = time.time()
        
        # Sample only flat pixels (where PRNU correlation is strongest)
        # PRNU lives in: flat walls, skies, backgrounds, smooth skin regions
        # NOT in: eyes, lips, hair, edges (these have scene leakage → noise-like → ρ ≈ 0)
        flat_pixel_coords = np.column_stack(np.where(flat_mask))
        
        if len(flat_pixel_coords) < 100:
            logger.warning(f"Insufficient flat pixels for correlation: {len(flat_pixel_coords)}")
            return 0.0, 1.0, noise_residual, autocorr_region
        
        # Sample flat pixels to avoid excessive computation
        sample_rate = max(1, len(flat_pixel_coords) // self.NOISE_SAMPLE_RATE_FACTOR)
        sample_indices = np.arange(0, len(flat_pixel_coords), sample_rate)
        sampled_coords = flat_pixel_coords[sample_indices]
        
        # Filter out boundary pixels (need neighbors)
        valid_coord_mask = (
            (sampled_coords[:, 0] > 0) & (sampled_coords[:, 0] < h - 1) &
            (sampled_coords[:, 1] > 0) & (sampled_coords[:, 1] < w - 1)
        )
        sampled_coords = sampled_coords[valid_coord_mask]
        
        if len(sampled_coords) < 10:
            logger.warning(f"Insufficient valid flat pixels for correlation: {len(sampled_coords)}")
            return 0.0, 1.0, noise_residual, autocorr_region
        
        n_samples = len(sampled_coords)
        logger.debug(f"  Sampling {n_samples} flat pixels for correlation (from {len(flat_pixel_coords)} total flat pixels)")
        
        y_flat = sampled_coords[:, 0]
        x_flat = sampled_coords[:, 1]
        
        logger.debug(f"  Extracting center and neighbor values...")
        # Get center values for all sampled flat pixels (vectorized)
        center_vals = noise_residual[y_flat, x_flat]
        
        # OPTIMIZED: Get all 8 neighbors at once using array slicing
        # Shape: (n_samples, 8) for neighbor values
        neighbors = np.zeros((n_samples, 8), dtype=noise_residual.dtype)
        
        # Extract all 8 neighbors using vectorized indexing
        # Top row
        neighbors[:, 0] = noise_residual[y_flat - 1, x_flat - 1]  # (-1, -1)
        neighbors[:, 1] = noise_residual[y_flat - 1, x_flat]      # (-1, 0)
        neighbors[:, 2] = noise_residual[y_flat - 1, x_flat + 1]  # (-1, 1)
        # Middle row
        neighbors[:, 3] = noise_residual[y_flat, x_flat - 1]     # (0, -1)
        neighbors[:, 4] = noise_residual[y_flat, x_flat + 1]     # (0, 1)
        # Bottom row
        neighbors[:, 5] = noise_residual[y_flat + 1, x_flat - 1]  # (1, -1)
        neighbors[:, 6] = noise_residual[y_flat + 1, x_flat]      # (1, 0)
        neighbors[:, 7] = noise_residual[y_flat + 1, x_flat + 1]  # (1, 1)
        
        logger.debug(f"  Computing correlation coefficients (correct Pearson formula, no mean-centering)...")
        # FIXED: Use correct Pearson correlation formula
        # ρ = cov(X, Y) / (σ_X σ_Y) = E[XY] / (σ_X σ_Y) when E[X] = E[Y] = 0 (already normalized)
        # Do NOT subtract neighbor means - this removes the covariance we want to measure
        # Formula: corr = mean(center_val * neighbor_val) / (std(center_val) * std(neighbor_val))
        # Do not locally mean-center - this collapses correlation
        
        # For each pixel, compute correlation between center and its 8 neighbors
        # Since center is a single value, we compute correlation across all pixels
        # OR: for each pixel, compute: corr = mean(center * neighbors) / (|center| * std(neighbors))
        # The user's formula suggests: corr = mean(center_val * neighbor_val) / (std(center_val) * std(neighbor_val))
        # This implies computing correlation across all pixels, not per-pixel
        
        # Vectorized approach: compute correlation across all sampled pixels
        # Stack center values and neighbor values, then compute correlation
        # For each neighbor position, compute correlation with center across all pixels
        
        correlations = []
        for neighbor_idx in range(8):
            neighbor_vals = neighbors[:, neighbor_idx]
            
            # Compute correlation between center_vals and neighbor_vals across all pixels
            # Pearson correlation: ρ = cov(X, Y) / (σ_X σ_Y)
            # = mean((X - μ_X)(Y - μ_Y)) / (σ_X σ_Y)
            # But since we don't want to mean-center, use: ρ = mean(X * Y) / (σ_X σ_Y)
            # when X and Y are already normalized (mean ≈ 0)
            
            # Compute std of center and neighbors
            center_std = np.std(center_vals)
            neighbor_std = np.std(neighbor_vals)
            
            if center_std < 1e-6 or neighbor_std < 1e-6:
                continue
            
            # Compute correlation: mean(center * neighbor) / (std(center) * std(neighbor))
            mean_product = np.mean(center_vals * neighbor_vals)
            corr = mean_product / (center_std * neighbor_std + 1e-10)
            
            # Use absolute correlation (we care about magnitude, not sign)
            correlations.append(abs(corr))
        
        # Average correlation across all 8 neighbor positions
        if len(correlations) > 0:
            neighbor_correlations = np.array(correlations)
        else:
            neighbor_correlations = np.array([])
        
        elapsed = time.time() - start_time
        logger.debug(f"  ✓ 8-neighbor correlation computed in {elapsed:.3f}s ({len(neighbor_correlations)} valid correlations)")

        # Compute mean spatial correlation
        logger.debug(f"  Computing statistics from {len(neighbor_correlations)} correlation values...")
        if len(neighbor_correlations) > 0:
            mean_correlation = np.mean(neighbor_correlations)
            std_correlation = np.std(neighbor_correlations)
        else:
            mean_correlation = 0.0
            std_correlation = 0.0

        # Decorrelation factor: inverse of correlation (0 = correlated, 1 = decorrelated)
        decorrelation_factor = max(0.0, min(1.0, 1.0 - mean_correlation))

        # Noise consistency score: higher correlation = higher score (real sensor)
        # Real sensors: ρ > 0.15, AI: ρ < 0.15
        if mean_correlation >= self.correlation_threshold:
            # Real sensor: high correlation
            noise_consistency_score = min(1.0, mean_correlation / self.CORRELATION_SCALE_FACTOR)
        else:
            # AI/synthetic: low correlation
            noise_consistency_score = max(0.0, mean_correlation / self.correlation_threshold)

        logger.debug(
            f"  ✓ Noise residual analysis complete: mean_correlation={mean_correlation:.4f}, "
            f"decorrelation_factor={decorrelation_factor:.4f}, "
            f"score={noise_consistency_score:.4f}"
        )

        return (
            noise_consistency_score,
            decorrelation_factor,
            noise_residual,
            autocorr_region,
        )

    def test(self, image_rgb: np.ndarray) -> OpticsTestResult:
        """
        Test sensor noise residual consistency.

        Args:
            image_rgb: RGB image (H, W, 3), float32 [0, 1]

        Returns:
            OpticsTestResult with score and violations
        """
        logger.debug("Running sensor noise residual test")

        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            return OpticsTestResult(
                score=0.5,
                violations=["Requires RGB image for noise residual test"],
                diagnostic_data={},
            )

        try:
            (
                noise_consistency_score,
                decorrelation_factor,
                noise_residual,
                autocorr_region,
            ) = self.analyze_noise_residual(image_rgb)

            violations = []
            score = noise_consistency_score

            # Compute mean correlation from decorrelation factor
            # decorrelation_factor = 1.0 - mean_correlation, so mean_correlation = 1.0 - decorrelation_factor
            mean_correlation = 1.0 - decorrelation_factor
            
            # Flag as AI if mean correlation falls below threshold
            if mean_correlation < self.correlation_threshold:
                violations.append(
                    f"Low spatial correlation detected (ρ={mean_correlation:.3f} < {self.correlation_threshold}) - "
                    f"noise is decorrelated, suggesting AI-generated image rather than real sensor data"
                )
                # Strong penalty for decorrelated noise
                score *= 0.3

            # Additional check: very high decorrelation (near-random noise)
            if decorrelation_factor > 0.9:
                violations.append(
                    f"Extremely decorrelated noise (decorrelation={decorrelation_factor:.3f}) - "
                    f"noise appears random, consistent with AI latent space reconstruction"
                )
                score *= 0.2

            # Check for structural patterns in autocorrelation
            # Real sensors: autocorrelation should have structure (Bayer patterns)
            # AI: autocorrelation should be near-delta function (no structure)
            # autocorr_region is already extracted (11×11) from analyze_noise_residual()
            region_size = 5
            autocorr_region_copy = autocorr_region.copy()
            autocorr_region_copy[region_size, region_size] = 0.0  # Exclude center
            
            # Real sensors: should have some structure (non-zero off-center values)
            # AI: should be near-zero everywhere except center
            off_center_strength = np.std(autocorr_region_copy)
            
            if off_center_strength < 0.05:  # Very weak structure
                violations.append(
                    f"Lack of autocorrelation structure (off-center std={off_center_strength:.4f}) - "
                    f"suggests AI-generated image with decorrelated noise"
                )
                score *= max(0.4, off_center_strength * 10)

            if not violations:
                violations.append("Passes sensor noise residual test")

            return OpticsTestResult(
                score=score,
                violations=violations,
                diagnostic_data={
                    "noise_residual": noise_residual,
                    "autocorrelation_region": autocorr_region,
                    "mean_correlation": mean_correlation,
                    "decorrelation_factor": decorrelation_factor,
                    "noise_consistency_score": noise_consistency_score,
                },
            )

        except Exception as e:
            logger.warning(f"Failed to analyze noise residual: {e}")
            return OpticsTestResult(
                score=0.5,
                violations=[f"Error in noise residual analysis: {str(e)}"],
                diagnostic_data={},
            )


# ============================================================================
# TEST 4: CHROMATIC ABERRATION TEST
# ============================================================================
# Computes edge offsets between R/G/B channels and checks for
# radial consistency. Note: Modern phones correct CA in ISP.
@pydantic_dataclass
class ChromaticAberrationTest:
    """Test 4: Chromatic aberration test.

    Computes edge offsets between R/G/B channels and checks for
    radial consistency. Note: Modern phones correct CA in ISP, and
    resizing/re-encoding can destroy subpixel CA cues.

    IMPORTANT: This test is conditional:
    - Low CA magnitude is NOT suspicious (expected after ISP correction)
    - Radial consistency only evaluated on original resolution images
    - Heavy processing/re-encode makes test low-confidence
    """

    edge_threshold: float = Field(default=0.1, gt=0.0)
    
    # Constants for chromatic aberration analysis
    ALIGNMENT_NORMALIZATION_FACTOR = 0.7  # Normalization factor for alignment scores
    MIN_VECTOR_NORM = 1e-6  # Minimum vector norm for normalization
    
    # Resolution thresholds
    MIN_RESOLUTION_FOR_RADIAL_TEST = 1024  # Minimum resolution to trust radial consistency (subpixel CA needs high res)
    LOW_CONFIDENCE_RESOLUTION = 512  # Below this, test is low-confidence (likely resized)

    def test(
        self, image_rgb: np.ndarray, original_resolution: Optional[Tuple[int, int]] = None
    ) -> OpticsTestResult:
        """
        Test chromatic aberration consistency (CONDITIONAL TEST).

        IMPORTANT CHANGES:
        - Low CA magnitude is NOT suspicious (phones correct CA in ISP)
        - Radial consistency only evaluated on original resolution images
        - Heavy processing/re-encode makes test low-confidence

        Args:
            image_rgb: RGB image (H, W, 3), float32 [0, 1]
            original_resolution: Optional (height, width) of original image before resizing.
                                If None, assumes image may have been resized.

        Returns:
            OpticsTestResult with score and violations
        """
        logger.debug("Running chromatic aberration test (conditional)")

        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            return OpticsTestResult(
                score=0.5,
                violations=["Requires RGB image for CA test"],
                diagnostic_data={},
            )

        h, w = image_rgb.shape[:2]
        
        # Check if image has been resized (likely destroyed subpixel CA cues)
        current_resolution = min(h, w)
        is_low_resolution = current_resolution < self.LOW_CONFIDENCE_RESOLUTION
        
        # Check original resolution if provided
        original_min_res = None
        if original_resolution is not None:
            original_min_res = min(original_resolution[0], original_resolution[1])
            is_low_resolution = original_min_res < self.LOW_CONFIDENCE_RESOLUTION
        
        # Determine test confidence
        if is_low_resolution:
            logger.debug(
                f"Low-resolution image detected (current: {current_resolution}, "
                f"original: {original_min_res}) - CA test will be low-confidence"
            )
            test_confidence = 0.3  # Low confidence for resized images
        elif original_resolution is None:
            logger.debug(
                "Original resolution unknown - assuming image may have been resized/re-encoded"
            )
            test_confidence = 0.5  # Medium confidence
        else:
            test_confidence = 1.0  # High confidence for original resolution
        
        # Check if resolution is sufficient for radial consistency test
        can_test_radial = False
        if original_resolution is not None:
            can_test_radial = original_min_res >= self.MIN_RESOLUTION_FOR_RADIAL_TEST
        else:
            # If original resolution unknown, assume it's been resized
            can_test_radial = False
        
        logger.debug(
            f"CA test confidence: {test_confidence:.2f}, can_test_radial: {can_test_radial}, "
            f"original_res: {original_resolution}"
        )
        center_y, center_x = get_center_coords((h, w))

        # Extract R, G, B channels
        r_channel = image_rgb[:, :, 0]
        g_channel = image_rgb[:, :, 1]
        b_channel = image_rgb[:, :, 2]

        # REFINED METHOD: Detect edges only in Green channel, then find gradient shifts in R/B
        # This ensures we compare the same physical edge across channels, avoiding
        # false offsets from noise-induced edge detection differences.
        
        # Convert green channel to uint8 for edge detection
        g_gray = (g_channel * 255).astype(np.uint8)
        
        # Detect edges ONLY in green channel (reference)
        edges_g = cv2.Canny(g_gray, 50, 150)
        edge_coords = np.column_stack(np.where(edges_g > 0))

        if len(edge_coords) < 20:
            return OpticsTestResult(
                score=0.5,
                violations=["Insufficient edges for CA analysis"],
                diagnostic_data={},
            )

        # Sample edges and compute offsets using gradient-based matching
        sample_indices = np.linspace(
            0, len(edge_coords) - 1, min(100, len(edge_coords)), dtype=int
        )
        logger.debug(f"Processing {len(sample_indices)} edge samples for CA detection")

        rg_offsets = []
        bg_offsets = []
        rg_radial_distances = []  # Radial distances for R-G offsets
        bg_radial_distances = []  # Radial distances for B-G offsets
        rg_offset_vectors = []  # Store tuples: (offset_vec, (y, x)) for radial alignment check
        bg_offset_vectors = []  # Store tuples: (offset_vec, (y, x)) for radial alignment check

        # Compute gradients for R and B channels (for gradient shift detection)
        # Use Sobel operator for more robust gradient computation
        logger.debug("Computing gradients for R, G, B channels...")
        r_grad_y = cv2.Sobel(r_channel, cv2.CV_64F, 0, 1, ksize=3)
        r_grad_x = cv2.Sobel(r_channel, cv2.CV_64F, 1, 0, ksize=3)
        r_grad_mag = np.sqrt(r_grad_x**2 + r_grad_y**2)
        
        b_grad_y = cv2.Sobel(b_channel, cv2.CV_64F, 0, 1, ksize=3)
        b_grad_x = cv2.Sobel(b_channel, cv2.CV_64F, 1, 0, ksize=3)
        b_grad_mag = np.sqrt(b_grad_x**2 + b_grad_y**2)
        
        # Also compute gradient direction in Green channel (for perpendicular search)
        g_grad_y = cv2.Sobel(g_channel, cv2.CV_64F, 0, 1, ksize=3)
        g_grad_x = cv2.Sobel(g_channel, cv2.CV_64F, 1, 0, ksize=3)
        logger.debug("  ✓ Gradients computed")

        # OPTIMIZED: Reduce logging frequency in tight loop
        log_interval = max(20, len(sample_indices) // 5)  # Log 5 times total
        if len(sample_indices) > 10:
            logger.debug(f"Processing {len(sample_indices)} edge samples for CA detection...")
        for sample_idx, idx in enumerate(sample_indices):
            if (sample_idx + 1) % log_interval == 0:
                logger.debug(f"  Processing edge {sample_idx + 1}/{len(sample_indices)}...")
            y, x = edge_coords[idx]
            
            # Skip if too close to image boundaries
            if y < 2 or y >= h - 2 or x < 2 or x >= w - 2:
                continue

            # Compute radial distance from center
            dy = y - center_y
            dx = x - center_x
            radial_dist = np.sqrt(dx**2 + dy**2)

            # Get edge orientation from Green channel gradient
            # Perpendicular direction (rotate gradient 90 degrees) is along the edge
            gx = g_grad_x[y, x]
            gy = g_grad_y[y, x]
            
            if abs(gx) < 1e-6 and abs(gy) < 1e-6:
                continue
            
            # Perpendicular direction (along which CA shift occurs)
            perp_x = -gy  # Rotate 90 degrees
            perp_y = gx
            perp_norm = np.sqrt(perp_x**2 + perp_y**2)
            if perp_norm < 1e-6:
                continue
            
            perp_x /= perp_norm
            perp_y /= perp_norm

            # Search for maximum gradient shift in R and B channels
            # along the perpendicular direction (where CA shift occurs)
            search_radius = 3
            search_positions = np.arange(-search_radius, search_radius + 0.5, 0.5)  # Sub-pixel resolution
            
            # R-G offset detection
            r_grad_along_perp = []
            r_positions = []
            for shift in search_positions:
                # Sample at sub-pixel location using bilinear interpolation
                y_shifted = y + shift * perp_y
                x_shifted = x + shift * perp_x
                
                # Bounds check
                if y_shifted < 1 or y_shifted >= h - 1 or x_shifted < 1 or x_shifted >= w - 1:
                    continue
                
                # Bilinear interpolation for gradient magnitude
                y0, y1 = int(np.floor(y_shifted)), int(np.ceil(y_shifted))
                x0, x1 = int(np.floor(x_shifted)), int(np.ceil(x_shifted))
                dy_frac = y_shifted - y0
                dx_frac = x_shifted - x0
                
                grad_val = (
                    r_grad_mag[y0, x0] * (1 - dx_frac) * (1 - dy_frac) +
                    r_grad_mag[y0, x1] * dx_frac * (1 - dy_frac) +
                    r_grad_mag[y1, x0] * (1 - dx_frac) * dy_frac +
                    r_grad_mag[y1, x1] * dx_frac * dy_frac
                )
                
                r_grad_along_perp.append(grad_val)
                r_positions.append(shift)
            
            # Find position of maximum gradient (edge location in R channel)
            if len(r_grad_along_perp) > 0:
                max_idx = np.argmax(r_grad_along_perp)
                r_shift = r_positions[max_idx]
                
                # Compute offset vector
                r_offset_x = r_shift * perp_x
                r_offset_y = r_shift * perp_y
                rg_offset = abs(r_shift)  # Magnitude
                
                rg_offsets.append(rg_offset)
                rg_radial_distances.append(radial_dist)  # Track radial distance for R-G offset
                offset_vec = np.array([r_offset_x, r_offset_y])
                rg_offset_vectors.append((offset_vec, (y, x)))
            else:
                rg_offset_vectors.append((np.array([0.0, 0.0]), (y, x)))
            
            # B-G offset detection (same method)
            b_grad_along_perp = []
            b_positions = []
            for shift in search_positions:
                y_shifted = y + shift * perp_y
                x_shifted = x + shift * perp_x
                
                if y_shifted < 1 or y_shifted >= h - 1 or x_shifted < 1 or x_shifted >= w - 1:
                    continue
                
                y0, y1 = int(np.floor(y_shifted)), int(np.ceil(y_shifted))
                x0, x1 = int(np.floor(x_shifted)), int(np.ceil(x_shifted))
                dy_frac = y_shifted - y0
                dx_frac = x_shifted - x0
                
                grad_val = (
                    b_grad_mag[y0, x0] * (1 - dx_frac) * (1 - dy_frac) +
                    b_grad_mag[y0, x1] * dx_frac * (1 - dy_frac) +
                    b_grad_mag[y1, x0] * (1 - dx_frac) * dy_frac +
                    b_grad_mag[y1, x1] * dx_frac * dy_frac
                )
                
                b_grad_along_perp.append(grad_val)
                b_positions.append(shift)
            
            if len(b_grad_along_perp) > 0:
                max_idx = np.argmax(b_grad_along_perp)
                b_shift = b_positions[max_idx]
                
                b_offset_x = b_shift * perp_x
                b_offset_y = b_shift * perp_y
                bg_offset = abs(b_shift)
                
                bg_offsets.append(bg_offset)
                bg_radial_distances.append(radial_dist)  # Track radial distance for B-G offset
                offset_vec = np.array([b_offset_x, b_offset_y])
                bg_offset_vectors.append((offset_vec, (y, x)))
            else:
                bg_offset_vectors.append((np.array([0.0, 0.0]), (y, x)))

        if len(rg_offsets) < 5 or len(bg_offsets) < 5:
            return OpticsTestResult(
                score=0.5,
                violations=["Insufficient CA offset measurements"],
                diagnostic_data={},
            )

        violations = []
        score = 1.0
        confidence_factor = test_confidence  # Will scale final score
        
        # Initialize proportionality fit results (for diagnostic data)
        proportionality_fits = {
            'rg_slope': None,
            'rg_intercept': None,
            'rg_r_squared': None,
            'bg_slope': None,
            'bg_intercept': None,
            'bg_r_squared': None,
        }

        # REMOVED: Zero CA check
        # Modern phones correct CA in ISP, so low/zero CA is expected, not suspicious
        mean_rg_offset = np.mean(rg_offsets)
        mean_bg_offset = np.mean(bg_offsets)
        
        logger.debug(
            f"CA offsets: R-G={mean_rg_offset:.3f}, B-G={mean_bg_offset:.3f} "
            f"(low magnitude is normal after ISP correction)"
        )

        # RADIAL PROPORTIONALITY TEST: Real lateral CA magnitude is proportional to radial distance
        # 
        # PHYSICS: Real lateral CA follows: |offset| ∝ r (proportional, not just linear)
        # This means: offset = k * r (intercept should be near zero)
        # 
        # FORENSIC TEST: Fit offset = a * r + b and check:
        # 1. Slope a > 0 (positive, increasing with distance)
        # 2. Intercept b ≈ 0 (proportional relationship, not just linear)
        # 3. Good fit quality (R² > 0.3) - validates the relationship exists
        #
        # IMPORTANT: Only evaluate on original resolution images (subpixel CA destroyed by resizing)
        if can_test_radial and len(rg_radial_distances) >= 10 and len(bg_radial_distances) >= 10:
            logger.debug("Testing CA radial proportionality (magnitude ∝ r)...")
            
            # Helper function to compute R² (coefficient of determination)
            def compute_r_squared(y_true, y_pred):
                ss_res = np.sum((y_true - y_pred) ** 2)
                ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
                if ss_tot < 1e-10:
                    return 0.0
                return 1.0 - (ss_res / ss_tot)
            
            # Test R-G proportionality
            try:
                rg_radial = np.array(rg_radial_distances)
                rg_offsets_arr = np.array(rg_offsets)
                
                # Fit: offset = a * r + b
                rg_slope, rg_intercept = np.polyfit(rg_radial, rg_offsets_arr, deg=1)
                rg_predicted = rg_slope * rg_radial + rg_intercept
                rg_r_squared = compute_r_squared(rg_offsets_arr, rg_predicted)
                
                # Store for diagnostic data
                proportionality_fits['rg_slope'] = float(rg_slope)
                proportionality_fits['rg_intercept'] = float(rg_intercept)
                proportionality_fits['rg_r_squared'] = float(rg_r_squared)
                
                logger.debug(
                    f"R-G CA fit: slope={rg_slope:.6f}, intercept={rg_intercept:.6f}, R²={rg_r_squared:.3f}"
                )
                
                # Check 1: Negative slope is non-physical
                if rg_slope < -0.01:
                    violations.append(
                        f"Non-physical CA radial variation (R-G slope: {rg_slope:.4f} < 0) - "
                        f"CA magnitude should increase with distance from center"
                    )
                    score *= 0.5
                
                # Check 2: Non-proportional relationship (intercept too large)
                # For proportionality, intercept should be near zero relative to typical offset
                mean_rg_offset = np.mean(rg_offsets_arr)
                if mean_rg_offset > 1e-3:  # Only check if there's meaningful CA
                    intercept_ratio = abs(rg_intercept) / mean_rg_offset
                    if intercept_ratio > 0.3:  # Intercept > 30% of mean offset (non-proportional)
                        violations.append(
                            f"Non-proportional CA relationship (R-G intercept ratio: {intercept_ratio:.2f}) - "
                            f"real CA should follow |offset| ∝ r (intercept ≈ 0)"
                        )
                        score *= max(0.4, 1.0 - intercept_ratio)
                
                # Check 3: Poor fit quality (weak relationship)
                if rg_r_squared < 0.3 and mean_rg_offset > 0.1:  # Only penalize if CA is significant
                    violations.append(
                        f"Weak CA radial relationship (R-G R²: {rg_r_squared:.3f} < 0.3) - "
                        f"CA magnitude should correlate with radial distance"
                    )
                    score *= max(0.5, rg_r_squared * 2)
                    
            except Exception as e:
                logger.warning(f"Failed to fit R-G CA proportionality: {e}")
            
            # Test B-G proportionality (same checks)
            try:
                bg_radial = np.array(bg_radial_distances)
                bg_offsets_arr = np.array(bg_offsets)
                
                bg_slope, bg_intercept = np.polyfit(bg_radial, bg_offsets_arr, deg=1)
                bg_predicted = bg_slope * bg_radial + bg_intercept
                bg_r_squared = compute_r_squared(bg_offsets_arr, bg_predicted)
                
                # Store for diagnostic data
                proportionality_fits['bg_slope'] = float(bg_slope)
                proportionality_fits['bg_intercept'] = float(bg_intercept)
                proportionality_fits['bg_r_squared'] = float(bg_r_squared)
                
                logger.debug(
                    f"B-G CA fit: slope={bg_slope:.6f}, intercept={bg_intercept:.6f}, R²={bg_r_squared:.3f}"
                )
                
                if bg_slope < -0.01:
                    violations.append(
                        f"Non-physical CA radial variation (B-G slope: {bg_slope:.4f} < 0)"
                    )
                    score *= 0.5
                
                mean_bg_offset = np.mean(bg_offsets_arr)
                if mean_bg_offset > 1e-3:
                    intercept_ratio = abs(bg_intercept) / mean_bg_offset
                    if intercept_ratio > 0.3:
                        violations.append(
                            f"Non-proportional CA relationship (B-G intercept ratio: {intercept_ratio:.2f})"
                        )
                        score *= max(0.4, 1.0 - intercept_ratio)
                
                if bg_r_squared < 0.3 and mean_bg_offset > 0.1:
                    violations.append(
                        f"Weak CA radial relationship (B-G R²: {bg_r_squared:.3f} < 0.3)"
                    )
                    score *= max(0.5, bg_r_squared * 2)
                    
            except Exception as e:
                logger.warning(f"Failed to fit B-G CA proportionality: {e}")
                
        else:
            # Cannot test radial consistency on resized images
            logger.debug(
                "Skipping radial proportionality test - image likely resized/re-encoded "
                "(subpixel CA cues destroyed) or insufficient samples"
            )

        # Check for symmetric fake CA (all offsets in same direction)
        # Real CA varies with angle
        rg_std = np.std(rg_offsets)
        bg_std = np.std(bg_offsets)

        if rg_std < 0.05 and bg_std < 0.05:
            violations.append("Suspiciously uniform CA (possible fake)")
            score *= 0.6

        # Check spatial coherence (CA should be smooth, not random)
        # High variance in offsets indicates non-coherent CA
        if rg_std > 1.0 or bg_std > 1.0:
            violations.append("Non-coherent chromatic aberration")
            score *= max(0.5, 1.0 - (max(rg_std, bg_std) - 0.5) * 0.2)

        # RADIAL ALIGNMENT TEST: Lateral CA must be radial (ONLY on original resolution)
        #
        # PHYSICS: Real lateral CA is radial - the offset vector (Δx, Δy) must point
        # directly toward or away from the image center (c_x, c_y).
        #
        # FORENSIC TEST: Calculate alignment = (Offset · Radius) / (|Offset| |Radius|)
        # - Alignment ≈ ±1: Radial (correct, physical)
        # - Alignment ≈ 0: Tangential (non-physical, AI artifact)
        #
        # IMPORTANT: Only evaluate on original resolution images (subpixel CA destroyed by resizing)
        alignments_rg = []
        alignments_bg = []
        
        if can_test_radial:
            # Check R-G alignment
            for offset_vec, (y, x) in rg_offset_vectors:
                offset_magnitude = np.linalg.norm(offset_vec) if offset_vec.size == 2 else 0.0
                
                # REFINEMENT: Add magnitude threshold - if ISP removed 99% of CA, remaining is mostly noise
                # Don't penalize alignment if total offset is < 0.5 pixels (likely noise, not real CA)
                if offset_magnitude < 0.5:
                    continue  # Skip small offsets that are likely noise
                
                if offset_vec.size == 2 and offset_magnitude > 0.01:
                    # Compute radial vector from center to edge location
                    dy = y - center_y
                    dx = x - center_x
                    radial_vec = np.array([dx, dy])
                    radial_vec_norm = np.linalg.norm(radial_vec)
                    
                    if radial_vec_norm > 0.01:
                        # Normalize both vectors
                        radial_vec_normalized = radial_vec / radial_vec_norm
                        offset_vec_normalized = offset_vec / offset_magnitude
                        
                        # Compute alignment: dot product (should be ±1 for radial)
                        # Alignment = (Offset · Radius) / (|Offset| |Radius|)
                        alignment = np.dot(offset_vec_normalized, radial_vec_normalized)
                        alignments_rg.append(alignment)
            
            # Check B-G alignment
            for offset_vec, (y, x) in bg_offset_vectors:
                offset_magnitude = np.linalg.norm(offset_vec) if offset_vec.size == 2 else 0.0
                
                # REFINEMENT: Add magnitude threshold - skip small offsets (< 0.5 pixels) that are likely noise
                if offset_magnitude < 0.5:
                    continue  # Skip small offsets that are likely noise
                
                if offset_vec.size == 2 and offset_magnitude > 0.01:
                    dy = y - center_y
                    dx = x - center_x
                    radial_vec = np.array([dx, dy])
                    radial_vec_norm = np.linalg.norm(radial_vec)
                    
                    if radial_vec_norm > 0.01:
                        radial_vec_normalized = radial_vec / radial_vec_norm
                        offset_vec_normalized = offset_vec / offset_magnitude
                        alignment = np.dot(offset_vec_normalized, radial_vec_normalized)
                        alignments_bg.append(alignment)
            
            # Check alignment scores (only if we have enough data)
            # Real CA: alignment close to ±1 (radial)
            # AI/fake CA: alignment close to 0 (tangential/sideways)
            is_radial = True  # Default to True (radial) if we can't test
            mean_alignment_rg = 0.0
            mean_alignment_bg = 0.0
            
            if len(alignments_rg) > 5:
                mean_alignment_rg = np.mean(np.abs(alignments_rg))
                # If mean alignment < 0.7, CA is too tangential (non-physical)
                if mean_alignment_rg < 0.7:
                    violations.append(
                        f"Non-radial CA detected (R-G alignment: {mean_alignment_rg:.2f}) - "
                        f"offset vectors are tangential (sideways), not radial - AI artifact"
                    )
                    score *= max(0.3, mean_alignment_rg / self.ALIGNMENT_NORMALIZATION_FACTOR)
                    is_radial = False  # Non-radial detected
            
            if len(alignments_bg) > 5:
                mean_alignment_bg = np.mean(np.abs(alignments_bg))
                if mean_alignment_bg < 0.7:
                    violations.append(
                        f"Non-radial CA detected (B-G alignment: {mean_alignment_bg:.2f}) - "
                        f"offset vectors are tangential (sideways), not radial - AI artifact"
                    )
                    score *= max(0.3, mean_alignment_bg / self.ALIGNMENT_NORMALIZATION_FACTOR)
                    is_radial = False  # Non-radial detected
            
            # If both alignments are good (>= 0.7), CA is radial
            if len(alignments_rg) > 5 and len(alignments_bg) > 5:
                is_radial = (mean_alignment_rg >= 0.7) and (mean_alignment_bg >= 0.7)
        else:
            # Cannot test radial alignment on resized images
            logger.debug(
                "Skipping radial alignment test - image likely resized/re-encoded "
                "(subpixel CA cues destroyed)"
            )
            is_radial = None  # Unknown (can't test)

        # COLOR ORDER CONSISTENCY TEST: Refractive index relationship
        #
        # PHYSICS: Refractive indices follow Cauchy's equation: n_blue > n_green > n_red
        # This means:
        # - Blue light is refracted MORE than green (B-G offset larger, opposite direction)
        # - Red light is refracted LESS than green (R-G offset smaller, opposite direction)
        # - B-G and R-G offsets should typically point in OPPOSITE directions
        #
        # FORENSIC TEST: If Blue and Red are both shifted the same way relative to Green,
        # the "lens" is physically impossible.
        #
        if len(rg_offset_vectors) > 5 and len(bg_offset_vectors) > 5:
            # Compare directions of R-G and B-G offsets
            # They should point in opposite directions (or have specific ratio)
            same_direction_count = 0
            total_comparisons = 0
            
            # Match R-G and B-G vectors by edge location
            rg_dict = {(y, x): vec for vec, (y, x) in rg_offset_vectors if np.linalg.norm(vec) > 0.01}
            bg_dict = {(y, x): vec for vec, (y, x) in bg_offset_vectors if np.linalg.norm(vec) > 0.01}
            
            # Find common edge locations
            common_locations = set(rg_dict.keys()) & set(bg_dict.keys())
            
            if len(common_locations) > 5:
                for y, x in common_locations:
                    rg_vec = rg_dict[(y, x)]
                    bg_vec = bg_dict[(y, x)]
                    
                    # Normalize vectors (with protection against zero vectors)
                    rg_norm = np.linalg.norm(rg_vec)
                    bg_norm = np.linalg.norm(bg_vec)
                    
                    if rg_norm < self.MIN_VECTOR_NORM or bg_norm < self.MIN_VECTOR_NORM:
                        continue  # Skip if vector is too small
                    
                    rg_vec_norm = rg_vec / rg_norm
                    bg_vec_norm = bg_vec / bg_norm
                    
                    # Compute dot product to check direction
                    # Positive = same direction, negative = opposite direction
                    direction_dot = np.dot(rg_vec_norm, bg_vec_norm)
                    
                    # For real CA: n_blue > n_green > n_red
                    # B-G and R-G should point in opposite directions (negative dot product)
                    # Or at least not strongly in the same direction
                    if direction_dot > 0.5:  # Strongly same direction = impossible
                        same_direction_count += 1
                    total_comparisons += 1
                
                if total_comparisons > 0:
                    same_direction_ratio = same_direction_count / total_comparisons
                    
                    # If >30% of offsets point in same direction, suspicious
                    if same_direction_ratio > 0.3:
                        violations.append(
                            f"Color order inconsistency detected ({same_direction_ratio:.1%} same-direction offsets) - "
                            f"B-G and R-G offsets should point opposite (n_blue > n_green > n_red) - physically impossible"
                        )
                        score *= max(0.2, 1.0 - same_direction_ratio * 2)
                    
                    # Also check magnitude relationship
                    # Typically: |B-G offset| > |R-G offset| (blue refracted more)
                    rg_magnitudes = [np.linalg.norm(rg_dict[loc]) for loc in common_locations]
                    bg_magnitudes = [np.linalg.norm(bg_dict[loc]) for loc in common_locations]
                    
                    if len(rg_magnitudes) > 5 and len(bg_magnitudes) > 5:
                        mean_rg_mag = np.mean(rg_magnitudes)
                        mean_bg_mag = np.mean(bg_magnitudes)
                        
                        # Blue should typically be refracted more (larger offset)
                        # But if R-G is consistently larger, suspicious
                        if mean_rg_mag > mean_bg_mag * 1.5:
                            violations.append(
                                f"Unusual CA magnitude relationship (R-G: {mean_rg_mag:.2f}, B-G: {mean_bg_mag:.2f}) - "
                                f"red refracted more than blue contradicts n_blue > n_red"
                            )
                            score *= 0.7

        # Apply confidence factor (low confidence for resized/re-encoded images)
        score = max(0.0, min(1.0, score * confidence_factor))
        
        # Add confidence warning if low resolution
        if is_low_resolution:
            violations.append(
                f"CA test low-confidence: image likely resized/re-encoded "
                f"(current: {current_resolution}, original: {original_min_res}) - "
                f"subpixel CA cues may be destroyed"
            )
            # Don't penalize score further, just note low confidence

        score = max(0.0, min(1.0, score))

        if not violations:
            violations.append("Passes chromatic aberration test")

        return OpticsTestResult(
            score=score,
            violations=violations,
            diagnostic_data={
                "rg_offsets": rg_offsets,
                "bg_offsets": bg_offsets,
                "rg_radial_distances": rg_radial_distances,
                "bg_radial_distances": bg_radial_distances,
                "test_confidence": test_confidence,
                "can_test_radial": can_test_radial,
                "original_resolution": original_resolution,
                "current_resolution": (h, w),
                "mean_rg_offset": mean_rg_offset,
                "mean_bg_offset": mean_bg_offset,
                "is_radial": is_radial,  # True if CA is radial, False if tangential, None if can't test
                # Proportionality fit results (None if not computed)
                "rg_slope": proportionality_fits['rg_slope'],
                "rg_intercept": proportionality_fits['rg_intercept'],
                "rg_r_squared": proportionality_fits['rg_r_squared'],
                "bg_slope": proportionality_fits['bg_slope'],
                "bg_intercept": proportionality_fits['bg_intercept'],
                "bg_r_squared": proportionality_fits['bg_r_squared'],
            },
        )


# ============================================================================
# MAIN DETECTOR: OPTICS CONSISTENCY DETECTOR
# ============================================================================
# Combines all five tests to produce an overall optics consistency score.
@pydantic_dataclass
class OpticsConsistencyDetector:
    """Main detector for optics consistency analysis.

    Combines all four tests to produce an overall optics consistency score.
    """

    frequency_weight: float = Field(default=0.25, ge=0.0, le=1.0)
    edge_psf_weight: float = Field(default=0.2, ge=0.0, le=1.0)
    dof_weight: float = Field(default=0.2, ge=0.0, le=1.0)
    ca_weight: float = Field(default=0.15, ge=0.0, le=1.0)
    noise_residual_weight: float = Field(default=0.2, ge=0.0, le=1.0)
    
    # Gradient cache to avoid recomputing gradients on the same image
    _gradient_cache: dict = Field(default_factory=dict, repr=False)

    def _get_cached_gradient(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get cached gradient or compute and cache it.
        
        Args:
            image: Grayscale image (H, W), float32 [0, 1]
            
        Returns:
            Tuple of (gy, gx) gradient arrays
        """
        # Create a simple hash of the image for caching
        # Use image shape and a sample of pixel values as key
        image_hash = hash((image.shape, image.dtype, tuple(image.flat[:100])))
        
        if image_hash in self._gradient_cache:
            logger.debug("  Using cached gradient")
            return self._gradient_cache[image_hash]
        
        # Compute gradient
        logger.debug("  Computing gradient (caching for reuse)")
        gy, gx = np.gradient(image)
        self._gradient_cache[image_hash] = (gy, gx)
        
        # Limit cache size to prevent memory issues
        if len(self._gradient_cache) > 10:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self._gradient_cache))
            del self._gradient_cache[oldest_key]
        
        return gy, gx

    def __post_init__(self):
        """Initialize test components and normalize weights."""
        self.frequency_test = FrequencyDomainOpticsTest()
        self.edge_psf_test = EdgePSFTest()
        self.dof_test = DepthOfFieldConsistencyTest()
        self.ca_test = ChromaticAberrationTest()
        self.noise_residual_test = SensorNoiseResidualTest()

        # Normalize weights
        total_weight = (
            self.frequency_weight
            + self.edge_psf_weight
            + self.dof_weight
            + self.ca_weight
            + self.noise_residual_weight
        )
        if total_weight > 0:
            self.frequency_weight /= total_weight
            self.edge_psf_weight /= total_weight
            self.dof_weight /= total_weight
            self.ca_weight /= total_weight
            self.noise_residual_weight /= total_weight

    def analyze(
        self, image_path: str, load_rgb: bool = True
    ) -> OpticsConsistencyResult:
        """
        Analyze image for optics consistency.

        Args:
            image_path: Path to image file
            load_rgb: If True, load RGB for CA test; if False, use grayscale only

        Returns:
            OpticsConsistencyResult with scores and explanations
        """
        logger.info(f"Analyzing optics consistency: {image_path}")

        # Load and preprocess image
        logger.info("Step 1/6: Loading and preprocessing image...")
        preprocessor = ImagePreprocessor(target_size=512, normalize_to_float=True)
        preprocessed = preprocessor.preprocess(image_path)
        logger.info(f"  ✓ Image preprocessed: shape={preprocessed.shape}")

        # Load RGB if needed for CA test
        image_rgb = None
        original_resolution = None
        if load_rgb:
            logger.info("Step 2/6: Loading RGB image...")
            try:
                from PIL import Image

                img = Image.open(image_path)
                # Store original resolution BEFORE resizing (needed for CA test)
                original_resolution = (img.height, img.width)
                logger.debug(f"Original image resolution: {original_resolution}")
                
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img_array = np.array(img, dtype=np.float32) / 255.0

                # Resize to match preprocessed size
                from PIL import Image as PILImage

                img_pil = PILImage.fromarray((img_array * 255).astype(np.uint8))
                img_pil = img_pil.resize((512, 512), PILImage.Resampling.LANCZOS)
                image_rgb = np.array(img_pil, dtype=np.float32) / 255.0
                logger.info(f"  ✓ RGB image loaded: shape={image_rgb.shape}, original={original_resolution}")
            except Exception as e:
                logger.warning(f"Failed to load RGB image: {e}")
                image_rgb = None

        # Run all tests
        logger.info("Step 3/6: Running frequency domain test...")
        frequency_result = self.frequency_test.test(preprocessed)
        logger.info(f"  ✓ Frequency test complete: score={frequency_result.score:.4f}")

        # Run CA test early if RGB available (needed for edge PSF test CA integration)
        ca_result = None
        ca_is_radial = None
        if image_rgb is not None:
            logger.info("Step 4/6: Running chromatic aberration test (this may take a moment)...")
            ca_result = self.ca_test.test(image_rgb, original_resolution=original_resolution)
            logger.info(f"  ✓ CA test complete: score={ca_result.score:.4f}")
            # Extract CA radial alignment result from diagnostic data
            ca_is_radial = ca_result.diagnostic_data.get('is_radial', None)

        logger.info("Step 5/6: Running edge PSF test...")
        # Pass CA radial alignment result to edge PSF test for integrated scoring
        edge_psf_result = self.edge_psf_test.test(preprocessed, ca_is_radial=ca_is_radial)
        logger.info(f"  ✓ Edge PSF test complete: score={edge_psf_result.score:.4f}")

        logger.info("Step 6/6: Running DOF consistency test (this may take a moment)...")
        # OPTIMIZED: Precompute gradient once and cache it (for potential reuse)
        self._get_cached_gradient(preprocessed)  # Precompute and cache
        # Note: DOF test will use its own gradient computation for now
        # Future: Refactor to pass gradient_cache to test methods
        dof_result = self.dof_test.test(preprocessed)
        logger.info(f"  ✓ DOF test complete: score={dof_result.score:.4f}")

        if image_rgb is not None:
            if ca_result is None:
                # CA test not run yet (shouldn't happen, but handle gracefully)
                logger.info("Step 6/6: Running chromatic aberration test (this may take a moment)...")
                ca_result = self.ca_test.test(image_rgb, original_resolution=original_resolution)
                logger.info(f"  ✓ CA test complete: score={ca_result.score:.4f}")
                ca_is_radial = ca_result.diagnostic_data.get('is_radial', None)
            
            logger.info("Step 7/7: Running noise residual test (this may take a moment)...")
            noise_residual_result = self.noise_residual_test.test(image_rgb)
            logger.info(f"  ✓ Noise residual test complete: score={noise_residual_result.score:.4f}")
        else:
            if ca_result is None:
                ca_result = OpticsTestResult(
                    score=0.5,
                    violations=["RGB image required for CA test"],
                    diagnostic_data={},
                )
            noise_residual_result = OpticsTestResult(
                score=0.5,
                violations=["RGB image required for noise residual test"],
                diagnostic_data={},
            )

        # Compute weighted overall score
        optics_score = (
            frequency_result.score * self.frequency_weight
            + edge_psf_result.score * self.edge_psf_weight
            + dof_result.score * self.dof_weight
            + ca_result.score * self.ca_weight
            + noise_residual_result.score * self.noise_residual_weight
        )

        # Generate explanation
        all_violations = []
        if frequency_result.violations:
            all_violations.extend(
                [f"Frequency: {v}" for v in frequency_result.violations if "Passes" not in v]
            )
        if edge_psf_result.violations:
            all_violations.extend(
                [f"PSF: {v}" for v in edge_psf_result.violations if "Passes" not in v]
            )
        if dof_result.violations:
            all_violations.extend(
                [f"DOF: {v}" for v in dof_result.violations if "Passes" not in v]
            )
        if ca_result.violations:
            all_violations.extend(
                [f"CA: {v}" for v in ca_result.violations if "Passes" not in v]
            )
        if noise_residual_result.violations:
            all_violations.extend(
                [f"Noise: {v}" for v in noise_residual_result.violations if "Passes" not in v]
            )

        if all_violations:
            explanation = "; ".join(all_violations)
        else:
            explanation = "Passes all optics consistency tests"

        return OpticsConsistencyResult(
            optics_score=optics_score,
            frequency_test=frequency_result,
            edge_psf_test=edge_psf_result,
            dof_consistency_test=dof_result,
            chromatic_aberration_test=ca_result,
            noise_residual_test=noise_residual_result,
            explanation=explanation,
        )

