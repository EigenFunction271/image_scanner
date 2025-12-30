"""Optics Consistency Detector - Physical optical law validation.

This module implements detectors that check if an image follows physical
optical laws expected from real cameras:
1. Monotonic low-pass Optical Transfer Function (OTF)
2. Blur occurs before detail and noise injection
3. Depth-of-field blur varies continuously with depth
4. Chromatic aberration is small but non-zero and spatially coherent
"""

import logging
from typing import List, NamedTuple, Tuple

import cv2
import numpy as np
from pydantic import Field
from pydantic.dataclasses import dataclass as pydantic_dataclass
from scipy import signal

from image_screener.dft import DFTProcessor
from image_screener.preprocessing import ImagePreprocessor

logger = logging.getLogger(__name__)


class OpticsTestResult(NamedTuple):
    """Result from a single optics test."""

    score: float  # 0.0 (fails) to 1.0 (passes)
    violations: List[str]  # List of detected violations
    diagnostic_data: dict  # Additional data for visualization


class OpticsConsistencyResult(NamedTuple):
    """Complete optics consistency analysis result."""

    optics_score: float  # Overall score (0.0-1.0)
    frequency_test: OpticsTestResult
    edge_psf_test: OpticsTestResult
    dof_consistency_test: OpticsTestResult
    chromatic_aberration_test: OpticsTestResult
    explanation: str  # Human-readable explanation


@pydantic_dataclass
class FrequencyDomainOpticsTest:
    """Test 1: Frequency-domain optics test.

    Checks for monotonic OTF decay by analyzing radial power spectrum.
    Real cameras apply a monotonic low-pass OTF, so the power spectrum
    should decay smoothly without mid-frequency bumps or high-frequency suppression.
    Also checks for missing stochastic noise floor (over-clean spectra).
    """

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
        # Focus on high frequencies (radii > 0.7)
        high_freq_mask = radii > 0.7
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
        center_y, center_x = h // 2, w // 2

        # Create mask for high-frequency region in 2D
        y_coords, x_coords = np.ogrid[:h, :w]
        distances = np.sqrt((y_coords - center_y) ** 2 + (x_coords - center_x) ** 2)
        max_distance = np.sqrt(center_y ** 2 + center_x ** 2)
        normalized_distances = distances / max_distance

        high_freq_2d_mask = normalized_distances > 0.7
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
        valid_mask = (radii > 0.1) & (radii < 0.9)
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
        bump_threshold = np.std(residuals) * 1.5
        bumps = np.sum(residuals > bump_threshold)
        bump_ratio = bumps / len(residuals)

        # Detect high-frequency suppression (negative residuals at high freq)
        high_freq_mask = radii[valid_mask] > 0.7
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

        if slope > -0.1:  # Should be clearly negative
            violations.append("Non-monotonic OTF decay (slope too shallow)")
            score *= 0.5

        if bump_ratio > 0.1:  # More than 10% of points are bumps
            violations.append(f"Mid-frequency bumps detected ({bump_ratio:.1%})")
            score *= max(0.3, 1.0 - bump_ratio * 2)

        if suppression_ratio > 0.15:
            violations.append(
                f"High-frequency suppression detected ({suppression_ratio:.1%})"
            )
            score *= max(0.3, 1.0 - suppression_ratio * 2)

        # Check smoothness of decay (low variance in residuals)
        residual_std = np.std(residuals)
        if residual_std > 0.5:  # High variance indicates non-smooth decay
            violations.append("Non-smooth power spectrum decay")
            score *= max(0.5, 1.0 - (residual_std - 0.3) * 0.5)

        # Test for missing stochastic noise floor (over-clean spectra)
        # Real images have sensor noise that appears as variance in high frequencies
        # AI-generated images may be too clean (near-zero variance)
        high_freq_variance = self._estimate_high_frequency_variance(
            log_magnitude, radii, radial_power
        )
        expected_noise_floor = self._estimate_expected_noise_floor(image)
        
        if high_freq_variance < expected_noise_floor * 0.1:  # Less than 10% of expected
            violations.append(
                f"Missing noise floor detected (variance: {high_freq_variance:.4f}, "
                f"expected: {expected_noise_floor:.4f})"
            )
            # Penalize more severely for very clean spectra
            variance_ratio = high_freq_variance / (expected_noise_floor + 1e-10)
            score *= max(0.2, variance_ratio * 2)  # Scale penalty with how clean it is

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
            },
        )


@pydantic_dataclass
class EdgePSFTest:
    """Test 2: Edge Spread Function / Point Spread Function test.

    Detects strong edges, extracts Edge Spread Functions (ESF),
    differentiates to get Line Spread Function (LSF), and flags
    ringing, negative lobes, or inconsistent PSF width.
    """

    edge_threshold: float = Field(default=0.1, gt=0.0)
    min_edge_length: int = Field(default=20, gt=0)

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

    def test(self, image: np.ndarray) -> OpticsTestResult:
        """
        Test edge spread function consistency.

        Args:
            image: Grayscale image (H, W), float32 [0, 1]

        Returns:
            OpticsTestResult with score and violations
        """
        logger.debug("Running edge PSF test")

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

            if has_left_negative and has_right_negative:
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

        # Check for negative lobes in LSF (using relative threshold)
        # Normalize LSF to peak=1.0, then check for negative values relative to peak
        # This adapts to different ISO noise levels
        negative_lobe_scores = []
        for esf in esf_samples:
            lsf = self._compute_lsf(esf, normalize=True)  # Normalized to peak=1.0

            if len(lsf) == 0:
                continue

            # Use relative threshold: -0.05 * max(LSF) = -0.05 (since normalized)
            # This adapts to image brightness/ISO noise levels
            negative_threshold = -0.05  # Relative to normalized peak
            negative_ratio = np.sum(lsf < negative_threshold) / len(lsf)
            negative_lobe_scores.append(negative_ratio)

        avg_negative = np.mean(negative_lobe_scores) if negative_lobe_scores else 0.0
        if avg_negative > 0.15:  # More than 15% negative indicates non-physical PSF
            violations.append(
                f"Negative lobes in LSF detected ({avg_negative:.1%}, threshold: -5% of peak)"
            )
            score *= max(0.4, 1.0 - avg_negative * 2)

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
            },
        )


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

        impossible_patterns = 0
        for i in range(len(sample_coords) - 2):
            for j in range(i + 1, len(sample_coords) - 1):
                for k in range(j + 1, len(sample_coords)):
                    # Get distances and blur for triplet
                    dist_ij = np.linalg.norm(sample_coords[i] - sample_coords[j])
                    dist_jk = np.linalg.norm(sample_coords[j] - sample_coords[k])
                    dist_ik = np.linalg.norm(sample_coords[i] - sample_coords[k])

                    blur_i, blur_j, blur_k = (
                        sample_blur[i],
                        sample_blur[j],
                        sample_blur[k],
                    )

                    # Check for impossible pattern: if i and k are close spatially
                    # but have very different blur, while j (between them) has opposite blur
                    # This violates monotonicity
                    if dist_ik < dist_ij and dist_ik < dist_jk:
                        # i and k are close, j is farther
                        # Expected: blur(i) ≈ blur(k), and both different from blur(j)
                        blur_diff_ik = abs(blur_i - blur_k)
                        blur_diff_ij = abs(blur_i - blur_j)
                        blur_diff_jk = abs(blur_j - blur_k)

                        # Impossible: i and k close but very different blur,
                        # while j (farther) has blur similar to one of them
                        if (
                            blur_diff_ik > max(blur_diff_ij, blur_diff_jk) * 1.5
                            and min(blur_diff_ij, blur_diff_jk) < blur_diff_ik * 0.5
                        ):
                            impossible_patterns += 1

        if impossible_patterns > len(sample_coords) * 0.1:  # >10% of triplets
            violations.append(
                f"Non-monotonic blur patterns detected ({impossible_patterns} violations) - "
                f"inconsistent with thin lens equation R ∝ |D - D_focus|"
            )

        return violations

    def estimate_local_blur(self, image: np.ndarray, y: int, x: int) -> float:
        """
        Estimate local blur radius at a point using edge-based methods.

        Only estimates blur at edges to ensure content-independence.
        Uses gradient kurtosis and edge width measurements.

        Args:
            image: Grayscale image
            y, x: Coordinates

        Returns:
            Estimated blur radius in pixels, or NaN if no edges found
        """
        h, w = image.shape
        half_window = self.blur_window_size // 2

        y_min = max(0, y - half_window)
        y_max = min(h, y + half_window + 1)
        x_min = max(0, x - half_window)
        x_max = min(w, x + half_window + 1)

        patch = image[y_min:y_max, x_min:x_max]

        if patch.size == 0:
            return np.nan

        # Convert patch to uint8 for edge detection
        patch_uint8 = (patch * 255).astype(np.uint8)

        # Detect edges in the patch
        edges = cv2.Canny(patch_uint8, 50, 150)

        # If no edges found, cannot estimate blur (content-dependent)
        if np.sum(edges) == 0:
            return np.nan

        # Method 1: Gradient kurtosis at edge pixels
        # Sharp edges have high kurtosis (peaked distribution)
        # Blurry edges have low kurtosis (flatter distribution)
        gy, gx = np.gradient(patch)
        gradient_mag = np.sqrt(gx**2 + gy**2)

        # Only consider gradients at edge pixels
        edge_gradients = gradient_mag[edges > 0]

        if len(edge_gradients) < 3:
            return np.nan

        # Compute kurtosis of edge gradients
        # High kurtosis = sharp (peaked), low kurtosis = blurry (flat)
        from scipy import stats

        try:
            kurtosis = stats.kurtosis(edge_gradients, fisher=True)  # Fisher's definition
            # Convert kurtosis to blur radius
            # Sharp edges: kurtosis > 0, blurry edges: kurtosis < 0
            # Map to blur radius: kurtosis of 2 = sharp (blur ~0), kurtosis of -1 = blurry (blur ~5)
            blur_from_kurtosis = max(0.0, 5.0 - (kurtosis + 1.0) * 2.0)
        except Exception:
            blur_from_kurtosis = np.nan

        # Method 2: Edge width measurement (ESF-based)
        # Extract edge profiles and measure width
        edge_coords = np.column_stack(np.where(edges > 0))
        edge_widths = []

        # Sample a few edges for width measurement
        sample_size = min(5, len(edge_coords))
        sample_indices = np.linspace(0, len(edge_coords) - 1, sample_size, dtype=int)

        for idx in sample_indices:
            ey, ex = edge_coords[idx]
            ey_global = y_min + ey
            ex_global = x_min + ex

            # Get edge orientation
            if (
                ey > 0
                and ey < patch.shape[0] - 1
                and ex > 0
                and ex < patch.shape[1] - 1
            ):
                gy_local = patch[ey + 1, ex] - patch[ey - 1, ex]
                gx_local = patch[ey, ex + 1] - patch[ey, ex - 1]

                if abs(gx_local) < 1e-6 and abs(gy_local) < 1e-6:
                    continue

                # Perpendicular direction
                perp_x = -gy_local
                perp_y = gx_local
                norm = np.sqrt(perp_x**2 + perp_y**2)
                if norm < 1e-6:
                    continue

                perp_x /= norm
                perp_y /= norm

                # Extract profile perpendicular to edge
                profile_length = 21
                profile = []

                for i in range(profile_length):
                    offset = i - profile_length // 2
                    py = ey_global + offset * perp_y
                    px = ex_global + offset * perp_x

                    if 0.5 <= py < h - 0.5 and 0.5 <= px < w - 0.5:
                        # Bilinear interpolation
                        y0 = int(np.floor(py))
                        y1 = min(y0 + 1, h - 1)
                        x0 = int(np.floor(px))
                        x1 = min(x0 + 1, w - 1)

                        dy = py - y0
                        dx = px - x0

                        val = (
                            image[y0, x0] * (1 - dx) * (1 - dy)
                            + image[y0, x1] * dx * (1 - dy)
                            + image[y1, x0] * (1 - dx) * dy
                            + image[y1, x1] * dx * dy
                        )
                        profile.append(val)
                    else:
                        break

                if len(profile) == profile_length:
                    profile = np.array(profile)
                    # Smooth profile
                    try:
                        profile_smooth = signal.savgol_filter(
                            profile, window_length=5, polyorder=2
                        )
                    except Exception:
                        profile_smooth = profile

                    # Measure edge width (10%-90% rise distance)
                    profile_min = np.min(profile_smooth)
                    profile_max = np.max(profile_smooth)
                    profile_range = profile_max - profile_min

                    if profile_range > 0.1:  # Significant edge
                        threshold_10 = profile_min + 0.1 * profile_range
                        threshold_90 = profile_min + 0.9 * profile_range

                        idx_10 = np.where(profile_smooth >= threshold_10)[0]
                        idx_90 = np.where(profile_smooth >= threshold_90)[0]

                        if len(idx_10) > 0 and len(idx_90) > 0:
                            width = idx_90[0] - idx_10[0]
                            if width > 0:
                                edge_widths.append(width)

        # Combine methods: prefer edge width if available, fallback to kurtosis
        if len(edge_widths) > 0:
            blur_radius = np.median(edge_widths)
        elif not np.isnan(blur_from_kurtosis):
            blur_radius = blur_from_kurtosis
        else:
            return np.nan

        return min(blur_radius, 10.0)  # Cap at reasonable value

    def test(self, image: np.ndarray) -> OpticsTestResult:
        """
        Test depth-of-field consistency.

        Args:
            image: Grayscale image (H, W), float32 [0, 1]

        Returns:
            OpticsTestResult with score and violations
        """
        logger.debug("Running depth-of-field consistency test")

        h, w = image.shape

        # Sample points on a grid
        grid_spacing = max(10, min(h, w) // 20)
        y_coords = np.arange(0, h, grid_spacing)
        x_coords = np.arange(0, w, grid_spacing)

        blur_map = np.zeros((len(y_coords), len(x_coords)))
        blur_map[:] = np.nan  # Initialize with NaN

        for i, y in enumerate(y_coords):
            for j, x in enumerate(x_coords):
                blur_est = self.estimate_local_blur(image, int(y), int(x))
                if not np.isnan(blur_est):
                    blur_map[i, j] = blur_est

        # Filter out NaN values for analysis
        valid_blur = blur_map[~np.isnan(blur_map)]

        if len(valid_blur) < 5:
            return OpticsTestResult(
                score=0.5,
                violations=["Insufficient edges for DOF analysis"],
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

        if max_gradient > 2.0:  # Large jumps indicate discrete blur regions (AI artifact)
            violations.append(
                f"Discrete blur regions detected (max gradient: {max_gradient:.2f} pixels/grid) - "
                f"non-physical transition consistent with AI segmentation+blur pipeline"
            )
            # Strong penalty for discrete jumps (strong AI indicator)
            score *= max(0.3, 1.0 - (max_gradient - 1.5) * 0.15)

        # MEAN GRADIENT TEST: Overall smoothness check
        # Real DOF: mean gradient typically < 0.3 (smooth variation)
        # AI-generated: higher mean gradient from artificial boundaries
        if mean_gradient > 0.5:
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
        # Real blur: High entropy (subtle variations across space)
        # AI blur: Low entropy (uniform regions with identical values)
        from scipy.stats import entropy as scipy_entropy

        # Compute local entropy using sliding window
        # High entropy = varied blur (natural), low entropy = uniform blur (AI)
        local_entropy_values = []
        window_size = 3  # 3×3 window for local entropy

        for i in range(1, blur_map.shape[0] - 1):
            for j in range(1, blur_map.shape[1] - 1):
                if valid_mask[i, j]:
                    # Extract local window
                    window = blur_map[
                        max(0, i - 1) : min(blur_map.shape[0], i + 2),
                        max(0, j - 1) : min(blur_map.shape[1], j + 2),
                    ]
                    window_valid = window[~np.isnan(window)]

                    if len(window_valid) >= 3:
                        # Compute histogram entropy
                        hist, _ = np.histogram(
                            window_valid, bins=min(10, len(window_valid)), density=True
                        )
                        hist = hist[hist > 0]  # Remove zeros
                        if len(hist) > 0:
                            local_ent = scipy_entropy(hist)
                            local_entropy_values.append(local_ent)

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
            score *= 0.4  # Strong penalty for physics violations

        score = max(0.0, min(1.0, score))

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
            },
        )


@pydantic_dataclass
class ChromaticAberrationTest:
    """Test 4: Chromatic aberration test.

    Computes edge offsets between R/G/B channels and checks for
    radial consistency and non-zero magnitude. Real cameras have
    small but non-zero CA that varies radially.
    """

    edge_threshold: float = Field(default=0.1, gt=0.0)

    def test(self, image_rgb: np.ndarray) -> OpticsTestResult:
        """
        Test chromatic aberration consistency.

        Args:
            image_rgb: RGB image (H, W, 3), float32 [0, 1]

        Returns:
            OpticsTestResult with score and violations
        """
        logger.debug("Running chromatic aberration test")

        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            return OpticsTestResult(
                score=0.5,
                violations=["Requires RGB image for CA test"],
                diagnostic_data={},
            )

        h, w = image_rgb.shape[:2]
        center_y, center_x = h // 2, w // 2

        # Extract R, G, B channels
        r_channel = image_rgb[:, :, 0]
        g_channel = image_rgb[:, :, 1]
        b_channel = image_rgb[:, :, 2]

        # Convert to grayscale for edge detection
        r_gray = (r_channel * 255).astype(np.uint8)
        g_gray = (g_channel * 255).astype(np.uint8)
        b_gray = (b_channel * 255).astype(np.uint8)

        # Detect edges in each channel
        edges_r = cv2.Canny(r_gray, 50, 150)
        edges_g = cv2.Canny(g_gray, 50, 150)
        edges_b = cv2.Canny(b_gray, 50, 150)

        # Find edge pixels in green channel (reference)
        edge_coords = np.column_stack(np.where(edges_g > 0))

        if len(edge_coords) < 20:
            return OpticsTestResult(
                score=0.5,
                violations=["Insufficient edges for CA analysis"],
                diagnostic_data={},
            )

        # Sample edges and compute offsets
        sample_indices = np.linspace(
            0, len(edge_coords) - 1, min(100, len(edge_coords)), dtype=int
        )

        rg_offsets = []
        bg_offsets = []
        radial_distances = []
        rg_offset_vectors = []  # Store tuples: (offset_vec, (y, x)) for radial alignment check
        bg_offset_vectors = []  # Store tuples: (offset_vec, (y, x)) for radial alignment check

        for idx in sample_indices:
            y, x = edge_coords[idx]

            # Compute radial distance from center
            dy = y - center_y
            dx = x - center_x
            radial_dist = np.sqrt(dx**2 + dy**2)
            radial_distances.append(radial_dist)

            # Check for corresponding edge in R and B channels
            # Look in small neighborhood
            search_radius = 3
            y_min = max(0, y - search_radius)
            y_max = min(h, y + search_radius + 1)
            x_min = max(0, x - search_radius)
            x_max = min(w, x + search_radius + 1)

            r_patch = edges_r[y_min:y_max, x_min:x_max]
            b_patch = edges_b[y_min:y_max, x_min:x_max]

            # Find closest edge pixel
            r_coords = np.column_stack(np.where(r_patch > 0))
            b_coords = np.column_stack(np.where(b_patch > 0))

            # Store offset vectors with corresponding edge locations
            if len(r_coords) > 0:
                # Offset relative to patch center
                r_offset_y = r_coords[0, 0] - search_radius
                r_offset_x = r_coords[0, 1] - search_radius
                rg_offset = np.sqrt(r_offset_y**2 + r_offset_x**2)
                rg_offsets.append(rg_offset)
                
                # Store offset vector with edge location for radial alignment check
                if rg_offset > 0:
                    offset_vec = np.array([r_offset_x, r_offset_y])
                    rg_offset_vectors.append((offset_vec, (y, x)))
                else:
                    rg_offset_vectors.append((np.array([0.0, 0.0]), (y, x)))
            else:
                # No R edge found, but store location for consistency
                rg_offset_vectors.append((np.array([0.0, 0.0]), (y, x)))

            if len(b_coords) > 0:
                b_offset_y = b_coords[0, 0] - search_radius
                b_offset_x = b_coords[0, 1] - search_radius
                bg_offset = np.sqrt(b_offset_y**2 + b_offset_x**2)
                bg_offsets.append(bg_offset)
                
                # Store offset vector with edge location for radial alignment check
                if bg_offset > 0:
                    offset_vec = np.array([b_offset_x, b_offset_y])
                    bg_offset_vectors.append((offset_vec, (y, x)))
                else:
                    bg_offset_vectors.append((np.array([0.0, 0.0]), (y, x)))
            else:
                # No B edge found, but store location for consistency
                bg_offset_vectors.append((np.array([0.0, 0.0]), (y, x)))

        if len(rg_offsets) < 5 or len(bg_offsets) < 5:
            return OpticsTestResult(
                score=0.5,
                violations=["Insufficient CA offset measurements"],
                diagnostic_data={},
            )

        violations = []
        score = 1.0

        # Check for zero CA (suspicious - real cameras have some CA)
        mean_rg_offset = np.mean(rg_offsets)
        mean_bg_offset = np.mean(bg_offsets)

        if mean_rg_offset < 0.1 and mean_bg_offset < 0.1:
            violations.append("Zero chromatic aberration detected (suspicious)")
            score *= 0.3

        # Check for radial consistency
        # CA should increase with distance from center
        if len(radial_distances) == len(rg_offsets):
            # Fit linear relationship: offset = a * radius + b
            try:
                rg_slope, _ = np.polyfit(radial_distances[: len(rg_offsets)], rg_offsets, deg=1)
                if rg_slope < -0.01:  # Negative slope is non-physical
                    violations.append("Non-physical CA radial variation (negative slope)")
                    score *= 0.5
            except Exception:
                pass

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

        # RADIAL ALIGNMENT TEST: Lateral CA must be radial (toward/away from center)
        #
        # PHYSICS: Real lateral CA is radial - the offset vector (Δx, Δy) must point
        # directly toward or away from the image center (c_x, c_y).
        #
        # FORENSIC TEST: Calculate alignment = (Offset · Radius) / (|Offset| |Radius|)
        # - Alignment ≈ ±1: Radial (correct, physical)
        # - Alignment ≈ 0: Tangential (non-physical, AI artifact)
        #
        alignments_rg = []
        alignments_bg = []
        
        # Check R-G alignment
        for offset_vec, (y, x) in rg_offset_vectors:
            if offset_vec.size == 2 and np.linalg.norm(offset_vec) > 0.01:
                # Compute radial vector from center to edge location
                dy = y - center_y
                dx = x - center_x
                radial_vec = np.array([dx, dy])
                radial_vec_norm = np.linalg.norm(radial_vec)
                
                if radial_vec_norm > 0.01:
                    # Normalize both vectors
                    radial_vec_normalized = radial_vec / radial_vec_norm
                    offset_vec_normalized = offset_vec / np.linalg.norm(offset_vec)
                    
                    # Compute alignment: dot product (should be ±1 for radial)
                    # Alignment = (Offset · Radius) / (|Offset| |Radius|)
                    alignment = np.dot(offset_vec_normalized, radial_vec_normalized)
                    alignments_rg.append(alignment)
        
        # Check B-G alignment
        for offset_vec, (y, x) in bg_offset_vectors:
            if offset_vec.size == 2 and np.linalg.norm(offset_vec) > 0.01:
                dy = y - center_y
                dx = x - center_x
                radial_vec = np.array([dx, dy])
                radial_vec_norm = np.linalg.norm(radial_vec)
                
                if radial_vec_norm > 0.01:
                    radial_vec_normalized = radial_vec / radial_vec_norm
                    offset_vec_normalized = offset_vec / np.linalg.norm(offset_vec)
                    alignment = np.dot(offset_vec_normalized, radial_vec_normalized)
                    alignments_bg.append(alignment)
        
        # Check alignment scores
        # Real CA: alignment close to ±1 (radial)
        # AI/fake CA: alignment close to 0 (tangential/sideways)
        if len(alignments_rg) > 5:
            mean_alignment_rg = np.mean(np.abs(alignments_rg))
            # If mean alignment < 0.7, CA is too tangential (non-physical)
            if mean_alignment_rg < 0.7:
                violations.append(
                    f"Non-radial CA detected (R-G alignment: {mean_alignment_rg:.2f}) - "
                    f"offset vectors are tangential (sideways), not radial - AI artifact"
                )
                score *= max(0.3, mean_alignment_rg / 0.7)
        
        if len(alignments_bg) > 5:
            mean_alignment_bg = np.mean(np.abs(alignments_bg))
            if mean_alignment_bg < 0.7:
                violations.append(
                    f"Non-radial CA detected (B-G alignment: {mean_alignment_bg:.2f}) - "
                    f"offset vectors are tangential (sideways), not radial - AI artifact"
                )
                score *= max(0.3, mean_alignment_bg / 0.7)

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
                    
                    # Normalize vectors
                    rg_vec_norm = rg_vec / np.linalg.norm(rg_vec)
                    bg_vec_norm = bg_vec / np.linalg.norm(bg_vec)
                    
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

        score = max(0.0, min(1.0, score))

        if not violations:
            violations.append("Passes chromatic aberration test")

        return OpticsTestResult(
            score=score,
            violations=violations,
            diagnostic_data={
                "rg_offsets": rg_offsets,
                "bg_offsets": bg_offsets,
                "radial_distances": radial_distances[: len(rg_offsets)],
                "mean_rg_offset": mean_rg_offset,
                "mean_bg_offset": mean_bg_offset,
            },
        )


@pydantic_dataclass
class OpticsConsistencyDetector:
    """Main detector for optics consistency analysis.

    Combines all four tests to produce an overall optics consistency score.
    """

    frequency_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    edge_psf_weight: float = Field(default=0.25, ge=0.0, le=1.0)
    dof_weight: float = Field(default=0.25, ge=0.0, le=1.0)
    ca_weight: float = Field(default=0.2, ge=0.0, le=1.0)

    def __post_init__(self):
        """Initialize test components."""
        self.frequency_test = FrequencyDomainOpticsTest()
        self.edge_psf_test = EdgePSFTest()
        self.dof_test = DepthOfFieldConsistencyTest()
        self.ca_test = ChromaticAberrationTest()

        # Normalize weights
        total_weight = (
            self.frequency_weight
            + self.edge_psf_weight
            + self.dof_weight
            + self.ca_weight
        )
        if total_weight > 0:
            self.frequency_weight /= total_weight
            self.edge_psf_weight /= total_weight
            self.dof_weight /= total_weight
            self.ca_weight /= total_weight

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
        preprocessor = ImagePreprocessor(target_size=512, normalize_to_float=True)
        preprocessed = preprocessor.preprocess(image_path)

        # Load RGB if needed for CA test
        image_rgb = None
        if load_rgb:
            try:
                from PIL import Image

                img = Image.open(image_path)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img_array = np.array(img, dtype=np.float32) / 255.0

                # Resize to match preprocessed size
                from PIL import Image as PILImage

                img_pil = PILImage.fromarray((img_array * 255).astype(np.uint8))
                img_pil = img_pil.resize((512, 512), PILImage.Resampling.LANCZOS)
                image_rgb = np.array(img_pil, dtype=np.float32) / 255.0
            except Exception as e:
                logger.warning(f"Failed to load RGB image: {e}")
                image_rgb = None

        # Run all tests
        frequency_result = self.frequency_test.test(preprocessed)
        edge_psf_result = self.edge_psf_test.test(preprocessed)
        dof_result = self.dof_test.test(preprocessed)

        if image_rgb is not None:
            ca_result = self.ca_test.test(image_rgb)
        else:
            ca_result = OpticsTestResult(
                score=0.5,
                violations=["RGB image required for CA test"],
                diagnostic_data={},
            )

        # Compute weighted overall score
        optics_score = (
            frequency_result.score * self.frequency_weight
            + edge_psf_result.score * self.edge_psf_weight
            + dof_result.score * self.dof_weight
            + ca_result.score * self.ca_weight
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
            explanation=explanation,
        )

