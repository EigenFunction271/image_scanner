"""2D Discrete Fourier Transform module for frequency domain analysis."""

import logging
from dataclasses import dataclass
from typing import NamedTuple, Tuple

import numpy as np
from pydantic import Field, field_validator
from pydantic.dataclasses import dataclass as pydantic_dataclass
from scipy import signal
from scipy.ndimage import maximum_filter

logger = logging.getLogger(__name__)

__all__ = ['DFTProcessor', 'SpectralPeak', 'ProcessImageResult']

# Peak detection parameters
MAX_PEAKS_TO_RETAIN = 1000  # Limit peaks to avoid noise and improve performance
MIN_INTERVAL_THRESHOLD = 5  # Minimum pixel interval for grid detection (avoids noise from nearby peaks)
GRID_ALIGNMENT_TOLERANCE = 3  # Pixel tolerance for grid alignment checks
EXPONENTIAL_SCALE_FACTOR = 2.0  # Controls exponential scaling of grid scores when strong patterns detected

# Autocorrelation parameters
PEAK_DETECTION_HEIGHT_FACTOR = 0.3  # Fraction of max for peak detection in histograms
AUTOCORR_REGION_SIZE = 256  # Maximum region size for autocorrelation computation


def get_center_coords(shape: Tuple[int, int]) -> Tuple[int, int]:
    """Get center coordinates for an image shape.
    
    Args:
        shape: (H, W) image shape
        
    Returns:
        (center_y, center_x) tuple
    """
    return shape[0] // 2, shape[1] // 2


class SpectralPeak(NamedTuple):
    """Represents a detected peak in the frequency spectrum."""

    u: int  # Frequency coordinate (horizontal)
    v: int  # Frequency coordinate (vertical)
    magnitude: float  # Log-scaled magnitude at this coordinate
    distance_from_center: float  # Distance from DC component


@dataclass
class ProcessImageResult:
    """Result of processing an image through the DFT pipeline."""

    log_magnitude_spectrum: np.ndarray
    peaks: list[SpectralPeak]
    artifact_score: float
    grid_strength: float
    grid_interval_u: float
    grid_interval_v: float
    azimuthal_radii: np.ndarray
    azimuthal_average: np.ndarray
    grid_consistency: float
    nyquist_symmetry: float


@pydantic_dataclass
class DFTProcessor:
    """Computes 2D DFT and extracts spectral features."""

    sensitivity: float = Field(default=1.0, gt=0.0)
    high_freq_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    peak_threshold_percentile: float = Field(default=98.0, ge=0.0, le=100.0)

    @field_validator("sensitivity")
    @classmethod
    def validate_sensitivity(cls, v: float) -> float:
        """Ensure sensitivity is positive."""
        if v <= 0:
            raise ValueError("Sensitivity must be positive")
        return v

    def compute_dft(self, image: np.ndarray) -> np.ndarray:
        """
        Compute 2D Discrete Fourier Transform of the image.

        Args:
            image: Grayscale image array (H, W), float32 [0, 1]

        Returns:
            Complex-valued FFT result (H, W)

        Raises:
            ValueError: If image is not 2D or is empty
        """
        if image.size == 0:
            raise ValueError("Image array is empty")
        if image.ndim != 2:
            raise ValueError(f"Expected 2D array, got {image.ndim}D array with shape {image.shape}")

        if image.dtype != np.float32:
            logger.warning(f"Converting image from {image.dtype} to float32")
            image = image.astype(np.float32)

        logger.debug(f"Computing 2D FFT for image shape: {image.shape}")
        fft_result = np.fft.fft2(image)
        return fft_result

    def shift_spectrum(self, fft_result: np.ndarray) -> np.ndarray:
        """
        Shift the FFT spectrum to center the DC component.

        Args:
            fft_result: Complex-valued FFT result

        Returns:
            Shifted FFT spectrum with DC at center
        """
        shifted = np.fft.fftshift(fft_result)
        logger.debug("Applied FFT shift to center DC component")
        return shifted

    def compute_magnitude_spectrum(
        self, fft_shifted: np.ndarray, log_scale: bool = True
    ) -> np.ndarray:
        """
        Compute magnitude spectrum, optionally log-scaled.

        Args:
            fft_shifted: Shifted FFT spectrum (complex)
            log_scale: If True, apply log scaling: log(1 + magnitude)

        Returns:
            Magnitude spectrum (real-valued)
        """
        magnitude = np.abs(fft_shifted)

        if log_scale:
            # Use log1p to avoid log(0) and handle small values better
            magnitude = np.log1p(magnitude)
            logger.debug("Applied log scaling to magnitude spectrum")

        return magnitude

    def get_high_frequency_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        """
        Create a mask for high-frequency regions of the spectrum.

        High frequencies are defined as regions beyond high_freq_threshold
        distance from the center (DC component).

        Args:
            shape: Shape of the spectrum (H, W)

        Returns:
            Boolean mask where True indicates high-frequency region
        """
        h, w = shape
        center_y, center_x = get_center_coords(shape)

        # Create coordinate grids
        y_coords, x_coords = np.ogrid[:h, :w]

        # Distance from center (normalized to [0, 1])
        max_distance = np.sqrt(center_y**2 + center_x**2)
        distances = np.sqrt((y_coords - center_y) ** 2 + (x_coords - center_x) ** 2)
        normalized_distances = distances / max_distance

        # High frequency mask: regions beyond threshold
        mask = normalized_distances > self.high_freq_threshold

        logger.debug(
            f"High-frequency mask: {mask.sum()} pixels ({mask.sum() / mask.size * 100:.1f}%) "
            f"beyond threshold {self.high_freq_threshold}"
        )

        return mask

    def detect_peaks(
        self, log_magnitude: np.ndarray, high_freq_mask: np.ndarray
    ) -> list[SpectralPeak]:
        """
        Detect peaks in high-frequency regions of the spectrum.

        Uses local maxima detection combined with thresholding to find
        significant peaks (not just all pixels above threshold).

        Args:
            log_magnitude: Log-scaled magnitude spectrum
            high_freq_mask: Boolean mask for high-frequency regions

        Returns:
            List of detected SpectralPeak objects
        """
        # Extract high-frequency values
        high_freq_values = log_magnitude[high_freq_mask]

        if len(high_freq_values) == 0:
            logger.warning("No high-frequency regions to analyze")
            return []

        # Calculate threshold: percentile * sensitivity
        threshold = np.percentile(high_freq_values, self.peak_threshold_percentile)
        threshold = threshold * self.sensitivity

        logger.debug(
            f"Peak detection threshold: {threshold:.4f} "
            f"(percentile: {self.peak_threshold_percentile}%, sensitivity: {self.sensitivity})"
        )

        h, w = log_magnitude.shape
        center_y, center_x = get_center_coords(log_magnitude.shape)

        # Use scipy's peak detection for 2D arrays (finds local maxima)
        # This is more accurate than simple thresholding
        # Find local maxima in the high-frequency region
        # Use a 3x3 neighborhood for peak detection
        local_maxima = maximum_filter(log_magnitude, size=3) == log_magnitude
        
        # Combine with threshold and high-frequency mask
        peak_mask = (
            local_maxima & 
            high_freq_mask & 
            (log_magnitude > threshold)
        )
        
        # Extract peak coordinates
        peak_coords = np.argwhere(peak_mask)
        
        # Convert to SpectralPeak objects
        peaks = []
        for v, u in peak_coords:
            magnitude = log_magnitude[v, u]
            distance = np.sqrt((v - center_y) ** 2 + (u - center_x) ** 2)
            peaks.append(SpectralPeak(u=u, v=v, magnitude=magnitude, distance_from_center=distance))

        # Sort by magnitude (descending)
        peaks.sort(key=lambda p: p.magnitude, reverse=True)
        
        # Limit to top peaks to avoid noise (keep top MAX_PEAKS_TO_RETAIN or all if fewer)
        if len(peaks) > MAX_PEAKS_TO_RETAIN:
            peaks = peaks[:MAX_PEAKS_TO_RETAIN]
            logger.debug(f"Limited peaks to top {MAX_PEAKS_TO_RETAIN} by magnitude")

        logger.info(f"Detected {len(peaks)} spectral peaks in high-frequency regions")

        return peaks

    def compute_spatial_autocorrelation(
        self, peaks: list[SpectralPeak], image_shape: Tuple[int, int]
    ) -> Tuple[float, float, float]:
        """
        Compute spatial autocorrelation of peak positions to detect grid patterns.

        Grid patterns indicate periodic upsampling artifacts common in AI-generated images.
        This analyzes the distribution of peak positions to find repeating intervals.

        Uses three complementary methods:
        1. Histogram-based interval detection: Finds dominant spacing intervals
           by histogramming pairwise peak distances (vectorized for efficiency).
        2. Grid alignment check: Counts how many peaks align to detected intervals
           within tolerance.
        3. 2D autocorrelation: Computes full autocorrelation of peak map to detect
           periodic patterns (secondary peaks indicate periodicity).

        Args:
            peaks: List of detected spectral peaks
            image_shape: (H, W) shape of the image

        Returns:
            Tuple of (grid_strength, dominant_interval_u, dominant_interval_v)
            - grid_strength: 0.0-1.0, strength of grid pattern (1.0 = perfect grid)
            - dominant_interval_u: Dominant horizontal interval in pixels
            - dominant_interval_v: Dominant vertical interval in pixels
        """
        if len(peaks) < 4:
            # Need at least 4 peaks to detect a grid pattern
            return 0.0, 0.0, 0.0

        h, w = image_shape
        center_y, center_x = get_center_coords(image_shape)

        # Convert peak positions to relative coordinates (centered at origin)
        peak_coords = np.array([(p.u - center_x, p.v - center_y) for p in peaks])

        # Compute pairwise distances and differences using vectorized operations
        n_peaks = len(peak_coords)
        grid_strength = 0.0
        dominant_interval_u = 0.0
        dominant_interval_v = 0.0

        # Method 1: Histogram of distance intervals (OPTIMIZED: vectorized)
        # Use broadcasting to compute all pairwise differences at once
        # Shape: (n_peaks, n_peaks) for u and v differences
        u_coords = peak_coords[:, 0:1]  # Shape: (n_peaks, 1)
        v_coords = peak_coords[:, 1:2]  # Shape: (n_peaks, 1)
        
        # Broadcast to get all pairwise differences
        u_diffs_matrix = np.abs(u_coords - u_coords.T)  # Shape: (n_peaks, n_peaks)
        v_diffs_matrix = np.abs(v_coords - v_coords.T)  # Shape: (n_peaks, n_peaks)
        
        # Extract upper triangle (avoid duplicates and self-comparisons)
        triu_indices = np.triu_indices(n_peaks, k=1)
        intervals_u = u_diffs_matrix[triu_indices]
        intervals_v = v_diffs_matrix[triu_indices]
        
        # Filter significant intervals (avoid noise from nearby peaks)
        intervals_u = intervals_u[intervals_u > MIN_INTERVAL_THRESHOLD]
        intervals_v = intervals_v[intervals_v > MIN_INTERVAL_THRESHOLD]

        if len(intervals_u) > 0 and len(intervals_v) > 0:
            # Find dominant intervals using histogram
            bins_u = np.arange(0, max(intervals_u) + 10, 5)
            bins_v = np.arange(0, max(intervals_v) + 10, 5)
            
            hist_u, _ = np.histogram(intervals_u, bins=bins_u)
            hist_v, _ = np.histogram(intervals_v, bins=bins_v)
            
            # Find peaks in histograms (dominant intervals)
            if len(hist_u) > 1:
                peaks_u_indices = signal.find_peaks(
                    hist_u, height=max(hist_u) * PEAK_DETECTION_HEIGHT_FACTOR
                )[0]
                if len(peaks_u_indices) > 0:
                    dominant_idx_u = peaks_u_indices[np.argmax(hist_u[peaks_u_indices])]
                    dominant_interval_u = bins_u[dominant_idx_u]
            
            if len(hist_v) > 1:
                peaks_v_indices = signal.find_peaks(
                    hist_v, height=max(hist_v) * PEAK_DETECTION_HEIGHT_FACTOR
                )[0]
                if len(peaks_v_indices) > 0:
                    dominant_idx_v = peaks_v_indices[np.argmax(hist_v[peaks_v_indices])]
                    dominant_interval_v = bins_v[dominant_idx_v]

            # Method 2: Check alignment to grid (OPTIMIZED: vectorized)
            # Count how many peaks align to a grid with the dominant intervals
            if dominant_interval_u > 0 and dominant_interval_v > 0:
                # Vectorized alignment check
                u_rel = peak_coords[:, 0]
                v_rel = peak_coords[:, 1]
                
                # Compute modulo for all peaks at once
                u_mod = np.abs(u_rel % dominant_interval_u)
                v_mod = np.abs(v_rel % dominant_interval_v)
                
                # Check alignment (within tolerance or near next grid point)
                u_aligned = (u_mod < GRID_ALIGNMENT_TOLERANCE) | \
                           (np.abs(u_mod - dominant_interval_u) < GRID_ALIGNMENT_TOLERANCE)
                v_aligned = (v_mod < GRID_ALIGNMENT_TOLERANCE) | \
                           (np.abs(v_mod - dominant_interval_v) < GRID_ALIGNMENT_TOLERANCE)
                
                aligned_count = np.sum(u_aligned | v_aligned)
                grid_strength = aligned_count / n_peaks

        # Method 3: Autocorrelation of peak positions
        # Create a binary map of peak positions
        peak_map = np.zeros((h, w), dtype=np.float32)
        for peak in peaks:
            if 0 <= peak.v < h and 0 <= peak.u < w:
                peak_map[peak.v, peak.u] = 1.0

        # Compute 2D autocorrelation (only in a region around center to save computation)
        # Focus on high-frequency region
        region_size = min(AUTOCORR_REGION_SIZE, h // 2, w // 2)
        y_start = max(0, center_y - region_size)
        y_end = min(h, center_y + region_size)
        x_start = max(0, center_x - region_size)
        x_end = min(w, center_x + region_size)

        region = peak_map[y_start:y_end, x_start:x_end]
        
        if np.sum(region) > 3:  # Need at least a few peaks
            # Compute autocorrelation
            autocorr = signal.correlate2d(region, region, mode='same')
            autocorr = autocorr / np.max(autocorr)  # Normalize
            
            # Look for periodic patterns in autocorrelation
            # Check for secondary peaks (indicating periodicity)
            center_autocorr = autocorr[region_size, region_size]
            
            # Find secondary peaks away from center
            autocorr_flat = autocorr.flatten()
            sorted_indices = np.argsort(autocorr_flat)[::-1]
            
            # Count significant secondary peaks (indicating periodicity)
            secondary_peaks = 0
            for idx in sorted_indices[:10]:  # Top 10 peaks
                y_idx, x_idx = np.unravel_index(idx, autocorr.shape)
                dist_from_center = np.sqrt(
                    (y_idx - region_size) ** 2 + (x_idx - region_size) ** 2
                )
                if dist_from_center > 10:  # Away from center
                    peak_value = autocorr_flat[idx]
                    if peak_value > 0.3:  # Significant peak
                        secondary_peaks += 1

            # Grid strength based on secondary peaks
            autocorr_grid_strength = min(secondary_peaks / 5.0, 1.0)
            
            # Combine grid strength measures
            grid_strength = max(grid_strength, autocorr_grid_strength * 0.7)

        logger.debug(
            f"Spatial autocorrelation: grid_strength={grid_strength:.4f}, "
            f"interval_u={dominant_interval_u:.1f}, interval_v={dominant_interval_v:.1f}"
        )

        return grid_strength, dominant_interval_u, dominant_interval_v

    def compute_grid_consistency_score(
        self, peaks: list[SpectralPeak], image_shape: Tuple[int, int]
    ) -> Tuple[float, float]:
        """
        Compute Grid Consistency Score using Nyquist Folding Symmetry analysis.

        This analyzes the spatial distribution of spectral peaks to detect if they
        align with a synthetic lattice (powers of 2: 4, 8, 16) which indicates
        AI upsampling artifacts.

        Args:
            peaks: List of detected spectral peaks
            image_shape: (H, W) shape of the image

        Returns:
            Tuple of (grid_consistency, nyquist_symmetry)
            - grid_consistency: 0.0-1.0, how well peaks align to power-of-2 grid
            - nyquist_symmetry: 0.0-1.0, symmetry across Nyquist boundaries
        """
        if len(peaks) < 4:
            return 0.0, 0.0

        h, w = image_shape
        center_y, center_x = get_center_coords(image_shape)

        # Extract peak coordinates relative to center
        peak_coords = np.array([(p.u - center_x, p.v - center_y) for p in peaks])

        # 1. Grid Consistency: Check if spacing aligns to powers of 2
        u_coords = peak_coords[:, 0]
        v_coords = peak_coords[:, 1]

        # Calculate spacing deltas (distances between sorted coordinates)
        u_sorted = np.sort(np.abs(u_coords))
        v_sorted = np.sort(np.abs(v_coords))

        u_diffs = np.diff(u_sorted)
        v_diffs = np.diff(v_sorted)

        # Filter out zero and very small differences
        u_diffs = u_diffs[u_diffs > 2]
        v_diffs = v_diffs[v_diffs > 2]

        grid_consistency = 0.0

        if len(u_diffs) > 0:
            # Check alignment to powers of 2 (4, 8, 16, 32)
            # AI upsampling typically uses these stride values
            alignment_scores = []
            for stride in [4, 8, 16, 32]:
                # Count how many differences are multiples of this stride
                u_aligned = np.sum((u_diffs % stride) < 2) / len(u_diffs)
                v_aligned = np.sum((v_diffs % stride) < 2) / len(v_diffs)
                alignment_scores.append((u_aligned + v_aligned) / 2.0)

            # Take the best alignment score
            grid_consistency = max(alignment_scores) if alignment_scores else 0.0

        # 2. Nyquist Folding Symmetry
        # Check for conjugate symmetry: peaks should have twins across the origin
        # In AI images, aliasing creates symmetric clusters
        nyquist_symmetry = 0.0

        if len(peaks) > 2:
            # Create a set of peak positions for fast lookup
            peak_positions = {(p.u, p.v) for p in peaks}

            symmetric_pairs = 0
            checked = set()

            for peak in peaks:
                peak_key = (peak.u, peak.v)
                if peak_key in checked:
                    continue

                # Check for symmetric peak: (-u, -v) relative to center
                sym_u = 2 * center_x - peak.u
                sym_v = 2 * center_y - peak.v

                # Check exact symmetry (conjugate symmetry)
                if (sym_u, sym_v) in peak_positions and (sym_u, sym_v) != peak_key:
                    if (sym_u, sym_v) not in checked:
                        symmetric_pairs += 1
                        checked.add((sym_u, sym_v))
                        checked.add(peak_key)
                        continue

                # Check diagonal symmetry (common in upsampling)
                diag_sym_u = center_x - (peak.u - center_x)
                diag_sym_v = center_y - (peak.v - center_y)
                
                if (diag_sym_u, diag_sym_v) in peak_positions and (diag_sym_u, diag_sym_v) != peak_key:
                    if (diag_sym_u, diag_sym_v) not in checked:
                        symmetric_pairs += 1
                        checked.add((diag_sym_u, diag_sym_v))
                        checked.add(peak_key)
                        continue

                checked.add(peak_key)

            # Normalize: symmetric_pairs can be at most len(peaks)/2 (if all peaks have pairs)
            # So we divide by len(peaks)/2 to get a score in [0, 1]
            max_possible_pairs = len(peaks) / 2
            if max_possible_pairs > 0:
                nyquist_symmetry = symmetric_pairs / max_possible_pairs
            else:
                nyquist_symmetry = 0.0
            nyquist_symmetry = min(nyquist_symmetry, 1.0)  # Cap at 1.0

        logger.debug(
            f"Grid consistency: {grid_consistency:.4f}, "
            f"Nyquist symmetry: {nyquist_symmetry:.4f}"
        )

        return grid_consistency, nyquist_symmetry

    def compute_azimuthal_average(
        self, log_magnitude: np.ndarray, num_bins: int = 256
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute azimuthal average of the magnitude spectrum.

        Integrates the magnitude spectrum along circles of different radii.
        This reveals periodic artifacts as sharp spikes in the 1D profile.

        Args:
            log_magnitude: Log-scaled magnitude spectrum (H, W)
            num_bins: Number of radial bins for the average

        Returns:
            Tuple of (radii, azimuthal_average)
            - radii: Radial distances from center (normalized to [0, 1])
            - azimuthal_average: Average magnitude at each radius

        Raises:
            ValueError: If log_magnitude is not 2D, is empty, or num_bins is invalid
        """
        if log_magnitude.size == 0:
            raise ValueError("Log magnitude spectrum array is empty")
        if log_magnitude.ndim != 2:
            raise ValueError(
                f"Expected 2D array for log_magnitude, got {log_magnitude.ndim}D array with shape {log_magnitude.shape}"
            )
        if num_bins <= 0:
            raise ValueError(f"num_bins must be positive, got {num_bins}")

        h, w = log_magnitude.shape
        center_y, center_x = get_center_coords(log_magnitude.shape)

        # Create coordinate grids
        y_coords, x_coords = np.ogrid[:h, :w]
        
        # Distance from center
        distances = np.sqrt((y_coords - center_y) ** 2 + (x_coords - center_x) ** 2)
        
        # Maximum distance (corner to center)
        max_distance = np.sqrt(center_y ** 2 + center_x ** 2)
        
        # Normalize distances to [0, 1]
        normalized_distances = distances / max_distance
        
        # Create radial bins
        bin_edges = np.linspace(0, 1, num_bins + 1)
        radii = (bin_edges[:-1] + bin_edges[1:]) / 2  # Bin centers
        
        # Compute azimuthal average using vectorized binning (OPTIMIZED)
        # Use digitize to assign each pixel to a bin
        bin_indices = np.digitize(normalized_distances.flatten(), bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, num_bins - 1)
        
        # Compute mean for each bin using bincount
        magnitude_flat = log_magnitude.flatten()
        bin_sums = np.bincount(bin_indices, weights=magnitude_flat, minlength=num_bins)
        bin_counts = np.bincount(bin_indices, minlength=num_bins)
        
        # Avoid division by zero and handle empty bins
        azimuthal_avg = bin_sums / (bin_counts + 1e-10)
        
        # Interpolate empty bins (forward fill)
        empty_bins = bin_counts == 0
        if np.any(empty_bins):
            # Find last non-empty value for forward fill
            for i in range(num_bins):
                if empty_bins[i]:
                    if i > 0:
                        azimuthal_avg[i] = azimuthal_avg[i - 1]
                    else:
                        azimuthal_avg[i] = 0.0
        
        logger.debug(f"Computed azimuthal average with {num_bins} bins")
        
        return radii, azimuthal_avg

    def detect_azimuthal_peaks(
        self, radii: np.ndarray, azimuthal_avg: np.ndarray, min_radius: float = 0.3
    ) -> list[Tuple[float, float]]:
        """
        Detect peaks in the azimuthal average profile.

        Sharp spikes in the azimuthal average indicate periodic artifacts
        from upsampling operations.

        Args:
            radii: Radial distances (normalized)
            azimuthal_avg: Azimuthal average values
            min_radius: Minimum radius to consider (ignore low frequencies)

        Returns:
            List of (radius, magnitude) tuples for detected peaks
        """
        # Focus on high-frequency region (beyond min_radius)
        high_freq_mask = radii >= min_radius
        if not np.any(high_freq_mask):
            return []
        
        high_freq_radii = radii[high_freq_mask]
        high_freq_avg = azimuthal_avg[high_freq_mask]
        
        # Find peaks using scipy
        if len(high_freq_avg) < 3:
            return []
        
        # Use prominence to find significant peaks
        prominence = np.std(high_freq_avg) * 0.5
        peak_indices = signal.find_peaks(
            high_freq_avg,
            prominence=prominence,
            distance=max(1, len(high_freq_avg) // 50)  # Minimum distance between peaks
        )[0]
        
        peaks = [(high_freq_radii[idx], high_freq_avg[idx]) for idx in peak_indices]
        
        logger.debug(f"Detected {len(peaks)} peaks in azimuthal average")
        
        return peaks

    def compute_spectral_artifact_score(
        self,
        peaks: list[SpectralPeak],
        image_shape: Tuple[int, int],
        grid_strength: float,
        grid_consistency: float,
        nyquist_symmetry: float,
    ) -> float:
        """
        Compute the Frequency Artifact Score (F_a) based on detected peaks.

        The score reflects the presence of artificial symmetry and periodic
        artifacts in the spectral power distribution.

        Args:
            peaks: List of detected spectral peaks
            image_shape: (H, W) shape of the image
            grid_strength: Precomputed grid strength from spatial autocorrelation
            grid_consistency: Precomputed grid consistency score
            nyquist_symmetry: Precomputed Nyquist symmetry score

        Returns:
            Frequency artifact score (higher = more likely AI-generated)
        """
        if not peaks:
            return 0.0

        # Score based on:
        # 1. Number of peaks (more peaks = more artifacts)
        # 2. Average magnitude of peaks (stronger peaks = more artifacts)
        # 3. Symmetry: check for peaks at symmetric positions

        num_peaks = len(peaks)
        avg_magnitude = np.mean([p.magnitude for p in peaks])

        h, w = image_shape
        center_y, center_x = get_center_coords(image_shape)

        symmetric_pairs = 0
        peak_positions = {(p.u, p.v) for p in peaks}

        for peak in peaks:
            # Check for symmetric peak across center
            sym_u = 2 * center_x - peak.u
            sym_v = 2 * center_y - peak.v

            if (sym_u, sym_v) in peak_positions:
                symmetric_pairs += 1

        symmetry_score = symmetric_pairs / max(num_peaks, 1)

        # Base score: weighted combination of peak count, magnitude, and symmetry
        base_score = (num_peaks / 100.0) * 0.25 + (avg_magnitude / 10.0) * 0.4 + symmetry_score * 0.15
        base_score = min(base_score, 1.0)  # Clamp to [0, 1]

        # Combine grid indicators: spatial autocorrelation + grid consistency
        # Grid consistency (Nyquist symmetry) is a stronger indicator
        combined_grid_score = max(grid_strength, grid_consistency * 0.8 + nyquist_symmetry * 0.2)

        # Exponential scaling based on combined grid pattern strength
        # If combined_grid_score > 0.5, apply exponential boost
        if combined_grid_score > 0.5:
            # Exponential factor: e^(combined_grid_score * EXPONENTIAL_SCALE_FACTOR)
            # Stronger grid patterns get exponentially higher scores
            grid_multiplier = np.exp(combined_grid_score * EXPONENTIAL_SCALE_FACTOR)
            # Normalize: e^2 ≈ 7.4, so we scale to reasonable range
            grid_boost = (grid_multiplier - 1.0) / (np.exp(2.0) - 1.0)  # Maps to [0, 1]
            grid_boost = min(grid_boost, 1.0)  # Cap at 1.0
            
            # Combine base score with exponential grid boost
            # Grid pattern is a strong indicator, so it gets significant weight
            score = base_score * 0.3 + grid_boost * 0.7
        else:
            # No significant grid pattern, use base score with grid contribution
            # Grid consistency still contributes even if below threshold
            score = base_score * 0.7 + combined_grid_score * 0.2 + grid_consistency * 0.1

        score = min(score, 1.0)  # Final clamp to [0, 1]

        logger.debug(
            f"Spectral artifact score: {score:.4f} "
            f"(peaks: {num_peaks}, avg_mag: {avg_magnitude:.4f}, "
            f"symmetry: {symmetry_score:.4f}, grid_strength: {grid_strength:.4f}, "
            f"grid_consistency: {grid_consistency:.4f}, nyquist_symmetry: {nyquist_symmetry:.4f})"
        )

        return score

    def process_image(self, image: np.ndarray) -> ProcessImageResult:
        """
        Complete DFT processing pipeline for an image.

        Args:
            image: Preprocessed grayscale image (H, W, float32)

        Returns:
            ProcessImageResult containing all processing results
        """
        # Compute DFT
        fft_result = self.compute_dft(image)

        # Shift spectrum
        fft_shifted = self.shift_spectrum(fft_result)

        # Compute log-scaled magnitude
        log_magnitude = self.compute_magnitude_spectrum(fft_shifted, log_scale=True)

        # Compute azimuthal average
        azimuthal_radii, azimuthal_avg = self.compute_azimuthal_average(log_magnitude)

        # Get high-frequency mask
        high_freq_mask = self.get_high_frequency_mask(log_magnitude.shape)

        # Detect peaks
        peaks = self.detect_peaks(log_magnitude, high_freq_mask)

        # Compute spatial autocorrelation (grid pattern)
        grid_strength, interval_u, interval_v = self.compute_spatial_autocorrelation(
            peaks, image.shape
        )

        # Compute grid consistency (Nyquist folding symmetry)
        grid_consistency, nyquist_symmetry = self.compute_grid_consistency_score(
            peaks, image.shape
        )

        # Compute artifact score (pass all precomputed values to avoid recomputation)
        score = self.compute_spectral_artifact_score(
            peaks,
            image_shape=image.shape,
            grid_strength=grid_strength,
            grid_consistency=grid_consistency,
            nyquist_symmetry=nyquist_symmetry,
        )

        return ProcessImageResult(
            log_magnitude_spectrum=log_magnitude,
            peaks=peaks,
            artifact_score=score,
            grid_strength=grid_strength,
            grid_interval_u=interval_u,
            grid_interval_v=interval_v,
            azimuthal_radii=azimuthal_radii,
            azimuthal_average=azimuthal_avg,
            grid_consistency=grid_consistency,
            nyquist_symmetry=nyquist_symmetry,
        )

