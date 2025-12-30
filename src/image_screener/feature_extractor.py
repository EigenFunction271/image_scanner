"""Wavelet-based feature extraction for AI image detection.

This module implements Filter 02: Residual Noise Entropy analysis using
wavelet decomposition to extract features that distinguish AI-generated
images from natural photographs.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pywt
from scipy import stats

logger = logging.getLogger(__name__)

__all__ = [
    'extract_all_features',
    'compute_energy_features',
    'compute_statistical_moments',
    'estimate_ggd_parameters',
    'compute_ggd_features',
    'compute_cross_scale_correlation',
    'detect_periodic_artifacts',
    'compute_noise_consistency',
]

# Periodic artifact detection parameters
PERIODIC_ARTIFACT_LAGS = [1, 2, 4, 8]  # Lags for autocorrelation to detect checkerboard patterns


def compute_energy_features(subbands: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Compute energy-based features from wavelet subbands.

    For each detail subband (LH1, HL1, HH1, LH2...):
    - Compute normalized energy
    - Compute energy ratios between levels

    Args:
        subbands: Dictionary of subbands {LL3, LH1, LH2, LH3, HL1, ...}

    Returns:
        Array of ~15 energy-based features
    """
    features = []

    # Get detail subbands in order (LH, HL, HH for each level)
    detail_subbands = []
    for level in [1, 2, 3]:
        for orientation in ['LH', 'HL', 'HH']:
            key = f"{orientation}{level}"
            if key in subbands:
                detail_subbands.append((key, subbands[key]))

    if not detail_subbands:
        logger.warning("No detail subbands found for energy computation")
        return np.zeros(15)

    # Compute total energy across all detail subbands
    total_energy = sum(np.sum(np.square(sb)) for _, sb in detail_subbands)

    if total_energy == 0:
        logger.warning("Zero total energy in subbands")
        return np.zeros(15)

    # Normalized energy for each subband
    for key, subband in detail_subbands:
        energy = np.sum(np.square(subband))
        normalized_energy = energy / total_energy
        features.append(normalized_energy)

    # Energy ratios between adjacent levels (for each orientation)
    for orientation in ['LH', 'HL', 'HH']:
        for level in [1, 2]:
            key1 = f"{orientation}{level}"
            key2 = f"{orientation}{level + 1}"
            if key1 in subbands and key2 in subbands:
                e1 = np.sum(np.square(subbands[key1]))
                e2 = np.sum(np.square(subbands[key2]))
                if e2 > 0:
                    ratio = e1 / e2
                    features.append(ratio)
                else:
                    features.append(0.0)

    # Pad to 15 features if needed
    while len(features) < 15:
        features.append(0.0)

    return np.array(features[:15])


def compute_statistical_moments(subbands: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Compute statistical moments for each detail subband.

    For each of 9 detail subbands:
    - Mean, std, skewness, kurtosis

    Args:
        subbands: Dictionary of subbands

    Returns:
        Array of 36 features (4 moments × 9 subbands)
    """
    features = []

    # Get detail subbands in order
    detail_keys = []
    for level in [1, 2, 3]:
        for orientation in ['LH', 'HL', 'HH']:
            key = f"{orientation}{level}"
            if key in subbands:
                detail_keys.append(key)

    for key in detail_keys:
        coeffs = subbands[key].flatten()

        # Mean
        mean_val = np.mean(coeffs)
        features.append(mean_val)

        # Standard deviation
        std_val = np.std(coeffs)
        features.append(std_val)

        # Skewness
        if std_val > 1e-10:
            skew_val = stats.skew(coeffs)
            features.append(skew_val)
        else:
            features.append(0.0)

        # Kurtosis
        if std_val > 1e-10:
            kurt_val = stats.kurtosis(coeffs)
            features.append(kurt_val)
        else:
            features.append(0.0)

    # Pad to 36 features if needed (9 subbands × 4 moments)
    while len(features) < 36:
        features.append(0.0)

    return np.array(features[:36])


def estimate_ggd_parameters(coefficients: np.ndarray) -> Tuple[float, float]:
    """
    Estimate Generalized Gaussian Distribution (GGD) parameters.

    Uses moment-matching method to estimate shape (beta) and scale (alpha).

    Args:
        coefficients: Wavelet coefficients (1D array)

    Returns:
        Tuple of (alpha, beta) parameters
    """
    coeffs = coefficients.flatten()
    coeffs = coeffs[coeffs != 0]  # Remove zeros

    if len(coeffs) < 10:
        return 1.0, 1.0

    # Compute moments
    m2 = np.mean(np.square(coeffs))
    m4 = np.mean(coeffs ** 4)

    if m2 < 1e-10 or m4 < 1e-10:
        return 1.0, 1.0

    # Compute r = m2^2 / m4
    r = (m2 ** 2) / m4

    # Estimate beta using approximation
    # For GGD: r ranges from 0 (Laplace) to 1 (Gaussian)
    if r < 0.15:
        beta = 0.5  # Laplace-like
    elif r < 0.25:
        beta = 0.7
    elif r < 0.33:
        beta = 1.0  # Gaussian
    elif r < 0.5:
        beta = 1.5
    else:
        beta = 2.0  # More Gaussian

    # Estimate alpha (scale parameter)
    # For GGD: alpha = sqrt(m2 * Gamma(1/beta) / Gamma(3/beta))
    # Simplified approximation
    try:
        from scipy.special import gamma
        gamma_ratio = gamma(1.0 / beta) / gamma(3.0 / beta)
        alpha = np.sqrt(m2 * gamma_ratio)
        if not np.isfinite(alpha) or alpha < 1e-10:
            alpha = np.sqrt(m2)
    except (ImportError, ValueError, ZeroDivisionError):
        # Fallback to simple approximation
        alpha = np.sqrt(m2)

    return float(alpha), float(beta)


def compute_ggd_features(subbands: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Compute GGD parameters for each detail subband.

    Args:
        subbands: Dictionary of subbands

    Returns:
        Array of 18 features (2 parameters × 9 subbands)
    """
    features = []

    # Get detail subbands in order
    detail_keys = []
    for level in [1, 2, 3]:
        for orientation in ['LH', 'HL', 'HH']:
            key = f"{orientation}{level}"
            if key in subbands:
                detail_keys.append(key)

    for key in detail_keys:
        coeffs = subbands[key]
        alpha, beta = estimate_ggd_parameters(coeffs)
        features.append(alpha)
        features.append(beta)

    # Pad to 18 features if needed (9 subbands × 2 parameters)
    while len(features) < 18:
        features.append(0.0)

    return np.array(features[:18])


def compute_cross_scale_correlation(subbands: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Compute cross-scale correlation for each orientation.

    For each orientation (LH, HL, HH):
    - Upsample coarser level to match finer level
    - Compute Pearson correlation between adjacent levels

    Args:
        subbands: Dictionary of subbands

    Returns:
        Array of 9 correlation values
    """
    features = []

    for orientation in ['LH', 'HL', 'HH']:
        for level in [1, 2]:
            key1 = f"{orientation}{level}"
            key2 = f"{orientation}{level + 1}"

            if key1 in subbands and key2 in subbands:
                fine = subbands[key1]
                coarse = subbands[key2]

                # Upsample coarse to match fine dimensions
                from scipy.ndimage import zoom
                zoom_factor = (
                    fine.shape[0] / coarse.shape[0],
                    fine.shape[1] / coarse.shape[1]
                )
                coarse_upsampled = zoom(coarse, zoom_factor, order=1)

                # Flatten and compute correlation
                fine_flat = fine.flatten()
                coarse_flat = coarse_upsampled.flatten()

                if len(fine_flat) > 0 and np.std(fine_flat) > 1e-10 and np.std(coarse_flat) > 1e-10:
                    corr, _ = stats.pearsonr(fine_flat, coarse_flat)
                    if np.isnan(corr):
                        corr = 0.0
                    features.append(corr)
                else:
                    features.append(0.0)
            else:
                features.append(0.0)

    # Pad to 9 features if needed
    while len(features) < 9:
        features.append(0.0)

    return np.array(features[:9])


def detect_periodic_artifacts(hh1_subband: np.ndarray) -> np.ndarray:
    """
    Detect periodic artifacts in HH1 subband (high-frequency detail).

    Computes 2D autocorrelation at specific lags to detect checkerboard patterns
    common in AI-generated images.

    Args:
        hh1_subband: HH1 subband coefficients

    Returns:
        Array of 4-6 artifact indicators
    """
    features = []

    if hh1_subband.size < 16:
        return np.zeros(6)

    # Compute 2D autocorrelation at specific lags
    for lag in PERIODIC_ARTIFACT_LAGS:
        if lag >= min(hh1_subband.shape):
            features.append(0.0)
            continue

        # Compute autocorrelation at this lag
        # Shift and correlate
        shifted = np.roll(hh1_subband, lag, axis=0)
        shifted = np.roll(shifted, lag, axis=1)

        # Compute correlation coefficient
        orig_flat = hh1_subband.flatten()
        shift_flat = shifted.flatten()

        if np.std(orig_flat) > 1e-10 and np.std(shift_flat) > 1e-10:
            corr = np.corrcoef(orig_flat, shift_flat)[0, 1]
            if np.isnan(corr):
                corr = 0.0
            features.append(abs(corr))
        else:
            features.append(0.0)

    # Additional: Check for checkerboard pattern using FFT
    # High energy at specific frequencies indicates periodic patterns
    fft_hh1 = np.fft.fft2(hh1_subband)
    magnitude = np.abs(np.fft.fftshift(fft_hh1))

    # Check for peaks at grid frequencies
    h, w = magnitude.shape
    center_y, center_x = h // 2, w // 2

    # Sample points that would indicate checkerboard
    grid_points = [
        (center_y + h // 4, center_x),
        (center_y, center_x + w // 4),
    ]

    for py, px in grid_points:
        if 0 <= py < h and 0 <= px < w:
            features.append(magnitude[py, px] / (np.max(magnitude) + 1e-10))
        else:
            features.append(0.0)

    # Pad to 6 features
    while len(features) < 6:
        features.append(0.0)

    return np.array(features[:6])


def compute_noise_consistency(hh1_subband: np.ndarray) -> np.ndarray:
    """
    Compute noise consistency metrics from HH1 subband.

    Divides into blocks and measures variance consistency across blocks.
    Natural images have more consistent noise; AI images may have structured patterns.

    Args:
        hh1_subband: HH1 subband coefficients

    Returns:
        Array of [mean_variance, variance_of_variances, coefficient_of_variation]
    """
    if hh1_subband.size < 16:
        return np.array([0.0, 0.0, 0.0])

    # Divide into 4×4 blocks
    h, w = hh1_subband.shape
    block_h = max(4, h // 4)
    block_w = max(4, w // 4)

    block_variances = []

    for i in range(0, h, block_h):
        for j in range(0, w, block_w):
            block = hh1_subband[i:i + block_h, j:j + block_w]
            if block.size > 0:
                block_var = np.var(block)
                block_variances.append(block_var)

    if len(block_variances) == 0:
        return np.array([0.0, 0.0, 0.0])

    block_variances = np.array(block_variances)

    # Mean variance
    mean_var = np.mean(block_variances)

    # Variance of variances (consistency measure)
    var_of_vars = np.var(block_variances)

    # Coefficient of variation
    if mean_var > 1e-10:
        cv = np.std(block_variances) / mean_var
    else:
        cv = 0.0

    return np.array([mean_var, var_of_vars, cv])


def extract_all_features(subbands: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Extract complete feature vector from wavelet subbands.

    Combines all feature extraction functions to create a comprehensive
    feature vector for classification.

    Args:
        subbands: Dictionary of wavelet subbands

    Returns:
        Complete feature vector as numpy array (~85-90 features)
    """
    all_features = []

    # 1. Energy features (~15)
    energy_feat = compute_energy_features(subbands)
    all_features.extend(energy_feat)

    # 2. Statistical moments (~36)
    moment_feat = compute_statistical_moments(subbands)
    all_features.extend(moment_feat)

    # 3. GGD parameters (~18)
    ggd_feat = compute_ggd_features(subbands)
    all_features.extend(ggd_feat)

    # 4. Cross-scale correlation (~9)
    corr_feat = compute_cross_scale_correlation(subbands)
    all_features.extend(corr_feat)

    # 5. Periodic artifacts from HH1 (~6)
    if 'HH1' in subbands:
        artifact_feat = detect_periodic_artifacts(subbands['HH1'])
        all_features.extend(artifact_feat)
    else:
        all_features.extend([0.0] * 6)

    # 6. Noise consistency from HH1 (~3)
    if 'HH1' in subbands:
        noise_feat = compute_noise_consistency(subbands['HH1'])
        all_features.extend(noise_feat)
    else:
        all_features.extend([0.0] * 3)

    feature_vector = np.array(all_features)

    # Replace any NaN or Inf values
    feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)

    logger.debug(f"Extracted {len(feature_vector)} features from wavelet subbands")

    return feature_vector

