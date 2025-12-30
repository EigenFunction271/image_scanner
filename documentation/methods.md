# Methods Documentation

This document provides detailed descriptions of all detection methods implemented in the Image Screener project.

## Table of Contents

1. [Filter 01: Spectral Peak Detector (FFT)](#filter-01-spectral-peak-detector-fft)
2. [Filter 02: Wavelet-based Noise Analysis](#filter-02-wavelet-based-noise-analysis)
3. [Optics Consistency Detector](#optics-consistency-detector)
   - [Frequency-Domain Optics Test](#1-frequency-domain-optics-test)
   - [Edge Spread Function (ESF) Test](#2-edge-spread-function-esf-test)
   - [Depth-of-Field Consistency Test](#3-depth-of-field-consistency-test)
   - [Chromatic Aberration Test](#4-chromatic-aberration-test)

---

## Filter 01: Spectral Peak Detector (FFT)

### Overview

The Spectral Peak Detector identifies periodic grid artifacts introduced by AI upsampling operations. AI models (Stable Diffusion, Midjourney, DALL-E) use transposed convolutions that upscale by factors of 2, 4, or 8, creating periodic patterns that are invisible in the spatial domain but detectable in the frequency domain.

### Mathematical Foundation

**Spectral Replication**: When an image is upsampled using transposed convolutions, the frequency domain undergoes spectral replication. The original spectrum is compressed and repeated, leaving residual peaks at specific frequencies.

**Nyquist Folding**: Upsampling operations create aliasing artifacts that manifest as conjugate symmetry across Nyquist boundaries in the frequency domain.

### Algorithm

1. **Preprocessing**
   - Load image and convert to grayscale
   - Resize to 512×512 (maintains aspect ratio with padding)
   - Normalize to float32 [0, 1]

2. **2D Discrete Fourier Transform**
   - Compute `FFT(image)` using `numpy.fft.fft2`
   - Shift spectrum to center DC component: `np.fft.fftshift(fft_result)`
   - Compute log-magnitude: `log(1 + |FFT|)`

3. **Peak Detection**
   - Apply high-frequency mask (frequencies > 30% of Nyquist)
   - Use 98th percentile threshold to identify candidate peaks
   - Apply `scipy.ndimage.maximum_filter` for local maxima detection
   - Retain top 1000 peaks by magnitude

4. **Grid Pattern Analysis**
   - **Spatial Autocorrelation**: Analyze peak positions for repeating intervals
     - Compute pairwise distances between peaks
     - Detect dominant spacing intervals (modulo 4, 8, 16, 32)
   - **Grid Consistency**: Check alignment to power-of-2 grid
     - Count peaks aligned to stride values (4, 8, 16, 32 pixels)
     - Score based on alignment percentage
   - **Nyquist Symmetry**: Detect conjugate symmetry
     - For each peak at (u, v), check for symmetric peak at (-u, -v)
     - Count symmetric pairs

5. **Azimuthal Average**
   - Integrate magnitude spectrum along radial circles
   - Create 1D profile showing power vs. radius
   - AI images show sharp spikes at specific radii (periodic artifacts)

6. **Scoring**
   - Base score from peak count and magnitude
   - Exponential scaling when strong grid patterns detected
   - Final artifact score: 0.0 (real) to 1.0 (AI-generated)

### Key Metrics

- **Artifact Score** (0.0-1.0): Overall likelihood of AI generation
- **Grid Strength**: Spatial autocorrelation of peak positions
- **Grid Consistency**: Power-of-2 alignment score (0.0-1.0)
- **Nyquist Symmetry**: Conjugate symmetry score (0.0-1.0)
- **Num Peaks**: Number of detected spectral peaks

### Performance

- Processing time: ~1-3 seconds per 512×512 image
- Memory: Minimal (in-place operations where possible)
- Default settings: 98th percentile threshold, top 1000 peaks

---

## Filter 02: Wavelet-based Noise Analysis

### Overview

The Wavelet Detector analyzes residual noise patterns to detect AI-generated images. Real cameras have Photo-Response Non-Uniformity (PRNU) - unique sensor noise patterns. AI images lack this physical signature and often show structured noise or hyper-smoothed regions.

### Algorithm

1. **Wavelet Decomposition**
   - Apply multi-level Discrete Wavelet Transform (DWT)
   - Default: Daubechies 4 ('db4') wavelet, 3 levels
   - Extract subbands: LL3, LH1-3, HL1-3, HH1-3

2. **Feature Extraction**
   - **Energy Features**: Normalized energy per subband, energy ratios between levels
   - **Statistical Moments**: Mean, std, skewness, kurtosis for each detail subband
   - **Noise Residual**: Extract residual after denoising
   - **Periodic Artifact Detection**: Autocorrelation at lags [1, 2, 4, 8]

3. **Classification**
   - Use Random Forest or SVM classifier
   - Train on labeled dataset (real vs. AI images)
   - Output: Probability of AI generation

### Key Features

- **Noise Entropy**: Real images have high-entropy, spatially white noise
- **Structured Noise**: AI images show structured patterns or zero entropy regions
- **PRNU Absence**: AI images lack sensor-specific noise signatures

---

## Optics Consistency Detector

The Optics Consistency Detector validates whether an image follows physical optical laws that real cameras must obey. It consists of four complementary tests that check different aspects of optical physics.

### Core Assumptions

1. **Monotonic OTF**: Real cameras apply a monotonic low-pass Optical Transfer Function (OTF)
2. **Blur Ordering**: Blur occurs **before** detail and noise injection
3. **Continuous DOF**: Depth-of-field blur varies **continuously with depth**
4. **Non-zero CA**: Chromatic aberration is small but non-zero and spatially coherent

---

### 1. Frequency-Domain Optics Test

**Purpose**: Validate monotonic OTF decay in the frequency domain.

**Method**:
1. Compute 2D FFT of grayscale image
2. Compute radial power spectrum (azimuthal average)
3. Fit log-log slope: `log(power) = a * log(radius) + b`
4. Check for deviations from smooth decay

**Violations Detected**:
- **Non-monotonic decay**: Positive or shallow slope (should be negative)
- **Mid-frequency bumps**: Positive residuals indicating artificial enhancements
- **High-frequency suppression**: Negative residuals at high frequencies (unnatural filtering)
- **Non-smooth decay**: High variance in residuals

**Scoring**:
- Penalize bumps, suppression, and non-smooth decay
- Score: 0.0 (fails) to 1.0 (passes)

**Physical Basis**: Real camera optics create smooth, monotonic frequency response. AI post-processing or synthetic generation often introduces non-physical frequency characteristics.

---

### 2. Edge Spread Function (ESF) Test

**Purpose**: Analyze Point Spread Function (PSF) consistency by examining edge transitions.

**Method**:
1. Detect strong edges using Canny edge detection
2. For each edge pixel, extract perpendicular profile (ESF)
3. Differentiate ESF to get Line Spread Function (LSF)
4. Analyze LSF for physical consistency

**Violations Detected**:
- **Ringing**: Oscillations in ESF (alternating signs in LSF)
- **Negative lobes**: Significant negative values in LSF (non-physical PSF)
- **Inconsistent PSF width**: High variation in PSF width across image

**Scoring**:
- Penalize ringing (oscillation ratio > 30%)
- Penalize negative lobes (> 15% negative values)
- Penalize inconsistent width (CV > 0.5)
- Score: 0.0 (fails) to 1.0 (passes)

**Physical Basis**: Real camera PSF is smooth and positive. Ringing and negative lobes indicate non-physical processing or synthetic generation.

---

### 3. Depth-of-Field Consistency Test

**Purpose**: Check spatial smoothness of blur variation (depth-of-field).

**Method**:
1. Estimate local blur radius at sampled grid points
   - Compute gradient magnitude in local window
   - Blur radius inversely related to gradient strength
2. Create blur map across image
3. Compute spatial gradient of blur map
4. Check for discrete jumps or semantic patterns

**Violations Detected**:
- **Discrete blur regions**: Large jumps in blur (max gradient > 2.0)
- **Non-smooth variation**: High mean gradient (> 0.5)
- **Semantic blur patterns**: Bimodal distribution (e.g., all foreground sharp, all background blurry)

**Scoring**:
- Penalize discrete jumps and high variation
- Penalize semantic patterns (CV > 0.8)
- Score: 0.0 (fails) to 1.0 (passes)

**Physical Basis**: Real DOF varies continuously with depth. Discrete or semantic blur regions indicate post-processing or synthetic generation.

---

### 4. Chromatic Aberration Test

**Purpose**: Validate non-zero, spatially coherent chromatic aberration.

**Method**:
1. Load RGB image and extract R, G, B channels
2. Detect edges in each channel using Canny
3. Use green channel as reference
4. For each edge in green, find corresponding edge in R and B channels
5. Compute edge offsets (R-G and B-G)
6. Analyze offset patterns

**Violations Detected**:
- **Zero CA**: Mean offsets < 0.1 pixels (suspicious - real cameras have CA)
- **Non-physical radial variation**: Negative slope (CA should increase with radius)
- **Uniform CA**: Low variance in offsets (suspiciously uniform)
- **Non-coherent CA**: High variance (> 1.0 pixel) indicating random offsets

**Scoring**:
- Penalize zero CA (score × 0.3)
- Penalize non-physical variation
- Penalize uniform or non-coherent CA
- Score: 0.0 (fails) to 1.0 (passes)

**Physical Basis**: Real cameras have small but non-zero chromatic aberration that varies radially and spatially. Zero CA or perfectly uniform CA indicates synthetic generation or heavy post-processing.

---

## Combined Scoring

The Optics Consistency Detector combines all four tests with configurable weights:

```
optics_score = w1 * frequency_score + w2 * psf_score + w3 * dof_score + w4 * ca_score
```

**Default weights**:
- Frequency: 0.3
- PSF: 0.25
- DOF: 0.25
- CA: 0.2

**Interpretation**:
- **Score ≥ 0.7**: Likely real image (passes most tests)
- **Score 0.4-0.7**: Suspicious (some violations)
- **Score < 0.4**: Likely AI-generated (fails multiple tests)

---

## Implementation Details

### Dependencies

- **numpy**: Array operations, FFT
- **scipy**: Signal processing, peak detection, optimization
- **opencv-python**: Edge detection (Canny)
- **PIL/Pillow**: Image loading
- **matplotlib**: Visualization (optional)

### Performance

- **Optics Detector**: ~2-5 seconds per 512×512 image
- **Spectral Peak Detector**: ~1-3 seconds per image
- **Wavelet Detector**: ~1-2 seconds per image (plus training time)

### Limitations

1. **Image Quality**: Works best on high-quality images (not heavily compressed)
2. **Post-processing**: Heavy JPEG compression or filtering can affect results
3. **CA Test**: Requires RGB image (skipped for grayscale)
4. **Edge Detection**: Requires sufficient edges for PSF analysis

---

## References

1. **Spectral Replication**: Theory of upsampling artifacts in frequency domain
2. **Nyquist Folding**: Aliasing and conjugate symmetry in sampled signals
3. **Optical Transfer Function**: Camera lens physics and frequency response
4. **Point Spread Function**: Edge spread and line spread functions
5. **Chromatic Aberration**: Physical lens properties and radial variation
6. **Photo-Response Non-Uniformity (PRNU)**: Sensor-specific noise patterns

---

## Future Work

- **Filter 03**: Geometric Gradient Consistency (checkerboard effect detection)
- **Filter 04**: Chromatic Channel Misalignment (radial displacement vectors)
- **Improved CA Detection**: More robust edge matching algorithms
- **Adaptive Thresholding**: Dynamic thresholds based on image characteristics
- **Multi-scale Analysis**: Analysis at multiple resolutions

