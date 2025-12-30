# Methods Documentation

This document provides detailed descriptions of all detection methods implemented in the Image Screener project.

## Table of Contents

1. [Filter 01: Spectral Peak Detector (FFT)](#filter-01-spectral-peak-detector-fft)
2. [Filter 02: Wavelet-based Noise Analysis](#filter-02-wavelet-based-noise-analysis)
3. [Optics Consistency Detector](#optics-consistency-detector)
   - [Frequency-Domain Optics Test](#1-frequency-domain-optics-test)
   - [Edge Spread Function (ESF) Test](#2-edge-spread-function-esf-test)
   - [Depth-of-Field Consistency Test](#3-depth-of-field-consistency-test-conditional)
   - [Chromatic Aberration Test](#4-chromatic-aberration-test-conditional)
   - [Sensor Noise Residual Test](#5-sensor-noise-residual-test)

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
- **Unnaturally low high-frequency energy**: Over-clean noise floor (AI signature)
- **Unnaturally uniform high-frequency tail**: Artificial smoothing (low variance in tail)
- **Mid-band energy bumps**: Elevated mid-band energy relative to expected decay (upsampling artifacts)
- **Uniform spectral shape across image**: Low spatial variance in spectral structure (AI uniformity)

**Enhanced Metrics**:
1. **High-frequency noise floor energy**: Real sensors have stochastic variation; AI often has unnaturally low or uniform tail
2. **Mid-band bump ratio**: Detects elevated mid-band energy (characteristic of upsampling artifacts)
3. **Spatial stationarity**: Measures variance in spectral shape across image quadrants (AI tends to be more uniform)

**Scoring**:
- Penalize bumps, suppression, and non-smooth decay
- Penalize unnaturally clean or uniform noise floor
- Penalize mid-band bumps and uniform spatial structure
- Score: 0.0 (fails) to 1.0 (passes)

**Physical Basis**: Real camera optics create smooth, monotonic frequency response with natural noise floor variation. AI post-processing or synthetic generation often introduces non-physical frequency characteristics, unnaturally clean tails, or uniform spectral structure.

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
- **Even-symmetric ringing**: Low asymmetry with negative lobes on both sides (AI diffusion artifacts)
- **Inconsistent PSF width**: High variation in PSF width across image

**Enhanced Metric - Asymmetry Analysis**:
- **Asymmetry score**: Measures symmetry of LSF around peak
  - Low asymmetry (< 0.3) with negative lobes on both sides → AI diffusion ringing (symmetric)
  - High asymmetry (> 0.5) → Real ISP sharpening (one-sided halos, less suspicious)
  - Medium asymmetry (0.3-0.5) → Ambiguous, less penalty
- Replaces simple negative lobe detection with forensic asymmetry analysis

**Scoring**:
- Penalize ringing (oscillation ratio > 30%)
- Strongly penalize symmetric ringing (low asymmetry + negative lobes on both sides)
- Penalize inconsistent width (CV > 0.5)
- Score: 0.0 (fails) to 1.0 (passes)

**Physical Basis**: Real camera PSF is smooth and positive. ISP sharpening creates asymmetric halos. AI diffusion models create even-symmetric ringing (negative lobes on both sides with low asymmetry), which is a strong AI signature.

---

### 3. Depth-of-Field Consistency Test (Conditional)

**Purpose**: Check spatial smoothness of blur variation (depth-of-field).

**Conditional Testing**: 
- First checks for usable blur evidence:
  - Presence of textured background (high local variance)
  - Presence of strong defocus gradients (spatial variation in blur)
- If insufficient evidence, test is skipped (neutral score) to avoid false positives
- Prevents penalizing deep-focus or clean-background images

**Method**:
1. **Blur Evidence Check**: Verify image has sufficient texture or defocus gradients
2. Estimate local blur radius using **frequency attenuation method**:
   - Compute Laplacian energy (high-frequency content)
   - Compute gradient energy (low-frequency content)
   - Blur radius from ratio: Laplacian/Gradient
   - Sharp regions: High ratio (> 2.0) → blur ≈ 0-1 pixels
   - Blurry regions: Low ratio (< 0.5) → blur ≈ 3-8 pixels
3. Create blur map across image at grid points
4. Compute spatial gradient of blur map
5. Check for discrete jumps or semantic patterns

**Violations Detected**:
- **Discrete blur regions**: Large jumps in blur (max gradient > 2.0)
- **Non-smooth variation**: High mean gradient (> 0.5)
- **Semantic blur patterns**: Bimodal distribution (e.g., all foreground sharp, all background blurry)

**Scoring**:
- If insufficient blur evidence: Score = 1.0 (neutral, test skipped)
- Otherwise: Penalize discrete jumps and high variation
- Penalize semantic patterns (CV > 0.8)
- Score: 0.0 (fails) to 1.0 (passes or skipped)

**Physical Basis**: Real DOF varies continuously with depth. Discrete or semantic blur regions indicate post-processing or synthetic generation. However, deep-focus images or clean backgrounds may legitimately have little blur variation.

---

### 4. Chromatic Aberration Test (Conditional)

**Purpose**: Validate non-zero, spatially coherent chromatic aberration.

**Conditional Testing**:
- **Resolution-aware**: Subpixel CA requires high resolution
  - Below 512px: Test is low-confidence (likely resized/re-encoded)
  - Below 1024px: Radial consistency test may be unreliable
- **ISP Correction Aware**: Modern phones correct CA in ISP
  - Low CA magnitude is NOT suspicious (expected after correction)
  - Test focuses on radial consistency rather than magnitude

**Method**:
1. Load RGB image and extract R, G, B channels
2. Detect edges ONLY in green channel (reference)
3. For each edge in green, find corresponding edge in R and B using gradient-based matching
4. Compute edge offsets (R-G and B-G) along perpendicular direction
5. Analyze offset patterns:
   - Radial consistency (CA should increase with radius)
   - Spatial coherence (offsets should be spatially correlated)

**Violations Detected**:
- **Non-physical radial variation**: Negative slope (CA should increase with radius)
- **Uniform CA**: Low variance in offsets (suspiciously uniform)
- **Non-coherent CA**: High variance (> 1.0 pixel) indicating random offsets
- **Note**: Zero CA is NOT penalized (expected after ISP correction)

**Scoring**:
- Low resolution (< 512px): Test marked as low-confidence
- Penalize non-physical variation and non-coherent CA
- Score: 0.0 (fails) to 1.0 (passes)

**Physical Basis**: Real cameras have small but non-zero chromatic aberration that varies radially and spatially. However, modern ISP correction often removes CA, so low magnitude is not suspicious. Radial consistency and spatial coherence are more reliable indicators.

---

### 5. Sensor Noise Residual Test

**Purpose**: Analyze spatial correlation structure of noise residuals to distinguish real camera sensor data from AI-generated images.

**Physical Basis**:
- **Real sensors**: Have structural correlation in noise due to:
  - Bayer demosaicing process (inter-pixel dependencies)
  - Physical sensor patterns (readout noise, pixel crosstalk)
  - ISP processing (color interpolation creates correlations)
- **AI-generated images**: Have decorrelated noise because:
  - Latent space reconstruction destroys inter-pixel phase relationships
  - Generative models produce independent pixel values
  - No physical sensor structure to preserve

**Method**:
1. **Noise Extraction**:
   - Use 3×3 median filter to estimate signal
   - Subtract from original to get noise residual
   - Normalize by standard deviation
2. **2D Autocorrelation** (FFT-optimized):
   - Compute autocorrelation using FFT: `IFFT(FFT(x) * conj(FFT(x)))`
   - O(n log n) complexity vs O(n²) for direct correlation
   - Extract 11×11 region around center for structure analysis
3. **8-Neighbor Correlation**:
   - Sample pixels (adaptive sampling rate)
   - Compute correlation with 8 immediate neighbors (vectorized)
   - Measure spatial correlation strength
4. **Structure Analysis**:
   - Check off-center autocorrelation strength
   - Real sensors: Should have structure (Bayer patterns)
   - AI: Should be near-delta function (no structure)

**Violations Detected**:
- **Low spatial correlation**: Mean correlation < 0.15 (decorrelated noise)
- **Extremely decorrelated noise**: Decorrelation factor > 0.9 (near-random)
- **Lack of autocorrelation structure**: Low off-center std (< 0.05)

**Scoring**:
- Real sensors: High correlation (ρ > 0.15) → High score
- AI/synthetic: Low correlation (ρ < 0.15) → Low score
- Score: 0.0 (AI) to 1.0 (real sensor)

**Performance**:
- FFT-based autocorrelation: ~0.1-0.2 seconds for 512×512 image
- 100x faster than direct correlation method
- Vectorized neighbor correlation computation

---

## Combined Scoring

The Optics Consistency Detector combines all five tests with configurable weights:

```
optics_score = w1 * frequency_score + w2 * psf_score + w3 * dof_score + w4 * ca_score + w5 * noise_score
```

**Default weights**:
- Frequency: 0.25
- PSF: 0.2
- DOF: 0.2
- CA: 0.15
- Noise Residual: 0.2

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

- **Optics Detector**: ~2-4 seconds per 512×512 image
  - Frequency test: ~0.5-1s
  - Edge PSF test: ~0.5-1s
  - DOF test: ~0.5-1s (conditional, may skip)
  - CA test: ~0.5-1s (conditional, requires RGB)
  - Noise residual test: ~0.1-0.2s (FFT-optimized)
- **Spectral Peak Detector**: ~1-3 seconds per image
- **Wavelet Detector**: ~1-2 seconds per image (plus training time)

**Optimizations**:
- FFT-based autocorrelation for noise residual (100x faster)
- Vectorized operations for grid pattern analysis
- Conditional testing to skip tests when evidence is insufficient
- Adaptive sampling for neighbor correlation

### Limitations

1. **Image Quality**: Works best on high-quality images (not heavily compressed)
2. **Post-processing**: Heavy JPEG compression or filtering can affect results
3. **CA Test**: Requires RGB image (skipped for grayscale), resolution-aware
4. **Edge Detection**: Requires sufficient edges for PSF analysis
5. **DOF Test**: Conditional - requires blur evidence (may skip deep-focus images)
6. **Noise Residual Test**: Requires sufficient image size (minimum 3×3)

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

