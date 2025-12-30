# Image Screener - AI Image Detection Tool

Experimental image forensics tool designed to distinguish between natural photography and AI-generated imagery using frequency domain analysis. Based on the mathematical principle that AI upsampling operations introduce periodic artifacts invisible to the naked eye but detectable in the Fourier domain.

## Features

### Filter 01: Spectral Peak Detector (FFT)
- Identifies periodic grid artifacts from AI upsampling
- **2D Discrete Fourier Transform** - Full frequency domain analysis
- **Spatial Autocorrelation** - Detects repeating grid patterns in peak positions
- **Grid Consistency Score** - Identifies power-of-2 alignment (4, 8, 16, 32) characteristic of upsampling strides
- **Nyquist Folding Symmetry** - Detects conjugate symmetry across Nyquist boundaries
- **Azimuthal Average** - 1D profile showing periodic artifacts as sharp spikes
- **Exponential Scoring** - Artifact scores scale exponentially when strong grid patterns are detected
- **Local Maxima Peak Detection** - Accurate peak finding using scipy's maximum filter

### Filter 02: Wavelet-based Noise Analysis
- Residual noise entropy analysis using wavelet decomposition
- Detects lack of sensor PRNU and VAE artifacts
- Statistical moment extraction from wavelet subbands

### Optics Consistency Detector
- **Frequency-domain optics test** - Validates monotonic OTF decay with enhanced metrics:
  - High-frequency noise floor energy analysis
  - Mid-band bump detection (upsampling artifacts)
  - Spatial stationarity analysis (uniform spectral structure detection)
- **Edge Spread Function (ESF) test** - Analyzes PSF consistency:
  - Asymmetry metric to distinguish AI diffusion ringing (symmetric) from ISP sharpening (asymmetric)
  - Ringing detection with improved forensic logic
- **Depth-of-field consistency** (Conditional Test):
  - Checks for usable blur evidence before running test
  - Frequency attenuation method (Laplacian/Gradient energy ratio)
  - Prevents false positives on deep-focus images
- **Chromatic aberration test** (Conditional Test):
  - Resolution-aware testing (subpixel CA needs high resolution)
  - Accounts for modern ISP correction
- **Sensor noise residual test** - Analyzes spatial correlation of noise:
  - FFT-based autocorrelation (100x faster than direct correlation)
  - 8-neighbor correlation analysis
  - Detects decorrelated noise patterns (AI signature)
- Physics-first approach based on optical laws that real cameras must follow

## Installation

### Option 1: Install as package (recommended)

```bash
pip3 install -e .
```

For development:

```bash
pip3 install -e ".[dev]"
```

### Option 2: Install dependencies only

```bash
pip3 install -r requirements.txt
```

Then run scripts with:
```bash
PYTHONPATH=src python3 examples/compare_test_images.py
```

**Note:** On macOS, use `pip3` instead of `pip`. Alternatively, use `python3 -m pip`.

## Usage

### Visual FFT Tool (Single Image Analysis)

Analyze a single image and visualize frequency domain peaks:

```bash
# Display visualization interactively
PYTHONPATH=src python3 examples/visualize_fft.py path/to/image.jpg

# Save visualization to file
PYTHONPATH=src python3 examples/visualize_fft.py path/to/image.jpg -o output.png

# Custom settings
PYTHONPATH=src python3 examples/visualize_fft.py path/to/image.jpg --sensitivity 1.5 --threshold 0.4
```

The visualization shows:
- **Original grayscale image** - Preprocessed input
- **Frequency spectrum** - Log-magnitude heatmap of 2D FFT
- **Spectrum with peaks** - Detected peaks highlighted and labeled
- **3D surface plot** - Zoomed view of frequency domain
- **Azimuthal average** - 1D profile showing periodic spikes (key indicator of AI generation)

### Two Image Comparison Tool

Compare two images side-by-side (e.g., real vs fake):

```bash
PYTHONPATH=src python3 examples/compare_two_images.py \
  "documentation/test images/03/real.jpg" \
  "documentation/test images/03/fake.png"

# With custom labels
PYTHONPATH=src python3 examples/compare_two_images.py \
  image1.jpg image2.png --label1 real --label2 fake
```

This tool:
- Analyzes both images with full FFT visualization
- Saves outputs to `documentation/test images/comparison_fft/`
- Displays side-by-side comparison metrics
- Provides interpretation of differences

### Batch Comparison Tool

Compare multiple real vs fake images:

```bash
PYTHONPATH=src python3 examples/compare_test_images.py
```

### Optics Consistency Detector

Analyze images for physical optical law violations:

```bash
# Basic usage
PYTHONPATH=src python3 examples/detect_optics.py image.jpg

# Save diagnostics to output directory
PYTHONPATH=src python3 examples/detect_optics.py image.jpg --output outputs/

# Skip chromatic aberration test (grayscale only, faster)
PYTHONPATH=src python3 examples/detect_optics.py image.jpg --no-rgb

# Test script (runs on default test images)
PYTHONPATH=src python3 examples/test_optics.py
```

The optics detector:
- Produces an **optics consistency score** (0.0-1.0)
- Generates diagnostic plots showing all test results
- Provides human-readable explanations of violations
- Validates physical optical laws that real cameras must follow
- Uses conditional testing to avoid false positives:
  - DOF test only runs when blur evidence is present
  - CA test accounts for resolution and ISP correction

### Programmatic Usage

#### Spectral Peak Detector

```python
from image_screener.spectral_peak_detector import SpectralPeakDetector

detector = SpectralPeakDetector(
    target_size=512,
    sensitivity=1.0,
    high_freq_threshold=0.3,
    peak_threshold_percentile=98.0
)

result = detector.analyze("path/to/image.jpg")

# Core metrics
print(f"Artifact Score: {result.artifact_score:.4f}")
print(f"Number of Peaks: {result.num_peaks}")

# Grid pattern detection
print(f"Grid Strength: {result.grid_strength:.4f}")
print(f"Grid Consistency (Power-of-2): {result.grid_consistency:.4f}")
print(f"Nyquist Symmetry: {result.nyquist_symmetry:.4f}")

# Access full data
print(f"Top 5 Peaks: {result.peaks[:5]}")
print(f"Azimuthal Average: {result.azimuthal_average}")
```

#### Optics Consistency Detector

```python
from image_screener.optics_consistency import OpticsConsistencyDetector

detector = OpticsConsistencyDetector(
    frequency_weight=0.25,
    edge_psf_weight=0.2,
    dof_weight=0.2,
    ca_weight=0.15,
    noise_residual_weight=0.2
)

result = detector.analyze("path/to/image.jpg")

# Overall score
print(f"Optics Score: {result.optics_score:.4f}")

# Individual test scores
print(f"Frequency Test: {result.frequency_test.score:.4f}")
print(f"Edge PSF Test: {result.edge_psf_test.score:.4f}")
print(f"DOF Test: {result.dof_consistency_test.score:.4f}")
print(f"CA Test: {result.chromatic_aberration_test.score:.4f}")
print(f"Noise Residual Test: {result.noise_residual_test.score:.4f}")

# Violations
print(f"Explanation: {result.explanation}")
```

### Detection Metrics Explained

- **Artifact Score** (0.0-1.0): Overall likelihood of AI generation. Higher = more likely AI.
- **Grid Strength**: Spatial autocorrelation of peak positions (repeating intervals).
- **Grid Consistency**: How well peaks align to power-of-2 grid (4, 8, 16, 32 pixels).
- **Nyquist Symmetry**: Conjugate symmetry across Nyquist boundaries (characteristic of aliasing).
- **Azimuthal Average**: 1D profile of frequency spectrum - AI images show sharp spikes.

### How It Works

1. **Upsampling Artifacts**: AI models use transposed convolutions that upscale by factors of 2, 4, or 8, creating periodic "grid" patterns.

2. **Spectral Replication**: In the frequency domain, upsampling causes spectral replication - the original spectrum is compressed and repeated, leaving residual peaks.

3. **Grid Detection**: We detect these patterns by:
   - Finding peaks in high-frequency regions
   - Analyzing spacing between peaks (modulo 4, 8, 16, 32)
   - Checking for conjugate symmetry (Nyquist folding)

4. **Exponential Scoring**: When strong grid patterns are detected, scores scale exponentially to clearly separate AI from real images.

## Testing

```bash
pytest
```

## Project Structure

- `src/image_screener/` - Main package
  - `preprocessing.py` - Image loading, grayscale conversion, resizing
  - `dft.py` - 2D DFT computation, peak detection, grid analysis, azimuthal average
  - `spectral_peak_detector.py` - Main Filter 01 implementation
  - `wavelet_detector.py` - Filter 02: Wavelet-based noise analysis
  - `optics_consistency.py` - Optics consistency detector (4 physics-based tests)
  - `optics_visualization.py` - Diagnostic plot generation for optics tests
- `examples/` - Example scripts
  - `visualize_fft.py` - Single image FFT visualization tool
  - `compare_test_images.py` - Batch comparison of real vs fake images
  - `compare_two_images.py` - Side-by-side comparison of two images
  - `detect_optics.py` - CLI tool for optics consistency detection
  - `test_optics.py` - Test script for optics diagnostics
- `tests/` - Test suite (pytest)
- `documentation/` - PRD, filter specifications, methods documentation, and test images

## Technical Details

### Detection Algorithm

The tool uses multiple complementary methods:

1. **Local Maxima Peak Detection**: Uses `scipy.ndimage.maximum_filter` to find actual peaks (not just threshold crossings)
2. **Spatial Autocorrelation**: Analyzes peak positions for repeating intervals
3. **Grid Consistency**: Checks if peak spacing aligns to powers of 2 (upsampling stride detection)
4. **Nyquist Symmetry**: Detects conjugate symmetry indicating aliasing artifacts
5. **Azimuthal Integration**: Integrates magnitude spectrum along radial circles to reveal periodic spikes

### Performance

- Default settings: 98th percentile threshold, top 1000 peaks
- Processing time: ~1-3 seconds per 512×512 image (Spectral Peak Detector)
- Optics Consistency Detector: ~2-4 seconds per 512×512 image
  - Noise residual test: ~0.1-0.2 seconds (optimized with FFT-based autocorrelation)
- Memory: Minimal (in-place operations where possible)
- Optimizations:
  - Vectorized operations for grid pattern analysis
  - FFT-based autocorrelation for noise residual (100x faster)
  - Conditional testing to skip tests when evidence is insufficient

## References

Based on the mathematical principles described in:
- Spectral replication from transposed convolutions
- Nyquist folding symmetry in aliased signals
- Grid saliency detection for upsampling artifacts

See `documentation/prd.md`, `documentation/filter_spec.md`, and `documentation/methods.md` for detailed specifications and method descriptions.

