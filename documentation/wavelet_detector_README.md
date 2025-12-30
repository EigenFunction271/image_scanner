# Wavelet-Based AI Image Detector (Filter 02)

This module implements **Filter 02: Residual Noise Entropy** analysis using multi-level discrete wavelet transforms to detect AI-generated images by analyzing noise patterns and texture characteristics.

## Overview

The wavelet detector uses the mathematical principle that AI-generated images have different noise characteristics than natural photographs:

- **Natural images**: High-entropy, spatially white noise with consistent variance
- **AI images**: Structured noise, hyper-smoothed regions, or periodic artifacts from VAE reconstruction

## Installation

Install additional dependencies:

```bash
pip3 install -r requirements.txt
```

Required packages:
- `PyWavelets` - Wavelet transforms
- `scikit-learn` - Machine learning classifiers
- `joblib` - Model persistence
- `tqdm` - Progress bars
- `seaborn` - Visualization (optional)

## Quick Start

### 1. Prepare Training Data

Organize your images into two directories:

```
data/
  real/     # Natural/photographic images
    img1.jpg
    img2.png
    ...
  ai/       # AI-generated images
    img1.jpg
    img2.png
    ...
```

### 2. Train a Model

```bash
PYTHONPATH=src python3 examples/train_wavelet.py \
  --real-dir data/real \
  --ai-dir data/ai \
  --output wavelet_model.pkl \
  --classifier rf
```

Options:
- `--classifier`: `rf` (Random Forest) or `svm` (Support Vector Machine)
- `--tune`: Perform hyperparameter tuning (slower but better results)
- `--test-split`: Test set fraction (default: 0.15)
- `--val-split`: Validation set fraction (default: 0.15)

### 3. Classify Images

**Single image:**
```bash
PYTHONPATH=src python3 examples/inference_wavelet.py \
  --image path/to/image.jpg \
  --model wavelet_model.pkl
```

**With visualization:**
```bash
PYTHONPATH=src python3 examples/inference_wavelet.py \
  --image path/to/image.jpg \
  --model wavelet_model.pkl \
  --visualize
```

**Batch classification:**
```bash
PYTHONPATH=src python3 examples/inference_wavelet.py \
  --dir path/to/images \
  --model wavelet_model.pkl
```

## Feature Extraction

The system extracts ~85-90 features from wavelet subbands:

1. **Energy Features** (~15): Normalized energy and energy ratios across subbands
2. **Statistical Moments** (~36): Mean, std, skewness, kurtosis for each detail subband
3. **GGD Parameters** (~18): Generalized Gaussian Distribution shape and scale parameters
4. **Cross-Scale Correlation** (~9): Correlation between adjacent decomposition levels
5. **Periodic Artifacts** (~6): Detection of checkerboard patterns in high-frequency subbands
6. **Noise Consistency** (~3): Variance consistency across image blocks

## Wavelet Decomposition

The system performs 3-level 2D discrete wavelet transform using Daubechies 4 (`db4`) wavelet:

- **Level 1**: LH1, HL1, HH1 (finest detail)
- **Level 2**: LH2, HL2, HH2 (medium detail)
- **Level 3**: LH3, HL3, HH3, LL3 (coarsest detail + approximation)

## Model Training

### Default Settings

- **Classifier**: Random Forest
  - `n_estimators`: 200
  - `max_depth`: 20
  - `min_samples_split`: 5

### Hyperparameter Tuning

Use `--tune` flag to perform grid search:

```bash
python train_wavelet.py --real-dir data/real --ai-dir data/ai --tune
```

This will search over:
- `n_estimators`: [100, 200]
- `max_depth`: [15, 20, 25]
- `min_samples_split`: [3, 5, 7]

### Expected Performance

With a balanced dataset of 1000+ images per class:
- **Accuracy**: 85-95%
- **AUC-ROC**: 0.90-0.98
- **False Positive Rate**: < 5% (on high-quality natural photography)

## Programmatic Usage

```python
from image_screener.wavelet_detector import WaveletDetector

# Load trained model
detector = WaveletDetector()
detector.load_model('wavelet_model.pkl')

# Classify image
prediction, probability = detector.predict('path/to/image.jpg')

if prediction == 1:
    print(f"AI-generated (confidence: {probability*100:.2f}%)")
else:
    print(f"Real image (confidence: {probability*100:.2f}%)")
```

## Feature Details

### GGD Parameter Estimation

The Generalized Gaussian Distribution (GGD) models the distribution of wavelet coefficients:

- **Natural images**: Coefficients follow a GGD with beta ≈ 1.0 (Gaussian-like)
- **AI images**: May show different beta values indicating structured patterns

### Noise Consistency

Natural images have consistent noise variance across blocks. AI images may show:
- Hyper-smoothed regions (very low variance)
- Structured patterns (high variance of variances)
- Inconsistent noise characteristics

### Periodic Artifacts

Detects checkerboard patterns common in:
- Transposed convolution upsampling
- GAN-generated images
- Diffusion model artifacts

## Troubleshooting

### "Image too small" error
- Minimum image size: 64×64 pixels
- Images are automatically padded to power-of-2 dimensions

### Low accuracy
- Ensure balanced training data (similar number of real and AI images)
- Try hyperparameter tuning with `--tune`
- Check image quality (avoid heavily compressed images)

### Memory issues
- Process images in batches
- Reduce image resolution if needed
- Use fewer decomposition levels (modify `WaveletDetector(levels=2)`)

## Integration with Filter 01

The wavelet detector (Filter 02) can be combined with the spectral peak detector (Filter 01) for improved accuracy:

```python
from image_screener.spectral_peak_detector import SpectralPeakDetector
from image_screener.wavelet_detector import WaveletDetector

# Run both detectors
fft_detector = SpectralPeakDetector()
wavelet_detector = WaveletDetector()
wavelet_detector.load_model('wavelet_model.pkl')

fft_result = fft_detector.analyze('image.jpg')
wavelet_pred, wavelet_prob = wavelet_detector.predict('image.jpg')

# Combine scores (weighted average)
combined_score = (fft_result.artifact_score * 0.5) + (wavelet_prob * 0.5)
```

## References

Based on the PRD Section 3.2: Noise Residual Analysis (Wavelet Filter)
- Uses BayesShrink-like wavelet denoising principles
- Analyzes Photo-Response Non-Uniformity (PRNU) absence
- Detects VAE reconstruction artifacts

