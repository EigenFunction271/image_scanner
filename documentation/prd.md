# PRD: Project "SpamBlamThankYouMaam" — Non-Watermark AI Detection

## 1. Executive Summary

**SpamBlamThankYouMaam** is an experimental image forensics tool designed to distinguish between natural photography and AI-generated imagery. Unlike watermark-based detection, SignalTrace analyzes low-level architectural artifacts—specifically frequency spikes from upsampling, VAE reconstruction noise, and the absence of physical sensor signatures (PRNU).

---

## 2. Core Problem Statement

AI image generators (Stable Diffusion, Midjourney, DALL-E 3) do not create images pixel-by-pixel in the way a sensor captures light. Instead:

1. They upscale latent representations, leaving **periodic spectral artifacts**.
2. They use decoders (VAEs) that struggle with **stochastic micro-textures**.
3. They lack the **physical lens/sensor imperfections** (Chromatic Aberration, PRNU) found in real cameras.

---

## 3. Technical Requirements & Model Architecture

### 3.1 Frequency Domain Analysis (FFT Filter)

* **Requirement:** Identify periodic "grid" artifacts caused by transposed convolutions.
* **Logic:** * Compute 2D Discrete Fourier Transform (DFT).
* Magnitude spectrum must be shifted and log-scaled.
* **Feature Extraction:** Detect "peaks" in high-frequency coordinates  where  (where  is a sensitivity constant).


* **Success Metric:** Detection of artificial symmetry in the spectral power distribution.

### 3.2 Noise Residual Analysis (Wavelet Filter)

* **Requirement:** Isolate the "noise" of the image to check for Photo-Response Non-Uniformity (PRNU).
* **Logic:** * Apply a Denoising Filter (e.g., BayesShrink Wavelet Denoising) to extract the residual .
* **Feature Extraction:** Calculate the Kurtosis and Skewness of the noise distribution.


* **Hypothesis:** Natural images exhibit high-entropy, spatially white noise. AI images exhibit "structured" noise or hyper-smoothed regions with zero entropy.

### 3.3 Geometric Gradient Consistency

* **Requirement:** Detect the "Checkerboard Effect."
* **Logic:** * Apply Sobel operators to find the gradient magnitude.
* Perform an Autocorrelation on the gradient map.


* **Feature Extraction:** Search for repeating peaks at  pixel intervals (common in power-of-two upsampling architectures).

---

## 4. User Stories

| ID | User | Requirement | Goal |
| --- | --- | --- | --- |
| **US.1** | Researcher | Upload a directory of mixed images. | Generate a "Synthetic Probability Score" based on spectral peaks. |
| **US.2** | Developer | Access a CLI tool. | Pipe image buffers into the forensic filters and get JSON metadata. |
| **US.3** | Analyst | Visualize the frequency spectrum. | Manually confirm the presence of "grid dots" in the Fourier domain. |

---

## 5. Functional Specifications

### 5.1 Input Processing

* Support for `.jpg`, `.png`, and `.webp`.
* Mandatory conversion to floating-point grayscale for mathematical precision.
* Normalization of image dimensions to  or  for consistent FFT binning.

### 5.2 The Scoring Engine

The final output should be a weighted index ():



Where:

*  = Frequency Artifact Score
*  = Noise Entropy Score
*  = Chromatic Consistency Score

---

## 6. Implementation Milestones

### Phase 1: Signal Extraction (Week 1)

* Build the FFT visualization module.
* Implement the Wavelet-based noise extractor.

### Phase 2: Calibration (Week 2)

* Run the script against a "Clean Dataset" (COCO or ImageNet) vs. a "Synthetic Dataset" (Stable Diffusion XL).
* Define the threshold values for "AI-positive" spectral peaks.

### Phase 3: Reporting (Week 3)

* Build a Markdown generator that outputs forensic reports with embedded plots.

---

## 7. Success Criteria

* **False Positive Rate (FPR):** < 5% on high-quality natural photography.
* **Detection Rate:** > 85% for images generated via Latent Diffusion Models without heavy post-processing.

---

**Next Step:** Would you like me to write the **Python Class structure** for the `SignalTrace` engine, including the boilerplate for the NumPy/SciPy Fourier transformations?