# AI Image Detection Filter Specifications

## [Filter 01] - Spectral Peak Detector (FFT)
- **Target:** Upsampling artifacts / Periodic grid noise.
- **Domain:** Frequency (Fourier).
- **Metric:** Detection of Dirac-like spikes in the high-frequency spectrum.
- **Sensitivity:** High for Diffusion models (Stable Diffusion/Midjourney).

## [Filter 02] - Residual Noise Entropy
- **Target:** Lack of sensor PRNU / VAE artifacts.
- **Domain:** Spatial (Noise Residual).
- **Metric:** Variance of the noise residual after Wavelet denoising.
- **Expectation:** AI images show lower entropy and higher spatial correlation in noise.

## [Filter 03] - Geometric Gradient Distribution
- **Target:** GAN/Diffusion "checkerboard" artifacts.
- **Domain:** Gradient (Sobel/Scharr).
- **Metric:** Histogram of Oriented Gradients (HOG) at a micro-scale.
- **Expectation:** Synthetic images show unnatural alignment of gradients with the pixel grid.

## [Filter 04] - Chromatic Channel Misalignment
- **Target:** Absence of physical lens properties.
- **Domain:** Color Space (RGB).
- **Metric:** Radial displacement vector field between R and B channels.
- **Expectation:** Natural photos show radial divergence; AI shows static or zero divergence.