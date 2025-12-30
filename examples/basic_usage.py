"""Basic usage example for Spectral Peak Detector."""

import logging
from pathlib import Path

from image_screener.spectral_peak_detector import SpectralPeakDetector

# Configure logging
logging.basicConfig(level=logging.INFO)

# Example: Analyze a single image
def analyze_single_image():
    """Analyze a single image for spectral artifacts."""
    detector = SpectralPeakDetector(
        target_size=512,
        sensitivity=1.0,
        high_freq_threshold=0.3,
        peak_threshold_percentile=95.0,
    )

    # Use test images from documentation
    image_path = Path("documentation/test images/01/real.jpg")

    if not image_path.exists():
        print(f"Image not found: {image_path}")
        print("Please provide a valid image path")
        return

    result = detector.analyze(image_path)

    print(f"\nAnalysis Results for: {result.image_path}")
    print(f"Artifact Score (F_a): {result.artifact_score:.4f}")
    print(f"Number of Detected Peaks: {result.num_peaks}")
    print(f"\nTop 5 Peaks:")
    for i, peak in enumerate(result.peaks[:5], 1):
        print(
            f"  {i}. Position: ({peak.u}, {peak.v}), "
            f"Magnitude: {peak.magnitude:.4f}, "
            f"Distance: {peak.distance_from_center:.2f}"
        )


# Example: Batch analyze multiple images
def batch_analyze():
    """Analyze multiple images in batch."""
    detector = SpectralPeakDetector()

    # Find all test images
    test_dir = Path("documentation/test images")
    image_paths = []
    for ext in ["*.jpg", "*.png"]:
        image_paths.extend(test_dir.rglob(ext))

    if not image_paths:
        print("No test images found")
        return

    print(f"Analyzing {len(image_paths)} images...\n")
    results = detector.batch_analyze(image_paths)

    print("Batch Analysis Results:")
    print("-" * 60)
    for result in results:
        print(
            f"{Path(result.image_path).name:30s} | "
            f"Score: {result.artifact_score:.4f} | "
            f"Peaks: {result.num_peaks:3d}"
        )


if __name__ == "__main__":
    print("=" * 60)
    print("Spectral Peak Detector - Basic Usage Example")
    print("=" * 60)

    # Run single image analysis
    analyze_single_image()

    print("\n" + "=" * 60)
    print("\n")

    # Run batch analysis
    batch_analyze()

