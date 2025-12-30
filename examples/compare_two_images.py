"""Compare FFT analysis for two images and save visualizations.

This script runs the FFT visualization tool on two images (typically real vs fake)
and saves the outputs to the comparison_fft folder for side-by-side comparison.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

from image_screener.spectral_peak_detector import SpectralPeakDetector

# Import the visualization function dynamically
spec = importlib.util.spec_from_file_location(
    "visualize_fft",
    Path(__file__).parent / "visualize_fft.py"
)
visualize_fft_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(visualize_fft_module)
create_fft_visualization = visualize_fft_module.create_fft_visualization


def compare_images(
    image1_path: Path,
    image2_path: Path,
    output_dir: Path,
    label1: str = None,
    label2: str = None,
    **detector_kwargs
):
    """
    Compare two images using FFT analysis and save visualizations.

    Args:
        image1_path: Path to first image
        image2_path: Path to second image
        output_dir: Directory to save output visualizations
        label1: Optional label for first image (used in filename)
        label2: Optional label for second image (used in filename)
        **detector_kwargs: Additional arguments for SpectralPeakDetector
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize detector
    detector = SpectralPeakDetector(**detector_kwargs)

    # Determine labels from filenames if not provided
    if label1 is None:
        label1 = image1_path.stem
    if label2 is None:
        label2 = image2_path.stem

    print("=" * 80)
    print("FFT COMPARISON: Two Image Analysis")
    print("=" * 80)

    # Analyze first image
    print(f"\n[1/2] Analyzing: {image1_path.name}")
    print("-" * 80)
    output1 = output_dir / f"{label1}_analysis.png"
    
    try:
        result1 = create_fft_visualization(
            image1_path,
            detector,
            output_path=output1,
            show_peaks=True,
            show_original=True,
        )
        print(f"✓ Saved: {output1}")
        print(f"  Artifact Score: {result1.artifact_score:.4f}")
        print(f"  Peaks: {result1.num_peaks}")
        print(f"  Grid Strength: {result1.grid_strength:.4f}")
        print(f"  Grid Consistency: {result1.grid_consistency:.4f}")
        print(f"  Nyquist Symmetry: {result1.nyquist_symmetry:.4f}")
    except Exception as e:
        print(f"✗ Error analyzing {image1_path}: {e}")
        result1 = None

    # Analyze second image
    print(f"\n[2/2] Analyzing: {image2_path.name}")
    print("-" * 80)
    output2 = output_dir / f"{label2}_analysis.png"
    
    try:
        result2 = create_fft_visualization(
            image2_path,
            detector,
            output_path=output2,
            show_peaks=True,
            show_original=True,
        )
        print(f"✓ Saved: {output2}")
        print(f"  Artifact Score: {result2.artifact_score:.4f}")
        print(f"  Peaks: {result2.num_peaks}")
        print(f"  Grid Strength: {result2.grid_strength:.4f}")
        print(f"  Grid Consistency: {result2.grid_consistency:.4f}")
        print(f"  Nyquist Symmetry: {result2.nyquist_symmetry:.4f}")
    except Exception as e:
        print(f"✗ Error analyzing {image2_path}: {e}")
        result2 = None

    # Print comparison summary
    if result1 and result2:
        print("\n" + "=" * 80)
        print("COMPARISON SUMMARY")
        print("=" * 80)
        print(f"\n{'Metric':<30} {'Image 1':<25} {'Image 2':<25}")
        print("-" * 80)
        print(f"{'Artifact Score':<30} {result1.artifact_score:<25.4f} {result2.artifact_score:<25.4f}")
        print(f"{'Number of Peaks':<30} {result1.num_peaks:<25} {result2.num_peaks:<25}")
        print(f"{'Grid Strength':<30} {result1.grid_strength:<25.4f} {result2.grid_strength:<25.4f}")
        print(f"{'Grid Consistency':<30} {result1.grid_consistency:<25.4f} {result2.grid_consistency:<25.4f}")
        print(f"{'Nyquist Symmetry':<30} {result1.nyquist_symmetry:<25.4f} {result2.nyquist_symmetry:<25.4f}")
        
        if result1.grid_strength > 0.1 or result2.grid_strength > 0.1:
            grid_int1 = f"({result1.grid_interval_u:.1f}, {result1.grid_interval_v:.1f})"
            grid_int2 = f"({result2.grid_interval_u:.1f}, {result2.grid_interval_v:.1f})"
            print(f"{'Grid Interval (u, v)':<30} {grid_int1:<25} {grid_int2:<25}")
        
        # Interpretation
        print("\n" + "-" * 80)
        print("INTERPRETATION:")
        
        score_diff = abs(result1.artifact_score - result2.artifact_score)
        if score_diff > 0.3:
            higher = "Image 1" if result1.artifact_score > result2.artifact_score else "Image 2"
            print(f"  ⚠️  Significant difference detected ({score_diff:.4f})")
            print(f"  {higher} shows stronger AI generation signatures")
        else:
            print(f"  Similar artifact scores - both images may be from the same source type")
        
        if result1.grid_strength > 0.5 or result2.grid_strength > 0.5:
            strong_grid = "Image 1" if result1.grid_strength > result2.grid_strength else "Image 2"
            print(f"  🔴 Strong grid pattern detected in {strong_grid}")
            print(f"     This is a strong indicator of AI upsampling artifacts")

    print("\n" + "=" * 80)
    print(f"Visualizations saved to: {output_dir}")
    print("=" * 80)


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Compare FFT analysis for two images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two specific images
  python compare_two_images.py image1.jpg image2.png

  # Compare real vs fake from test images
  python compare_two_images.py \\
    "documentation/test images/01/real.jpg" \\
    "documentation/test images/01/fake.png"

  # With custom labels
  python compare_two_images.py image1.jpg image2.png --label1 real --label2 fake
        """
    )
    
    parser.add_argument(
        'image1',
        type=str,
        help='Path to first image'
    )
    
    parser.add_argument(
        'image2',
        type=str,
        help='Path to second image'
    )
    
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default='documentation/test images/comparison_fft',
        help='Output directory for visualizations (default: documentation/test images/comparison_fft)'
    )
    
    parser.add_argument(
        '--label1',
        type=str,
        default=None,
        help='Label for first image (used in output filename)'
    )
    
    parser.add_argument(
        '--label2',
        type=str,
        default=None,
        help='Label for second image (used in output filename)'
    )
    
    parser.add_argument(
        '--target-size',
        type=int,
        default=512,
        choices=[256, 512, 1024],
        help='Target size for image preprocessing (default: 512)'
    )
    
    parser.add_argument(
        '--sensitivity',
        type=float,
        default=1.0,
        help='Sensitivity constant for peak detection (default: 1.0)'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.3,
        help='High-frequency threshold (0.0-1.0, default: 0.3)'
    )
    
    parser.add_argument(
        '--percentile',
        type=float,
        default=95.0,
        help='Peak threshold percentile (default: 95.0)'
    )
    
    args = parser.parse_args()
    
    # Validate image paths
    image1_path = Path(args.image1)
    image2_path = Path(args.image2)
    
    if not image1_path.exists():
        print(f"Error: Image not found: {image1_path}", file=sys.stderr)
        sys.exit(1)
    
    if not image2_path.exists():
        print(f"Error: Image not found: {image2_path}", file=sys.stderr)
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    
    # Run comparison
    compare_images(
        image1_path,
        image2_path,
        output_dir,
        label1=args.label1,
        label2=args.label2,
        target_size=args.target_size,
        sensitivity=args.sensitivity,
        high_freq_threshold=args.threshold,
        peak_threshold_percentile=args.percentile,
    )


if __name__ == "__main__":
    main()

