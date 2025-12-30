"""Visual FFT tool - Analyze a single image and visualize frequency domain peaks.

This tool is designed to visually demonstrate the difference between AI-generated
and real images by showing periodic artifacts in the frequency domain.
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)

from image_screener.spectral_peak_detector import SpectralPeakDetector

# Configure logging
logging.basicConfig(level=logging.WARNING)


def create_fft_visualization(
    image_path: Path,
    detector: SpectralPeakDetector,
    output_path: Path = None,
    show_peaks: bool = True,
    show_original: bool = True,
):
    """
    Create a comprehensive FFT visualization for a single image.

    Args:
        image_path: Path to the image file
        detector: SpectralPeakDetector instance
        output_path: Optional path to save the visualization
        show_peaks: Whether to highlight detected peaks
        show_original: Whether to show the original image
    """
    print(f"Analyzing image: {image_path}")
    
    # Analyze the image
    result = detector.analyze(image_path)
    
    print(f"  Artifact Score: {result.artifact_score:.4f}")
    print(f"  Detected Peaks: {result.num_peaks}")
    
    # Create figure with all plots in one file
    # Layout: [Original | Spectrum] on top, [Azimuthal] below
    if show_original:
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3, height_ratios=[1, 0.6])
        ax_original = fig.add_subplot(gs[0, 0])
        ax_spectrum_peaks = fig.add_subplot(gs[0, 1])
        ax_azimuthal = fig.add_subplot(gs[1, :])  # Full width for azimuthal
    else:
        fig = plt.figure(figsize=(20, 10))
        gs = fig.add_gridspec(2, 1, hspace=0.3, wspace=0.3, height_ratios=[1, 0.6])
        ax_spectrum_peaks = fig.add_subplot(gs[0, 0])
        ax_azimuthal = fig.add_subplot(gs[1, 0])
    
    # Build title with grid information
    title = (
        f'FFT Analysis: {Path(image_path).name}\n'
        f'Artifact Score: {result.artifact_score:.4f} | Peaks: {result.num_peaks}'
    )
    if result.grid_strength > 0.1 or result.grid_consistency > 0.1:
        title += (
            f'\nGrid Pattern: Strength={result.grid_strength:.3f} | '
            f'Consistency={result.grid_consistency:.3f} | '
            f'Symmetry={result.nyquist_symmetry:.3f}'
        )
        if result.grid_interval_u > 0 or result.grid_interval_v > 0:
            title += f' | Interval: ({result.grid_interval_u:.1f}, {result.grid_interval_v:.1f})'
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # 1. Original Image (if requested)
    if show_original:
        ax_original.imshow(result.preprocessed_image, cmap='gray')
        ax_original.set_title('Original Image (Grayscale)', fontsize=12, fontweight='bold')
        ax_original.axis('off')
    
    # 2. Spectrum with Peak Markers (only this spectrum view)
    spectrum_display = result.log_magnitude_spectrum
    h, w = spectrum_display.shape
    ax_spectrum_peaks.imshow(
        spectrum_display,
        cmap='hot',
        aspect='auto',
        interpolation='bilinear'
    )
    ax_spectrum_peaks.set_title(
        f'Frequency Spectrum with Detected Peaks ({result.num_peaks} peaks)',
        fontsize=12,
        fontweight='bold'
    )
    ax_spectrum_peaks.set_xlabel('Frequency (u)', fontsize=10)
    ax_spectrum_peaks.set_ylabel('Frequency (v)', fontsize=10)
    
    # Mark detected peaks
    if show_peaks and result.peaks:
        peak_u = [p.u for p in result.peaks]
        peak_v = [p.v for p in result.peaks]
        peak_mags = [p.magnitude for p in result.peaks]
        
        # Scatter plot of peaks, sized by magnitude
        scatter = ax_spectrum_peaks.scatter(
            peak_u,
            peak_v,
            c=peak_mags,
            cmap='cool',
            s=100,
            edgecolors='white',
            linewidths=2,
            alpha=0.8,
            zorder=10
        )
        
        # Add colorbar for peak magnitudes
        cbar2 = plt.colorbar(scatter, ax=ax_spectrum_peaks)
        cbar2.set_label('Peak Magnitude', fontsize=10)
        
        # Mark top 5 peaks with labels
        top_peaks = sorted(result.peaks, key=lambda p: p.magnitude, reverse=True)[:5]
        for i, peak in enumerate(top_peaks, 1):
            ax_spectrum_peaks.annotate(
                f'#{i}',
                (peak.u, peak.v),
                xytext=(5, 5),
                textcoords='offset points',
                color='white',
                fontsize=10,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7)
            )
    
    # Add center crosshair
    ax_spectrum_peaks.axhline(y=h//2, color='cyan', linestyle='--', linewidth=1, alpha=0.5)
    ax_spectrum_peaks.axvline(x=w//2, color='cyan', linestyle='--', linewidth=1, alpha=0.5)
    
    # 3. Azimuthal Average Plot (1D profile showing periodic artifacts) - in separate figure
    ax_azimuthal.plot(
        result.azimuthal_radii,
        result.azimuthal_average,
        linewidth=2,
        color='#2c3e50',
        label='Azimuthal Average'
    )
    
    # Detect and mark peaks in azimuthal average
    from image_screener.dft import DFTProcessor
    temp_processor = DFTProcessor()
    azimuthal_peaks = temp_processor.detect_azimuthal_peaks(
        result.azimuthal_radii, result.azimuthal_average, min_radius=0.3
    )
    
    if azimuthal_peaks:
        peak_radii = [p[0] for p in azimuthal_peaks]
        peak_mags = [p[1] for p in azimuthal_peaks]
        ax_azimuthal.scatter(
            peak_radii,
            peak_mags,
            color='red',
            s=100,
            zorder=10,
            label=f'Detected Peaks ({len(azimuthal_peaks)})',
            edgecolors='white',
            linewidths=2
        )
        
        # Annotate significant peaks
        for i, (radius, mag) in enumerate(azimuthal_peaks[:5], 1):
            ax_azimuthal.annotate(
                f'Peak {i}',
                (radius, mag),
                xytext=(10, 10),
                textcoords='offset points',
                fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
            )
    
    ax_azimuthal.set_xlabel('Normalized Radius (0 = center, 1 = corner)', fontsize=12, fontweight='bold')
    ax_azimuthal.set_ylabel('Average Log Magnitude', fontsize=12, fontweight='bold')
    ax_azimuthal.set_title(
        'Azimuthal Average: Periodic Artifacts Appear as Sharp Spikes',
        fontsize=14,
        fontweight='bold'
    )
    ax_azimuthal.grid(True, alpha=0.3)
    ax_azimuthal.legend()
    
    # Add interpretation text
    if azimuthal_peaks:
        interpretation = (
            f"⚠️ {len(azimuthal_peaks)} periodic spikes detected! "
            "Sharp bumps indicate upsampling artifacts from AI generation."
        )
    else:
        interpretation = "✓ Smooth decay - typical of natural images."
    
    ax_azimuthal.text(
        0.02, 0.98,
        interpretation,
        transform=ax_azimuthal.transAxes,
        fontsize=11,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    )
    
    # Layout and save main figure
    plt.tight_layout()
    
    # Save or show figure
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Visualization saved to: {output_path}")
    else:
        plt.show()
    
    plt.close(fig)
    
    return result


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Visual FFT tool - Analyze single image and visualize frequency domain peaks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze an image and display interactively
  python visualize_fft.py path/to/image.jpg

  # Save visualization to file
  python visualize_fft.py path/to/image.jpg -o output.png

  # Analyze with custom settings
  python visualize_fft.py path/to/image.jpg --sensitivity 1.5 --threshold 0.4
        """
    )
    
    parser.add_argument(
        'image',
        type=str,
        help='Path to the image file to analyze'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output path for the visualization (default: display interactively)'
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
    
    parser.add_argument(
        '--no-peaks',
        action='store_true',
        help='Do not highlight detected peaks'
    )
    
    parser.add_argument(
        '--no-original',
        action='store_true',
        help='Do not show original image'
    )
    
    args = parser.parse_args()
    
    # Validate image path
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}", file=sys.stderr)
        sys.exit(1)
    
    # Initialize detector
    detector = SpectralPeakDetector(
        target_size=args.target_size,
        sensitivity=args.sensitivity,
        high_freq_threshold=args.threshold,
        peak_threshold_percentile=args.percentile,
    )
    
    # Create visualization
    output_path = Path(args.output) if args.output else None
    result = create_fft_visualization(
        image_path,
        detector,
        output_path=output_path,
        show_peaks=not args.no_peaks,
        show_original=not args.no_original,
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Image: {image_path}")
    print(f"Artifact Score: {result.artifact_score:.4f}")
    print(f"Number of Peaks: {result.num_peaks}")
    print(f"Grid Pattern Strength: {result.grid_strength:.4f}")
    print(f"Grid Consistency (Power-of-2): {result.grid_consistency:.4f}")
    print(f"Nyquist Symmetry: {result.nyquist_symmetry:.4f}")
    if result.grid_strength > 0.1:
        print(f"Grid Interval: ({result.grid_interval_u:.1f}, {result.grid_interval_v:.1f}) pixels")
    
    if result.peaks:
        print(f"\nTop 5 Peaks:")
        for i, peak in enumerate(result.peaks[:5], 1):
            print(
                f"  {i}. Position: ({peak.u:4d}, {peak.v:4d}), "
                f"Magnitude: {peak.magnitude:8.4f}, "
                f"Distance from center: {peak.distance_from_center:6.2f}"
            )
    
    # Interpretation
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    if result.grid_consistency > 0.5 or result.nyquist_symmetry > 0.5:
        print("🔴 NYQUIST FOLDING SYMMETRY DETECTED - Very likely AI-generated!")
        print("   Peaks align to power-of-2 grid (4, 8, 16) - characteristic of upsampling")
        print(f"   Grid Consistency: {result.grid_consistency:.3f}")
        print(f"   Nyquist Symmetry: {result.nyquist_symmetry:.3f}")
        if result.grid_interval_u > 0 or result.grid_interval_v > 0:
            print(f"   Grid interval: ({result.grid_interval_u:.1f}, {result.grid_interval_v:.1f}) pixels")
    elif result.grid_strength > 0.5:
        print("🔴 STRONG GRID PATTERN DETECTED - Very likely AI-generated!")
        print("   Repeating spatial intervals indicate upsampling artifacts")
        print(f"   Grid interval: ({result.grid_interval_u:.1f}, {result.grid_interval_v:.1f}) pixels")
    elif result.artifact_score > 0.5:
        print("⚠️  HIGH artifact score - Likely AI-generated image")
        if result.grid_strength > 0.3 or result.grid_consistency > 0.3:
            print("   Grid pattern detected - strong indicator of AI generation")
        print("   Look for periodic patterns and symmetric peaks in the spectrum")
    elif result.artifact_score > 0.3:
        print("⚠️  MODERATE artifact score - Possibly AI-generated")
        print("   Check for clusters of peaks in high-frequency regions")
    else:
        print("✓  LOW artifact score - Likely real/natural image")
        print("   Natural images typically show more uniform frequency distribution")


if __name__ == "__main__":
    main()

