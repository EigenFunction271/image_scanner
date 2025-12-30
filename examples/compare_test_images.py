"""Compare real vs fake images from test images folder."""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from image_screener.spectral_peak_detector import DetectionResult, SpectralPeakDetector

# Configure logging
logging.basicConfig(level=logging.WARNING)  # Reduce noise for comparison output


def find_test_images(test_dir: Path) -> Dict[str, List[Path]]:
    """
    Find all real and fake images in test directory structure.

    Args:
        test_dir: Path to test images directory

    Returns:
        Dictionary with 'real' and 'fake' keys containing lists of image paths
    """
    images = defaultdict(list)

    # Look for images matching "real" or "fake" patterns
    for image_path in test_dir.rglob("*.png"):
        name_lower = image_path.stem.lower()
        if "real" in name_lower:
            images["real"].append(image_path)
        elif "fake" in name_lower:
            images["fake"].append(image_path)

    for image_path in test_dir.rglob("*.jpg"):
        name_lower = image_path.stem.lower()
        if "real" in name_lower:
            images["real"].append(image_path)

    return dict(images)


def analyze_images(
    detector: SpectralPeakDetector, image_paths: List[Path]
) -> List[DetectionResult]:
    """Analyze a list of images and return results."""
    results = []
    for img_path in image_paths:
        try:
            result = detector.analyze(img_path)
            results.append(result)
        except Exception as e:
            print(f"Error analyzing {img_path}: {e}")
    return results


def get_relative_path(image_path: str) -> str:
    """
    Convert image path to relative path from current working directory.
    
    Handles both absolute and relative paths gracefully.
    """
    path = Path(image_path)
    try:
        # Try to make it relative to cwd
        if path.is_absolute():
            rel_path = path.relative_to(Path.cwd())
        else:
            # Already relative, but resolve it to handle .. properly
            rel_path = path
        return str(rel_path)
    except ValueError:
        # If it's not a subpath, just return the path as-is
        return str(path)


def print_comparison_table(results: Dict[str, List[DetectionResult]]):
    """Print a formatted comparison table of results."""
    print("\n" + "=" * 100)
    print("SPECTRAL PEAK DETECTOR - REAL vs FAKE COMPARISON")
    print("=" * 100)

    # Print header
    print(f"\n{'Category':<15} {'Image Path':<40} {'Score':<10} {'Peaks':<8} {'Grid':<8} {'Grid Int':<12}")
    print("-" * 100)

    # Print real images
    print("\n📸 REAL IMAGES:")
    for result in results.get("real", []):
        rel_path = get_relative_path(result.image_path)
        grid_info = f"{result.grid_strength:.3f}" if result.grid_strength > 0.01 else "0.000"
        grid_int = f"({result.grid_interval_u:.0f},{result.grid_interval_v:.0f})" if result.grid_strength > 0.1 else "N/A"
        print(
            f"{'REAL':<15} {rel_path:<40} "
            f"{result.artifact_score:<10.4f} {result.num_peaks:<8} {grid_info:<8} {grid_int:<12}"
        )

    # Print fake images
    print("\n🤖 FAKE IMAGES:")
    for result in results.get("fake", []):
        rel_path = get_relative_path(result.image_path)
        grid_info = f"{result.grid_strength:.3f}" if result.grid_strength > 0.01 else "0.000"
        grid_int = f"({result.grid_interval_u:.0f},{result.grid_interval_v:.0f})" if result.grid_strength > 0.1 else "N/A"
        print(
            f"{'FAKE':<15} {rel_path:<40} "
            f"{result.artifact_score:<10.4f} {result.num_peaks:<8} {grid_info:<8} {grid_int:<12}"
        )


def print_statistics(results: Dict[str, List[DetectionResult]]):
    """Print statistical comparison between real and fake images."""
    print("\n" + "=" * 100)
    print("STATISTICAL SUMMARY")
    print("=" * 100)

    for category in ["real", "fake"]:
        category_results = results.get(category, [])
        if not category_results:
            continue

        scores = [r.artifact_score for r in category_results]
        peak_counts = [r.num_peaks for r in category_results]
        grid_strengths = [r.grid_strength for r in category_results]

        print(f"\n{category.upper()} IMAGES ({len(category_results)} total):")
        print(f"  Artifact Score:")
        print(f"    Mean:   {sum(scores) / len(scores):.4f}")
        print(f"    Min:    {min(scores):.4f}")
        print(f"    Max:    {max(scores):.4f}")
        print(f"    StdDev: {(sum((x - sum(scores)/len(scores))**2 for x in scores) / len(scores))**0.5:.4f}")

        print(f"  Peak Count:")
        print(f"    Mean:   {sum(peak_counts) / len(peak_counts):.1f}")
        print(f"    Min:    {min(peak_counts)}")
        print(f"    Max:    {max(peak_counts)}")

        print(f"  Grid Pattern Strength:")
        print(f"    Mean:   {sum(grid_strengths) / len(grid_strengths):.4f}")
        print(f"    Min:    {min(grid_strengths):.4f}")
        print(f"    Max:    {max(grid_strengths):.4f}")
        strong_grid_count = sum(1 for g in grid_strengths if g > 0.5)
        print(f"    Images with strong grid (>0.5): {strong_grid_count}/{len(grid_strengths)}")

    # Comparison
    real_results = results.get("real", [])
    fake_results = results.get("fake", [])

    if real_results and fake_results:
        real_avg_score = sum(r.artifact_score for r in real_results) / len(real_results)
        fake_avg_score = sum(r.artifact_score for r in fake_results) / len(fake_results)
        real_avg_grid = sum(r.grid_strength for r in real_results) / len(real_results)
        fake_avg_grid = sum(r.grid_strength for r in fake_results) / len(fake_results)

        print("\n" + "-" * 100)
        print("COMPARISON:")
        print(f"  Artifact Score:")
        print(f"    Real avg: {real_avg_score:.4f}")
        print(f"    Fake avg: {fake_avg_score:.4f}")
        print(f"    Difference: {fake_avg_score - real_avg_score:.4f}")
        print(f"    Ratio: {fake_avg_score / real_avg_score:.2f}x" if real_avg_score > 0 else "    Ratio: N/A")
        print(f"  Grid Pattern Strength:")
        print(f"    Real avg: {real_avg_grid:.4f}")
        print(f"    Fake avg: {fake_avg_grid:.4f}")
        print(f"    Difference: {fake_avg_grid - real_avg_grid:.4f}")
        if fake_avg_grid > 0.5:
            print(f"    ⚠️  Strong grid patterns detected in fake images!")


def print_detailed_peaks(results: Dict[str, List[DetectionResult]], top_n: int = 3):
    """Print detailed peak information for each image."""
    print("\n" + "=" * 100)
    print(f"DETAILED PEAK INFORMATION (Top {top_n} peaks per image)")
    print("=" * 100)

    for category in ["real", "fake"]:
        category_results = results.get(category, [])
        if not category_results:
            continue

        print(f"\n{category.upper()} IMAGES:")
        for result in category_results:
            rel_path = get_relative_path(result.image_path)
            print(f"\n  {rel_path}")
            print(f"    Artifact Score: {result.artifact_score:.4f}")
            print(f"    Total Peaks: {result.num_peaks}")

            if result.peaks:
                print(f"    Top {min(top_n, len(result.peaks))} Peaks:")
                for i, peak in enumerate(result.peaks[:top_n], 1):
                    print(
                        f"      {i}. Position: ({peak.u:4d}, {peak.v:4d}), "
                        f"Magnitude: {peak.magnitude:8.4f}, "
                        f"Distance: {peak.distance_from_center:6.2f}"
                    )
            else:
                print("    No peaks detected")


def create_visualizations(results: Dict[str, List[DetectionResult]], output_dir: Path = None):
    """
    Create visualization graphs comparing real vs fake images.
    
    Args:
        results: Dictionary with 'real' and 'fake' keys containing DetectionResult lists
        output_dir: Optional directory to save plots. If None, displays interactively.
    """
    real_results = results.get("real", [])
    fake_results = results.get("fake", [])
    
    if not real_results and not fake_results:
        print("No results to visualize")
        return
    
    # Extract data
    real_scores = [r.artifact_score for r in real_results]
    fake_scores = [r.artifact_score for r in fake_results]
    real_peaks = [r.num_peaks for r in real_results]
    fake_peaks = [r.num_peaks for r in fake_results]
    real_top_mags = [r.peaks[0].magnitude if r.peaks else 0.0 for r in real_results]
    fake_top_mags = [r.peaks[0].magnitude if r.peaks else 0.0 for r in fake_results]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Bar chart: Average scores comparison
    ax1 = plt.subplot(2, 3, 1)
    categories = ["Real", "Fake"]
    avg_scores = [
        np.mean(real_scores) if real_scores else 0,
        np.mean(fake_scores) if fake_scores else 0
    ]
    std_scores = [
        np.std(real_scores) if real_scores else 0,
        np.std(fake_scores) if fake_scores else 0
    ]
    
    bars = ax1.bar(categories, avg_scores, yerr=std_scores, capsize=5, 
                   color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Artifact Score', fontsize=12, fontweight='bold')
    ax1.set_title('Average Artifact Score Comparison', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(bottom=0)
    
    # Add value labels on bars
    for i, (bar, avg, std) in enumerate(zip(bars, avg_scores, std_scores)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                f'{avg:.4f}\n(±{std:.4f})',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 2. Box plot: Score distribution
    ax2 = plt.subplot(2, 3, 2)
    box_data = []
    labels = []
    if real_scores:
        box_data.append(real_scores)
        labels.append('Real')
    if fake_scores:
        box_data.append(fake_scores)
        labels.append('Fake')
    
    bp = ax2.boxplot(box_data, labels=labels, patch_artist=True,
                     boxprops=dict(facecolor='lightblue', alpha=0.7),
                     medianprops=dict(color='red', linewidth=2))
    ax2.set_ylabel('Artifact Score', fontsize=12, fontweight='bold')
    ax2.set_title('Score Distribution (Box Plot)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Histogram: Score distribution overlay
    ax3 = plt.subplot(2, 3, 3)
    if real_scores:
        ax3.hist(real_scores, bins=10, alpha=0.6, label='Real', color='#2ecc71', edgecolor='black')
    if fake_scores:
        ax3.hist(fake_scores, bins=10, alpha=0.6, label='Fake', color='#e74c3c', edgecolor='black')
    ax3.set_xlabel('Artifact Score', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax3.set_title('Score Distribution (Histogram)', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Scatter: Score vs Peak Count
    ax4 = plt.subplot(2, 3, 4)
    if real_results:
        ax4.scatter(real_peaks, real_scores, alpha=0.6, s=100, label='Real', 
                   color='#2ecc71', edgecolor='black', linewidth=1.5)
    if fake_results:
        ax4.scatter(fake_peaks, fake_scores, alpha=0.6, s=100, label='Fake', 
                   color='#e74c3c', edgecolor='black', linewidth=1.5, marker='^')
    ax4.set_xlabel('Number of Peaks', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Artifact Score', fontsize=12, fontweight='bold')
    ax4.set_title('Score vs Peak Count', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Bar chart: Average peak counts
    ax5 = plt.subplot(2, 3, 5)
    avg_peaks = [
        np.mean(real_peaks) if real_peaks else 0,
        np.mean(fake_peaks) if fake_peaks else 0
    ]
    std_peaks = [
        np.std(real_peaks) if real_peaks else 0,
        np.std(fake_peaks) if fake_peaks else 0
    ]
    
    bars = ax5.bar(categories, avg_peaks, yerr=std_peaks, capsize=5,
                   color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black', linewidth=1.5)
    ax5.set_ylabel('Number of Peaks', fontsize=12, fontweight='bold')
    ax5.set_title('Average Peak Count Comparison', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.set_ylim(bottom=0)
    
    # Add value labels
    for bar, avg, std in zip(bars, avg_peaks, std_peaks):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + std + 1,
                f'{avg:.1f}\n(±{std:.1f})',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 6. Individual image scores
    ax6 = plt.subplot(2, 3, 6)
    all_scores = []
    all_labels = []
    all_colors = []
    
    # Add real scores
    for i, score in enumerate(real_scores):
        all_scores.append(score)
        all_labels.append(f'R{i+1}')
        all_colors.append('#2ecc71')
    
    # Add fake scores
    for i, score in enumerate(fake_scores):
        all_scores.append(score)
        all_labels.append(f'F{i+1}')
        all_colors.append('#e74c3c')
    
    if all_scores:
        x_pos = range(len(all_scores))
        bars = ax6.bar(x_pos, all_scores, color=all_colors, alpha=0.7, 
                     edgecolor='black', linewidth=1.5)
        ax6.set_xticks(x_pos)
        ax6.set_xticklabels(all_labels, rotation=45, ha='right')
        ax6.set_ylabel('Artifact Score', fontsize=12, fontweight='bold')
        ax6.set_title('Individual Image Scores', fontsize=14, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='y')
        ax6.set_ylim(bottom=0)
        
        # Add value labels
        for bar, score in zip(bars, all_scores):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}',
                    ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Save or show
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "comparison_results.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 Visualization saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    """Main comparison function."""
    # Find test images
    test_dir = Path("documentation/test images")
    if not test_dir.exists():
        print(f"Test directory not found: {test_dir}")
        print("Please run from project root directory")
        return

    images = find_test_images(test_dir)

    if not images:
        print(f"No test images found in {test_dir}")
        return

    print(f"Found {len(images.get('real', []))} real images and {len(images.get('fake', []))} fake images")

    # Initialize detector
    detector = SpectralPeakDetector(
        target_size=512,
        sensitivity=1.0,
        high_freq_threshold=0.3,
        peak_threshold_percentile=95.0,
    )

    # Analyze all images
    print("\nAnalyzing images...")
    results = {}
    for category, image_paths in images.items():
        print(f"  Processing {category} images...")
        results[category] = analyze_images(detector, image_paths)

    # Print results
    print_comparison_table(results)
    print_statistics(results)
    print_detailed_peaks(results, top_n=5)

    # Create visualizations
    print("\n" + "=" * 100)
    print("Generating visualizations...")
    print("=" * 100)
    
    output_dir = Path("output")
    create_visualizations(results, output_dir=output_dir)

    print("\n" + "=" * 100)
    print("Analysis complete!")
    print("=" * 100)


if __name__ == "__main__":
    main()

