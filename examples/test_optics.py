#!/usr/bin/env python3
"""Test script for optics consistency diagnostics.

This script tests the optics consistency detector on test images
and generates diagnostic plots.

Usage:
    PYTHONPATH=src python3 examples/test_optics.py
    PYTHONPATH=src python3 examples/test_optics.py --image path/to/image.jpg
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

# Use non-interactive backend
matplotlib.use("Agg")

from image_screener.optics_consistency import OpticsConsistencyDetector
from image_screener.optics_visualization import create_optics_diagnostics_plot

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_single_image(image_path: Path, output_dir: Path):
    """Test optics detector on a single image."""
    print(f"\n{'='*80}")
    print(f"Testing: {image_path.name}")
    print(f"{'='*80}")

    detector = OpticsConsistencyDetector()

    try:
        result = detector.analyze(str(image_path), load_rgb=True)
    except Exception as e:
        logger.error(f"Failed to analyze {image_path}: {e}", exc_info=True)
        return None

    # Print results
    print(f"\nOverall Optics Score: {result.optics_score:.4f} / 1.0")
    print("\nTest Scores:")
    print(f"  Frequency Domain Test:     {result.frequency_test.score:.4f}")
    print(f"  Edge PSF Test:              {result.edge_psf_test.score:.4f}")
    print(f"  DOF Consistency Test:      {result.dof_consistency_test.score:.4f}")
    print(f"  Chromatic Aberration Test:  {result.chromatic_aberration_test.score:.4f}")

    print("\nViolations:")
    for violation in result.frequency_test.violations:
        print(f"  Frequency: {violation}")
    for violation in result.edge_psf_test.violations:
        print(f"  PSF: {violation}")
    for violation in result.dof_consistency_test.violations:
        print(f"  DOF: {violation}")
    for violation in result.chromatic_aberration_test.violations:
        print(f"  CA: {violation}")

    print(f"\nExplanation: {result.explanation}")

    # Create visualization
    output_path = output_dir / f"{image_path.stem}_optics_diagnostics.png"
    try:
        create_optics_diagnostics_plot(
            result,
            str(image_path),
            output_path=output_path,
            show_plot=False,
        )
        print(f"\n✓ Saved diagnostics to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to create visualization: {e}", exc_info=True)
        return None

    return result


def test_default_images(output_dir: Path):
    """Test on default test images if they exist."""
    project_root = Path(__file__).parent.parent
    test_image_dirs = [
        project_root / "documentation" / "test images" / "01",
        project_root / "documentation" / "test images" / "02",
        project_root / "documentation" / "test images" / "03",
    ]

    results = []

    for test_dir in test_image_dirs:
        if not test_dir.exists():
            continue

        # Look for real and fake images
        for pattern in ["real.jpg", "real.png", "fake.png"]:
            image_path = test_dir / pattern
            if image_path.exists():
                result = test_single_image(image_path, output_dir)
                if result:
                    results.append((image_path.name, result.optics_score))

    return results


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(
        description="Test optics consistency diagnostics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--image",
        "-i",
        type=str,
        help="Path to specific image to test (if not provided, tests default images)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="output",
        help="Output directory for diagnostics plots (default: 'output')",
    )

    parser.add_argument(
        "--no-rgb",
        action="store_true",
        help="Skip chromatic aberration test (use grayscale only)",
    )

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("OPTICS CONSISTENCY DETECTOR - TEST SCRIPT")
    print("=" * 80)

    results = []

    if args.image:
        # Test single image
        image_path = Path(args.image)
        if not image_path.exists():
            logger.error(f"Image not found: {image_path}")
            sys.exit(1)

        detector = OpticsConsistencyDetector()
        result = detector.analyze(str(image_path), load_rgb=not args.no_rgb)

        test_single_image(image_path, output_dir)
        results.append((image_path.name, result.optics_score))

    else:
        # Test default images
        print("\nTesting default test images...")
        results = test_default_images(output_dir)

        if not results:
            print("\nNo test images found. Use --image to specify an image.")
            print("\nExample:")
            print("  PYTHONPATH=src python3 examples/test_optics.py --image path/to/image.jpg")
            sys.exit(0)

    # Summary
    if results:
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        print(f"\n{'Image':<40} {'Score':<10} {'Status':<15}")
        print("-" * 80)

        for image_name, score in results:
            if score >= 0.7:
                status = "✓ PASS"
            elif score >= 0.4:
                status = "⚠ SUSPICIOUS"
            else:
                status = "✗ FAIL"
            print(f"{image_name:<40} {score:<10.4f} {status:<15}")

        avg_score = sum(s for _, s in results) / len(results)
        print("-" * 80)
        print(f"{'Average Score':<40} {avg_score:<10.4f}")
        print("=" * 80)

        print(f"\n✓ All diagnostics saved to: {output_dir}")


if __name__ == "__main__":
    main()

