#!/usr/bin/env python3
"""CLI tool for optics consistency detection.

Usage:
    python detect_optics.py image.jpg
    python detect_optics.py image.jpg --output outputs/
    python detect_optics.py image.jpg --no-rgb  # Skip CA test
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

# Use non-interactive backend if not displaying
matplotlib.use("Agg")

from image_screener.optics_consistency import OpticsConsistencyDetector
from image_screener.optics_visualization import create_optics_diagnostics_plot

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Detect optics consistency violations in images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python detect_optics.py image.jpg
  python detect_optics.py image.jpg --output outputs/
  python detect_optics.py image.jpg --no-rgb --output outputs/diagnostics.png
        """,
    )

    parser.add_argument(
        "image_path",
        type=str,
        help="Path to input image file (.jpg, .png, .webp)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="output",
        help="Output directory or file path (default: 'output')",
    )

    parser.add_argument(
        "--no-rgb",
        action="store_true",
        help="Skip chromatic aberration test (use grayscale only)",
    )

    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Don't display plot interactively (save only)",
    )

    parser.add_argument(
        "--weights",
        nargs=4,
        type=float,
        metavar=("FREQ", "PSF", "DOF", "CA"),
        help="Custom weights for tests (frequency, psf, dof, ca). Must sum to 1.0",
    )

    args = parser.parse_args()

    # Validate image path
    image_path = Path(args.image_path)
    if not image_path.exists():
        logger.error(f"Image not found: {image_path}")
        sys.exit(1)

    # Determine output path
    output_path = Path(args.output)
    if output_path.suffix in [".png", ".jpg", ".pdf", ".svg"]:
        # Direct file path
        diagnostics_path = output_path
    else:
        # Directory path
        output_path.mkdir(parents=True, exist_ok=True)
        diagnostics_path = output_path / f"{image_path.stem}_optics_diagnostics.png"

    # Initialize detector
    detector_kwargs = {}
    if args.weights:
        if len(args.weights) != 4:
            logger.error("Must provide exactly 4 weights")
            sys.exit(1)
        if abs(sum(args.weights) - 1.0) > 0.01:
            logger.warning(
                f"Weights sum to {sum(args.weights)}, normalizing to 1.0"
            )
        detector_kwargs = {
            "frequency_weight": args.weights[0],
            "edge_psf_weight": args.weights[1],
            "dof_weight": args.weights[2],
            "ca_weight": args.weights[3],
        }

    detector = OpticsConsistencyDetector(**detector_kwargs)

    # Run analysis
    logger.info(f"Analyzing optics consistency: {image_path}")
    try:
        result = detector.analyze(str(image_path), load_rgb=not args.no_rgb)
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        sys.exit(1)

    # Print results
    print("\n" + "=" * 80)
    print("OPTICS CONSISTENCY ANALYSIS RESULTS")
    print("=" * 80)
    print(f"\nImage: {image_path.name}")
    print(f"\nOverall Optics Score: {result.optics_score:.4f} / 1.0")
    print("\nTest Scores:")
    print(f"  Frequency Domain Test:     {result.frequency_test.score:.4f}")
    print(f"  Edge PSF Test:              {result.edge_psf_test.score:.4f}")
    print(f"  DOF Consistency Test:      {result.dof_consistency_test.score:.4f}")
    print(f"  Chromatic Aberration Test:  {result.chromatic_aberration_test.score:.4f}")

    print("\nViolations:")
    print(f"  Frequency: {', '.join(result.frequency_test.violations)}")
    print(f"  PSF: {', '.join(result.edge_psf_test.violations)}")
    print(f"  DOF: {', '.join(result.dof_consistency_test.violations)}")
    print(f"  CA: {', '.join(result.chromatic_aberration_test.violations)}")

    print(f"\nExplanation: {result.explanation}")
    print("=" * 80)

    # Create visualization
    logger.info(f"Creating diagnostics plot: {diagnostics_path}")
    try:
        create_optics_diagnostics_plot(
            result,
            str(image_path),
            output_path=diagnostics_path,
            show_plot=False,  # Always False when using Agg backend
        )
        print(f"\n✓ Saved diagnostics to: {diagnostics_path}")
    except Exception as e:
        logger.error(f"Failed to create visualization: {e}", exc_info=True)
        sys.exit(1)

    # Exit code based on score
    if result.optics_score < 0.5:
        sys.exit(2)  # Low score - likely AI generated
    elif result.optics_score < 0.7:
        sys.exit(1)  # Medium score - suspicious
    else:
        sys.exit(0)  # High score - likely real


if __name__ == "__main__":
    main()

