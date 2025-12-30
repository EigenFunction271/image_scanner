"""Inference script for wavelet-based AI image detector."""

import argparse
import logging
import sys
from pathlib import Path

from image_screener.wavelet_detector import WaveletDetector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def classify_image(model_path: Path, image_path: Path, visualize: bool = False) -> None:
    """
    Classify a single image using trained model.

    Args:
        model_path: Path to trained model file
        image_path: Path to image to classify
        visualize: Whether to visualize wavelet decomposition
    """
    # Load model
    detector = WaveletDetector()
    try:
        detector.load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)

    # Classify
    try:
        prediction, probability = detector.predict(image_path)
    except Exception as e:
        print(f"Error classifying image: {e}", file=sys.stderr)
        sys.exit(1)

    # Display result
    result = "AI-generated" if prediction == 1 else "Real/Natural"
    confidence = probability * 100

    print("\n" + "=" * 60)
    print("CLASSIFICATION RESULT")
    print("=" * 60)
    print(f"Image:      {image_path}")
    print(f"Prediction: {result}")
    print(f"Confidence: {confidence:.2f}%")
    print("=" * 60)

    # Visualize if requested
    if visualize:
        try:
            visualize_decomposition(detector, image_path)
        except Exception as e:
            logger.warning(f"Visualization failed: {e}")


def visualize_decomposition(detector: WaveletDetector, image_path: Path) -> None:
    """
    Visualize wavelet decomposition subbands.

    Args:
        detector: WaveletDetector instance
        image_path: Path to image
    """
    import matplotlib.pyplot as plt

    # Preprocess and decompose
    image = detector.preprocess_image(image_path)
    subbands = detector.decompose(image)

    # Create visualization
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    # Original image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Subbands
    subband_order = ['LL3', 'LH1', 'HL1', 'HH1', 'LH2', 'HL2', 'HH2', 'LH3', 'HL3', 'HH3']
    for i, key in enumerate(subband_order, 1):
        if i < len(axes) and key in subbands:
            # Normalize for display
            subband = subbands[key]
            subband_display = np.abs(subband)
            subband_display = (subband_display - subband_display.min()) / (
                subband_display.max() - subband_display.min() + 1e-10
            )

            axes[i].imshow(subband_display, cmap='gray')
            axes[i].set_title(key)
            axes[i].axis('off')

    plt.tight_layout()

    # Save visualization
    output_path = Path(image_path).parent / f"{Path(image_path).stem}_wavelet_decomp.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nWavelet decomposition visualization saved to: {output_path}")


def batch_classify(model_path: Path, image_dir: Path) -> None:
    """
    Classify multiple images in a directory.

    Args:
        model_path: Path to trained model file
        image_dir: Directory containing images
    """
    from tqdm import tqdm

    # Load model
    detector = WaveletDetector()
    try:
        detector.load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)

    # Find images
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    images = []
    for ext in image_extensions:
        images.extend(image_dir.glob(f'*{ext}'))
        images.extend(image_dir.glob(f'*{ext.upper()}'))

    if len(images) == 0:
        print(f"No images found in {image_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"\nClassifying {len(images)} images...\n")

    results = []
    for img_path in tqdm(images, desc="Processing"):
        try:
            prediction, probability = detector.predict(img_path)
            result = "AI" if prediction == 1 else "Real"
            results.append((img_path.name, result, probability * 100))
        except Exception as e:
            logger.warning(f"Failed to classify {img_path}: {e}")

    # Print results
    print("\n" + "=" * 80)
    print("BATCH CLASSIFICATION RESULTS")
    print("=" * 80)
    print(f"{'Image':<40} {'Prediction':<15} {'Confidence':<10}")
    print("-" * 80)

    for name, pred, conf in results:
        print(f"{name:<40} {pred:<15} {conf:>6.2f}%")

    # Summary
    ai_count = sum(1 for _, pred, _ in results if pred == "AI")
    real_count = len(results) - ai_count

    print("\n" + "-" * 80)
    print(f"Summary: {real_count} Real, {ai_count} AI-generated")
    print("=" * 80)


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(
        description='Classify images using trained wavelet detector',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Classify single image
  python inference_wavelet.py --image photo.jpg --model model.pkl

  # Classify with visualization
  python inference_wavelet.py --image photo.jpg --model model.pkl --visualize

  # Batch classify directory
  python inference_wavelet.py --dir ./images --model model.pkl
        """
    )

    parser.add_argument(
        '--image',
        type=str,
        help='Path to single image to classify'
    )

    parser.add_argument(
        '--dir',
        type=str,
        help='Directory containing images to classify (batch mode)'
    )

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model file (.pkl)'
    )

    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate wavelet decomposition visualization'
    )

    args = parser.parse_args()

    if not args.image and not args.dir:
        print("Error: Must specify either --image or --dir", file=sys.stderr)
        sys.exit(1)

    if args.image and args.dir:
        print("Error: Cannot specify both --image and --dir", file=sys.stderr)
        sys.exit(1)

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    # Classify
    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"Error: Image not found: {image_path}", file=sys.stderr)
            sys.exit(1)

        classify_image(model_path, image_path, visualize=args.visualize)
    else:
        image_dir = Path(args.dir)
        if not image_dir.exists():
            print(f"Error: Directory not found: {image_dir}", file=sys.stderr)
            sys.exit(1)

        batch_classify(model_path, image_dir)


if __name__ == '__main__':
    main()

