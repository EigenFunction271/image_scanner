"""Training script for wavelet-based AI image detector."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from tqdm import tqdm

from image_screener.wavelet_detector import WaveletDetector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_images_from_directory(directory: Path, label: int) -> list[Tuple[Path, int]]:
    """
    Load all images from a directory with their labels.

    Args:
        directory: Path to directory containing images
        label: Label for images in this directory (0=real, 1=AI)

    Returns:
        List of (image_path, label) tuples
    """
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    images = []

    for ext in image_extensions:
        images.extend(directory.glob(f'*{ext}'))
        images.extend(directory.glob(f'*{ext.upper()}'))

    logger.info(f"Found {len(images)} images in {directory}")

    return [(img, label) for img in images]


def extract_features_from_images(
    image_list: list[Tuple[Path, int]],
    detector: WaveletDetector,
    n_jobs: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features from all images.

    Args:
        image_list: List of (image_path, label) tuples
        detector: WaveletDetector instance
        n_jobs: Number of parallel workers (1 = sequential)

    Returns:
        Tuple of (feature_matrix, labels)
    """
    features_list = []
    labels_list = []

    logger.info(f"Extracting features from {len(image_list)} images...")

    if n_jobs > 1 and len(image_list) > 50:
        # Use parallel processing for larger datasets
        from multiprocessing import Pool
        from functools import partial

        def extract_single(args):
            img_path, label = args
            try:
                image = detector.preprocess_image(img_path)
                subbands = detector.decompose(image)
                features = detector.extract_features(subbands)
                return features, label
            except Exception as e:
                logger.warning(f"Failed to process {img_path}: {e}")
                return None, None

        with Pool(processes=n_jobs) as pool:
            results = list(tqdm(
                pool.imap(extract_single, image_list),
                total=len(image_list),
                desc="Processing images"
            ))

        for features, label in results:
            if features is not None:
                features_list.append(features)
                labels_list.append(label)
    else:
        # Sequential processing
        for img_path, label in tqdm(image_list, desc="Processing images"):
            try:
                # Preprocess
                image = detector.preprocess_image(img_path)

                # Decompose
                subbands = detector.decompose(image)

                # Extract features
                features = detector.extract_features(subbands)
                features_list.append(features)
                labels_list.append(label)

            except Exception as e:
                logger.warning(f"Failed to process {img_path}: {e}")
                continue

    if len(features_list) == 0:
        raise ValueError("No features extracted. Check image paths and formats.")

    X = np.array(features_list)
    y = np.array(labels_list)

    logger.info(f"Extracted features: {X.shape}")

    return X, y


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    classifier_type: str = 'rf',
    tune_hyperparameters: bool = False
) -> WaveletDetector:
    """
    Train wavelet detector model.

    Args:
        X_train: Training features
        y_train: Training labels
        classifier_type: 'rf' or 'svm'
        tune_hyperparameters: Whether to perform grid search

    Returns:
        Trained WaveletDetector
    """
    detector = WaveletDetector()

    if tune_hyperparameters and classifier_type == 'rf':
        logger.info("Performing hyperparameter tuning...")
        from sklearn.ensemble import RandomForestClassifier

        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [15, 20, 25],
            'min_samples_split': [3, 5, 7]
        }

        base_classifier = RandomForestClassifier(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(
            base_classifier,
            param_grid,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1
        )

        # Normalize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        grid_search.fit(X_train_scaled, y_train)

        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")

        # Use best classifier
        detector.classifier = grid_search.best_estimator_
        detector.scaler = scaler
        detector.is_fitted = True
    else:
        detector.fit(X_train, y_train, classifier_type=classifier_type)

    return detector


def evaluate_model(
    detector: WaveletDetector,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: Path
) -> None:
    """
    Evaluate trained model and print metrics.

    Args:
        detector: Trained WaveletDetector
        X_test: Test features
        y_test: Test labels
        output_dir: Directory to save evaluation plots
    """
    # Normalize test features
    X_test_scaled = detector.scaler.transform(X_test)

    # Predictions
    y_pred = detector.classifier.predict(X_test_scaled)
    y_proba = detector.classifier.predict_proba(X_test_scaled)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print("\n" + "=" * 80)
    print("MODEL EVALUATION")
    print("=" * 80)
    print(f"\nAccuracy:  {accuracy:.4f}")
    print(f"AUC-ROC:   {auc:.4f}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Real', 'AI']))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"              Real    AI")
    print(f"Actual Real   {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"        AI    {cm[1,0]:4d}  {cm[1,1]:4d}")

    # Save confusion matrix plot
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.xticks([0.5, 1.5], ['Real', 'AI'])
        plt.yticks([0.5, 1.5], ['Real', 'AI'], rotation=0)

        cm_path = output_dir / 'confusion_matrix.png'
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\nConfusion matrix saved to: {cm_path}")

        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)

        roc_path = output_dir / 'roc_curve.png'
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"ROC curve saved to: {roc_path}")

    except ImportError:
        logger.warning("matplotlib/seaborn not available. Skipping plots.")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description='Train wavelet-based AI image detector',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--real-dir',
        type=str,
        required=True,
        help='Directory containing real/natural images'
    )

    parser.add_argument(
        '--ai-dir',
        type=str,
        required=True,
        help='Directory containing AI-generated images'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='wavelet_model.pkl',
        help='Output path for trained model (default: wavelet_model.pkl)'
    )

    parser.add_argument(
        '--classifier',
        type=str,
        default='rf',
        choices=['rf', 'svm'],
        help='Classifier type (default: rf)'
    )

    parser.add_argument(
        '--tune',
        action='store_true',
        help='Perform hyperparameter tuning (slower but better results)'
    )

    parser.add_argument(
        '--test-split',
        type=float,
        default=0.15,
        help='Test set fraction (default: 0.15)'
    )

    parser.add_argument(
        '--val-split',
        type=float,
        default=0.15,
        help='Validation set fraction (default: 0.15)'
    )

    parser.add_argument(
        '--n-jobs',
        type=int,
        default=1,
        help='Number of parallel workers for feature extraction (default: 1, use -1 for all cores)'
    )

    args = parser.parse_args()

    # Load images
    print("Loading images...")
    real_images = load_images_from_directory(Path(args.real_dir), label=0)
    ai_images = load_images_from_directory(Path(args.ai_dir), label=1)

    if len(real_images) == 0 or len(ai_images) == 0:
        print("Error: Need images in both directories", file=sys.stderr)
        sys.exit(1)

    print(f"\nDataset: {len(real_images)} real, {len(ai_images)} AI images")

    # Extract features
    detector = WaveletDetector()
    all_images = real_images + ai_images

    n_jobs = args.n_jobs
    if n_jobs == -1:
        import os
        n_jobs = os.cpu_count() or 1

    X, y = extract_features_from_images(all_images, detector, n_jobs=n_jobs)

    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=args.test_split, random_state=42, stratify=y
    )

    val_size = args.val_split / (1 - args.test_split)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=42, stratify=y_temp
    )

    print(f"\nData split:")
    print(f"  Training:   {X_train.shape[0]} samples")
    print(f"  Validation: {X_val.shape[0]} samples")
    print(f"  Test:       {X_test.shape[0]} samples")

    # Train model
    print("\nTraining model...")
    detector = train_model(
        X_train, y_train,
        classifier_type=args.classifier,
        tune_hyperparameters=args.tune
    )

    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    evaluate_model(detector, X_val, y_val, Path(args.output).parent)

    # Final evaluation on test set
    print("\nEvaluating on test set...")
    evaluate_model(detector, X_test, y_test, Path(args.output).parent)

    # Save model
    output_path = Path(args.output)
    detector.save_model(output_path)
    print(f"\n✓ Model saved to: {output_path}")

    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()

