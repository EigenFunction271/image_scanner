"""Wavelet-based AI image detector (Filter 02).

This module implements residual noise entropy analysis using wavelet
decomposition to detect AI-generated images by analyzing noise patterns
and texture characteristics.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, TypedDict, Tuple, Union

import numpy as np
import pywt
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from image_screener.feature_extractor import extract_all_features

logger = logging.getLogger(__name__)

__all__ = ['WaveletDetector', 'WaveletSubbands']


class WaveletSubbands(TypedDict):
    """Type definition for wavelet decomposition subbands."""
    LL3: np.ndarray
    LH1: np.ndarray
    LH2: np.ndarray
    LH3: np.ndarray
    HL1: np.ndarray
    HL2: np.ndarray
    HL3: np.ndarray
    HH1: np.ndarray
    HH2: np.ndarray
    HH3: np.ndarray


class WaveletDetector:
    """
    Wavelet-based detector for AI-generated images.

    Uses multi-level discrete wavelet transform to extract features
    that distinguish natural photography from AI-generated imagery.
    """

    def __init__(self, wavelet: str = 'db4', levels: int = 3):
        """
        Initialize wavelet detector.

        Args:
            wavelet: Wavelet type (default: 'db4' - Daubechies 4)
            levels: Number of decomposition levels (default: 3)
        """
        self.wavelet = wavelet
        self.levels = levels
        self.classifier = None
        self.scaler = StandardScaler()
        self.is_fitted = False

        logger.info(f"Initialized WaveletDetector with wavelet={wavelet}, levels={levels}")

    def preprocess_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Load and preprocess image for wavelet analysis.

        Args:
            image_path: Path to image file

        Returns:
            Preprocessed grayscale image as numpy array [0, 1]
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load image
        try:
            img = Image.open(path)
        except Exception as e:
            raise ValueError(f"Failed to load image {image_path}: {e}")

        # Convert to grayscale
        if img.mode != 'L':
            img = img.convert('L')

        # Convert to numpy array
        img_array = np.array(img, dtype=np.float32)

        # Validate dimensions
        if img_array.shape[0] < 64 or img_array.shape[1] < 64:
            raise ValueError(
                f"Image too small: {img_array.shape}. Minimum 64×64 required."
            )

        # Normalize to [0, 1]
        img_array = img_array / 255.0

        # Pad to nearest power of 2 for optimal wavelet decomposition
        h, w = img_array.shape
        target_h = 2 ** int(np.ceil(np.log2(h)))
        target_w = 2 ** int(np.ceil(np.log2(w)))

        if target_h != h or target_w != w:
            # Pad with edge values
            pad_h = target_h - h
            pad_w = target_w - w
            img_array = np.pad(
                img_array,
                ((0, pad_h), (0, pad_w)),
                mode='edge'
            )

        logger.debug(f"Preprocessed image: {img_array.shape}")

        return img_array

    def decompose(self, image: np.ndarray) -> WaveletSubbands:
        """
        Perform multi-level 2D discrete wavelet transform.

        Args:
            image: Preprocessed grayscale image

        Returns:
            Dictionary of subbands {LL3, LH1, LH2, LH3, HL1, HL2, HL3, HH1, HH2, HH3}
            
        Raises:
            ValueError: If image is not 2D or too small
        """
        if image.ndim != 2:
            raise ValueError(f"Expected 2D image, got {image.ndim}D")
        if image.size < 64:
            raise ValueError(f"Image too small: {image.shape}. Minimum 64×64 required.")
        subbands = {}

        current = image
        for level in range(1, self.levels + 1):
            # Perform 2D DWT
            coeffs = pywt.dwt2(current, self.wavelet, mode='symmetric')
            LL, (LH, HL, HH) = coeffs

            # Store detail subbands
            subbands[f'LH{level}'] = LH
            subbands[f'HL{level}'] = HL
            subbands[f'HH{level}'] = HH

            # Continue decomposition on LL subband
            current = LL

        # Store final approximation (LL at deepest level)
        subbands[f'LL{self.levels}'] = current

        logger.debug(f"Decomposed image into {len(subbands)} subbands")

        return subbands

    def extract_features(self, subbands: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Extract all features from wavelet subbands.

        Args:
            subbands: Dictionary of wavelet subbands

        Returns:
            1D numpy array of features (~85-90 features)
        """
        return extract_all_features(subbands)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        classifier_type: str = 'rf'
    ) -> None:
        """
        Train classifier on feature vectors.

        Args:
            X_train: Feature matrix (n_samples, n_features)
            y_train: Labels (0=real, 1=AI)
            classifier_type: 'rf' (Random Forest), 'svm', or 'xgboost'
        """
        if X_train.shape[0] == 0:
            raise ValueError("Training data is empty")

        logger.info(f"Training {classifier_type} classifier on {X_train.shape[0]} samples")

        # Normalize features
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Initialize classifier
        if classifier_type == 'rf':
            self.classifier = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
        elif classifier_type == 'svm':
            self.classifier = SVC(
                kernel='rbf',
                probability=True,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")

        # Train
        self.classifier.fit(X_train_scaled, y_train)
        self.is_fitted = True

        logger.info("Classifier training completed")

    def predict(self, image_path: Union[str, Path]) -> Tuple[int, float]:
        """
        Classify a single image.

        Args:
            image_path: Path to image file

        Returns:
            Tuple of (prediction, probability)
            - prediction: 0 (real) or 1 (AI-generated)
            - probability: Confidence score [0, 1]
        """
        if not self.is_fitted:
            raise ValueError("Model not trained. Call fit() first or load a model.")

        # Preprocess
        image = self.preprocess_image(image_path)

        # Decompose
        subbands = self.decompose(image)

        # Extract features
        features = self.extract_features(subbands)
        features = features.reshape(1, -1)

        # Normalize
        features_scaled = self.scaler.transform(features)

        # Predict
        prediction = self.classifier.predict(features_scaled)[0]
        probabilities = self.classifier.predict_proba(features_scaled)[0]

        # Get probability for predicted class
        prob = float(probabilities[prediction])

        return int(prediction), prob

    def save_model(self, filepath: Union[str, Path]) -> None:
        """
        Save trained model and normalization parameters.

        Args:
            filepath: Path to save model file
        """
        import joblib

        if not self.is_fitted:
            raise ValueError("No model to save. Train model first.")

        model_data = {
            'classifier': self.classifier,
            'scaler': self.scaler,
            'wavelet': self.wavelet,
            'levels': self.levels,
        }

        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: Union[str, Path]) -> None:
        """
        Load trained model and normalization parameters.

        Args:
            filepath: Path to model file
        """
        import joblib

        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        model_data = joblib.load(filepath)

        self.classifier = model_data['classifier']
        self.scaler = model_data['scaler']
        self.wavelet = model_data.get('wavelet', 'db4')
        self.levels = model_data.get('levels', 3)
        self.is_fitted = True

        logger.info(f"Model loaded from {filepath}")

