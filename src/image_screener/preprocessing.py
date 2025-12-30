"""Image preprocessing utilities for DFT analysis."""

import logging
from pathlib import Path
from typing import Tuple, Union

import numpy as np
from PIL import Image
from pydantic import Field, field_validator
from pydantic.dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ImagePreprocessor:
    """Preprocesses images for frequency domain analysis."""

    target_size: int = Field(default=512, ge=256, le=2048)
    normalize_to_float: bool = Field(default=True)

    @field_validator("target_size")
    @classmethod
    def validate_target_size(cls, v: int) -> int:
        """Ensure target size is a power of 2 for optimal FFT performance."""
        if not (v & (v - 1) == 0) and v != 0:
            logger.warning(f"Target size {v} is not a power of 2, may affect FFT performance")
        return v

    def load_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Load an image from disk and convert to grayscale.

        Args:
            image_path: Path to the image file (.jpg, .png, .webp)

        Returns:
            Grayscale image as numpy array (uint8 or float32)
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        if path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
            raise ValueError(f"Unsupported image format: {path.suffix}")

        logger.debug(f"Loading image: {image_path}")
        try:
            img = Image.open(path)
            # Verify image integrity (this may modify the image, so we need to reopen)
            img.verify()
            # Reopen after verify since verify() may modify the image
            img = Image.open(path)
        except Exception as e:
            raise ValueError(f"Failed to load or verify image {image_path}: {e}")

        # Convert to grayscale
        try:
            if img.mode != "L":
                img = img.convert("L")
                logger.debug(f"Converted image from {img.mode} to grayscale")
        except Exception as e:
            raise ValueError(f"Failed to convert image {image_path} to grayscale: {e}")

        try:
            img_array = np.array(img, dtype=np.uint8)
        except Exception as e:
            raise ValueError(f"Failed to convert image {image_path} to numpy array: {e}")

        if self.normalize_to_float:
            img_array = img_array.astype(np.float32) / 255.0
            logger.debug("Normalized image to float32 [0, 1]")

        return img_array

    def resize_with_padding(
        self, image: np.ndarray, target_size: int
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Resize image to target dimensions while maintaining aspect ratio.

        Pads with zeros (black) if needed to reach exact target size.

        Args:
            image: Input image array (H, W)
            target_size: Target size for both dimensions

        Returns:
            Tuple of (resized_image, original_shape)
        """
        original_shape = image.shape
        h, w = original_shape

        # Calculate scaling factor to fit within target_size
        scale = min(target_size / h, target_size / w)

        new_h = int(h * scale)
        new_w = int(w * scale)

        # Resize using nearest neighbor for simplicity (can use bilinear if needed)
        resized = Image.fromarray((image * 255).astype(np.uint8) if image.dtype == np.float32 else image)
        resized = resized.resize((new_w, new_h), Image.Resampling.LANCZOS)

        resized_array = np.array(resized, dtype=image.dtype)
        if image.dtype == np.float32:
            resized_array = resized_array.astype(np.float32) / 255.0

        # Pad to exact target_size
        pad_h = target_size - new_h
        pad_w = target_size - new_w

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        padded = np.pad(
            resized_array,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=0.0 if image.dtype == np.float32 else 0,
        )

        logger.debug(
            f"Resized image from {original_shape} to {padded.shape} "
            f"(scale: {scale:.3f}, padding: ({pad_top}, {pad_bottom}, {pad_left}, {pad_right}))"
        )

        return padded, original_shape

    def preprocess(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Complete preprocessing pipeline: load, convert, resize.

        Args:
            image_path: Path to the image file

        Returns:
            Preprocessed grayscale image array (target_size x target_size, float32)
        """
        image = self.load_image(image_path)
        resized, _ = self.resize_with_padding(image, self.target_size)
        return resized

