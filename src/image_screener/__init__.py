"""Image Screener - AI Image Detection Tool"""

__version__ = "0.1.0"

# Export main classes (using relative imports to avoid circular dependencies)
try:
    from .optics_consistency import OpticsConsistencyDetector
    from .spectral_peak_detector import SpectralPeakDetector

    __all__ = [
        "SpectralPeakDetector",
        "OpticsConsistencyDetector",
    ]
except ImportError:
    # Allow partial imports if dependencies are missing
    __all__ = []

