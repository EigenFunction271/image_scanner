"""Shared constants for optics consistency tests."""

# Frequency Domain Test Constants
HIGH_FREQ_THRESHOLD = 0.7  # Radius threshold for high-frequency region
MIN_RADIUS_THRESHOLD = 0.1  # Minimum radius to consider
MAX_RADIUS_THRESHOLD = 0.9  # Maximum radius to consider
OTF_SLOPE_THRESHOLD = -0.5  # Minimum acceptable OTF decay slope
BUMP_RATIO_THRESHOLD = 0.1  # Maximum acceptable bump ratio (10%)
SUPPRESSION_RATIO_THRESHOLD = 0.15  # Maximum acceptable suppression ratio (15%)
RESIDUAL_STD_THRESHOLD = 0.5  # High variance threshold for non-smooth decay
RESIDUAL_STD_BASE = 0.3  # Base value for residual std penalty calculation
NOISE_FLOOR_RATIO_THRESHOLD = 0.1  # Minimum noise floor ratio (10% of expected)
BUMP_DETECTION_FACTOR = 1.5  # Factor for bump detection threshold

# Edge PSF Test Constants
DEFAULT_EDGE_THRESHOLD = 0.1
DEFAULT_MIN_EDGE_LENGTH = 20

# DOF Test Constants
DEFAULT_BLUR_WINDOW_SIZE = 21
DOF_VIOLATION_RATIO_THRESHOLD = 0.1  # Maximum acceptable violation ratio (10% of triplets)
EDGE_DENSITY_THRESHOLD = 0.1  # Minimum edge density in window to estimate blur
MAX_GRADIENT_THRESHOLD = 2.0  # Large jumps indicate discrete blur regions
MEAN_GRADIENT_THRESHOLD = 0.5  # Non-smooth blur variation threshold
THIN_LENS_VIOLATION_PENALTY = 0.4  # Strong penalty for physics violations
TEXTURE_THRESHOLD = 0.01  # Minimum texture variance for blur evidence
DEFOCUS_GRADIENT_THRESHOLD = 0.05  # Minimum defocus gradient for blur evidence
MIN_EVIDENCE_RATIO = 0.1  # Minimum ratio of image with blur evidence

# CA Test Constants
MIN_RESOLUTION_FOR_RADIAL_TEST = 1024  # Minimum resolution for radial consistency test
CA_MAGNITUDE_THRESHOLD = 0.5  # Minimum CA offset magnitude (pixels) for alignment checks
CA_ALIGNMENT_THRESHOLD = 0.7  # Minimum radial alignment score
CA_COLOR_ORDER_THRESHOLD = 0.3  # Maximum acceptable color order violation

# Noise Residual Test Constants
NOISE_SAMPLE_RATE_FACTOR = 1000  # Sample 1 in N pixels for correlation
MIN_RESIDUAL_STD = 0.001  # Minimum residual std for normalization
MIN_AUTOCORR_VALUE = 1e-10  # Minimum autocorrelation value for normalization
SPATIAL_CORRELATION_THRESHOLD = 0.15  # Threshold for AI detection (ρ < 0.15)

