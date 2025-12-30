# Code Review: Image Screener

## Executive Summary

This codebase implements an AI image detection tool using frequency domain analysis. The code is generally well-structured with good use of type hints, logging, and Pydantic validation. However, there are several areas for improvement including dependency management, type safety, error handling, and code organization.

---

## Critical Issues

### 1. **Dependency Mismatch Between `requirements.txt` and `pyproject.toml`**

**Issue**: `requirements.txt` includes dependencies (`matplotlib`, `PyWavelets`, `scikit-learn`, `joblib`, `tqdm`, `seaborn`) that are not listed in `pyproject.toml`. This creates inconsistency and potential installation issues.

**Location**: 
- `requirements.txt` (lines 5-10)
- `pyproject.toml` (lines 11-16)

**Impact**: Users installing via `pip install -e .` won't get all required dependencies.

**Recommendation**: 
- Move all dependencies to `pyproject.toml` under `[project.dependencies]`
- Mark visualization dependencies (`matplotlib`, `seaborn`) as optional extras
- Remove `requirements.txt` or keep it only for legacy compatibility with a note

### 2. **Unsafe Return Type in `DFTProcessor.process_image()`**

**Issue**: The method returns a 10-element tuple which is error-prone and hard to maintain.

**Location**: `src/image_screener/dft.py:662-721`

**Current**:
```python
def process_image(self, image: np.ndarray) -> Tuple[np.ndarray, list[SpectralPeak], float, float, float, float, np.ndarray, np.ndarray, float, float]:
```

**Recommendation**: Create a `ProcessImageResult` NamedTuple or dataclass:
```python
@dataclass
class ProcessImageResult:
    log_magnitude_spectrum: np.ndarray
    peaks: list[SpectralPeak]
    artifact_score: float
    grid_strength: float
    grid_interval_u: float
    grid_interval_v: float
    azimuthal_radii: np.ndarray
    azimuthal_average: np.ndarray
    grid_consistency: float
    nyquist_symmetry: float
```

---

## High Priority Issues

### 3. **Missing Error Handling for Image Loading**

**Issue**: `ImagePreprocessor.load_image()` doesn't handle corrupted images or PIL exceptions gracefully.

**Location**: `src/image_screener/preprocessing.py:48`

**Current**:
```python
img = Image.open(path)
```

**Recommendation**: Wrap in try-except:
```python
try:
    img = Image.open(path)
    img.verify()  # Verify image integrity
    img = Image.open(path)  # Reopen after verify
except Exception as e:
    raise ValueError(f"Failed to load or verify image {image_path}: {e}")
```

### 4. **Inconsistent Default Values**

**Issue**: `SpectralPeakDetector` uses `peak_threshold_percentile=95.0` but `DFTProcessor` uses `98.0` as default.

**Location**: 
- `src/image_screener/spectral_peak_detector.py:47`
- `src/image_screener/dft.py:30`

**Recommendation**: Align defaults or document the difference clearly.

### 5. **Missing Input Validation**

**Issue**: Several methods don't validate array shapes or data types before processing.

**Location**: 
- `DFTProcessor.compute_dft()` - doesn't check for 2D array
- `DFTProcessor.compute_azimuthal_average()` - doesn't validate input shape

**Recommendation**: Add validation:
```python
if image.ndim != 2:
    raise ValueError(f"Expected 2D array, got {image.ndim}D")
if image.size == 0:
    raise ValueError("Image array is empty")
```

### 6. **Hardcoded Magic Numbers**

**Issue**: Many magic numbers throughout the codebase without constants or documentation.

**Examples**:
- `dft.py:190` - `max_peaks = 1000`
- `dft.py:245` - `if du > 5:  # Minimum interval threshold`
- `dft.py:275` - `tolerance = 3  # pixels`
- `dft.py:637` - `scale_factor = 2.0`
- `feature_extractor.py:296` - `lags = [1, 2, 4, 8]`

**Recommendation**: Extract to module-level constants with documentation:
```python
# Peak detection parameters
MAX_PEAKS_TO_RETAIN = 1000  # Limit peaks to avoid noise
MIN_INTERVAL_THRESHOLD = 5  # Minimum pixel interval for grid detection
GRID_ALIGNMENT_TOLERANCE = 3  # Pixel tolerance for grid alignment
EXPONENTIAL_SCALE_FACTOR = 2.0  # Controls exponential scaling of grid scores
```

---

## Medium Priority Issues

### 7. **Inefficient Operations**

**Issue**: Some operations could be optimized.

**Location**: `src/image_screener/dft.py:199-346` - `compute_spatial_autocorrelation()`

**Problems**:
- Nested loops for pairwise comparisons: O(n²) complexity
- Multiple redundant computations
- Large autocorrelation computation on full image

**Recommendation**: 
- Use vectorized operations where possible
- Consider using `scipy.spatial.distance` for pairwise distances
- Cache intermediate results

### 8. **Missing Type Hints**

**Issue**: Some functions lack complete type hints.

**Examples**:
- `feature_extractor.py:135` - `estimate_ggd_parameters()` return type could be more specific
- `wavelet_detector.py:103` - `decompose()` return type uses `Dict[str, np.ndarray]` but could be `Dict[str, np.ndarray]` with TypedDict

**Recommendation**: Add complete type hints using `typing` module and consider `TypedDict` for structured dictionaries.

### 9. **Incomplete Test Coverage**

**Issue**: Missing tests for several critical functions.

**Missing Tests**:
- `DFTProcessor.compute_spatial_autocorrelation()` - complex logic, needs edge cases
- `DFTProcessor.compute_grid_consistency_score()` - needs tests for various grid patterns
- `DFTProcessor.compute_azimuthal_average()` - needs tests for edge cases
- `feature_extractor.py` - many functions lack tests
- Error handling paths in preprocessing

**Recommendation**: Add comprehensive test coverage, especially for edge cases (empty arrays, single peaks, extreme values).

### 10. **Logging Configuration**

**Issue**: Logging is configured but not consistently used. Some debug logs might be too verbose for production.

**Location**: Throughout codebase

**Recommendation**: 
- Use structured logging levels appropriately
- Consider adding a logging configuration module
- Remove or gate verbose debug logs behind a flag

### 11. **Documentation Gaps**

**Issue**: Some complex algorithms lack sufficient documentation.

**Examples**:
- `compute_spatial_autocorrelation()` - the three methods used aren't well explained
- `compute_spectral_artifact_score()` - the exponential scaling logic needs better documentation
- `estimate_ggd_parameters()` - the approximation method should be documented

**Recommendation**: Add detailed docstrings explaining the mathematical principles and algorithm choices.

---

## Low Priority / Code Quality

### 12. **Code Duplication**

**Issue**: Some repeated patterns could be extracted.

**Examples**:
- Center calculation: `center_y, center_x = h // 2, w // 2` appears multiple times
- Peak coordinate conversion patterns

**Recommendation**: Extract to helper functions:
```python
def get_center_coords(shape: Tuple[int, int]) -> Tuple[int, int]:
    """Get center coordinates for an image shape."""
    return shape[0] // 2, shape[1] // 2
```

### 13. **Inconsistent Naming**

**Issue**: Some inconsistencies in variable naming.

**Examples**:
- `u, v` vs `x, y` for coordinates (FFT convention vs image convention)
- `h, w` vs `height, width`

**Recommendation**: Document the convention (FFT uses `u, v` for frequency coordinates) or use more descriptive names.

### 14. **Missing `__all__` Exports**

**Issue**: Modules don't define `__all__`, making it unclear what the public API is.

**Recommendation**: Add `__all__` to each module:
```python
__all__ = ['SpectralPeakDetector', 'DetectionResult']
```

### 15. **Unused Imports**

**Issue**: Some imports may be unused (need to verify with linter).

**Recommendation**: Run `ruff check --select F401` to find unused imports.

### 16. **Missing Validation for Edge Cases**

**Issue**: Some functions don't handle edge cases well.

**Examples**:
- Empty image arrays
- Single-pixel images
- Very large images (memory issues)
- Images with all zeros or all ones

**Recommendation**: Add validation and appropriate error messages.

---

## Security Considerations

### 17. **Path Traversal Risk**

**Issue**: Image paths are used directly without sanitization in some places.

**Location**: `examples/` scripts

**Recommendation**: Use `pathlib.Path.resolve()` to prevent path traversal attacks.

### 18. **Resource Limits**

**Issue**: No limits on image size or processing time.

**Recommendation**: Add configurable limits:
```python
MAX_IMAGE_SIZE = 4096  # pixels
MAX_PROCESSING_TIME = 30  # seconds
```

---

## Performance Optimizations

### 19. **Memory Efficiency**

**Issue**: Some operations create unnecessary copies.

**Examples**:
- `dft.py:88` - `magnitude = np.abs(fft_shifted)` creates a copy
- Multiple array conversions in preprocessing

**Recommendation**: Use in-place operations where possible and document when copies are necessary.

### 20. **Batch Processing**

**Issue**: `batch_analyze()` processes images sequentially.

**Location**: `src/image_screener/spectral_peak_detector.py:112-126`

**Recommendation**: Consider parallel processing with `multiprocessing` or `concurrent.futures` for large batches.

---

## Architecture Suggestions

### 21. **Separation of Concerns**

**Issue**: `DFTProcessor` does too much - computation, analysis, and scoring.

**Recommendation**: Consider splitting into:
- `DFTComputer` - pure FFT operations
- `PeakDetector` - peak detection logic
- `GridAnalyzer` - grid pattern analysis
- `ScoreCalculator` - scoring logic

### 22. **Configuration Management**

**Issue**: Parameters are scattered across classes.

**Recommendation**: Create a configuration dataclass or use a config file:
```python
@dataclass
class DetectionConfig:
    target_size: int = 512
    sensitivity: float = 1.0
    high_freq_threshold: float = 0.3
    peak_threshold_percentile: float = 98.0
    max_peaks: int = 1000
    # ... etc
```

### 23. **Missing Integration Tests**

**Issue**: No end-to-end integration tests.

**Recommendation**: Add integration tests that:
- Test the full pipeline from image to result
- Test with real test images
- Validate expected scores for known real/fake images

---

## Positive Aspects

✅ **Good use of type hints** - Most functions have proper type annotations
✅ **Pydantic validation** - Good use of field validators
✅ **Logging** - Comprehensive logging throughout
✅ **Modular design** - Well-separated concerns
✅ **Documentation** - Good docstrings on most functions
✅ **Testing** - Basic test coverage exists
✅ **Error handling** - Some error handling in place

---

## Recommended Action Plan

### Phase 1 (Critical - Do First)
1. Fix dependency mismatch between `requirements.txt` and `pyproject.toml`
2. Replace tuple return type with NamedTuple/dataclass
3. Add error handling for image loading
4. Add input validation for array operations

### Phase 2 (High Priority - Do Soon)
5. Extract magic numbers to constants
6. Align default values
7. Add missing type hints
8. Improve test coverage for edge cases

### Phase 3 (Medium Priority - Do When Time Permits)
9. Optimize inefficient operations
10. Add comprehensive documentation for algorithms
11. Refactor to reduce code duplication
12. Add integration tests

### Phase 4 (Nice to Have)
13. Performance optimizations
14. Architecture improvements
15. Security hardening

---

## Summary Statistics

- **Total Issues Found**: 23
- **Critical**: 2
- **High Priority**: 4
- **Medium Priority**: 6
- **Low Priority**: 11

**Estimated Effort**:
- Phase 1: 4-6 hours
- Phase 2: 8-12 hours
- Phase 3: 16-24 hours
- Phase 4: 20-30 hours

**Overall Code Quality**: 7/10 - Good foundation with room for improvement in robustness and maintainability.

