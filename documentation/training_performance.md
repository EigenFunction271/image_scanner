# Training Performance Analysis

## Computational Complexity

### Per-Image Feature Extraction

**Time Complexity:**
- **Wavelet Decomposition**: O(N log N) where N = image pixels
  - 3-level 2D DWT: ~3 × (N/4 + N/16 + N/64) operations
  - For 512×512 image: ~0.1-0.3 seconds per image
- **Feature Computation**: O(N) for most operations
  - Statistical moments: O(N)
  - Cross-scale correlation: O(N) with upsampling
  - Periodic artifact detection: O(N log N) for FFT
  - **Total per image**: ~0.2-0.5 seconds

**Memory per Image:**
- Input image: ~1 MB (512×512 float32)
- Wavelet subbands: ~1.3 MB (10 subbands, decreasing sizes)
- Feature vector: ~0.0003 MB (87 features)
- **Peak memory**: ~2-3 MB per image (temporary)

### Training Process

**Feature Extraction Phase:**
- **Time**: ~0.3-0.5 seconds per image
- **1000 images**: ~5-8 minutes (sequential)
- **10,000 images**: ~50-80 minutes
- **Parallelization**: Can process multiple images in parallel (use multiprocessing)

**Classifier Training:**
- **Random Forest (200 trees)**:
  - Time: O(n_samples × n_features × n_estimators × log(n_samples))
  - 1000 samples: ~10-30 seconds
  - 10,000 samples: ~2-5 minutes
  - Memory: ~50-200 MB (depends on tree depth)

- **SVM (RBF kernel)**:
  - Time: O(n_samples² × n_features) - much slower for large datasets
  - 1000 samples: ~1-5 minutes
  - 10,000 samples: ~2-4 hours (not recommended)
  - Memory: ~100-500 MB

**Hyperparameter Tuning:**
- **GridSearchCV with 5-fold CV**:
  - Multiplies training time by: (n_combinations × 5)
  - Example: 18 combinations × 5 folds = 90× training time
  - 1000 samples: ~15-45 minutes
  - 10,000 samples: ~3-7 hours

## Performance Benchmarks

### Typical Training Times (CPU, single-threaded)

| Dataset Size | Feature Extraction | RF Training | RF + Tuning | SVM Training |
|--------------|-------------------|-------------|-------------|--------------|
| 100 images   | ~30 seconds       | ~2 seconds  | ~3 minutes  | ~10 seconds  |
| 1,000 images | ~5 minutes        | ~20 seconds | ~20 minutes | ~3 minutes   |
| 10,000 images| ~50 minutes       | ~3 minutes  | ~4 hours    | ~2 hours     |

### Memory Requirements

| Dataset Size | Feature Matrix | RF Model | Peak Memory |
|--------------|----------------|----------|-------------|
| 1,000 images | ~0.7 MB        | ~50 MB   | ~200 MB     |
| 10,000 images| ~7 MB          | ~100 MB  | ~500 MB     |
| 100,000 images| ~70 MB        | ~200 MB  | ~2 GB       |

## Optimization Strategies

### 1. Parallel Feature Extraction

The bottleneck is feature extraction. You can parallelize this:

```python
from multiprocessing import Pool
from functools import partial

def extract_features_worker(args):
    img_path, label, detector = args
    try:
        image = detector.preprocess_image(img_path)
        subbands = detector.decompose(image)
        features = detector.extract_features(subbands)
        return features, label
    except Exception as e:
        return None, None

# Use multiprocessing
with Pool(processes=4) as pool:
    results = pool.map(extract_features_worker, image_list)
```

**Speedup**: ~3-4× with 4 cores (due to overhead)

### 2. Reduce Image Resolution

If images are large, resize before processing:

```python
# In preprocess_image, add:
if max(h, w) > 1024:
    scale = 1024 / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
```

**Speedup**: ~4× for 2048×2048 → 1024×1024

### 3. Reduce Decomposition Levels

```python
detector = WaveletDetector(levels=2)  # Instead of 3
```

**Speedup**: ~1.5×, but may reduce accuracy slightly

### 4. Skip Hyperparameter Tuning

Use default parameters for initial training:

```bash
python train_wavelet.py --real-dir data/real --ai-dir data/ai --output model.pkl
# Skip --tune flag
```

**Speedup**: ~90× for training phase

### 5. Use Smaller Random Forest

```python
# In train.py, modify:
RandomForestClassifier(n_estimators=100, max_depth=15, ...)
```

**Speedup**: ~2× for training, minimal accuracy loss

## Recommended Training Setup

### For Small Datasets (< 1,000 images)
- **Time**: ~10-15 minutes total
- **Memory**: < 500 MB
- **Strategy**: Use default settings, no tuning
- **Hardware**: Any modern laptop/desktop

### For Medium Datasets (1,000 - 10,000 images)
- **Time**: ~1-2 hours total
- **Memory**: ~1-2 GB
- **Strategy**: 
  - Parallel feature extraction (4-8 cores)
  - Default RF parameters
  - Optional: Tune on subset (1,000 samples)
- **Hardware**: Multi-core CPU recommended

### For Large Datasets (> 10,000 images)
- **Time**: ~4-8 hours total
- **Memory**: ~2-4 GB
- **Strategy**:
  - Parallel feature extraction (8+ cores)
  - Resize large images to 1024×1024
  - Train on subset or use incremental learning
  - Consider GPU acceleration (if available)
- **Hardware**: Multi-core CPU or cloud instance

## GPU Acceleration

Currently, the implementation uses CPU-only operations. For GPU acceleration:

1. **PyWavelets**: No GPU support (CPU-only)
2. **scikit-learn**: No GPU support (CPU-only)
3. **NumPy/SciPy**: Can use CuPy for GPU arrays, but requires significant refactoring

**Recommendation**: For datasets < 50,000 images, CPU is sufficient. For larger datasets, consider:
- Distributed processing across multiple machines
- Incremental learning (train on batches)
- Feature caching to disk

## Inference Performance

Once trained, inference is very fast:
- **Single image**: ~0.2-0.5 seconds
- **Batch of 100**: ~20-50 seconds
- **Memory**: ~50-100 MB (model + temporary arrays)

## Summary

**Training is moderately computationally intensive:**

1. **Feature extraction**: ~0.3-0.5 sec/image (can parallelize)
2. **Classifier training**: Fast for RF (~seconds to minutes)
3. **Hyperparameter tuning**: Very slow (hours for large datasets)

**Recommendations:**
- Start with default settings (no tuning)
- Use parallel processing for feature extraction
- For > 5,000 images, consider training on a subset first
- Hyperparameter tuning is optional and can be done later

