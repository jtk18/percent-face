# percent-face

Pure Rust facial landmark detection using Ensemble of Regression Trees (ERT).

This crate implements the algorithm from ["One Millisecond Face Alignment with an Ensemble of Regression Trees"](https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Kazemi_One_Millisecond_Face_2014_CVPR_paper.pdf) (Kazemi & Sullivan, CVPR 2014).

## Status

**Work in Progress** - Core inference implemented, testing against dlib models in progress.

## Algorithm Overview

The ERT algorithm achieves real-time facial landmark detection through:

1. **Initial Estimate**: Start with mean face shape scaled to the detected face bounding box
2. **Cascade Refinement**: For each of ~10 cascade levels:
   - Sample sparse pixel intensity differences as features
   - Each regression tree in the ensemble votes on landmark position adjustments
   - Sum tree predictions to get shape update
   - Apply update to refine current shape estimate
3. **Output**: Final 68-point (or 5-point) facial landmarks

### Why This Works

- **Sparse features**: Only ~400 pixel comparisons per cascade level (not dense HOG/SIFT)
- **Shape-indexed features**: Pixel locations are relative to current landmark estimates, providing implicit alignment
- **Gradient boosting**: Each cascade level corrects errors from previous levels
- **Simple inference**: Just tree traversal and additions - no convolutions or matrix operations

## Crate Structure

```
src/
├── lib.rs          # Public API and module exports
├── types.rs        # Core types: Point, Shape, BoundingBox
├── tree.rs         # RegressionTree, TreeEnsemble, SplitFeature
├── features.rs     # Pixel feature extraction, ImageAccess trait
├── model.rs        # ShapePredictor (main entry point)
├── dlib.rs         # dlib .dat format loader + JSON loader
└── error.rs        # Error types
```

## Usage

```rust
use percent_face::{ShapePredictor, BoundingBox, GrayImage};

// Load a model (converted from dlib format)
let model = percent_face::dlib::load_json_model("model.json")?;

// Create/load a grayscale image
let image = GrayImage::new(pixels, width, height);

// Face bounding box from your face detector
let face_rect = BoundingBox::new(x, y, width, height);

// Predict landmarks
let landmarks = model.predict(&image, &face_rect);

for (i, point) in landmarks.points.iter().enumerate() {
    println!("Landmark {}: ({}, {})", i, point.x, point.y);
}
```

### Custom Image Types

Implement the `ImageAccess` trait for your own image types:

```rust
use percent_face::ImageAccess;

impl ImageAccess for MyImage {
    fn get_pixel(&self, x: i32, y: i32) -> u8 {
        // Return grayscale intensity, 0 for out-of-bounds
    }
    fn width(&self) -> u32 { self.width }
    fn height(&self) -> u32 { self.height }
}
```

## Loading dlib Models

Two options for loading dlib's pretrained shape predictors:

### Option 1: Python Converter (Recommended)

```bash
# Decompress model
bunzip2 shape_predictor_68_face_landmarks.dat.bz2

# Convert to JSON
python scripts/convert_dlib_model.py shape_predictor_68_face_landmarks.dat model.json
```

```rust
let model = percent_face::dlib::load_json_model("model.json")?;
```

### Option 2: Direct Binary Loading

```rust
let model = percent_face::dlib::load_dlib_model("shape_predictor_68_face_landmarks.dat")?;
```

Note: Binary loading may need adjustment for different dlib versions.

## Implementation Plan

### Phase 1: Core Implementation (Done)
- [x] Core data structures (Point, Shape, BoundingBox)
- [x] Regression tree structure and traversal
- [x] Tree ensemble (gradient boosting)
- [x] Pixel feature extraction
- [x] Cascade inference loop
- [x] Model serialization (bincode)
- [x] dlib format loader (binary + JSON)

### Phase 2: Accuracy & Compatibility (In Progress)
- [ ] Test against dlib models
- [ ] Bilinear interpolation for sub-pixel sampling
- [ ] Similarity transform normalization between cascade stages
- [ ] Validate output matches dlib reference

### Phase 3: Performance
- [ ] Benchmarks
- [ ] SIMD optimization for feature extraction
- [ ] Parallel tree evaluation
- [ ] Memory layout optimization

### Phase 4: Training (Optional)
- [ ] Training data loader (iBUG format)
- [ ] Gradient boosting trainer
- [ ] Model export

## References

- [One Millisecond Face Alignment with an Ensemble of Regression Trees](https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Kazemi_One_Millisecond_Face_2014_CVPR_paper.pdf) - Original paper
- [dlib shape_predictor](http://dlib.net/ml.html#shape_predictor) - Reference implementation
- [iBUG 300-W dataset](https://ibug.doc.ic.ac.uk/resources/300-W/) - Standard benchmark

## License

MIT
