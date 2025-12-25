# percent-face

Pure Rust facial landmark detection using Ensemble of Regression Trees (ERT).

This crate implements the algorithm from ["One Millisecond Face Alignment with an Ensemble of Regression Trees"](https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Kazemi_One_Millisecond_Face_2014_CVPR_paper.pdf) (Kazemi & Sullivan, CVPR 2014).

## Status

**Work in Progress** - Core inference implemented, dlib model loading fully working. Ready for accuracy testing against dlib reference.

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
├── dlib.rs         # dlib .dat/.dat.bz2 format loader
└── error.rs        # Error types
```

## GUI Demo

A GUI application is available for testing face detection and landmark visualization:

```bash
# Download the face detection model
git clone --depth 1 https://github.com/atomashpolskiy/rustface.git /tmp/rustface
cp /tmp/rustface/model/seeta_fd_frontal_v1.0.bin .
rm -rf /tmp/rustface

# Run the GUI
cargo run --features gui --bin percent-face-gui
```

1. Click "Load Face Detector" and "Load Landmark Model"
2. Open an image (File > Open or drag & drop)
3. Click "Detect Faces" to see landmarks overlaid

## Usage

```rust
use percent_face::{BoundingBox, GrayImage};

// Load a dlib model directly (supports .dat and .dat.bz2)
let model = percent_face::dlib::load_dlib_model("shape_predictor_68_face_landmarks.dat.bz2")?;

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

Load dlib's pretrained shape predictors directly - both compressed (`.dat.bz2`) and uncompressed (`.dat`) formats are supported:

```rust
// Load compressed model directly (no decompression needed)
let model = percent_face::dlib::load_dlib_model("shape_predictor_68_face_landmarks.dat.bz2")?;

// Or uncompressed
let model = percent_face::dlib::load_dlib_model("shape_predictor_68_face_landmarks.dat")?;
```

### Obtaining Models

Download pretrained models from the [dlib-models repository](https://github.com/davisking/dlib-models):

```bash
git clone --depth 1 git@github.com:davisking/dlib-models.git
```

Available shape predictors:
- `shape_predictor_5_face_landmarks.dat.bz2` - 5-point model (eyes + nose tip)
- `shape_predictor_68_face_landmarks.dat.bz2` - Full 68-point iBUG model

## Implementation Plan

### Phase 1: Core Implementation (Done)
- [x] Core data structures (Point, Shape, BoundingBox)
- [x] Regression tree structure and traversal
- [x] Tree ensemble (gradient boosting)
- [x] Pixel feature extraction framework
- [x] Cascade inference loop
- [x] Model serialization (bincode)
- [x] dlib .dat/.dat.bz2 format loader (pure Rust, no Python)

### Phase 2: Accuracy & Compatibility (Done)
- [x] Integrate anchor_idx and deltas into split features
- [x] Test inference against dlib reference output (sub-pixel accuracy achieved)
- [x] Bilinear interpolation for sub-pixel sampling
- [ ] Similarity transform normalization between cascade stages (optional, for rotated faces)

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
