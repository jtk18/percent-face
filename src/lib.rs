//! # percent-face
//!
//! Pure Rust facial landmark detection and facial feature measurement.
//!
//! This crate provides:
//! - **Landmark Detection**: ERT-based facial landmark detection (68 or 81 points)
//! - **Feature Metrics**: Area calculations for eyes, nose, mouth, eyebrows, forehead
//! - **Proportions**: Feature sizes relative to face area, symmetry measurements
//!
//! Implements the algorithm from "One Millisecond Face Alignment with an
//! Ensemble of Regression Trees" (Kazemi & Sullivan, 2014).
//!
//! ## Algorithm Overview
//!
//! 1. Start with mean face shape as initial estimate
//! 2. For each cascade level:
//!    - Extract sparse pixel intensity difference features
//!    - Each regression tree in the ensemble predicts shape deltas
//!    - Sum predictions to get shape update
//!    - Apply update to refine current shape estimate
//! 3. Return final 68/81-point facial landmarks
//! 4. Optionally calculate facial feature metrics from landmarks
//!
//! ## Quick Start
//!
//! ```rust
//! use percent_face::{ShapePredictor, BoundingBox, GrayImage};
//!
//! // Load a trained model
//! // let model = ShapePredictor::load("model.bin").unwrap();
//!
//! // Or create a test model for development
//! use percent_face::{
//!     default_68_point_mean_shape, ShapePredictorBuilder,
//!     TreeEnsemble, RegressionTree, TreeNode, Shape,
//! };
//!
//! let mean_shape = default_68_point_mean_shape();
//! let tree = RegressionTree::new(vec![TreeNode::Leaf {
//!     delta: Shape::zeros(68),
//! }]);
//! let ensemble = TreeEnsemble::new(vec![tree], 68);
//! let model = ShapePredictorBuilder::new()
//!     .mean_shape(mean_shape)
//!     .add_cascade_stage(ensemble)
//!     .build()
//!     .unwrap();
//!
//! // Create a grayscale image (or use your own via ImageAccess trait)
//! let image = GrayImage::from_fn(640, 480, |x, y| ((x + y) % 256) as u8);
//!
//! // Define face bounding box (from a face detector)
//! let face_rect = BoundingBox::new(100.0, 50.0, 200.0, 200.0);
//!
//! // Predict landmarks
//! let landmarks = model.predict(&image, &face_rect);
//! println!("Found {} landmarks", landmarks.num_landmarks());
//! ```
//!
//! ## Custom Image Types
//!
//! Implement the [`ImageAccess`] trait for your own image types:
//!
//! ```rust
//! use percent_face::ImageAccess;
//!
//! struct MyImage { /* ... */ }
//!
//! impl ImageAccess for MyImage {
//!     fn get_pixel(&self, x: i32, y: i32) -> u8 {
//!         // Return grayscale intensity at (x, y)
//!         // Return 0 for out-of-bounds
//!         0
//!     }
//!     fn width(&self) -> u32 { 640 }
//!     fn height(&self) -> u32 { 480 }
//! }
//! ```

pub mod dlib;
mod error;
mod features;
mod metrics;
mod model;
mod tree;
mod types;

pub use error::{Error, Result};
pub use features::{find_similarity_transform, GrayImage, ImageAccess, SimilarityTransform2D};
pub use metrics::{polygon_area, FaceMetrics};
pub use model::{
    default_5_point_mean_shape, default_68_point_mean_shape, ShapePredictor, ShapePredictorBuilder,
};
pub use tree::{RegressionTree, SplitFeature, TreeEnsemble, TreeNode};
pub use types::{BoundingBox, Point, Shape};
