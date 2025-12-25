use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::features::{find_similarity_transform, make_feature_extractor, ImageAccess};
use crate::tree::TreeEnsemble;
use crate::types::{BoundingBox, Point, Shape};

/// The main shape predictor model.
///
/// This implements the ERT algorithm for facial landmark detection.
/// The model consists of:
/// - A mean shape (initial estimate)
/// - A cascade of tree ensembles that iteratively refine the shape
///
/// # Usage
///
/// ```ignore
/// let model = ShapePredictor::load("model.bin")?;
/// let face_rect = BoundingBox::new(100.0, 100.0, 200.0, 200.0);
/// let landmarks = model.predict(&image, &face_rect);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapePredictor {
    /// The mean face shape in normalized [0,1] coordinates.
    /// This serves as the initial estimate before cascade refinement.
    mean_shape: Shape,

    /// Cascade of tree ensembles. Each ensemble refines the shape estimate.
    cascade: Vec<TreeEnsemble>,

    /// Number of landmark points (typically 68 for full face, 5 for simplified).
    num_landmarks: usize,
}

impl ShapePredictor {
    /// Create a new shape predictor with the given mean shape and cascade.
    pub fn new(mean_shape: Shape, cascade: Vec<TreeEnsemble>) -> Self {
        let num_landmarks = mean_shape.num_landmarks();
        Self {
            mean_shape,
            cascade,
            num_landmarks,
        }
    }

    /// Load a model from a binary file.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes)?;
        let model: Self = bincode::deserialize(&bytes)?;
        Ok(model)
    }

    /// Save the model to a binary file.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        let bytes = bincode::serialize(self)?;
        writer.write_all(&bytes)?;
        Ok(())
    }

    /// Get the number of landmarks this model predicts.
    pub fn num_landmarks(&self) -> usize {
        self.num_landmarks
    }

    /// Get the number of cascade stages.
    pub fn num_cascade_stages(&self) -> usize {
        self.cascade.len()
    }

    /// Predict facial landmarks for a face detected in the given bounding box.
    ///
    /// # Arguments
    ///
    /// * `image` - Grayscale image to analyze
    /// * `face_rect` - Bounding box of the detected face
    ///
    /// # Returns
    ///
    /// A `Shape` containing the predicted landmark positions in image coordinates.
    pub fn predict<I: ImageAccess>(&self, image: &I, face_rect: &BoundingBox) -> Shape {
        // Start with mean shape, scaled to the face bounding box
        let initial_shape = self.initialize_shape(face_rect);
        let mut current_shape = initial_shape.clone();

        // Apply each cascade stage
        for ensemble in &self.cascade {
            // Compute similarity transform from initial shape to current shape.
            // This allows feature extraction to adapt to the current face orientation.
            let tform = find_similarity_transform(&initial_shape, &current_shape);

            // Create feature extractor for current shape estimate with transform
            let get_feature = make_feature_extractor(&current_shape, face_rect, image, Some(tform));

            // Get shape delta from this ensemble
            let delta = ensemble.predict(get_feature);

            // Apply delta (scale it to image coordinates)
            self.apply_delta(&mut current_shape, &delta, face_rect);
        }

        current_shape
    }

    /// Initialize the shape by scaling the mean shape to the face bounding box.
    fn initialize_shape(&self, face_rect: &BoundingBox) -> Shape {
        let mut shape = Shape::with_capacity(self.num_landmarks);
        for p in &self.mean_shape.points {
            // Mean shape is in normalized [0,1] coordinates
            // Scale to image coordinates within the bounding box
            let img_point = face_rect.denormalize_point(*p);
            shape.points.push(img_point);
        }
        shape
    }

    /// Apply a normalized delta to the current shape.
    fn apply_delta(&self, shape: &mut Shape, delta: &Shape, face_rect: &BoundingBox) {
        for (point, delta_point) in shape.points.iter_mut().zip(delta.points.iter()) {
            // Delta is in normalized coordinates, scale to image space
            point.x += delta_point.x * face_rect.width;
            point.y += delta_point.y * face_rect.height;
        }
    }
}

/// Builder for creating a ShapePredictor model.
pub struct ShapePredictorBuilder {
    mean_shape: Option<Shape>,
    cascade: Vec<TreeEnsemble>,
}

impl ShapePredictorBuilder {
    pub fn new() -> Self {
        Self {
            mean_shape: None,
            cascade: Vec::new(),
        }
    }

    /// Set the mean shape (initial estimate).
    pub fn mean_shape(mut self, shape: Shape) -> Self {
        self.mean_shape = Some(shape);
        self
    }

    /// Add a tree ensemble to the cascade.
    pub fn add_cascade_stage(mut self, ensemble: TreeEnsemble) -> Self {
        self.cascade.push(ensemble);
        self
    }

    /// Build the ShapePredictor.
    pub fn build(self) -> Result<ShapePredictor> {
        let mean_shape = self
            .mean_shape
            .ok_or_else(|| crate::error::Error::InvalidModel("Missing mean shape".into()))?;

        if self.cascade.is_empty() {
            return Err(crate::error::Error::InvalidModel(
                "Cascade must have at least one stage".into(),
            ));
        }

        Ok(ShapePredictor::new(mean_shape, self.cascade))
    }
}

impl Default for ShapePredictorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Creates a simple 5-point mean face shape.
/// Points: left eye center, right eye center, nose tip, left mouth corner, right mouth corner.
pub fn default_5_point_mean_shape() -> Shape {
    Shape::new(vec![
        Point::new(0.30, 0.30), // left eye
        Point::new(0.70, 0.30), // right eye
        Point::new(0.50, 0.55), // nose tip
        Point::new(0.35, 0.75), // left mouth
        Point::new(0.65, 0.75), // right mouth
    ])
}

/// Creates a 68-point mean face shape based on the iBUG 68-point annotation scheme.
pub fn default_68_point_mean_shape() -> Shape {
    // Approximate positions for the 68 iBUG landmarks
    // In normalized [0,1] coordinates
    let points = vec![
        // Jaw line (0-16)
        Point::new(0.10, 0.35),
        Point::new(0.11, 0.45),
        Point::new(0.12, 0.55),
        Point::new(0.14, 0.65),
        Point::new(0.18, 0.73),
        Point::new(0.24, 0.80),
        Point::new(0.32, 0.85),
        Point::new(0.41, 0.88),
        Point::new(0.50, 0.89), // Chin center
        Point::new(0.59, 0.88),
        Point::new(0.68, 0.85),
        Point::new(0.76, 0.80),
        Point::new(0.82, 0.73),
        Point::new(0.86, 0.65),
        Point::new(0.88, 0.55),
        Point::new(0.89, 0.45),
        Point::new(0.90, 0.35),
        // Right eyebrow (17-21)
        Point::new(0.20, 0.26),
        Point::new(0.25, 0.22),
        Point::new(0.32, 0.21),
        Point::new(0.38, 0.23),
        Point::new(0.43, 0.27),
        // Left eyebrow (22-26)
        Point::new(0.57, 0.27),
        Point::new(0.62, 0.23),
        Point::new(0.68, 0.21),
        Point::new(0.75, 0.22),
        Point::new(0.80, 0.26),
        // Nose bridge (27-30)
        Point::new(0.50, 0.32),
        Point::new(0.50, 0.40),
        Point::new(0.50, 0.48),
        Point::new(0.50, 0.55),
        // Nose bottom (31-35)
        Point::new(0.40, 0.58),
        Point::new(0.45, 0.60),
        Point::new(0.50, 0.62),
        Point::new(0.55, 0.60),
        Point::new(0.60, 0.58),
        // Right eye (36-41)
        Point::new(0.24, 0.32),
        Point::new(0.28, 0.29),
        Point::new(0.34, 0.29),
        Point::new(0.38, 0.33),
        Point::new(0.34, 0.35),
        Point::new(0.28, 0.35),
        // Left eye (42-47)
        Point::new(0.62, 0.33),
        Point::new(0.66, 0.29),
        Point::new(0.72, 0.29),
        Point::new(0.76, 0.32),
        Point::new(0.72, 0.35),
        Point::new(0.66, 0.35),
        // Outer lip (48-59)
        Point::new(0.32, 0.72),
        Point::new(0.38, 0.68),
        Point::new(0.44, 0.66),
        Point::new(0.50, 0.67),
        Point::new(0.56, 0.66),
        Point::new(0.62, 0.68),
        Point::new(0.68, 0.72),
        Point::new(0.62, 0.78),
        Point::new(0.56, 0.80),
        Point::new(0.50, 0.81),
        Point::new(0.44, 0.80),
        Point::new(0.38, 0.78),
        // Inner lip (60-67)
        Point::new(0.36, 0.72),
        Point::new(0.44, 0.70),
        Point::new(0.50, 0.70),
        Point::new(0.56, 0.70),
        Point::new(0.64, 0.72),
        Point::new(0.56, 0.74),
        Point::new(0.50, 0.75),
        Point::new(0.44, 0.74),
    ];

    Shape::new(points)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::GrayImage;
    use crate::tree::{RegressionTree, TreeNode};

    fn create_dummy_model() -> ShapePredictor {
        let mean_shape = default_5_point_mean_shape();

        // Create a trivial ensemble with a single tree that returns zeros
        let tree = RegressionTree::new(vec![TreeNode::Leaf {
            delta: Shape::zeros(5),
        }]);

        let ensemble = TreeEnsemble::new(vec![tree], 5);

        ShapePredictor::new(mean_shape, vec![ensemble])
    }

    #[test]
    fn model_initialization() {
        let model = create_dummy_model();
        assert_eq!(model.num_landmarks(), 5);
        assert_eq!(model.num_cascade_stages(), 1);
    }

    #[test]
    fn predict_returns_correct_shape_size() {
        let model = create_dummy_model();
        let image = GrayImage::from_fn(100, 100, |_, _| 128);
        let face_rect = BoundingBox::new(10.0, 10.0, 80.0, 80.0);

        let landmarks = model.predict(&image, &face_rect);
        assert_eq!(landmarks.num_landmarks(), 5);
    }

    #[test]
    fn predict_landmarks_within_bbox() {
        let model = create_dummy_model();
        let image = GrayImage::from_fn(100, 100, |_, _| 128);
        let face_rect = BoundingBox::new(10.0, 10.0, 80.0, 80.0);

        let landmarks = model.predict(&image, &face_rect);

        // All landmarks should be roughly within the face bounding box
        for point in &landmarks.points {
            assert!(point.x >= 0.0 && point.x <= 100.0);
            assert!(point.y >= 0.0 && point.y <= 100.0);
        }
    }

    #[test]
    fn save_and_load_model() {
        let model = create_dummy_model();

        // Save to temp file
        let temp_path = std::env::temp_dir().join("test_model.bin");
        model.save(&temp_path).unwrap();

        // Load it back
        let loaded = ShapePredictor::load(&temp_path).unwrap();
        assert_eq!(loaded.num_landmarks(), model.num_landmarks());
        assert_eq!(loaded.num_cascade_stages(), model.num_cascade_stages());

        // Clean up
        std::fs::remove_file(temp_path).ok();
    }

    #[test]
    fn default_mean_shapes() {
        let shape_5 = default_5_point_mean_shape();
        assert_eq!(shape_5.num_landmarks(), 5);

        let shape_68 = default_68_point_mean_shape();
        assert_eq!(shape_68.num_landmarks(), 68);
    }
}
