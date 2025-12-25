//! Integration tests comparing percent-face output against dlib reference.

use percent_face::{dlib::load_dlib_model, BoundingBox, GrayImage};
use std::path::PathBuf;

fn dlib_models_dir() -> Option<PathBuf> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("dlib-models");
    if path.exists() {
        Some(path)
    } else {
        None
    }
}

/// Create the gradient test image: pixel[x,y] = (x + y) % 256
fn create_gradient_image(width: u32, height: u32) -> GrayImage {
    GrayImage::from_fn(width, height, |x, y| ((x + y) % 256) as u8)
}

/// Reference landmarks from dlib for the 5-point model on gradient image.
/// Image: 100x100, formula: (x + y) % 256
/// BBox: left=25, top=25, right=75, bottom=75 (i.e., x=25, y=25, w=50, h=50)
const DLIB_REFERENCE_5POINT: [(f32, f32); 5] = [
    (68.0, 46.0), // Landmark 0
    (60.0, 45.0), // Landmark 1
    (40.0, 40.0), // Landmark 2
    (48.0, 42.0), // Landmark 3
    (51.0, 60.0), // Landmark 4
];

#[test]
fn compare_5point_inference() {
    let Some(models_dir) = dlib_models_dir() else {
        eprintln!("Skipping test: dlib-models directory not found");
        return;
    };

    let model_path = models_dir.join("shape_predictor_5_face_landmarks.dat.bz2");
    if !model_path.exists() {
        eprintln!("Skipping test: model file not found");
        return;
    }

    // Load model
    let model = load_dlib_model(&model_path).expect("Failed to load model");
    assert_eq!(model.num_landmarks(), 5);

    // Create the same test image as dlib reference
    let image = create_gradient_image(100, 100);

    // Same bounding box: dlib uses (left, top, right, bottom) = (25, 25, 75, 75)
    // Our BoundingBox uses (x, y, width, height) = (25, 25, 50, 50)
    let bbox = BoundingBox::new(25.0, 25.0, 50.0, 50.0);

    // Run inference
    let landmarks = model.predict(&image, &bbox);

    println!("\nComparing percent-face vs dlib reference:");
    println!("{:<12} {:>12} {:>12} {:>12}", "Landmark", "percent-face", "dlib", "diff");
    println!("{:-<52}", "");

    let mut max_error = 0.0f32;
    let mut total_error = 0.0f32;

    for (i, (expected_x, expected_y)) in DLIB_REFERENCE_5POINT.iter().enumerate() {
        let actual = &landmarks.points[i];
        let error_x = (actual.x - expected_x).abs();
        let error_y = (actual.y - expected_y).abs();
        let error = (error_x * error_x + error_y * error_y).sqrt();

        max_error = max_error.max(error);
        total_error += error;

        println!(
            "Point {:<5} ({:>5.1}, {:>5.1}) ({:>5.1}, {:>5.1}) {:>6.2}px",
            i, actual.x, actual.y, expected_x, expected_y, error
        );
    }

    let avg_error = total_error / 5.0;
    println!("{:-<52}", "");
    println!("Max error: {:.2}px, Avg error: {:.2}px", max_error, avg_error);

    // Assert sub-pixel accuracy - allow small margin for floating point differences
    // and nearest-neighbor vs bilinear interpolation differences
    assert!(
        max_error < 1.5,
        "Max error {:.2}px exceeds threshold of 1.5px",
        max_error
    );
    assert!(
        avg_error < 1.0,
        "Avg error {:.2}px exceeds threshold of 1.0px",
        avg_error
    );
}

#[test]
fn inference_produces_valid_landmarks() {
    let Some(models_dir) = dlib_models_dir() else {
        eprintln!("Skipping test: dlib-models directory not found");
        return;
    };

    let model_path = models_dir.join("shape_predictor_5_face_landmarks.dat.bz2");
    if !model_path.exists() {
        return;
    }

    let model = load_dlib_model(&model_path).expect("Failed to load model");

    // Create test image
    let image = create_gradient_image(200, 200);
    let bbox = BoundingBox::new(50.0, 50.0, 100.0, 100.0);

    let landmarks = model.predict(&image, &bbox);

    // All landmarks should be within or near the bounding box
    for (i, point) in landmarks.points.iter().enumerate() {
        // Allow some margin outside bbox (landmarks can be slightly outside face box)
        let margin = 50.0;
        assert!(
            point.x >= bbox.x - margin && point.x <= bbox.x + bbox.width + margin,
            "Landmark {} x={} outside expected range",
            i,
            point.x
        );
        assert!(
            point.y >= bbox.y - margin && point.y <= bbox.y + bbox.height + margin,
            "Landmark {} y={} outside expected range",
            i,
            point.y
        );
    }

    println!("All landmarks within expected bounds");
}
