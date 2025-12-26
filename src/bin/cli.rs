//! CLI application for facial landmark detection and metrics.
//!
//! Usage:
//!   percent-face <image>                    # Human-readable output
//!   percent-face <image> --json             # JSON output
//!   percent-face <image> -o metrics.json    # Save to file

use clap::Parser;
use image::GenericImageView;
use percent_face::{dlib::load_dlib_model, BoundingBox, FaceMetrics, GrayImage};
use rustface::ImageData;
use serde::Serialize;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "percent-face")]
#[command(author, version, about = "Facial landmark detection and metrics", long_about = None)]
struct Args {
    /// Input image file
    #[arg(required = true)]
    image: PathBuf,

    /// Output as JSON
    #[arg(short, long)]
    json: bool,

    /// Output file (default: stdout)
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Face detector model path
    #[arg(long, default_value = "seeta_fd_frontal_v1.0.bin")]
    detector: PathBuf,

    /// Landmark model path
    #[arg(long, default_value = "shape_predictor_81_face_landmarks.dat")]
    landmarks: PathBuf,

    /// Minimum face size for detection
    #[arg(long, default_value = "20")]
    min_face_size: u32,

    /// Show verbose output
    #[arg(short, long)]
    verbose: bool,
}

/// Output structure for JSON serialization
#[derive(Serialize)]
struct Output {
    image: String,
    width: u32,
    height: u32,
    faces_detected: usize,
    faces: Vec<FaceOutput>,
}

#[derive(Serialize)]
struct FaceOutput {
    /// Face index (1-based)
    index: usize,
    /// Bounding box from detector
    bounding_box: BoundingBoxOutput,
    /// Landmark count (68 or 81)
    landmark_count: usize,
    /// Raw metrics (areas in pixels)
    areas: AreasOutput,
    /// Ratios (percentages)
    ratios: RatiosOutput,
}

#[derive(Serialize)]
struct BoundingBoxOutput {
    x: i32,
    y: i32,
    width: u32,
    height: u32,
    area_percent: f32,
}

#[derive(Serialize)]
struct AreasOutput {
    jawline: f32,
    head: f32,
    forehead: f32,
    left_eye: f32,
    right_eye: f32,
    left_eyebrow: f32,
    right_eyebrow: f32,
    nose: f32,
    outer_mouth: f32,
    inner_mouth: f32,
}

#[derive(Serialize)]
struct RatiosOutput {
    /// Features as % of face (jawline) area
    eyes_percent: f32,
    left_eye_percent: f32,
    right_eye_percent: f32,
    eyebrows_percent: f32,
    nose_percent: f32,
    mouth_percent: f32,
    lips_percent: f32,
    forehead_percent: f32,
    /// Head/face relationship
    face_to_head_percent: f32,
    /// Symmetry and inter-feature
    eye_symmetry: f32,
    eye_to_mouth_ratio: f32,
    /// Image coverage
    face_of_image_percent: f32,
    head_of_image_percent: f32,
    /// Model info
    has_forehead_landmarks: bool,
}

fn main() {
    let args = Args::parse();

    if let Err(e) = run(&args) {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn run(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    // Load models
    if args.verbose {
        eprintln!("Loading face detector from {:?}...", args.detector);
    }
    let detector_path = args.detector.to_str()
        .ok_or("Invalid detector path")?;
    let mut detector = rustface::create_detector(detector_path)
        .map_err(|e| format!("Failed to load face detector: {}", e))?;
    detector.set_min_face_size(args.min_face_size);
    detector.set_score_thresh(2.0);
    detector.set_pyramid_scale_factor(0.8);
    detector.set_slide_window_step(4, 4);

    if args.verbose {
        eprintln!("Loading landmark model from {:?}...", args.landmarks);
    }
    let landmark_model = load_dlib_model(&args.landmarks)?;

    // Load image
    if args.verbose {
        eprintln!("Loading image {:?}...", args.image);
    }
    let img = image::open(&args.image)?;
    let (width, height) = img.dimensions();
    let image_area = (width * height) as f32;

    // Convert to grayscale
    let gray_img = img.to_luma8();
    let gray = GrayImage::new(gray_img.to_vec(), width, height);
    let image_data = ImageData::new(gray_img.as_raw(), width, height);

    // Detect faces
    if args.verbose {
        eprintln!("Detecting faces...");
    }
    let faces = detector.detect(&image_data);

    if args.verbose {
        eprintln!("Found {} face(s)", faces.len());
    }

    // Process each face
    let mut face_outputs = Vec::new();

    for (i, face) in faces.iter().enumerate() {
        let bbox = face.bbox();
        let bbox_area = (bbox.width() * bbox.height()) as f32;

        let face_bbox = BoundingBox::new(
            bbox.x() as f32,
            bbox.y() as f32,
            bbox.width() as f32,
            bbox.height() as f32,
        );

        let landmarks = landmark_model.predict(&gray, &face_bbox);
        let landmark_count = landmarks.num_landmarks();

        let metrics = FaceMetrics::from_shape(&landmarks);

        if let Some(m) = metrics {
            let face_output = FaceOutput {
                index: i + 1,
                bounding_box: BoundingBoxOutput {
                    x: bbox.x(),
                    y: bbox.y(),
                    width: bbox.width(),
                    height: bbox.height(),
                    area_percent: bbox_area / image_area * 100.0,
                },
                landmark_count,
                areas: AreasOutput {
                    jawline: m.jawline_area,
                    head: m.head_area,
                    forehead: m.forehead_area,
                    left_eye: m.left_eye_area,
                    right_eye: m.right_eye_area,
                    left_eyebrow: m.left_eyebrow_area,
                    right_eyebrow: m.right_eyebrow_area,
                    nose: m.nose_area,
                    outer_mouth: m.outer_mouth_area,
                    inner_mouth: m.inner_mouth_area,
                },
                ratios: RatiosOutput {
                    eyes_percent: m.eyes_ratio(),
                    left_eye_percent: m.left_eye_ratio(),
                    right_eye_percent: m.right_eye_ratio(),
                    eyebrows_percent: m.eyebrows_ratio(),
                    nose_percent: m.nose_ratio(),
                    mouth_percent: m.mouth_ratio(),
                    lips_percent: m.lips_ratio(),
                    forehead_percent: m.forehead_ratio(),
                    face_to_head_percent: m.face_to_head_ratio(),
                    eye_symmetry: m.eye_symmetry(),
                    eye_to_mouth_ratio: m.eye_to_mouth_ratio() / 100.0,
                    face_of_image_percent: m.jawline_area / image_area * 100.0,
                    head_of_image_percent: m.head_area / image_area * 100.0,
                    has_forehead_landmarks: m.has_forehead_landmarks,
                },
            };
            face_outputs.push(face_output);
        }
    }

    let output = Output {
        image: args.image.display().to_string(),
        width,
        height,
        faces_detected: faces.len(),
        faces: face_outputs,
    };

    // Generate output
    let output_str = if args.json {
        serde_json::to_string_pretty(&output)?
    } else {
        format_human_readable(&output)
    };

    // Write output
    if let Some(ref path) = args.output {
        std::fs::write(path, &output_str)?;
        if args.verbose {
            eprintln!("Output written to {:?}", path);
        }
    } else {
        println!("{}", output_str);
    }

    Ok(())
}

fn format_human_readable(output: &Output) -> String {
    let mut s = String::new();

    s.push_str(&format!("Image: {} ({}x{})\n", output.image, output.width, output.height));
    s.push_str(&format!("Faces detected: {}\n", output.faces_detected));

    if output.faces.is_empty() {
        s.push_str("\nNo faces found.\n");
        return s;
    }

    for face in &output.faces {
        s.push_str(&format!("\n--- Face {} ---\n", face.index));
        s.push_str(&format!("Bounding box: {}x{} at ({}, {})\n",
            face.bounding_box.width, face.bounding_box.height,
            face.bounding_box.x, face.bounding_box.y));
        s.push_str(&format!("Landmarks: {} points\n", face.landmark_count));

        s.push_str("\nImage Coverage:\n");
        s.push_str(&format!("  Face: {:.1}% of image\n", face.ratios.face_of_image_percent));
        let head_label = if face.ratios.has_forehead_landmarks { "Head" } else { "Head (est.)" };
        s.push_str(&format!("  {}: {:.1}% of image\n", head_label, face.ratios.head_of_image_percent));

        s.push_str("\nFeatures (% of face):\n");
        s.push_str(&format!("  Eyes:     {:.1}% (L: {:.1}%, R: {:.1}%)\n",
            face.ratios.eyes_percent, face.ratios.left_eye_percent, face.ratios.right_eye_percent));
        s.push_str(&format!("  Eyebrows: {:.1}%\n", face.ratios.eyebrows_percent));
        s.push_str(&format!("  Nose:     {:.1}%\n", face.ratios.nose_percent));
        s.push_str(&format!("  Mouth:    {:.1}% (lips: {:.1}%)\n",
            face.ratios.mouth_percent, face.ratios.lips_percent));
        s.push_str(&format!("  Forehead: {:.1}%\n", face.ratios.forehead_percent));

        s.push_str("\nRatios:\n");
        s.push_str(&format!("  Face/Head:    {:.1}%\n", face.ratios.face_to_head_percent));
        s.push_str(&format!("  Eye symmetry: {:.1}%\n", face.ratios.eye_symmetry));
        s.push_str(&format!("  Eye/Mouth:    {:.2}x\n", face.ratios.eye_to_mouth_ratio));
    }

    s
}
