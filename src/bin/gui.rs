//! GUI application for visualizing facial landmark detection.
//!
//! Run with: cargo run --features gui --bin percent-face-gui

use eframe::egui;
use image::{DynamicImage, Rgba, RgbaImage};
use percent_face::{dlib::load_dlib_model, BoundingBox, FaceMetrics, GrayImage, ShapePredictor};
use rustface::{Detector, FaceInfo, ImageData};
use std::collections::HashMap;
use std::f32::consts::PI;
use std::fs::File;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread;

/// Information about a downloadable model
#[allow(dead_code)]
struct ModelInfo {
    name: &'static str,
    filename: &'static str,
    url: &'static str,
    description: &'static str,
    required: bool,
}

const AVAILABLE_MODELS: &[ModelInfo] = &[
    ModelInfo {
        name: "Face Detector",
        filename: "seeta_fd_frontal_v1.0.bin",
        url: "https://github.com/atomashpolskiy/rustface/raw/master/model/seeta_fd_frontal_v1.0.bin",
        description: "SeetaFace frontal face detector (~1 MB)",
        required: true,
    },
    ModelInfo {
        name: "81-point Landmarks",
        filename: "shape_predictor_81_face_landmarks.dat",
        url: "https://github.com/codeniko/shape_predictor_81_face_landmarks/raw/master/shape_predictor_81_face_landmarks.dat",
        description: "Full face with forehead (~19 MB)",
        required: true,
    },
    ModelInfo {
        name: "68-point Landmarks",
        filename: "shape_predictor_68_face_landmarks.dat.bz2",
        url: "https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2",
        description: "Standard iBUG model (~99 MB)",
        required: false,
    },
    ModelInfo {
        name: "5-point Landmarks",
        filename: "shape_predictor_5_face_landmarks.dat.bz2",
        url: "https://github.com/davisking/dlib-models/raw/master/shape_predictor_5_face_landmarks.dat.bz2",
        description: "Eyes + nose only (~9 MB)",
        required: false,
    },
];

#[derive(Clone, Debug)]
enum DownloadStatus {
    NotDownloaded,
    Downloading(f32), // progress 0.0 to 1.0
    Downloaded,
    Failed(String),
}

impl PartialEq for DownloadStatus {
    fn eq(&self, other: &Self) -> bool {
        matches!(
            (self, other),
            (DownloadStatus::NotDownloaded, DownloadStatus::NotDownloaded)
                | (DownloadStatus::Downloaded, DownloadStatus::Downloaded)
                | (DownloadStatus::Downloading(_), DownloadStatus::Downloading(_))
                | (DownloadStatus::Failed(_), DownloadStatus::Failed(_))
        )
    }
}

/// Download a model file from the given URL with progress reporting
fn download_model(
    filename: &str,
    url: &str,
    progress: Arc<Mutex<HashMap<String, DownloadStatus>>>,
    ctx: egui::Context,
) -> Result<(), String> {
    let response = ureq::get(url)
        .call()
        .map_err(|e| format!("HTTP error: {}", e))?;

    let content_length = response
        .header("Content-Length")
        .and_then(|s| s.parse::<u64>().ok());

    let mut file = File::create(filename)
        .map_err(|e| format!("Failed to create file: {}", e))?;

    let mut reader = response.into_reader();
    let mut buffer = [0u8; 32768];
    let mut downloaded: u64 = 0;

    loop {
        let bytes_read = reader
            .read(&mut buffer)
            .map_err(|e| format!("Read error: {}", e))?;
        if bytes_read == 0 {
            break;
        }
        file.write_all(&buffer[..bytes_read])
            .map_err(|e| format!("Write error: {}", e))?;

        downloaded += bytes_read as u64;

        // Update progress
        if let Some(total) = content_length {
            let progress_val = downloaded as f32 / total as f32;
            if let Ok(mut status) = progress.lock() {
                status.insert(filename.to_string(), DownloadStatus::Downloading(progress_val));
            }
            ctx.request_repaint();
        }
    }

    Ok(())
}

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1024.0, 768.0]),
        // Allow software rendering fallback for environments without GPU (WSL, SSH, etc.)
        hardware_acceleration: eframe::HardwareAcceleration::Preferred,
        ..Default::default()
    };

    eframe::run_native(
        "percent-face - Facial Landmark Detection",
        options,
        Box::new(|cc| Ok(Box::new(FaceApp::new(cc)))),
    )
}

/// Extended metrics for GUI display (adds image-relative calculations)
#[derive(Clone, Default)]
struct GuiFaceMetrics {
    /// Core metrics from the library
    metrics: FaceMetrics,
    /// Bounding box area from detector
    bbox_area: f32,
    /// Total image area for percentage calculations
    image_area: f32,
}

impl GuiFaceMetrics {
    fn bbox_percent(&self) -> f32 {
        if self.image_area > 0.0 { self.bbox_area / self.image_area * 100.0 } else { 0.0 }
    }
    fn face_percent(&self) -> f32 {
        if self.image_area > 0.0 { self.metrics.jawline_area / self.image_area * 100.0 } else { 0.0 }
    }
    fn head_percent(&self) -> f32 {
        if self.image_area > 0.0 { self.metrics.head_area / self.image_area * 100.0 } else { 0.0 }
    }
}

struct FaceApp {
    // Image state
    original_image: Option<DynamicImage>,
    rotated_image: Option<DynamicImage>,
    display_texture: Option<egui::TextureHandle>,
    image_path: Option<PathBuf>,

    // Models
    face_detector: Option<Box<dyn Detector>>,
    landmark_model: Option<ShapePredictor>,

    // Detection results
    faces: Vec<FaceInfo>,
    face_metrics: Vec<GuiFaceMetrics>,
    status: String,

    // Settings
    min_face_size: u32,
    rotation_degrees: f32,
    selected_landmark_idx: usize,  // Index into AVAILABLE_MODELS (1, 2, or 3)

    // Loading state
    frame_count: u32,
    auto_load_stage: u8, // 0=not started, 1=loading fd, 2=loading lm, 3=done

    // Download state
    download_status: Arc<Mutex<HashMap<String, DownloadStatus>>>,
}

impl FaceApp {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        // Initialize download status based on which files exist
        let mut download_status = HashMap::new();
        for model in AVAILABLE_MODELS {
            let status = if std::path::Path::new(model.filename).exists() {
                DownloadStatus::Downloaded
            } else {
                DownloadStatus::NotDownloaded
            };
            download_status.insert(model.filename.to_string(), status);
        }

        Self {
            original_image: None,
            rotated_image: None,
            display_texture: None,
            image_path: None,
            face_detector: None,
            landmark_model: None,
            faces: Vec::new(),
            face_metrics: Vec::new(),
            status: "Starting up...".to_string(),
            min_face_size: 20,
            rotation_degrees: 0.0,
            selected_landmark_idx: 1, // Default to 81-point model
            frame_count: 0,
            auto_load_stage: 0,
            download_status: Arc::new(Mutex::new(download_status)),
        }
    }

    fn face_detector_model(&self) -> &'static ModelInfo {
        &AVAILABLE_MODELS[0] // Face detector is always index 0
    }

    fn landmark_model_info(&self) -> &'static ModelInfo {
        &AVAILABLE_MODELS[self.selected_landmark_idx]
    }

    fn is_auto_loading(&self) -> bool {
        self.auto_load_stage > 0 && self.auto_load_stage < 3
    }

    fn step_auto_load(&mut self) {
        match self.auto_load_stage {
            0 => {
                // Check if we should auto-load
                let fd_exists = std::path::Path::new(self.face_detector_model().filename).exists();
                let lm_exists = std::path::Path::new(self.landmark_model_info().filename).exists();

                if fd_exists || lm_exists {
                    self.auto_load_stage = 1;
                    self.status = "Loading models... (1/2) Face detector".to_string();
                } else {
                    self.auto_load_stage = 3; // Skip to done
                    self.status = "Models not found - download them first".to_string();
                }
            }
            1 => {
                // Load face detector
                let fd_exists = std::path::Path::new(self.face_detector_model().filename).exists();
                if fd_exists {
                    self.load_face_detector();
                }
                self.auto_load_stage = 2;
                self.status = "Loading models... (2/2) Landmark model".to_string();
            }
            2 => {
                // Load landmark model
                let lm_exists = std::path::Path::new(self.landmark_model_info().filename).exists();
                if lm_exists {
                    self.load_landmark_model();
                }
                self.auto_load_stage = 3;

                // Final status
                let fd_ok = self.face_detector.is_some();
                let lm_ok = self.landmark_model.is_some();
                self.status = match (fd_ok, lm_ok) {
                    (true, true) => "Ready - load an image to begin".to_string(),
                    (true, false) => "Face detector loaded, landmark model missing".to_string(),
                    (false, true) => "Landmark model loaded, face detector missing".to_string(),
                    (false, false) => "No models loaded - download them first".to_string(),
                };
            }
            _ => {}
        }
    }

    fn apply_rotation(&mut self) {
        let Some(ref img) = self.original_image else {
            return;
        };

        if self.rotation_degrees.abs() < 0.01 {
            self.rotated_image = Some(img.clone());
        } else {
            // Rotate image around center
            let rotated = rotate_image(img, self.rotation_degrees);
            self.rotated_image = Some(rotated);
        }
        self.display_texture = None;
        self.faces.clear();
        self.face_metrics.clear();
    }

    fn load_face_detector(&mut self) {
        let filename = self.face_detector_model().filename;
        match rustface::create_detector(filename) {
            Ok(mut detector) => {
                detector.set_min_face_size(self.min_face_size);
                detector.set_score_thresh(2.0);
                detector.set_pyramid_scale_factor(0.8);
                detector.set_slide_window_step(4, 4);
                self.face_detector = Some(detector);
                self.status = "Face detector loaded".to_string();
            }
            Err(e) => {
                self.status = format!("Failed to load face detector: {}", e);
            }
        }
    }

    fn load_landmark_model(&mut self) {
        let filename = self.landmark_model_info().filename;
        match load_dlib_model(filename) {
            Ok(model) => {
                self.landmark_model = Some(model);
                self.status = format!(
                    "Landmark model loaded ({} points)",
                    self.landmark_model.as_ref().unwrap().num_landmarks()
                );
            }
            Err(e) => {
                self.status = format!("Failed to load landmark model: {}", e);
            }
        }
    }

    fn start_download(&mut self, filename: &str, url: &str, ctx: &egui::Context) {
        // Set status to downloading
        if let Ok(mut status) = self.download_status.lock() {
            status.insert(filename.to_string(), DownloadStatus::Downloading(0.0));
        }

        let filename = filename.to_string();
        let url = url.to_string();
        let status_map = Arc::clone(&self.download_status);
        let ctx = ctx.clone();

        thread::spawn(move || {
            let result = download_model(&filename, &url, Arc::clone(&status_map), ctx.clone());
            if let Ok(mut status) = status_map.lock() {
                match result {
                    Ok(()) => {
                        status.insert(filename, DownloadStatus::Downloaded);
                    }
                    Err(e) => {
                        status.insert(filename, DownloadStatus::Failed(e));
                    }
                }
            }
            ctx.request_repaint();
        });
    }

    fn load_image(&mut self, path: PathBuf) {
        match image::open(&path) {
            Ok(img) => {
                self.original_image = Some(img.clone());
                self.rotated_image = Some(img);
                self.image_path = Some(path.clone());
                self.faces.clear();
                self.face_metrics.clear();
                self.display_texture = None;
                self.rotation_degrees = 0.0;
                self.status = format!("Loaded: {}", path.display());
            }
            Err(e) => {
                self.status = format!("Failed to load image: {}", e);
            }
        }
    }

    fn detect_faces(&mut self) {
        let Some(ref img) = self.rotated_image else {
            self.status = "No image loaded".to_string();
            return;
        };

        let Some(ref mut detector) = self.face_detector else {
            self.status = "Face detector not loaded".to_string();
            return;
        };

        // Convert to grayscale for detection
        let gray = img.to_luma8();
        let (width, height) = gray.dimensions();

        let image_data = ImageData::new(gray.as_raw(), width, height);
        self.faces = detector.detect(&image_data);
        self.face_metrics.clear();
        self.status = format!("Detected {} face(s) (rotation: {:.1}°)", self.faces.len(), self.rotation_degrees);

        // Clear texture to force redraw
        self.display_texture = None;
    }

    fn render_results(&mut self, ctx: &egui::Context) {
        let Some(ref img) = self.rotated_image else {
            return;
        };

        // Convert image to RGBA
        let mut rgba: RgbaImage = img.to_rgba8();
        let (width, height) = rgba.dimensions();

        // Convert to grayscale for landmark detection
        let gray_img = img.to_luma8();
        let gray = GrayImage::new(gray_img.to_vec(), width, height);

        let image_area = (width * height) as f32;

        // Draw face boxes and landmarks
        for face in &self.faces {
            let bbox = face.bbox();

            // Draw bounding box (green)
            draw_rect(
                &mut rgba,
                bbox.x() as i32,
                bbox.y() as i32,
                bbox.width() as i32,
                bbox.height() as i32,
                Rgba([0, 255, 0, 255]),
            );

            let bbox_area = (bbox.width() * bbox.height()) as f32;
            let mut gui_metrics = GuiFaceMetrics {
                bbox_area,
                image_area,
                ..Default::default()
            };

            // Run landmark detection if model is loaded
            if let Some(ref model) = self.landmark_model {
                let face_bbox = BoundingBox::new(
                    bbox.x() as f32,
                    bbox.y() as f32,
                    bbox.width() as f32,
                    bbox.height() as f32,
                );

                let landmarks = model.predict(&gray, &face_bbox);

                // Calculate metrics using library
                if let Some(metrics) = FaceMetrics::from_shape(&landmarks) {
                    // Draw forehead outline (yellow)
                    if landmarks.points.len() == 81 {
                        for i in 68..80 {
                            draw_line(
                                &mut rgba,
                                landmarks.points[i].x as i32, landmarks.points[i].y as i32,
                                landmarks.points[i + 1].x as i32, landmarks.points[i + 1].y as i32,
                                Rgba([255, 255, 0, 255]),
                            );
                        }
                        // Connect forehead to jawline
                        draw_line(
                            &mut rgba,
                            landmarks.points[68].x as i32, landmarks.points[68].y as i32,
                            landmarks.points[16].x as i32, landmarks.points[16].y as i32,
                            Rgba([255, 255, 0, 255]),
                        );
                        draw_line(
                            &mut rgba,
                            landmarks.points[80].x as i32, landmarks.points[80].y as i32,
                            landmarks.points[0].x as i32, landmarks.points[0].y as i32,
                            Rgba([255, 255, 0, 255]),
                        );
                    } else {
                        // 68-point model: draw estimated head top
                        let eyebrow_top = landmarks.points[17..=26]
                            .iter()
                            .map(|p| p.y)
                            .fold(f32::MAX, f32::min);
                        let chin_y = landmarks.points[8].y;
                        let face_height = chin_y - eyebrow_top;
                        let head_top_y = eyebrow_top - face_height * 0.6;

                        draw_line(
                            &mut rgba,
                            landmarks.points[0].x as i32, head_top_y as i32,
                            landmarks.points[16].x as i32, head_top_y as i32,
                            Rgba([255, 255, 0, 255]),
                        );
                    }

                    gui_metrics.metrics = metrics;
                }

                // Draw landmarks (red dots)
                for point in &landmarks.points {
                    draw_circle(&mut rgba, point.x as i32, point.y as i32, 2, Rgba([255, 0, 0, 255]));
                }

                // Draw connections based on model type
                if landmarks.points.len() >= 68 {
                    draw_face_connections(&mut rgba, &landmarks.points);
                } else if landmarks.points.len() == 5 {
                    draw_5point_connections(&mut rgba, &landmarks.points);
                }
            }

            self.face_metrics.push(gui_metrics);
        }

        // Convert to egui texture
        let size = [width as usize, height as usize];
        let pixels: Vec<egui::Color32> = rgba
            .pixels()
            .map(|p| egui::Color32::from_rgba_unmultiplied(p[0], p[1], p[2], p[3]))
            .collect();

        let color_image = egui::ColorImage { size, pixels };
        self.display_texture = Some(ctx.load_texture("result", color_image, Default::default()));
    }
}

impl eframe::App for FaceApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Auto-load models after UI has rendered, one step per frame
        self.frame_count = self.frame_count.saturating_add(1);
        if self.frame_count >= 2 && self.auto_load_stage < 3 {
            self.step_auto_load();
            ctx.request_repaint(); // Continue to next stage
        }

        egui::TopBottomPanel::top("menu").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("Open Image...").clicked() {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("Images", &["png", "jpg", "jpeg", "bmp", "gif"])
                            .pick_file()
                        {
                            self.load_image(path);
                        }
                        ui.close_menu();
                    }
                    if ui.button("Quit").clicked() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });
            });
        });

        egui::SidePanel::left("controls").min_width(250.0).show(ctx, |ui| {
            ui.heading("Models");
            ui.separator();

            // Show progress bar during auto-loading
            if self.is_auto_loading() {
                let progress = match self.auto_load_stage {
                    1 => 0.25,
                    2 => 0.75,
                    _ => 0.0,
                };
                ui.add(egui::ProgressBar::new(progress).animate(true));
                ui.add_space(8.0);
            }

            // Get download status snapshot
            let status_snapshot: HashMap<String, DownloadStatus> =
                self.download_status.lock().unwrap().clone();

            // Face Detector
            ui.label("Face Detector:");
            let fd_model = self.face_detector_model();
            let fd_status = status_snapshot
                .get(fd_model.filename)
                .cloned()
                .unwrap_or(DownloadStatus::NotDownloaded);

            ui.horizontal(|ui| {
                ui.label(fd_model.name);
                if self.face_detector.is_some() {
                    ui.label("✓");
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    match &fd_status {
                        DownloadStatus::NotDownloaded | DownloadStatus::Failed(_) => {
                            if ui.small_button("Download").clicked() {
                                self.start_download(fd_model.filename, fd_model.url, ctx);
                            }
                        }
                        DownloadStatus::Downloading(_) => {}
                        DownloadStatus::Downloaded => {
                            let btn = ui.add_enabled(
                                !self.is_auto_loading() && self.face_detector.is_none(),
                                egui::Button::new("Load").small(),
                            );
                            if btn.clicked() {
                                self.load_face_detector();
                            }
                        }
                    }
                });
            });
            // Show progress bar if downloading
            if let DownloadStatus::Downloading(progress) = &fd_status {
                ui.add(egui::ProgressBar::new(*progress).show_percentage());
                ctx.request_repaint();
            }
            if let DownloadStatus::Failed(err) = &fd_status {
                ui.label(egui::RichText::new(err).small().color(egui::Color32::RED));
            }
            ui.add_space(8.0);

            // Landmark Model (dropdown)
            ui.label("Landmark Model:");
            let landmark_models: Vec<(usize, &ModelInfo)> = AVAILABLE_MODELS
                .iter()
                .enumerate()
                .filter(|(_, m)| m.filename.contains("shape_predictor"))
                .collect();

            let current_model = self.landmark_model_info();
            ui.horizontal(|ui| {
                egui::ComboBox::from_id_salt("landmark_model")
                    .selected_text(current_model.name)
                    .show_ui(ui, |ui| {
                        for (idx, model) in &landmark_models {
                            let is_downloaded = status_snapshot
                                .get(model.filename)
                                .map(|s| *s == DownloadStatus::Downloaded)
                                .unwrap_or(false);
                            let label = if is_downloaded {
                                format!("{} ✓", model.name)
                            } else {
                                model.name.to_string()
                            };
                            if ui.selectable_value(&mut self.selected_landmark_idx, *idx, label).clicked() {
                                // Clear loaded model when selection changes
                                self.landmark_model = None;
                            }
                        }
                    });
                if self.landmark_model.is_some() {
                    ui.label("✓");
                }
            });

            // Download/Load button for selected landmark model
            let lm_status = status_snapshot
                .get(current_model.filename)
                .cloned()
                .unwrap_or(DownloadStatus::NotDownloaded);

            ui.horizontal(|ui| {
                match &lm_status {
                    DownloadStatus::NotDownloaded | DownloadStatus::Failed(_) => {
                        if ui.button("Download").clicked() {
                            self.start_download(current_model.filename, current_model.url, ctx);
                        }
                    }
                    DownloadStatus::Downloading(_) => {}
                    DownloadStatus::Downloaded => {
                        let btn = ui.add_enabled(
                            !self.is_auto_loading() && self.landmark_model.is_none(),
                            egui::Button::new("Load"),
                        );
                        if btn.clicked() {
                            self.load_landmark_model();
                        }
                    }
                }
            });
            // Show progress bar if downloading
            if let DownloadStatus::Downloading(progress) = &lm_status {
                ui.add(egui::ProgressBar::new(*progress).show_percentage());
                ctx.request_repaint();
            }
            if let DownloadStatus::Failed(err) = &lm_status {
                ui.label(egui::RichText::new(err).small().color(egui::Color32::RED));
            }
            ui.add_space(16.0);

            ui.heading("Detection");
            ui.separator();

            ui.add(egui::Slider::new(&mut self.min_face_size, 10..=100).text("Min Face Size"));

            if ui.button("Detect Faces").clicked() {
                self.detect_faces();
            }
            ui.add_space(16.0);

            ui.heading("Image Transform");
            ui.separator();

            let old_rotation = self.rotation_degrees;

            ui.horizontal(|ui| {
                if ui.button("<<").clicked() {
                    self.rotation_degrees -= 10.0;
                }
                if ui.button("<").clicked() {
                    self.rotation_degrees -= 1.0;
                }
                ui.label(format!("{:+.0}°", self.rotation_degrees));
                if ui.button(">").clicked() {
                    self.rotation_degrees += 1.0;
                }
                if ui.button(">>").clicked() {
                    self.rotation_degrees += 10.0;
                }
            });

            ui.horizontal(|ui| {
                if ui.button("-90°").clicked() {
                    self.rotation_degrees = -90.0;
                }
                if ui.button("-45°").clicked() {
                    self.rotation_degrees = -45.0;
                }
                if ui.button("0°").clicked() {
                    self.rotation_degrees = 0.0;
                }
                if ui.button("+45°").clicked() {
                    self.rotation_degrees = 45.0;
                }
                if ui.button("+90°").clicked() {
                    self.rotation_degrees = 90.0;
                }
            });

            // Clamp to valid range
            self.rotation_degrees = self.rotation_degrees.clamp(-180.0, 180.0);

            if (self.rotation_degrees - old_rotation).abs() > 0.01 {
                self.apply_rotation();
            }
            ui.add_space(16.0);

            ui.heading("Status");
            ui.separator();
            ui.label(&self.status);

            if !self.faces.is_empty() {
                ui.add_space(8.0);
                ui.label(format!("Faces found: {}", self.faces.len()));

                for (i, (face, gui_metrics)) in self.faces.iter().zip(self.face_metrics.iter()).enumerate() {
                    let bbox = face.bbox();
                    let m = &gui_metrics.metrics;

                    ui.add_space(4.0);
                    ui.collapsing(format!("Face {}", i + 1), |ui| {
                        ui.label(format!("Detection box: {}x{}", bbox.width(), bbox.height()));
                        ui.label(format!("Box: {:.1}% of image", gui_metrics.bbox_percent()));

                        if m.jawline_area > 0.0 {
                            ui.add_space(4.0);
                            ui.label("Image coverage:");
                            ui.label(format!("  Face: {:.1}%", gui_metrics.face_percent()));
                            let head_label = if m.has_forehead_landmarks { "Head" } else { "Head (est.)" };
                            ui.label(format!("  {}: {:.1}%", head_label, gui_metrics.head_percent()));

                            ui.add_space(4.0);
                            ui.label("Features (% of face):");
                            ui.label(format!("  Eyes: {:.1}% (L:{:.1}% R:{:.1}%)",
                                m.eyes_ratio(), m.left_eye_ratio(), m.right_eye_ratio()));
                            ui.label(format!("  Eyebrows: {:.1}%", m.eyebrows_ratio()));
                            ui.label(format!("  Nose: {:.1}%", m.nose_ratio()));
                            ui.label(format!("  Mouth: {:.1}% (lips:{:.1}%)",
                                m.mouth_ratio(), m.lips_ratio()));
                            ui.label(format!("  Forehead: {:.1}%", m.forehead_ratio()));

                            ui.add_space(4.0);
                            ui.label("Ratios:");
                            ui.label(format!("  Lower face: {:.1}%", m.lower_face_ratio()));
                            ui.label(format!("  Eye symmetry: {:.1}%", m.eye_symmetry()));
                            ui.label(format!("  Eye/Mouth: {:.2}x", m.eye_to_mouth_ratio() / 100.0));
                        }
                    });
                }
            }
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            // Render if we have an image and faces detected
            if self.rotated_image.is_some() && self.display_texture.is_none() && !self.faces.is_empty()
            {
                self.render_results(ctx);
            }

            // Show the image
            if let Some(ref texture) = self.display_texture {
                let available_size = ui.available_size();
                let texture_size = texture.size_vec2();

                // Scale to fit
                let scale = (available_size.x / texture_size.x)
                    .min(available_size.y / texture_size.y)
                    .min(1.0);
                let display_size = texture_size * scale;

                ui.centered_and_justified(|ui| {
                    ui.image((texture.id(), display_size));
                });
            } else if let Some(ref img) = self.rotated_image {
                // Show rotated image without detections
                let rgba = img.to_rgba8();
                let (width, height) = rgba.dimensions();
                let size = [width as usize, height as usize];
                let pixels: Vec<egui::Color32> = rgba
                    .pixels()
                    .map(|p| egui::Color32::from_rgba_unmultiplied(p[0], p[1], p[2], p[3]))
                    .collect();

                let color_image = egui::ColorImage { size, pixels };
                let texture = ctx.load_texture("rotated", color_image, Default::default());

                let available_size = ui.available_size();
                let texture_size = texture.size_vec2();
                let scale = (available_size.x / texture_size.x)
                    .min(available_size.y / texture_size.y)
                    .min(1.0);
                let display_size = texture_size * scale;

                ui.centered_and_justified(|ui| {
                    ui.image((texture.id(), display_size));
                });
            } else {
                ui.centered_and_justified(|ui| {
                    ui.heading("Drag and drop an image or use File > Open");
                });
            }
        });

        // Handle drag and drop
        ctx.input(|i| {
            for file in &i.raw.dropped_files {
                if let Some(path) = &file.path {
                    self.load_image(path.clone());
                }
            }
        });
    }
}

// Drawing helpers

fn draw_rect(img: &mut RgbaImage, x: i32, y: i32, w: i32, h: i32, color: Rgba<u8>) {
    let (img_w, img_h) = img.dimensions();

    // Top and bottom edges
    for dx in 0..w {
        let px = x + dx;
        if px >= 0 && px < img_w as i32 {
            if y >= 0 && y < img_h as i32 {
                img.put_pixel(px as u32, y as u32, color);
            }
            let by = y + h - 1;
            if by >= 0 && by < img_h as i32 {
                img.put_pixel(px as u32, by as u32, color);
            }
        }
    }

    // Left and right edges
    for dy in 0..h {
        let py = y + dy;
        if py >= 0 && py < img_h as i32 {
            if x >= 0 && x < img_w as i32 {
                img.put_pixel(x as u32, py as u32, color);
            }
            let rx = x + w - 1;
            if rx >= 0 && rx < img_w as i32 {
                img.put_pixel(rx as u32, py as u32, color);
            }
        }
    }
}

fn draw_circle(img: &mut RgbaImage, cx: i32, cy: i32, radius: i32, color: Rgba<u8>) {
    let (img_w, img_h) = img.dimensions();

    for dy in -radius..=radius {
        for dx in -radius..=radius {
            if dx * dx + dy * dy <= radius * radius {
                let px = cx + dx;
                let py = cy + dy;
                if px >= 0 && px < img_w as i32 && py >= 0 && py < img_h as i32 {
                    img.put_pixel(px as u32, py as u32, color);
                }
            }
        }
    }
}

fn draw_line(img: &mut RgbaImage, x0: i32, y0: i32, x1: i32, y1: i32, color: Rgba<u8>) {
    let (img_w, img_h) = img.dimensions();

    let dx = (x1 - x0).abs();
    let dy = (y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx - dy;

    let mut x = x0;
    let mut y = y0;

    loop {
        if x >= 0 && x < img_w as i32 && y >= 0 && y < img_h as i32 {
            img.put_pixel(x as u32, y as u32, color);
        }

        if x == x1 && y == y1 {
            break;
        }

        let e2 = 2 * err;
        if e2 > -dy {
            err -= dy;
            x += sx;
        }
        if e2 < dx {
            err += dx;
            y += sy;
        }
    }
}

fn draw_5point_connections(img: &mut RgbaImage, points: &[percent_face::Point]) {
    let cyan = Rgba([0, 255, 255, 255]);

    // Connect eyes (0-1 are outer eye corners, 2 is nose, 3-4 are mouth corners based on 5-point)
    // Typical 5-point: left eye outer, right eye outer, nose tip, left mouth, right mouth
    if points.len() >= 5 {
        // Left eye to nose
        draw_line(
            img,
            points[0].x as i32, points[0].y as i32,
            points[2].x as i32, points[2].y as i32,
            cyan,
        );
        // Right eye to nose
        draw_line(
            img,
            points[1].x as i32, points[1].y as i32,
            points[2].x as i32, points[2].y as i32,
            cyan,
        );
        // Nose to mouth corners
        draw_line(
            img,
            points[2].x as i32, points[2].y as i32,
            points[3].x as i32, points[3].y as i32,
            cyan,
        );
        draw_line(
            img,
            points[2].x as i32, points[2].y as i32,
            points[4].x as i32, points[4].y as i32,
            cyan,
        );
        // Mouth line
        draw_line(
            img,
            points[3].x as i32, points[3].y as i32,
            points[4].x as i32, points[4].y as i32,
            cyan,
        );
    }
}

fn draw_face_connections(img: &mut RgbaImage, points: &[percent_face::Point]) {
    let cyan = Rgba([0, 255, 255, 255]);

    // 68-point landmark connections
    let connections: &[&[usize]] = &[
        // Jaw
        &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        // Right eyebrow
        &[17, 18, 19, 20, 21],
        // Left eyebrow
        &[22, 23, 24, 25, 26],
        // Nose bridge
        &[27, 28, 29, 30],
        // Nose bottom
        &[31, 32, 33, 34, 35],
        // Right eye
        &[36, 37, 38, 39, 40, 41, 36],
        // Left eye
        &[42, 43, 44, 45, 46, 47, 42],
        // Outer lip
        &[48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 48],
        // Inner lip
        &[60, 61, 62, 63, 64, 65, 66, 67, 60],
    ];

    for group in connections {
        for i in 0..group.len() - 1 {
            let p1 = &points[group[i]];
            let p2 = &points[group[i + 1]];
            draw_line(img, p1.x as i32, p1.y as i32, p2.x as i32, p2.y as i32, cyan);
        }
    }
}

/// Rotate an image by the given angle in degrees around its center.
fn rotate_image(img: &DynamicImage, degrees: f32) -> DynamicImage {
    let rgba = img.to_rgba8();
    let (width, height) = rgba.dimensions();

    // Calculate the size of the new image to fit the rotated content
    let radians = degrees * PI / 180.0;
    let cos_a = radians.cos().abs();
    let sin_a = radians.sin().abs();

    let new_width = (width as f32 * cos_a + height as f32 * sin_a).ceil() as u32;
    let new_height = (width as f32 * sin_a + height as f32 * cos_a).ceil() as u32;

    // Center of original and new image
    let cx = width as f32 / 2.0;
    let cy = height as f32 / 2.0;
    let new_cx = new_width as f32 / 2.0;
    let new_cy = new_height as f32 / 2.0;

    // Rotation matrix (inverse, to map destination to source)
    let cos_r = (-radians).cos();
    let sin_r = (-radians).sin();

    let mut output = RgbaImage::new(new_width, new_height);

    for dy in 0..new_height {
        for dx in 0..new_width {
            // Translate to center of new image
            let x = dx as f32 - new_cx;
            let y = dy as f32 - new_cy;

            // Rotate (inverse)
            let src_x = x * cos_r - y * sin_r + cx;
            let src_y = x * sin_r + y * cos_r + cy;

            // Bilinear interpolation
            if src_x >= 0.0 && src_x < width as f32 - 1.0 && src_y >= 0.0 && src_y < height as f32 - 1.0
            {
                let x0 = src_x.floor() as u32;
                let y0 = src_y.floor() as u32;
                let x1 = x0 + 1;
                let y1 = y0 + 1;

                let fx = src_x - x0 as f32;
                let fy = src_y - y0 as f32;

                let p00 = rgba.get_pixel(x0, y0);
                let p10 = rgba.get_pixel(x1, y0);
                let p01 = rgba.get_pixel(x0, y1);
                let p11 = rgba.get_pixel(x1, y1);

                let mut pixel = [0u8; 4];
                for i in 0..4 {
                    let top = p00[i] as f32 * (1.0 - fx) + p10[i] as f32 * fx;
                    let bottom = p01[i] as f32 * (1.0 - fx) + p11[i] as f32 * fx;
                    pixel[i] = (top * (1.0 - fy) + bottom * fy) as u8;
                }
                output.put_pixel(dx, dy, Rgba(pixel));
            }
        }
    }

    DynamicImage::ImageRgba8(output)
}
