//! GUI application for visualizing facial landmark detection.
//!
//! Run with: cargo run --features gui --bin percent-face-gui

use eframe::egui;
use image::{DynamicImage, Rgba, RgbaImage};
use percent_face::{dlib::load_dlib_model, BoundingBox, GrayImage, ShapePredictor};
use rustface::{Detector, FaceInfo, ImageData};
use std::path::PathBuf;

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1024.0, 768.0]),
        ..Default::default()
    };

    eframe::run_native(
        "percent-face - Facial Landmark Detection",
        options,
        Box::new(|cc| Ok(Box::new(FaceApp::new(cc)))),
    )
}

struct FaceApp {
    // Image state
    original_image: Option<DynamicImage>,
    display_texture: Option<egui::TextureHandle>,
    image_path: Option<PathBuf>,

    // Models
    face_detector: Option<Box<dyn Detector>>,
    landmark_model: Option<ShapePredictor>,

    // Detection results
    faces: Vec<FaceInfo>,
    status: String,

    // Settings
    min_face_size: u32,
    landmark_model_path: String,
    face_detector_model_path: String,
}

impl FaceApp {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        Self {
            original_image: None,
            display_texture: None,
            image_path: None,
            face_detector: None,
            landmark_model: None,
            faces: Vec::new(),
            status: "Load an image and models to begin".to_string(),
            min_face_size: 20,
            landmark_model_path: "dlib-models/shape_predictor_68_face_landmarks.dat.bz2"
                .to_string(),
            face_detector_model_path: "seeta_fd_frontal_v1.0.bin".to_string(),
        }
    }

    fn load_face_detector(&mut self) {
        match rustface::create_detector(&self.face_detector_model_path) {
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
        match load_dlib_model(&self.landmark_model_path) {
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

    fn load_image(&mut self, path: PathBuf) {
        match image::open(&path) {
            Ok(img) => {
                self.original_image = Some(img);
                self.image_path = Some(path.clone());
                self.faces.clear();
                self.display_texture = None;
                self.status = format!("Loaded: {}", path.display());
            }
            Err(e) => {
                self.status = format!("Failed to load image: {}", e);
            }
        }
    }

    fn detect_faces(&mut self) {
        let Some(ref img) = self.original_image else {
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
        self.status = format!("Detected {} face(s)", self.faces.len());

        // Clear texture to force redraw
        self.display_texture = None;
    }

    fn render_results(&mut self, ctx: &egui::Context) {
        let Some(ref img) = self.original_image else {
            return;
        };

        // Convert image to RGBA
        let mut rgba: RgbaImage = img.to_rgba8();
        let (width, height) = rgba.dimensions();

        // Convert to grayscale for landmark detection
        let gray_img = img.to_luma8();
        let gray = GrayImage::new(gray_img.to_vec(), width, height);

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

            // Run landmark detection if model is loaded
            if let Some(ref model) = self.landmark_model {
                let face_bbox = BoundingBox::new(
                    bbox.x() as f32,
                    bbox.y() as f32,
                    bbox.width() as f32,
                    bbox.height() as f32,
                );

                let landmarks = model.predict(&gray, &face_bbox);

                // Draw landmarks (red dots)
                for point in &landmarks.points {
                    draw_circle(&mut rgba, point.x as i32, point.y as i32, 2, Rgba([255, 0, 0, 255]));
                }

                // Draw connections for 68-point model
                if landmarks.points.len() == 68 {
                    draw_face_connections(&mut rgba, &landmarks.points);
                } else if landmarks.points.len() == 5 {
                    draw_5point_connections(&mut rgba, &landmarks.points);
                }
            }
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

            ui.label("Face Detector Model:");
            ui.text_edit_singleline(&mut self.face_detector_model_path);
            if ui.button("Load Face Detector").clicked() {
                self.load_face_detector();
            }
            ui.add_space(8.0);

            ui.label("Landmark Model:");
            ui.text_edit_singleline(&mut self.landmark_model_path);
            if ui.button("Load Landmark Model").clicked() {
                self.load_landmark_model();
            }
            ui.add_space(16.0);

            ui.heading("Detection");
            ui.separator();

            ui.add(egui::Slider::new(&mut self.min_face_size, 10..=100).text("Min Face Size"));

            if ui.button("Detect Faces").clicked() {
                self.detect_faces();
            }
            ui.add_space(16.0);

            ui.heading("Status");
            ui.separator();
            ui.label(&self.status);

            if !self.faces.is_empty() {
                ui.add_space(8.0);
                ui.label(format!("Faces found: {}", self.faces.len()));
                for (i, face) in self.faces.iter().enumerate() {
                    let bbox = face.bbox();
                    ui.label(format!(
                        "  Face {}: {}x{} at ({}, {})",
                        i + 1,
                        bbox.width(),
                        bbox.height(),
                        bbox.x(),
                        bbox.y()
                    ));
                }
            }
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            // Render if we have an image and faces detected
            if self.original_image.is_some() && self.display_texture.is_none() && !self.faces.is_empty()
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
            } else if let Some(ref img) = self.original_image {
                // Show original image without detections
                let rgba = img.to_rgba8();
                let (width, height) = rgba.dimensions();
                let size = [width as usize, height as usize];
                let pixels: Vec<egui::Color32> = rgba
                    .pixels()
                    .map(|p| egui::Color32::from_rgba_unmultiplied(p[0], p[1], p[2], p[3]))
                    .collect();

                let color_image = egui::ColorImage { size, pixels };
                let texture = ctx.load_texture("original", color_image, Default::default());

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
