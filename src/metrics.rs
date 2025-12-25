//! Facial feature metrics and area calculations.
//!
//! This module provides tools for measuring facial features from landmark points,
//! including area calculations for individual features and ratios between them.

use crate::types::{Point, Shape};

/// Facial feature areas calculated from landmarks.
///
/// All areas are in square pixels. Use the ratio methods to get
/// proportions relative to the face or head area.
#[derive(Debug, Clone, Default)]
pub struct FaceMetrics {
    /// Area enclosed by the jawline (points 0-16)
    pub jawline_area: f32,

    /// Full head area including forehead
    /// - 81-point model: actual forehead landmarks (68-80)
    /// - 68-point model: estimated from facial proportions
    pub head_area: f32,

    /// Whether head_area uses actual forehead landmarks (81-point) or estimation (68-point)
    pub has_forehead_landmarks: bool,

    /// Left eye area (points 36-41)
    pub left_eye_area: f32,

    /// Right eye area (points 42-47)
    pub right_eye_area: f32,

    /// Left eyebrow area (points 17-21, estimated as line with width)
    pub left_eyebrow_area: f32,

    /// Right eyebrow area (points 22-26, estimated as line with width)
    pub right_eyebrow_area: f32,

    /// Nose area (points 27-35)
    pub nose_area: f32,

    /// Outer mouth/lips area (points 48-59)
    pub outer_mouth_area: f32,

    /// Inner mouth area (points 60-67)
    pub inner_mouth_area: f32,

    /// Forehead area (points 68-80 for 81-point model, estimated otherwise)
    pub forehead_area: f32,
}

impl FaceMetrics {
    /// Calculate metrics from a shape with 68 or 81 landmarks.
    ///
    /// Returns `None` if the shape doesn't have at least 68 points.
    pub fn from_shape(shape: &Shape) -> Option<Self> {
        if shape.num_landmarks() < 68 {
            return None;
        }

        let points = &shape.points;
        let mut metrics = Self::default();

        // Jawline (points 0-16)
        let jawline: Vec<_> = points[0..=16].to_vec();
        metrics.jawline_area = polygon_area(&jawline);

        // Eyes (closed polygons)
        let left_eye: Vec<_> = points[36..=41].to_vec();
        let right_eye: Vec<_> = points[42..=47].to_vec();
        metrics.left_eye_area = polygon_area(&left_eye);
        metrics.right_eye_area = polygon_area(&right_eye);

        // Eyebrows (approximate as polygon with small height)
        // Points 17-21 (right eyebrow) and 22-26 (left eyebrow)
        metrics.right_eyebrow_area = eyebrow_area(&points[17..=21]);
        metrics.left_eyebrow_area = eyebrow_area(&points[22..=26]);

        // Nose (points 27-35)
        // Bridge (27-30) + bottom (31-35) form the nose region
        metrics.nose_area = nose_polygon_area(points);

        // Mouth
        let outer_mouth: Vec<_> = points[48..=59].to_vec();
        let inner_mouth: Vec<_> = points[60..=67].to_vec();
        metrics.outer_mouth_area = polygon_area(&outer_mouth);
        metrics.inner_mouth_area = polygon_area(&inner_mouth);

        // Head area with forehead
        if shape.num_landmarks() >= 81 {
            // 81-point model: use actual forehead landmarks
            let mut head_polygon = jawline.clone();
            for i in 68..=80 {
                head_polygon.push(points[i]);
            }
            metrics.head_area = polygon_area(&head_polygon);
            metrics.has_forehead_landmarks = true;

            // Forehead area (polygon from eyebrows to hairline)
            metrics.forehead_area = forehead_area_81(points);
        } else {
            // 68-point model: estimate forehead
            let eyebrow_top = points[17..=26]
                .iter()
                .map(|p| p.y)
                .fold(f32::MAX, f32::min);

            let chin_y = points[8].y;
            let face_height = chin_y - eyebrow_top;
            let forehead_height = face_height * 0.6;
            let head_top_y = eyebrow_top - forehead_height;

            let left_temple = Point::new(points[0].x, head_top_y);
            let right_temple = Point::new(points[16].x, head_top_y);

            let mut head_polygon = jawline.clone();
            head_polygon.push(right_temple);
            head_polygon.push(left_temple);
            metrics.head_area = polygon_area(&head_polygon);
            metrics.has_forehead_landmarks = false;

            // Estimate forehead area
            metrics.forehead_area = estimate_forehead_area(points, head_top_y);
        }

        Some(metrics)
    }

    /// Total eye area (both eyes combined)
    pub fn total_eye_area(&self) -> f32 {
        self.left_eye_area + self.right_eye_area
    }

    /// Total eyebrow area (both eyebrows combined)
    pub fn total_eyebrow_area(&self) -> f32 {
        self.left_eyebrow_area + self.right_eyebrow_area
    }

    /// Lip area (outer mouth minus inner mouth)
    pub fn lip_area(&self) -> f32 {
        (self.outer_mouth_area - self.inner_mouth_area).max(0.0)
    }

    // === Ratios relative to jawline (face) area ===

    /// Left eye as percentage of face area
    pub fn left_eye_ratio(&self) -> f32 {
        ratio(self.left_eye_area, self.jawline_area)
    }

    /// Right eye as percentage of face area
    pub fn right_eye_ratio(&self) -> f32 {
        ratio(self.right_eye_area, self.jawline_area)
    }

    /// Total eyes as percentage of face area
    pub fn eyes_ratio(&self) -> f32 {
        ratio(self.total_eye_area(), self.jawline_area)
    }

    /// Left eyebrow as percentage of face area
    pub fn left_eyebrow_ratio(&self) -> f32 {
        ratio(self.left_eyebrow_area, self.jawline_area)
    }

    /// Right eyebrow as percentage of face area
    pub fn right_eyebrow_ratio(&self) -> f32 {
        ratio(self.right_eyebrow_area, self.jawline_area)
    }

    /// Total eyebrows as percentage of face area
    pub fn eyebrows_ratio(&self) -> f32 {
        ratio(self.total_eyebrow_area(), self.jawline_area)
    }

    /// Nose as percentage of face area
    pub fn nose_ratio(&self) -> f32 {
        ratio(self.nose_area, self.jawline_area)
    }

    /// Outer mouth as percentage of face area
    pub fn mouth_ratio(&self) -> f32 {
        ratio(self.outer_mouth_area, self.jawline_area)
    }

    /// Lips (outer - inner mouth) as percentage of face area
    pub fn lips_ratio(&self) -> f32 {
        ratio(self.lip_area(), self.jawline_area)
    }

    /// Inner mouth (mouth opening) as percentage of face area
    pub fn mouth_opening_ratio(&self) -> f32 {
        ratio(self.inner_mouth_area, self.jawline_area)
    }

    /// Forehead as percentage of face area
    pub fn forehead_ratio(&self) -> f32 {
        ratio(self.forehead_area, self.jawline_area)
    }

    // === Ratios relative to head area ===

    /// Jawline (face) as percentage of head area
    pub fn face_to_head_ratio(&self) -> f32 {
        ratio(self.jawline_area, self.head_area)
    }

    /// Forehead as percentage of head area
    pub fn forehead_to_head_ratio(&self) -> f32 {
        ratio(self.forehead_area, self.head_area)
    }

    // === Inter-feature ratios ===

    /// Eye-to-mouth ratio (eyes area / mouth area)
    pub fn eye_to_mouth_ratio(&self) -> f32 {
        ratio(self.total_eye_area(), self.outer_mouth_area)
    }

    /// Nose-to-mouth ratio
    pub fn nose_to_mouth_ratio(&self) -> f32 {
        ratio(self.nose_area, self.outer_mouth_area)
    }

    /// Eye symmetry (left/right eye ratio, 1.0 = perfect symmetry)
    pub fn eye_symmetry(&self) -> f32 {
        if self.right_eye_area > self.left_eye_area {
            ratio(self.left_eye_area, self.right_eye_area)
        } else {
            ratio(self.right_eye_area, self.left_eye_area)
        }
    }
}

/// Calculate the area of a polygon using the shoelace formula.
pub fn polygon_area(points: &[Point]) -> f32 {
    if points.len() < 3 {
        return 0.0;
    }

    let mut area = 0.0;
    let n = points.len();

    for i in 0..n {
        let j = (i + 1) % n;
        area += points[i].x * points[j].y;
        area -= points[j].x * points[i].y;
    }

    (area / 2.0).abs()
}

/// Calculate percentage ratio, handling division by zero.
fn ratio(numerator: f32, denominator: f32) -> f32 {
    if denominator > 0.0 {
        (numerator / denominator) * 100.0
    } else {
        0.0
    }
}

/// Estimate eyebrow area from the 5 eyebrow points.
/// Eyebrows are thin, so we create a polygon by offsetting points up/down.
fn eyebrow_area(points: &[Point]) -> f32 {
    if points.len() < 2 {
        return 0.0;
    }

    // Estimate eyebrow thickness based on the arc length
    let mut length = 0.0;
    for i in 0..points.len() - 1 {
        let dx = points[i + 1].x - points[i].x;
        let dy = points[i + 1].y - points[i].y;
        length += (dx * dx + dy * dy).sqrt();
    }

    // Eyebrow thickness is roughly 1/10 of its length
    let thickness = length * 0.1;

    // Create a polygon by offsetting points
    let mut polygon = Vec::with_capacity(points.len() * 2);

    // Top edge (offset up)
    for p in points {
        polygon.push(Point::new(p.x, p.y - thickness / 2.0));
    }

    // Bottom edge (offset down, reversed)
    for p in points.iter().rev() {
        polygon.push(Point::new(p.x, p.y + thickness / 2.0));
    }

    polygon_area(&polygon)
}

/// Calculate nose area from the nose landmarks.
/// The nose is defined by the bridge (27-30) and bottom (31-35).
fn nose_polygon_area(points: &[Point]) -> f32 {
    // Create a nose polygon:
    // Start at bridge top (27), go down to nose tip (30),
    // then around the nostrils (31-35), back to bridge
    let nose_polygon = vec![
        points[27], // Bridge top
        points[28],
        points[29],
        points[30], // Nose tip
        points[35], // Right nostril outer
        points[34],
        points[33], // Bottom center
        points[32],
        points[31], // Left nostril outer
    ];

    polygon_area(&nose_polygon)
}

/// Calculate forehead area for 81-point model.
fn forehead_area_81(points: &[Point]) -> f32 {
    // Forehead is bounded by:
    // - Top: hairline points (68-80)
    // - Bottom: eyebrows (17-26)

    // Get the eyebrow points as bottom boundary
    let left_brow_center = points[19]; // Center of left eyebrow
    let right_brow_center = points[24]; // Center of right eyebrow

    // Create forehead polygon: hairline + eyebrow tops
    let mut forehead = Vec::new();

    // Add hairline points (68-80)
    for i in 68..=80 {
        forehead.push(points[i]);
    }

    // Connect back along eyebrow tops
    // Right eyebrow outer to left eyebrow outer
    forehead.push(points[26]); // Right eyebrow outer
    forehead.push(right_brow_center);
    forehead.push(points[22]); // Right eyebrow inner
    forehead.push(points[21]); // Left eyebrow inner
    forehead.push(left_brow_center);
    forehead.push(points[17]); // Left eyebrow outer

    polygon_area(&forehead)
}

/// Estimate forehead area for 68-point model.
fn estimate_forehead_area(points: &[Point], head_top_y: f32) -> f32 {
    // Forehead is bounded by:
    // - Top: estimated head top line
    // - Bottom: eyebrows
    // - Sides: temples (extrapolated from jawline endpoints)

    let forehead = vec![
        Point::new(points[0].x, head_top_y),  // Left temple top
        Point::new(points[16].x, head_top_y), // Right temple top
        points[26],                            // Right eyebrow outer
        points[22],                            // Right eyebrow inner
        points[21],                            // Left eyebrow inner
        points[17],                            // Left eyebrow outer
    ];

    polygon_area(&forehead)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_polygon_area_triangle() {
        let triangle = vec![
            Point::new(0.0, 0.0),
            Point::new(4.0, 0.0),
            Point::new(2.0, 3.0),
        ];
        // Area = 0.5 * base * height = 0.5 * 4 * 3 = 6
        assert!((polygon_area(&triangle) - 6.0).abs() < 0.01);
    }

    #[test]
    fn test_polygon_area_square() {
        let square = vec![
            Point::new(0.0, 0.0),
            Point::new(10.0, 0.0),
            Point::new(10.0, 10.0),
            Point::new(0.0, 10.0),
        ];
        assert!((polygon_area(&square) - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_ratio() {
        assert!((ratio(25.0, 100.0) - 25.0).abs() < 0.01);
        assert!((ratio(0.0, 100.0) - 0.0).abs() < 0.01);
        assert!((ratio(100.0, 0.0) - 0.0).abs() < 0.01); // Division by zero
    }
}
