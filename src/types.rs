use serde::{Deserialize, Serialize};

/// A 2D point with floating-point coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}

impl Point {
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    pub const fn zero() -> Self {
        Self { x: 0.0, y: 0.0 }
    }

    pub fn distance(&self, other: &Point) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }
}

impl std::ops::Add for Point {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

impl std::ops::AddAssign for Point {
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
    }
}

impl std::ops::Sub for Point {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

impl std::ops::Mul<f32> for Point {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
        }
    }
}

/// A bounding box defined by top-left corner, width, and height.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BoundingBox {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl BoundingBox {
    pub const fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    pub fn center(&self) -> Point {
        Point::new(self.x + self.width / 2.0, self.y + self.height / 2.0)
    }

    /// Convert a point from normalized coordinates [0,1] to image coordinates
    /// within this bounding box.
    pub fn denormalize_point(&self, p: Point) -> Point {
        Point::new(self.x + p.x * self.width, self.y + p.y * self.height)
    }

    /// Convert a point from image coordinates to normalized [0,1] coordinates
    /// relative to this bounding box.
    pub fn normalize_point(&self, p: Point) -> Point {
        Point::new(
            (p.x - self.x) / self.width,
            (p.y - self.y) / self.height,
        )
    }
}

/// A facial shape represented as a collection of landmark points.
/// Standard dlib model uses 68 landmarks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Shape {
    pub points: Vec<Point>,
}

impl Shape {
    pub fn new(points: Vec<Point>) -> Self {
        Self { points }
    }

    pub fn with_capacity(n: usize) -> Self {
        Self {
            points: Vec::with_capacity(n),
        }
    }

    pub fn num_landmarks(&self) -> usize {
        self.points.len()
    }

    /// Create a zeroed shape with n landmarks.
    pub fn zeros(n: usize) -> Self {
        Self {
            points: vec![Point::zero(); n],
        }
    }

    /// Add another shape's deltas to this shape.
    pub fn add_delta(&mut self, delta: &Shape) {
        debug_assert_eq!(self.points.len(), delta.points.len());
        for (p, d) in self.points.iter_mut().zip(delta.points.iter()) {
            *p += *d;
        }
    }

    /// Flatten shape to a vector of [x0, y0, x1, y1, ...] coordinates.
    pub fn to_flat_vec(&self) -> Vec<f32> {
        let mut v = Vec::with_capacity(self.points.len() * 2);
        for p in &self.points {
            v.push(p.x);
            v.push(p.y);
        }
        v
    }

    /// Create shape from a flat vector of [x0, y0, x1, y1, ...] coordinates.
    pub fn from_flat_vec(v: &[f32]) -> Self {
        debug_assert!(v.len() % 2 == 0);
        let points: Vec<Point> = v
            .chunks_exact(2)
            .map(|chunk| Point::new(chunk[0], chunk[1]))
            .collect();
        Self { points }
    }
}

impl std::ops::Index<usize> for Shape {
    type Output = Point;

    fn index(&self, idx: usize) -> &Self::Output {
        &self.points[idx]
    }
}

impl std::ops::IndexMut<usize> for Shape {
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        &mut self.points[idx]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point_arithmetic() {
        let a = Point::new(1.0, 2.0);
        let b = Point::new(3.0, 4.0);

        let sum = a + b;
        assert_eq!(sum.x, 4.0);
        assert_eq!(sum.y, 6.0);

        let diff = b - a;
        assert_eq!(diff.x, 2.0);
        assert_eq!(diff.y, 2.0);

        let scaled = a * 2.0;
        assert_eq!(scaled.x, 2.0);
        assert_eq!(scaled.y, 4.0);
    }

    #[test]
    fn bounding_box_normalization() {
        let bbox = BoundingBox::new(100.0, 100.0, 200.0, 200.0);

        // Center of bbox in normalized coords is (0.5, 0.5)
        let center_norm = Point::new(0.5, 0.5);
        let center_img = bbox.denormalize_point(center_norm);
        assert_eq!(center_img.x, 200.0);
        assert_eq!(center_img.y, 200.0);

        // And back
        let back = bbox.normalize_point(center_img);
        assert!((back.x - 0.5).abs() < 1e-6);
        assert!((back.y - 0.5).abs() < 1e-6);
    }

    #[test]
    fn shape_delta() {
        let mut shape = Shape::new(vec![Point::new(0.0, 0.0), Point::new(1.0, 1.0)]);
        let delta = Shape::new(vec![Point::new(0.1, 0.2), Point::new(0.3, 0.4)]);
        shape.add_delta(&delta);

        assert!((shape[0].x - 0.1).abs() < 1e-6);
        assert!((shape[0].y - 0.2).abs() < 1e-6);
        assert!((shape[1].x - 1.3).abs() < 1e-6);
        assert!((shape[1].y - 1.4).abs() < 1e-6);
    }
}
