use crate::tree::SplitFeature;
use crate::types::{BoundingBox, Point, Shape};

/// A 2x2 similarity transform matrix (rotation + uniform scale).
/// Represents the matrix:
/// | a  -b |
/// | b   a |
/// Where a = scale * cos(theta), b = scale * sin(theta)
#[derive(Debug, Clone, Copy)]
pub struct SimilarityTransform2D {
    pub a: f32,
    pub b: f32,
}

impl SimilarityTransform2D {
    /// Identity transform (no rotation, scale = 1)
    pub fn identity() -> Self {
        Self { a: 1.0, b: 0.0 }
    }

    /// Apply the transform to a point
    #[inline]
    pub fn apply(&self, p: Point) -> Point {
        Point::new(
            self.a * p.x - self.b * p.y,
            self.b * p.x + self.a * p.y,
        )
    }
}

/// Compute the similarity transform that best maps `from_shape` to `to_shape`.
/// This finds the 2x2 rotation+scale matrix M such that M * from ≈ to.
///
/// Uses the closed-form solution for Procrustes alignment.
pub fn find_similarity_transform(from_shape: &Shape, to_shape: &Shape) -> SimilarityTransform2D {
    debug_assert_eq!(from_shape.num_landmarks(), to_shape.num_landmarks());

    if from_shape.num_landmarks() == 0 {
        return SimilarityTransform2D::identity();
    }

    // Compute centroids
    let n = from_shape.num_landmarks() as f32;
    let mut from_cx = 0.0;
    let mut from_cy = 0.0;
    let mut to_cx = 0.0;
    let mut to_cy = 0.0;

    for (fp, tp) in from_shape.points.iter().zip(to_shape.points.iter()) {
        from_cx += fp.x;
        from_cy += fp.y;
        to_cx += tp.x;
        to_cy += tp.y;
    }

    from_cx /= n;
    from_cy /= n;
    to_cx /= n;
    to_cy /= n;

    // Compute covariance terms for the rotation+scale
    // We want to find (a, b) such that:
    // | a -b | * (from - from_center) ≈ (to - to_center)
    // | b  a |
    //
    // This minimizes sum of squared errors.
    // Solution: a = sum(fx*tx + fy*ty) / sum(fx*fx + fy*fy)
    //           b = sum(fx*ty - fy*tx) / sum(fx*fx + fy*fy)

    let mut numerator_a = 0.0;
    let mut numerator_b = 0.0;
    let mut denominator = 0.0;

    for (fp, tp) in from_shape.points.iter().zip(to_shape.points.iter()) {
        let fx = fp.x - from_cx;
        let fy = fp.y - from_cy;
        let tx = tp.x - to_cx;
        let ty = tp.y - to_cy;

        numerator_a += fx * tx + fy * ty;
        numerator_b += fx * ty - fy * tx;
        denominator += fx * fx + fy * fy;
    }

    if denominator < 1e-10 {
        return SimilarityTransform2D::identity();
    }

    SimilarityTransform2D {
        a: numerator_a / denominator,
        b: numerator_b / denominator,
    }
}

/// Trait for accessing pixel intensities from an image.
pub trait ImageAccess {
    /// Get the grayscale intensity at (x, y). Returns 0 for out-of-bounds pixels.
    /// Coordinates are in image space (not normalized).
    fn get_pixel(&self, x: i32, y: i32) -> u8;

    /// Image dimensions.
    fn width(&self) -> u32;
    fn height(&self) -> u32;
}

/// A simple grayscale image buffer implementing ImageAccess.
pub struct GrayImage {
    data: Vec<u8>,
    width: u32,
    height: u32,
}

impl GrayImage {
    pub fn new(data: Vec<u8>, width: u32, height: u32) -> Self {
        debug_assert_eq!(data.len(), (width * height) as usize);
        Self {
            data,
            width,
            height,
        }
    }

    pub fn from_fn<F>(width: u32, height: u32, f: F) -> Self
    where
        F: Fn(u32, u32) -> u8,
    {
        let mut data = Vec::with_capacity((width * height) as usize);
        for y in 0..height {
            for x in 0..width {
                data.push(f(x, y));
            }
        }
        Self { data, width, height }
    }
}

impl ImageAccess for GrayImage {
    fn get_pixel(&self, x: i32, y: i32) -> u8 {
        if x < 0 || y < 0 || x >= self.width as i32 || y >= self.height as i32 {
            return 0;
        }
        self.data[(y as u32 * self.width + x as u32) as usize]
    }

    fn width(&self) -> u32 {
        self.width
    }

    fn height(&self) -> u32 {
        self.height
    }
}

/// Computes the pixel coordinates for a split feature given the current shape.
///
/// The feature is defined by two anchor points in the shape, plus offsets.
/// The offsets are in a normalized coordinate system relative to the bounding box.
///
/// If `tform` is provided, it transforms the offsets from reference shape space
/// to current shape space (for handling rotated faces).
pub fn compute_feature_points(
    feature: &SplitFeature,
    shape: &Shape,
    bbox: &BoundingBox,
    tform: Option<&SimilarityTransform2D>,
) -> (Point, Point) {
    // Get anchor positions from current shape (in image coordinates)
    let anchor1 = shape[feature.anchor1_idx as usize];
    let anchor2 = shape[feature.anchor2_idx as usize];

    // The offsets are normalized, so scale them by the bbox size
    let mut offset1 = Point::new(
        feature.offset1_x * bbox.width,
        feature.offset1_y * bbox.height,
    );
    let mut offset2 = Point::new(
        feature.offset2_x * bbox.width,
        feature.offset2_y * bbox.height,
    );

    // Apply similarity transform to rotate/scale offsets if provided
    if let Some(t) = tform {
        offset1 = t.apply(offset1);
        offset2 = t.apply(offset2);
    }

    // Compute final pixel positions
    let p1 = anchor1 + offset1;
    let p2 = anchor2 + offset2;

    (p1, p2)
}

/// Sample a pixel with bilinear interpolation for sub-pixel accuracy.
#[inline]
fn sample_bilinear<I: ImageAccess>(image: &I, x: f32, y: f32) -> f32 {
    // Get integer coordinates of the four surrounding pixels
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    // Compute fractional parts
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;

    // Get the four surrounding pixel values
    let p00 = image.get_pixel(x0, y0) as f32;
    let p10 = image.get_pixel(x1, y0) as f32;
    let p01 = image.get_pixel(x0, y1) as f32;
    let p11 = image.get_pixel(x1, y1) as f32;

    // Bilinear interpolation
    let top = p00 * (1.0 - fx) + p10 * fx;
    let bottom = p01 * (1.0 - fx) + p11 * fx;
    top * (1.0 - fy) + bottom * fy
}

/// Compute the pixel intensity difference feature value.
pub fn compute_feature_value<I: ImageAccess>(
    feature: &SplitFeature,
    shape: &Shape,
    bbox: &BoundingBox,
    image: &I,
    tform: Option<&SimilarityTransform2D>,
) -> f32 {
    let (p1, p2) = compute_feature_points(feature, shape, bbox, tform);

    // Sample pixel intensities with bilinear interpolation
    let i1 = sample_bilinear(image, p1.x, p1.y);
    let i2 = sample_bilinear(image, p2.x, p2.y);

    // Return raw pixel difference (dlib uses unscaled values)
    i1 - i2
}

/// Creates a feature extractor closure for use with tree prediction.
pub fn make_feature_extractor<'a, I: ImageAccess>(
    shape: &'a Shape,
    bbox: &'a BoundingBox,
    image: &'a I,
    tform: Option<SimilarityTransform2D>,
) -> impl Fn(&SplitFeature) -> f32 + 'a {
    move |feature: &SplitFeature| compute_feature_value(feature, shape, bbox, image, tform.as_ref())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bilinear_interpolation() {
        // 2x2 image with known values
        let img = GrayImage::new(vec![0, 100, 200, 50], 2, 2);

        // At integer coordinates, should return exact pixel values
        assert!((sample_bilinear(&img, 0.0, 0.0) - 0.0).abs() < 0.01);
        assert!((sample_bilinear(&img, 1.0, 0.0) - 100.0).abs() < 0.01);
        assert!((sample_bilinear(&img, 0.0, 1.0) - 200.0).abs() < 0.01);
        assert!((sample_bilinear(&img, 1.0, 1.0) - 50.0).abs() < 0.01);

        // At center (0.5, 0.5), should be average of all four: (0+100+200+50)/4 = 87.5
        assert!((sample_bilinear(&img, 0.5, 0.5) - 87.5).abs() < 0.01);

        // At (0.5, 0.0), should be average of top row: (0+100)/2 = 50
        assert!((sample_bilinear(&img, 0.5, 0.0) - 50.0).abs() < 0.01);
    }

    #[test]
    fn gray_image_access() {
        // 3x3 checkerboard pattern
        let data = vec![
            0, 255, 0, //
            255, 0, 255, //
            0, 255, 0, //
        ];
        let img = GrayImage::new(data, 3, 3);

        assert_eq!(img.get_pixel(0, 0), 0);
        assert_eq!(img.get_pixel(1, 0), 255);
        assert_eq!(img.get_pixel(1, 1), 0);

        // Out of bounds returns 0
        assert_eq!(img.get_pixel(-1, 0), 0);
        assert_eq!(img.get_pixel(3, 0), 0);
    }

    #[test]
    fn feature_computation() {
        // Simple 10x10 gradient image
        let img = GrayImage::from_fn(10, 10, |x, _y| (x * 25) as u8);

        let bbox = BoundingBox::new(0.0, 0.0, 10.0, 10.0);
        let shape = Shape::new(vec![
            Point::new(2.0, 5.0), // landmark 0
            Point::new(7.0, 5.0), // landmark 1
        ]);

        let feature = SplitFeature {
            anchor1_idx: 0,
            offset1_x: 0.0,
            offset1_y: 0.0,
            anchor2_idx: 1,
            offset2_x: 0.0,
            offset2_y: 0.0,
        };

        let value = compute_feature_value(&feature, &shape, &bbox, &img, None);

        // pixel at x=2 is 50, pixel at x=7 is 175
        // difference = 50 - 175 = -125
        assert!((value - (-125.0)).abs() < 0.01);
    }

    #[test]
    fn similarity_transform() {
        // Test identity: same shape should give identity transform
        let shape = Shape::new(vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(0.5, 1.0),
        ]);
        let tform = find_similarity_transform(&shape, &shape);
        assert!((tform.a - 1.0).abs() < 0.01);
        assert!((tform.b - 0.0).abs() < 0.01);

        // Test 90 degree rotation
        let from = Shape::new(vec![
            Point::new(1.0, 0.0),
            Point::new(0.0, 0.0),
            Point::new(0.0, 1.0),
        ]);
        let to = Shape::new(vec![
            Point::new(0.0, 1.0),
            Point::new(0.0, 0.0),
            Point::new(-1.0, 0.0),
        ]);
        let tform = find_similarity_transform(&from, &to);
        // For 90° rotation: a ≈ 0, b ≈ 1
        assert!((tform.a - 0.0).abs() < 0.1);
        assert!((tform.b - 1.0).abs() < 0.1);
    }
}
