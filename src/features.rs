use crate::tree::SplitFeature;
use crate::types::{BoundingBox, Point, Shape};

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
pub fn compute_feature_points(
    feature: &SplitFeature,
    shape: &Shape,
    bbox: &BoundingBox,
) -> (Point, Point) {
    // Get anchor positions from current shape (in image coordinates)
    let anchor1 = shape[feature.anchor1_idx as usize];
    let anchor2 = shape[feature.anchor2_idx as usize];

    // The offsets are normalized, so scale them by the bbox size
    let offset1 = Point::new(
        feature.offset1_x * bbox.width,
        feature.offset1_y * bbox.height,
    );
    let offset2 = Point::new(
        feature.offset2_x * bbox.width,
        feature.offset2_y * bbox.height,
    );

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
) -> f32 {
    let (p1, p2) = compute_feature_points(feature, shape, bbox);

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
) -> impl Fn(&SplitFeature) -> f32 + 'a {
    move |feature: &SplitFeature| compute_feature_value(feature, shape, bbox, image)
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

        let value = compute_feature_value(&feature, &shape, &bbox, &img);

        // pixel at x=2 is 50, pixel at x=7 is 175
        // difference = 50 - 175 = -125
        assert!((value - (-125.0)).abs() < 0.01);
    }
}
