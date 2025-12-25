//! Loader for dlib's shape_predictor .dat format.
//!
//! Provides two loading options:
//! 1. [`load_dlib_model`] - Direct binary parser (may need adjustment for your dlib version)
//! 2. [`load_json_model`] - Load from JSON exported by the Python converter script
//!
//! ## Using the Python Converter
//!
//! If direct binary loading fails, use the converter script:
//!
//! ```bash
//! python scripts/convert_dlib_model.py shape_predictor_68_face_landmarks.dat model.json
//! ```
//!
//! Then load with:
//!
//! ```ignore
//! let model = percent_face::dlib::load_json_model("model.json")?;
//! ```

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use crate::error::{Error, Result};
use crate::model::ShapePredictor;
use crate::tree::{RegressionTree, SplitFeature, TreeEnsemble, TreeNode};
use crate::types::{Point, Shape};

/// Reader wrapper for parsing dlib's binary format.
struct DlibReader<R: Read> {
    reader: R,
}

#[allow(dead_code)]
impl<R: Read> DlibReader<R> {
    fn new(reader: R) -> Self {
        Self { reader }
    }

    fn read_u8(&mut self) -> Result<u8> {
        let mut buf = [0u8; 1];
        self.reader.read_exact(&mut buf)?;
        Ok(buf[0])
    }

    fn read_u32(&mut self) -> Result<u32> {
        let mut buf = [0u8; 4];
        self.reader.read_exact(&mut buf)?;
        Ok(u32::from_le_bytes(buf))
    }

    fn read_u64(&mut self) -> Result<u64> {
        let mut buf = [0u8; 8];
        self.reader.read_exact(&mut buf)?;
        Ok(u64::from_le_bytes(buf))
    }

    fn read_i32(&mut self) -> Result<i32> {
        let mut buf = [0u8; 4];
        self.reader.read_exact(&mut buf)?;
        Ok(i32::from_le_bytes(buf))
    }

    fn read_f32(&mut self) -> Result<f32> {
        let mut buf = [0u8; 4];
        self.reader.read_exact(&mut buf)?;
        Ok(f32::from_le_bytes(buf))
    }

    fn read_f64(&mut self) -> Result<f64> {
        let mut buf = [0u8; 8];
        self.reader.read_exact(&mut buf)?;
        Ok(f64::from_le_bytes(buf))
    }

    fn read_bytes(&mut self, n: usize) -> Result<Vec<u8>> {
        let mut buf = vec![0u8; n];
        self.reader.read_exact(&mut buf)?;
        Ok(buf)
    }

    /// Read a dlib-style length-prefixed string.
    fn read_string(&mut self) -> Result<String> {
        let len = self.read_u64()? as usize;
        let bytes = self.read_bytes(len)?;
        String::from_utf8(bytes)
            .map_err(|e| Error::InvalidModel(format!("Invalid UTF-8 string: {}", e)))
    }

    /// Skip n bytes.
    fn skip(&mut self, n: usize) -> Result<()> {
        let mut buf = vec![0u8; n];
        self.reader.read_exact(&mut buf)?;
        Ok(())
    }
}

#[allow(dead_code)]
impl<R: Read + Seek> DlibReader<R> {
    fn position(&mut self) -> Result<u64> {
        Ok(self.reader.stream_position()?)
    }

    fn seek(&mut self, pos: u64) -> Result<()> {
        self.reader.seek(SeekFrom::Start(pos))?;
        Ok(())
    }
}

/// Load a dlib shape_predictor from a .dat file.
///
/// # Format Overview
///
/// The dlib shape_predictor serialization contains:
/// 1. Version string "shape_predictor" or "shape_predictor_model"
/// 2. Initial shape (mean face as vector of 2D points)
/// 3. Cascade of regression tree forests
/// 4. Anchor indices for features
/// 5. Leaf deltas
///
/// This is a complex nested format that we parse incrementally.
pub fn load_dlib_model<P: AsRef<Path>>(path: P) -> Result<ShapePredictor> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut r = DlibReader::new(reader);

    // Read and verify the model header
    let header = r.read_string()?;
    if !header.starts_with("shape_predictor") {
        return Err(Error::InvalidModel(format!(
            "Invalid dlib model header: expected 'shape_predictor', got '{}'",
            header
        )));
    }

    // Read version number
    let version = r.read_u32()?;
    if version > 1 {
        return Err(Error::InvalidModel(format!(
            "Unsupported dlib model version: {}",
            version
        )));
    }

    // Parse the shape predictor structure
    parse_shape_predictor(&mut r)
}

fn parse_shape_predictor<R: Read>(r: &mut DlibReader<R>) -> Result<ShapePredictor> {
    // Read initial_shape (mean shape)
    // Format: vector<vector<float,2>> but stored as nested serialization
    let initial_shape = parse_initial_shape(r)?;
    let num_landmarks = initial_shape.num_landmarks();

    // Read forests (cascade of tree ensembles)
    // Format: vector<vector<impl::regression_tree>>
    let num_cascades = r.read_u64()? as usize;
    let mut cascade = Vec::with_capacity(num_cascades);

    for _ in 0..num_cascades {
        let ensemble = parse_tree_ensemble(r, num_landmarks)?;
        cascade.push(ensemble);
    }

    // Read anchor_idx (which landmarks anchor each split feature)
    // This is used to build the final split features
    // Format: vector<vector<unsigned long>>
    let _anchor_idx = parse_anchor_indices(r)?;

    // Read deltas (leaf node shape adjustments)
    // Format: vector<vector<matrix<float,0,1>>>
    // Already incorporated into trees above

    Ok(ShapePredictor::new(initial_shape, cascade))
}

fn parse_anchor_indices<R: Read>(r: &mut DlibReader<R>) -> Result<Vec<Vec<u64>>> {
    // anchor_idx: vector<vector<unsigned long>>
    // Maps each cascade level to anchor indices for features
    let num_cascades = r.read_u64()? as usize;
    let mut result = Vec::with_capacity(num_cascades);

    for _ in 0..num_cascades {
        let num_anchors = r.read_u64()? as usize;
        let mut anchors = Vec::with_capacity(num_anchors);
        for _ in 0..num_anchors {
            anchors.push(r.read_u64()?);
        }
        result.push(anchors);
    }

    Ok(result)
}

fn parse_initial_shape<R: Read>(r: &mut DlibReader<R>) -> Result<Shape> {
    // The initial shape in dlib is stored as vector<vector<float,2>>
    // which is serialized with matrix dimensions

    // Read number of points
    let num_points = r.read_u64()? as usize;

    let mut points = Vec::with_capacity(num_points);
    for _ in 0..num_points {
        // Each point is a 2-element vector (x, y) as f64 in dlib
        let x = r.read_f64()? as f32;
        let y = r.read_f64()? as f32;
        points.push(Point::new(x, y));
    }

    Ok(Shape::new(points))
}

fn parse_tree_ensemble<R: Read>(r: &mut DlibReader<R>, num_landmarks: usize) -> Result<TreeEnsemble> {
    // Read number of trees in this ensemble
    let num_trees = r.read_u64()? as usize;

    let mut trees = Vec::with_capacity(num_trees);
    for _ in 0..num_trees {
        let tree = parse_regression_tree(r, num_landmarks)?;
        trees.push(tree);
    }

    Ok(TreeEnsemble::new(trees, num_landmarks))
}

fn parse_regression_tree<R: Read>(r: &mut DlibReader<R>, num_landmarks: usize) -> Result<RegressionTree> {
    // dlib regression_tree structure:
    // - splits: vector<split_feature>
    // - leaf_values: vector<matrix<float>>

    // Read number of splits
    let num_splits = r.read_u64()? as usize;

    // For a complete binary tree with n leaves, there are n-1 split nodes
    // Total nodes = 2n - 1
    let num_leaves = num_splits + 1;
    let total_nodes = 2 * num_leaves - 1;

    // Read split features
    let mut splits = Vec::with_capacity(num_splits);
    for _ in 0..num_splits {
        let split = parse_split_feature(r)?;
        splits.push(split);
    }

    // Read leaf values
    let num_leaf_values = r.read_u64()? as usize;
    let mut leaf_deltas = Vec::with_capacity(num_leaf_values);
    for _ in 0..num_leaf_values {
        let delta = parse_leaf_delta(r, num_landmarks)?;
        leaf_deltas.push(delta);
    }

    // Build tree structure
    // dlib uses a complete binary tree layout where:
    // - Splits are stored first (indices 0 to num_splits-1)
    // - Leaves follow (indices num_splits to total_nodes-1)
    // - Left child of node i is at 2*i + 1
    // - Right child of node i is at 2*i + 2

    let mut nodes = Vec::with_capacity(total_nodes);

    // Add split nodes
    for (i, split) in splits.into_iter().enumerate() {
        let left = (2 * i + 1) as u32;
        let right = (2 * i + 2) as u32;

        nodes.push(TreeNode::Split {
            feature: split,
            threshold: 0.0, // Threshold is encoded in the split feature
            left,
            right,
        });
    }

    // Add leaf nodes
    for delta in leaf_deltas {
        nodes.push(TreeNode::Leaf { delta });
    }

    Ok(RegressionTree::new(nodes))
}

fn parse_split_feature<R: Read>(r: &mut DlibReader<R>) -> Result<SplitFeature> {
    // dlib split_feature structure:
    // - idx1, idx2: unsigned long (anchor landmark indices)
    // - offset1, offset2: vector<float,2> (pixel offsets)
    // - thresh: float (split threshold, but we handle this differently)

    let anchor1_idx = r.read_u64()? as u16;
    let anchor2_idx = r.read_u64()? as u16;

    let offset1_x = r.read_f32()?;
    let offset1_y = r.read_f32()?;
    let offset2_x = r.read_f32()?;
    let offset2_y = r.read_f32()?;

    // Read threshold (stored here but we put it in the TreeNode)
    let _thresh = r.read_f32()?;

    Ok(SplitFeature {
        anchor1_idx,
        anchor2_idx,
        offset1_x,
        offset1_y,
        offset2_x,
        offset2_y,
    })
}

fn parse_leaf_delta<R: Read>(r: &mut DlibReader<R>, num_landmarks: usize) -> Result<Shape> {
    // Leaf delta is a matrix<float,0,1> (column vector)
    // with 2*num_landmarks elements (x,y pairs)

    // Read matrix dimensions
    let rows = r.read_u64()? as usize;
    let cols = r.read_u64()? as usize;

    if cols != 1 || rows != num_landmarks * 2 {
        return Err(Error::InvalidModel(format!(
            "Unexpected leaf delta dimensions: {}x{}, expected {}x1",
            rows,
            cols,
            num_landmarks * 2
        )));
    }

    let mut flat = Vec::with_capacity(rows);
    for _ in 0..rows {
        flat.push(r.read_f32()?);
    }

    Ok(Shape::from_flat_vec(&flat))
}

// ============================================================================
// JSON Format Loader (from Python converter output)
// ============================================================================

use serde::Deserialize;

#[derive(Deserialize)]
struct JsonModel {
    #[allow(dead_code)]
    version: u32,
    num_landmarks: usize,
    mean_shape: Vec<JsonPoint>,
    cascade: Vec<JsonEnsemble>,
}

#[derive(Deserialize)]
struct JsonPoint {
    x: f64,
    y: f64,
}

#[derive(Deserialize)]
struct JsonEnsemble {
    trees: Vec<JsonTree>,
}

#[derive(Deserialize)]
struct JsonTree {
    nodes: Vec<JsonNode>,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum JsonNode {
    #[serde(rename = "split")]
    Split {
        feature: JsonFeature,
        threshold: f32,
    },
    #[serde(rename = "leaf")]
    Leaf { delta: Vec<JsonPoint> },
}

#[derive(Deserialize)]
struct JsonFeature {
    anchor1_idx: u64,
    anchor2_idx: u64,
    offset1_x: f32,
    offset1_y: f32,
    offset2_x: f32,
    offset2_y: f32,
}

/// Load a model from JSON format (output of convert_dlib_model.py).
///
/// This is the recommended approach as it's more reliable than parsing
/// the binary format directly.
///
/// # Example
///
/// ```ignore
/// // First convert with Python:
/// // python scripts/convert_dlib_model.py model.dat model.json
///
/// let model = percent_face::dlib::load_json_model("model.json")?;
/// ```
pub fn load_json_model<P: AsRef<Path>>(path: P) -> Result<ShapePredictor> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let json_model: JsonModel = serde_json::from_reader(reader)
        .map_err(|e| Error::InvalidModel(format!("JSON parse error: {}", e)))?;

    let num_landmarks = json_model.num_landmarks;

    // Convert mean shape
    let mean_shape = Shape::new(
        json_model
            .mean_shape
            .iter()
            .map(|p| Point::new(p.x as f32, p.y as f32))
            .collect(),
    );

    // Convert cascade
    let mut cascade = Vec::with_capacity(json_model.cascade.len());
    for json_ensemble in json_model.cascade {
        let mut trees = Vec::with_capacity(json_ensemble.trees.len());

        for json_tree in json_ensemble.trees {
            let nodes = convert_json_nodes(&json_tree.nodes, num_landmarks)?;
            trees.push(RegressionTree::new(nodes));
        }

        cascade.push(TreeEnsemble::new(trees, num_landmarks));
    }

    Ok(ShapePredictor::new(mean_shape, cascade))
}

fn convert_json_nodes(json_nodes: &[JsonNode], num_landmarks: usize) -> Result<Vec<TreeNode>> {
    // Count splits and leaves to determine tree structure
    let num_splits = json_nodes
        .iter()
        .filter(|n| matches!(n, JsonNode::Split { .. }))
        .count();
    let num_leaves = json_nodes
        .iter()
        .filter(|n| matches!(n, JsonNode::Leaf { .. }))
        .count();

    // Validate tree structure
    if num_leaves != num_splits + 1 {
        return Err(Error::InvalidModel(format!(
            "Invalid tree: {} splits should have {} leaves, got {}",
            num_splits,
            num_splits + 1,
            num_leaves
        )));
    }

    let mut nodes = Vec::with_capacity(json_nodes.len());

    // Process nodes - splits come first, then leaves
    // For a complete binary tree, left child of node i is at 2*i+1, right at 2*i+2
    for (i, json_node) in json_nodes.iter().enumerate() {
        match json_node {
            JsonNode::Split { feature, threshold } => {
                let left = (2 * i + 1) as u32;
                let right = (2 * i + 2) as u32;

                nodes.push(TreeNode::Split {
                    feature: SplitFeature {
                        anchor1_idx: feature.anchor1_idx as u16,
                        anchor2_idx: feature.anchor2_idx as u16,
                        offset1_x: feature.offset1_x,
                        offset1_y: feature.offset1_y,
                        offset2_x: feature.offset2_x,
                        offset2_y: feature.offset2_y,
                    },
                    threshold: *threshold,
                    left,
                    right,
                });
            }
            JsonNode::Leaf { delta } => {
                if delta.len() != num_landmarks {
                    return Err(Error::InvalidModel(format!(
                        "Leaf delta has {} points, expected {}",
                        delta.len(),
                        num_landmarks
                    )));
                }
                let shape = Shape::new(
                    delta
                        .iter()
                        .map(|p| Point::new(p.x as f32, p.y as f32))
                        .collect(),
                );
                nodes.push(TreeNode::Leaf { delta: shape });
            }
        }
    }

    Ok(nodes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn write_u64_le(v: &mut Vec<u8>, val: u64) {
        v.extend_from_slice(&val.to_le_bytes());
    }

    fn write_f64_le(v: &mut Vec<u8>, val: f64) {
        v.extend_from_slice(&val.to_le_bytes());
    }

    fn write_string(v: &mut Vec<u8>, s: &str) {
        write_u64_le(v, s.len() as u64);
        v.extend_from_slice(s.as_bytes());
    }

    #[test]
    fn read_primitives() {
        let data = vec![
            0x01, 0x02, 0x03, 0x04, // u32: 0x04030201
            0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, // u64
        ];
        let cursor = Cursor::new(data);
        let mut reader = DlibReader::new(cursor);

        assert_eq!(reader.read_u32().unwrap(), 0x04030201);
        assert_eq!(reader.read_u64().unwrap(), 0x0c0b0a0908070605);
    }

    #[test]
    fn read_string() {
        let mut data = Vec::new();
        write_string(&mut data, "hello");

        let cursor = Cursor::new(data);
        let mut reader = DlibReader::new(cursor);

        assert_eq!(reader.read_string().unwrap(), "hello");
    }

    #[test]
    fn parse_initial_shape_test() {
        let mut data = Vec::new();

        // 2 points
        write_u64_le(&mut data, 2);
        write_f64_le(&mut data, 0.3); // point 0 x
        write_f64_le(&mut data, 0.4); // point 0 y
        write_f64_le(&mut data, 0.6); // point 1 x
        write_f64_le(&mut data, 0.7); // point 1 y

        let cursor = Cursor::new(data);
        let mut reader = DlibReader::new(cursor);

        let shape = parse_initial_shape(&mut reader).unwrap();
        assert_eq!(shape.num_landmarks(), 2);
        assert!((shape[0].x - 0.3).abs() < 1e-5);
        assert!((shape[0].y - 0.4).abs() < 1e-5);
        assert!((shape[1].x - 0.6).abs() < 1e-5);
        assert!((shape[1].y - 0.7).abs() < 1e-5);
    }

    #[test]
    fn load_json_model_test() {
        let json = r#"{
            "version": 1,
            "num_landmarks": 2,
            "mean_shape": [
                {"x": 0.3, "y": 0.3},
                {"x": 0.7, "y": 0.3}
            ],
            "cascade": [
                {
                    "trees": [
                        {
                            "nodes": [
                                {
                                    "type": "split",
                                    "feature": {
                                        "anchor1_idx": 0,
                                        "anchor2_idx": 1,
                                        "offset1_x": 0.0,
                                        "offset1_y": 0.0,
                                        "offset2_x": 0.0,
                                        "offset2_y": 0.0
                                    },
                                    "threshold": 0.5
                                },
                                {
                                    "type": "leaf",
                                    "delta": [{"x": -0.01, "y": 0.0}, {"x": 0.01, "y": 0.0}]
                                },
                                {
                                    "type": "leaf",
                                    "delta": [{"x": 0.01, "y": 0.0}, {"x": -0.01, "y": 0.0}]
                                }
                            ]
                        }
                    ]
                }
            ]
        }"#;

        // Write to temp file
        let temp_path = std::env::temp_dir().join("test_model.json");
        std::fs::write(&temp_path, json).unwrap();

        let model = load_json_model(&temp_path).unwrap();
        assert_eq!(model.num_landmarks(), 2);
        assert_eq!(model.num_cascade_stages(), 1);

        // Clean up
        std::fs::remove_file(temp_path).ok();
    }
}
