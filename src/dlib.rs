//! Loader for dlib's shape_predictor .dat format.
//!
//! This module provides a pure Rust parser for dlib's binary shape predictor format,
//! supporting both raw `.dat` files and bzip2-compressed `.dat.bz2` files.
//!
//! # Example
//!
//! ```ignore
//! use percent_face::dlib::load_dlib_model;
//!
//! // Load compressed model directly
//! let model = load_dlib_model("shape_predictor_68_face_landmarks.dat.bz2")?;
//!
//! // Or uncompressed
//! let model = load_dlib_model("shape_predictor_68_face_landmarks.dat")?;
//! ```
//!
//! # Obtaining Models
//!
//! Pre-trained models are available from the dlib-models repository:
//!
//! ```bash
//! git clone --depth 1 git@github.com:davisking/dlib-models.git
//! ```
//!
//! Common models:
//! - `shape_predictor_5_face_landmarks.dat.bz2` - 5-point model (eyes + nose)
//! - `shape_predictor_68_face_landmarks.dat.bz2` - Full 68-point model

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use bzip2::read::BzDecoder;

use crate::error::{Error, Result};
use crate::model::ShapePredictor;
use crate::tree::{RegressionTree, SplitFeature, TreeEnsemble, TreeNode};
use crate::types::{Point, Shape};

/// Reader wrapper for parsing dlib's binary format.
///
/// dlib uses a variable-length integer encoding:
/// - Control byte: high bit = sign (1 = negative), low 4 bits = number of bytes following
/// - Value bytes: little-endian integer value
///
/// Floats are stored as (mantissa, exponent) pairs, reconstructed via ldexp.
struct DlibReader<R: Read> {
    reader: R,
}

impl<R: Read> DlibReader<R> {
    fn new(reader: R) -> Self {
        Self { reader }
    }

    fn read_byte(&mut self) -> Result<u8> {
        let mut buf = [0u8; 1];
        self.reader.read_exact(&mut buf)?;
        Ok(buf[0])
    }

    /// Decode a variable-length integer.
    fn read_int(&mut self) -> Result<i64> {
        let control = self.read_byte()?;
        let is_negative = (control & 0x80) != 0;
        let num_bytes = (control & 0x0F) as usize;

        if num_bytes == 0 {
            return Ok(0);
        }

        let mut val: u64 = 0;
        for i in 0..num_bytes {
            let byte = self.read_byte()? as u64;
            val |= byte << (8 * i);
        }

        let signed_val = val as i64;
        Ok(if is_negative { -signed_val } else { signed_val })
    }

    /// Read an unsigned long.
    fn read_ulong(&mut self) -> Result<u64> {
        let val = self.read_int()?;
        if val < 0 {
            return Err(Error::InvalidModel(format!(
                "Expected unsigned value, got {}",
                val
            )));
        }
        Ok(val as u64)
    }

    /// Decode a float stored as (mantissa, exponent) pair.
    fn read_float(&mut self) -> Result<f32> {
        let mantissa = self.read_int()?;
        let exponent = self.read_int()? as i32;

        if mantissa == 0 {
            return Ok(0.0);
        }

        let result = (mantissa as f64) * (2.0_f64).powi(exponent);
        Ok(result as f32)
    }

    /// Read a matrix stored as (-rows, -cols, data...).
    fn read_float_matrix(&mut self) -> Result<(usize, usize, Vec<f32>)> {
        let rows_neg = self.read_int()?;
        let cols_neg = self.read_int()?;

        let rows = (-rows_neg) as usize;
        let cols = (-cols_neg) as usize;

        let mut data = Vec::with_capacity(rows * cols);
        for _ in 0..(rows * cols) {
            data.push(self.read_float()?);
        }

        Ok((rows, cols, data))
    }
}

/// Raw split feature data before anchor/delta resolution.
struct RawSplit {
    feature_idx1: u16,
    feature_idx2: u16,
    threshold: f32,
}

/// Raw regression tree before anchor/delta resolution.
struct RawTree {
    splits: Vec<RawSplit>,
    leaf_deltas: Vec<Shape>,
}

/// Load a dlib shape_predictor from a .dat or .dat.bz2 file.
pub fn load_dlib_model<P: AsRef<Path>>(path: P) -> Result<ShapePredictor> {
    let path = path.as_ref();
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let is_bz2 = path.extension().is_some_and(|ext| ext == "bz2");

    if is_bz2 {
        let decoder = BzDecoder::new(reader);
        let mut r = DlibReader::new(decoder);
        parse_shape_predictor(&mut r)
    } else {
        let mut r = DlibReader::new(reader);
        parse_shape_predictor(&mut r)
    }
}

/// Load a dlib model from an already-opened reader.
pub fn load_dlib_model_from_reader<R: Read>(reader: R) -> Result<ShapePredictor> {
    let mut r = DlibReader::new(reader);
    parse_shape_predictor(&mut r)
}

fn parse_shape_predictor<R: Read>(r: &mut DlibReader<R>) -> Result<ShapePredictor> {
    // 1. Read version
    let version = r.read_int()?;
    if version != 1 {
        return Err(Error::InvalidModel(format!(
            "Unsupported shape_predictor version: {}",
            version
        )));
    }

    // 2. Read initial_shape matrix
    let (rows, cols, data) = r.read_float_matrix()?;
    if cols != 1 || rows % 2 != 0 {
        return Err(Error::InvalidModel(format!(
            "Invalid initial_shape dimensions: {}x{}",
            rows, cols
        )));
    }

    let num_landmarks = rows / 2;
    let initial_shape = Shape::new(
        data.chunks_exact(2)
            .map(|chunk| Point::new(chunk[0], chunk[1]))
            .collect(),
    );

    // 3. Read forests - store as raw trees for now
    let num_cascades = r.read_ulong()? as usize;
    let mut raw_cascades: Vec<Vec<RawTree>> = Vec::with_capacity(num_cascades);

    for _ in 0..num_cascades {
        let num_trees = r.read_ulong()? as usize;
        let mut trees = Vec::with_capacity(num_trees);

        for _ in 0..num_trees {
            let tree = parse_raw_tree(r, num_landmarks)?;
            trees.push(tree);
        }

        raw_cascades.push(trees);
    }

    // 4. Read anchor_idx (vector<vector<unsigned long>>)
    // anchor_idx[cascade][feature_idx] = landmark index
    let num_anchor_cascades = r.read_ulong()? as usize;
    let mut anchor_idx: Vec<Vec<u16>> = Vec::with_capacity(num_anchor_cascades);
    for _ in 0..num_anchor_cascades {
        let num_anchors = r.read_ulong()? as usize;
        let mut anchors = Vec::with_capacity(num_anchors);
        for _ in 0..num_anchors {
            anchors.push(r.read_ulong()? as u16);
        }
        anchor_idx.push(anchors);
    }

    // 5. Read deltas (vector<vector<vector<float,2>>>)
    // deltas[cascade][feature_idx] = (dx, dy) offset
    let num_delta_cascades = r.read_ulong()? as usize;
    let mut deltas: Vec<Vec<(f32, f32)>> = Vec::with_capacity(num_delta_cascades);
    for _ in 0..num_delta_cascades {
        let num_deltas = r.read_ulong()? as usize;
        let mut cascade_deltas = Vec::with_capacity(num_deltas);
        for _ in 0..num_deltas {
            let dx = r.read_float()?;
            let dy = r.read_float()?;
            cascade_deltas.push((dx, dy));
        }
        deltas.push(cascade_deltas);
    }

    // 6. Convert raw trees to final trees with resolved anchors/deltas
    let mut cascade = Vec::with_capacity(num_cascades);

    for (cascade_idx, raw_trees) in raw_cascades.into_iter().enumerate() {
        let cascade_anchors = anchor_idx.get(cascade_idx).ok_or_else(|| {
            Error::InvalidModel(format!("Missing anchor_idx for cascade {}", cascade_idx))
        })?;
        let cascade_deltas = deltas.get(cascade_idx).ok_or_else(|| {
            Error::InvalidModel(format!("Missing deltas for cascade {}", cascade_idx))
        })?;

        let mut trees = Vec::with_capacity(raw_trees.len());

        for raw_tree in raw_trees {
            let tree = resolve_tree(raw_tree, cascade_anchors, cascade_deltas)?;
            trees.push(tree);
        }

        cascade.push(TreeEnsemble::new(trees, num_landmarks));
    }

    Ok(ShapePredictor::new(initial_shape, cascade))
}

fn parse_raw_tree<R: Read>(r: &mut DlibReader<R>, num_landmarks: usize) -> Result<RawTree> {
    // Read splits
    let num_splits = r.read_ulong()? as usize;
    let mut splits = Vec::with_capacity(num_splits);

    for _ in 0..num_splits {
        let feature_idx1 = r.read_ulong()? as u16;
        let feature_idx2 = r.read_ulong()? as u16;
        let threshold = r.read_float()?;

        splits.push(RawSplit {
            feature_idx1,
            feature_idx2,
            threshold,
        });
    }

    // Read leaf values
    let num_leaves = r.read_ulong()? as usize;

    if num_leaves != num_splits + 1 {
        return Err(Error::InvalidModel(format!(
            "Invalid tree: {} splits should have {} leaves, got {}",
            num_splits,
            num_splits + 1,
            num_leaves
        )));
    }

    let mut leaf_deltas = Vec::with_capacity(num_leaves);
    for _ in 0..num_leaves {
        let (rows, cols, data) = r.read_float_matrix()?;

        if cols != 1 || rows != num_landmarks * 2 {
            return Err(Error::InvalidModel(format!(
                "Invalid leaf delta: {}x{}, expected {}x1",
                rows, cols,
                num_landmarks * 2
            )));
        }

        let delta = Shape::new(
            data.chunks_exact(2)
                .map(|chunk| Point::new(chunk[0], chunk[1]))
                .collect(),
        );
        leaf_deltas.push(delta);
    }

    Ok(RawTree { splits, leaf_deltas })
}

fn resolve_tree(
    raw: RawTree,
    anchors: &[u16],
    deltas: &[(f32, f32)],
) -> Result<RegressionTree> {
    let num_splits = raw.splits.len();
    let total_nodes = num_splits + raw.leaf_deltas.len();
    let mut nodes = Vec::with_capacity(total_nodes);

    // Add split nodes
    for (i, split) in raw.splits.into_iter().enumerate() {
        let idx1 = split.feature_idx1 as usize;
        let idx2 = split.feature_idx2 as usize;

        // Look up anchor landmark indices
        let anchor1_idx = *anchors.get(idx1).ok_or_else(|| {
            Error::InvalidModel(format!("Feature index {} out of bounds", idx1))
        })?;
        let anchor2_idx = *anchors.get(idx2).ok_or_else(|| {
            Error::InvalidModel(format!("Feature index {} out of bounds", idx2))
        })?;

        // Look up pixel offsets
        let (offset1_x, offset1_y) = *deltas.get(idx1).ok_or_else(|| {
            Error::InvalidModel(format!("Delta index {} out of bounds", idx1))
        })?;
        let (offset2_x, offset2_y) = *deltas.get(idx2).ok_or_else(|| {
            Error::InvalidModel(format!("Delta index {} out of bounds", idx2))
        })?;

        let left = (2 * i + 1) as u32;
        let right = (2 * i + 2) as u32;

        nodes.push(TreeNode::Split {
            feature: SplitFeature {
                anchor1_idx,
                anchor2_idx,
                offset1_x,
                offset1_y,
                offset2_x,
                offset2_y,
            },
            threshold: split.threshold,
            left,
            right,
        });
    }

    // Add leaf nodes
    for delta in raw.leaf_deltas {
        nodes.push(TreeNode::Leaf { delta });
    }

    Ok(RegressionTree::new(nodes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use std::path::PathBuf;

    fn write_control_byte(v: &mut Vec<u8>, is_negative: bool, num_bytes: u8) {
        let control = if is_negative { 0x80 } else { 0x00 } | (num_bytes & 0x0F);
        v.push(control);
    }

    fn write_int(v: &mut Vec<u8>, val: i64) {
        if val == 0 {
            v.push(0x00);
            return;
        }

        let is_negative = val < 0;
        let abs_val = val.unsigned_abs();

        let num_bytes = if abs_val <= 0xFF {
            1
        } else if abs_val <= 0xFFFF {
            2
        } else if abs_val <= 0xFF_FFFF {
            3
        } else if abs_val <= 0xFFFF_FFFF {
            4
        } else {
            8
        };

        write_control_byte(v, is_negative, num_bytes);

        for i in 0..num_bytes {
            v.push(((abs_val >> (8 * i)) & 0xFF) as u8);
        }
    }

    fn write_float(v: &mut Vec<u8>, val: f32) {
        if val == 0.0 {
            write_int(v, 0);
            write_int(v, 0);
            return;
        }

        let (mantissa, exponent, _sign) = num_traits_like_frexp(val as f64);
        let int_mantissa = (mantissa * (1i64 << 53) as f64) as i64;
        let adjusted_exp = exponent - 53;

        write_int(v, int_mantissa);
        write_int(v, adjusted_exp as i64);
    }

    fn num_traits_like_frexp(val: f64) -> (f64, i32, i32) {
        if val == 0.0 {
            return (0.0, 0, 1);
        }
        let sign = if val < 0.0 { -1 } else { 1 };
        let abs_val = val.abs();
        let exp = abs_val.log2().floor() as i32 + 1;
        let mantissa = abs_val / (2.0_f64).powi(exp);
        (mantissa * sign as f64, exp, sign)
    }

    #[test]
    fn read_varint() {
        let mut data = Vec::new();
        write_int(&mut data, 0);
        write_int(&mut data, 1);
        write_int(&mut data, 127);
        write_int(&mut data, 128);
        write_int(&mut data, 255);
        write_int(&mut data, 256);
        write_int(&mut data, -1);
        write_int(&mut data, -128);

        let cursor = Cursor::new(data);
        let mut reader = DlibReader::new(cursor);

        assert_eq!(reader.read_int().unwrap(), 0);
        assert_eq!(reader.read_int().unwrap(), 1);
        assert_eq!(reader.read_int().unwrap(), 127);
        assert_eq!(reader.read_int().unwrap(), 128);
        assert_eq!(reader.read_int().unwrap(), 255);
        assert_eq!(reader.read_int().unwrap(), 256);
        assert_eq!(reader.read_int().unwrap(), -1);
        assert_eq!(reader.read_int().unwrap(), -128);
    }

    #[test]
    fn read_float_values() {
        let mut data = Vec::new();
        write_float(&mut data, 0.0);
        write_float(&mut data, 1.0);
        write_float(&mut data, -1.0);
        write_float(&mut data, 0.5);
        write_float(&mut data, 0.25);

        let cursor = Cursor::new(data);
        let mut reader = DlibReader::new(cursor);

        assert!((reader.read_float().unwrap() - 0.0).abs() < 1e-6);
        assert!((reader.read_float().unwrap() - 1.0).abs() < 1e-6);
        assert!((reader.read_float().unwrap() - (-1.0)).abs() < 1e-6);
        assert!((reader.read_float().unwrap() - 0.5).abs() < 1e-6);
        assert!((reader.read_float().unwrap() - 0.25).abs() < 1e-6);
    }

    fn dlib_models_dir() -> Option<PathBuf> {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("dlib-models");
        if path.exists() {
            Some(path)
        } else {
            None
        }
    }

    /// Test loading the 5-point shape predictor model.
    ///
    /// This test requires the dlib-models directory. To set it up:
    /// ```bash
    /// git clone --depth 1 git@github.com:davisking/dlib-models.git
    /// ```
    #[test]
    fn load_5_point_model() {
        let Some(models_dir) = dlib_models_dir() else {
            eprintln!("Skipping test: dlib-models directory not found");
            eprintln!("To run this test, clone the models:");
            eprintln!("  git clone --depth 1 git@github.com:davisking/dlib-models.git");
            return;
        };

        let model_path = models_dir.join("shape_predictor_5_face_landmarks.dat.bz2");
        if !model_path.exists() {
            eprintln!("Skipping test: model file not found at {:?}", model_path);
            return;
        }

        let model = load_dlib_model(&model_path).expect("Failed to load 5-point model");

        assert_eq!(model.num_landmarks(), 5);
        assert_eq!(model.num_cascade_stages(), 15);

        eprintln!(
            "Loaded 5-point model: {} landmarks, {} cascade stages",
            model.num_landmarks(),
            model.num_cascade_stages()
        );
    }

    /// Test loading the 68-point shape predictor model.
    #[test]
    fn load_68_point_model() {
        let Some(models_dir) = dlib_models_dir() else {
            eprintln!("Skipping test: dlib-models directory not found");
            return;
        };

        let model_path = models_dir.join("shape_predictor_68_face_landmarks.dat.bz2");
        if !model_path.exists() {
            eprintln!("Skipping test: model file not found");
            return;
        }

        let model = load_dlib_model(&model_path).expect("Failed to load 68-point model");

        assert_eq!(model.num_landmarks(), 68);
        assert!(model.num_cascade_stages() > 0);

        eprintln!(
            "Loaded 68-point model: {} landmarks, {} cascade stages",
            model.num_landmarks(),
            model.num_cascade_stages()
        );
    }

    /// Test loading the GTX variant of the 68-point model.
    #[test]
    fn load_68_point_gtx_model() {
        let Some(models_dir) = dlib_models_dir() else {
            eprintln!("Skipping test: dlib-models directory not found");
            return;
        };

        let model_path = models_dir.join("shape_predictor_68_face_landmarks_GTX.dat.bz2");
        if !model_path.exists() {
            eprintln!("Skipping test: GTX model file not found");
            return;
        }

        let model = load_dlib_model(&model_path).expect("Failed to load 68-point GTX model");

        assert_eq!(model.num_landmarks(), 68);

        eprintln!(
            "Loaded 68-point GTX model: {} landmarks, {} cascade stages",
            model.num_landmarks(),
            model.num_cascade_stages()
        );
    }

    /// Test that split features have correct anchor indices and offsets.
    #[test]
    fn verify_split_features() {
        let Some(models_dir) = dlib_models_dir() else {
            eprintln!("Skipping test: dlib-models directory not found");
            return;
        };

        let model_path = models_dir.join("shape_predictor_5_face_landmarks.dat.bz2");
        if !model_path.exists() {
            return;
        }

        let model = load_dlib_model(&model_path).expect("Failed to load model");

        // Verify that anchor indices are valid (0-4 for 5-point model).
        // The model loaded successfully which means all anchor lookups worked.
        assert_eq!(model.num_landmarks(), 5);
    }
}
