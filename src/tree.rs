use serde::{Deserialize, Serialize};

use crate::types::Shape;

/// A split feature defined by two anchor points.
/// The feature value is the intensity difference between pixels
/// at positions determined by these anchors relative to the current shape.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SplitFeature {
    /// Index of the first anchor landmark in the shape.
    pub anchor1_idx: u16,
    /// Offset from anchor1 (in normalized coordinates).
    pub offset1_x: f32,
    pub offset1_y: f32,
    /// Index of the second anchor landmark in the shape.
    pub anchor2_idx: u16,
    /// Offset from anchor2 (in normalized coordinates).
    pub offset2_x: f32,
    pub offset2_y: f32,
}

/// A node in the regression tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TreeNode {
    /// Internal split node.
    Split {
        feature: SplitFeature,
        threshold: f32,
        left: u32,
        right: u32,
    },
    /// Leaf node containing shape delta.
    Leaf { delta: Shape },
}

/// A single regression tree.
///
/// The tree predicts a shape delta by:
/// 1. Starting at the root node
/// 2. At each split, computing a pixel intensity difference feature
/// 3. Going left if feature < threshold, right otherwise
/// 4. Returning the shape delta at the reached leaf
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionTree {
    pub nodes: Vec<TreeNode>,
}

impl RegressionTree {
    /// Create a new regression tree with the given nodes.
    /// Node 0 is the root.
    pub fn new(nodes: Vec<TreeNode>) -> Self {
        Self { nodes }
    }

    /// Traverse the tree and return a reference to the leaf delta.
    ///
    /// `get_feature` is a closure that computes the pixel intensity difference
    /// for a given split feature and returns the value.
    pub fn predict<F>(&self, get_feature: F) -> &Shape
    where
        F: Fn(&SplitFeature) -> f32,
    {
        let mut node_idx = 0usize;

        loop {
            match &self.nodes[node_idx] {
                TreeNode::Split {
                    feature,
                    threshold,
                    left,
                    right,
                } => {
                    let value = get_feature(feature);
                    // dlib uses: if (diff > threshold) go_left; else go_right;
                    node_idx = if value > *threshold {
                        *left as usize
                    } else {
                        *right as usize
                    };
                }
                TreeNode::Leaf { delta } => {
                    return delta;
                }
            }
        }
    }

    /// Get the number of nodes in the tree.
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Get the depth of the tree (for debugging/validation).
    pub fn depth(&self) -> usize {
        self.depth_from(0)
    }

    fn depth_from(&self, node_idx: usize) -> usize {
        match &self.nodes[node_idx] {
            TreeNode::Split { left, right, .. } => {
                1 + self
                    .depth_from(*left as usize)
                    .max(self.depth_from(*right as usize))
            }
            TreeNode::Leaf { .. } => 1,
        }
    }
}

/// An ensemble of regression trees that together predict a shape delta.
/// Each tree votes on adjustments, and the results are summed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeEnsemble {
    pub trees: Vec<RegressionTree>,
    pub num_landmarks: usize,
}

impl TreeEnsemble {
    pub fn new(trees: Vec<RegressionTree>, num_landmarks: usize) -> Self {
        Self {
            trees,
            num_landmarks,
        }
    }

    /// Predict the shape delta by summing predictions from all trees.
    pub fn predict<F>(&self, get_feature: F) -> Shape
    where
        F: Fn(&SplitFeature) -> f32,
    {
        let mut delta = Shape::zeros(self.num_landmarks);

        for tree in &self.trees {
            let tree_delta = tree.predict(&get_feature);
            delta.add_delta(tree_delta);
        }

        delta
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Point;

    #[test]
    fn simple_tree_traversal() {
        // Create a simple tree:
        //        [0: split]
        //       /          \
        //   [1: leaf]   [2: leaf]
        let nodes = vec![
            TreeNode::Split {
                feature: SplitFeature {
                    anchor1_idx: 0,
                    offset1_x: 0.0,
                    offset1_y: 0.0,
                    anchor2_idx: 1,
                    offset2_x: 0.0,
                    offset2_y: 0.0,
                },
                threshold: 50.0, // Raw pixel difference threshold
                left: 1,
                right: 2,
            },
            TreeNode::Leaf {
                delta: Shape::new(vec![Point::new(-0.1, -0.1)]),
            },
            TreeNode::Leaf {
                delta: Shape::new(vec![Point::new(0.1, 0.1)]),
            },
        ];

        let tree = RegressionTree::new(nodes);

        // Feature value > threshold => go left (dlib convention)
        let result = tree.predict(|_| 100.0);
        assert_eq!(result[0].x, -0.1);

        // Feature value <= threshold => go right
        let result = tree.predict(|_| 30.0);
        assert_eq!(result[0].x, 0.1);
    }

    #[test]
    fn ensemble_sums_predictions() {
        // Two trees, each with just a leaf
        let tree1 = RegressionTree::new(vec![TreeNode::Leaf {
            delta: Shape::new(vec![Point::new(0.1, 0.2)]),
        }]);

        let tree2 = RegressionTree::new(vec![TreeNode::Leaf {
            delta: Shape::new(vec![Point::new(0.3, 0.4)]),
        }]);

        let ensemble = TreeEnsemble::new(vec![tree1, tree2], 1);
        let delta = ensemble.predict(|_| 0.0);

        assert!((delta[0].x - 0.4).abs() < 1e-6);
        assert!((delta[0].y - 0.6).abs() < 1e-6);
    }
}
