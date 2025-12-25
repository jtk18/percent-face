#!/usr/bin/env python3
"""
Convert a dlib shape_predictor .dat file to percent-face JSON format.

This uses dlib's Python bindings to read the model, then exports it to
a JSON format that can be easily parsed by our Rust loader.

Usage:
    python convert_dlib_model.py shape_predictor_68_face_landmarks.dat output.json

Requirements:
    pip install dlib numpy
"""

import argparse
import json
import struct
import sys
from pathlib import Path

try:
    import dlib
    import numpy as np
except ImportError:
    print("Error: This script requires dlib and numpy.")
    print("Install with: pip install dlib numpy")
    sys.exit(1)


def extract_shape_predictor_data(model_path: str) -> dict:
    """
    Extract shape predictor data using dlib's Python API.

    Unfortunately, dlib doesn't expose the internal tree structure through
    its Python API, so we need to use an indirect approach.
    """
    predictor = dlib.shape_predictor(model_path)

    # Get number of parts (landmarks)
    # We need to run inference on a dummy image to determine this
    # since dlib doesn't expose num_parts directly on the predictor

    # The predictor itself doesn't expose its internal structure in Python,
    # so we'll need to parse the binary format directly or use a workaround.

    # For now, return what we can determine
    return {
        "model_path": model_path,
        "note": "dlib Python API doesn't expose internal tree structure"
    }


def parse_dlib_binary(model_path: str) -> dict:
    """
    Parse dlib's binary format directly.

    The dlib serialization format for shape_predictor:
    1. initial_shape: matrix<double> as column vector (2*num_landmarks x 1)
    2. forests: vector<vector<regression_tree>>
    3. anchor_idx: vector<vector<unsigned long>>
    4. deltas: vector<vector<matrix<float,0,1>>>

    Each regression_tree contains:
    - splits: vector<split_feature>
    - leaf_values: vector<matrix<float,0,1>>

    Each split_feature contains:
    - idx1, idx2: unsigned long (anchor indices)
    - w1, w2: vector<double,2> (offsets)
    - thresh: double
    """

    with open(model_path, 'rb') as f:
        data = f.read()

    pos = 0

    def read_u64():
        nonlocal pos
        val = struct.unpack('<Q', data[pos:pos+8])[0]
        pos += 8
        return val

    def read_f64():
        nonlocal pos
        val = struct.unpack('<d', data[pos:pos+8])[0]
        pos += 8
        return val

    def read_f32():
        nonlocal pos
        val = struct.unpack('<f', data[pos:pos+4])[0]
        pos += 4
        return val

    def read_matrix_f64():
        """Read a dlib matrix<double>"""
        rows = read_u64()
        cols = read_u64()
        values = []
        for _ in range(int(rows * cols)):
            values.append(read_f64())
        return {'rows': rows, 'cols': cols, 'data': values}

    def read_matrix_f32():
        """Read a dlib matrix<float>"""
        rows = read_u64()
        cols = read_u64()
        values = []
        for _ in range(int(rows * cols)):
            values.append(read_f32())
        return {'rows': rows, 'cols': cols, 'data': values}

    def read_split_feature():
        idx1 = read_u64()
        idx2 = read_u64()
        # w1, w2 are 2D vectors stored as floats
        w1_x = read_f32()
        w1_y = read_f32()
        w2_x = read_f32()
        w2_y = read_f32()
        thresh = read_f32()
        return {
            'anchor1_idx': idx1,
            'anchor2_idx': idx2,
            'offset1': [w1_x, w1_y],
            'offset2': [w2_x, w2_y],
            'threshold': thresh
        }

    def read_regression_tree():
        num_splits = read_u64()
        splits = [read_split_feature() for _ in range(int(num_splits))]

        num_leaves = read_u64()
        leaves = [read_matrix_f32() for _ in range(int(num_leaves))]

        return {'splits': splits, 'leaves': leaves}

    # Parse initial_shape
    initial_shape = read_matrix_f64()
    num_landmarks = initial_shape['rows'] // 2

    # Parse forests (cascade of tree ensembles)
    num_cascades = read_u64()
    cascades = []
    for _ in range(int(num_cascades)):
        num_trees = read_u64()
        trees = [read_regression_tree() for _ in range(int(num_trees))]
        cascades.append({'trees': trees})

    # Parse anchor_idx
    num_anchor_cascades = read_u64()
    anchor_idx = []
    for _ in range(int(num_anchor_cascades)):
        num_anchors = read_u64()
        anchors = [read_u64() for _ in range(int(num_anchors))]
        anchor_idx.append(anchors)

    # Convert initial_shape to points
    points = []
    for i in range(int(num_landmarks)):
        x = initial_shape['data'][i * 2]
        y = initial_shape['data'][i * 2 + 1]
        points.append({'x': x, 'y': y})

    return {
        'num_landmarks': int(num_landmarks),
        'initial_shape': points,
        'cascades': cascades,
        'anchor_idx': anchor_idx
    }


def convert_to_percent_face_format(dlib_data: dict) -> dict:
    """Convert parsed dlib data to percent-face format."""

    result = {
        'version': 1,
        'num_landmarks': dlib_data['num_landmarks'],
        'mean_shape': dlib_data['initial_shape'],
        'cascade': []
    }

    for cascade_idx, cascade in enumerate(dlib_data['cascades']):
        ensemble = {
            'trees': []
        }

        for tree_data in cascade['trees']:
            # Build tree nodes
            nodes = []

            # Add split nodes
            for split in tree_data['splits']:
                nodes.append({
                    'type': 'split',
                    'feature': {
                        'anchor1_idx': split['anchor1_idx'],
                        'anchor2_idx': split['anchor2_idx'],
                        'offset1_x': split['offset1'][0],
                        'offset1_y': split['offset1'][1],
                        'offset2_x': split['offset2'][0],
                        'offset2_y': split['offset2'][1],
                    },
                    'threshold': split['threshold'],
                })

            # Add leaf nodes
            for leaf in tree_data['leaves']:
                # Convert flat vector to points
                delta_points = []
                for i in range(len(leaf['data']) // 2):
                    delta_points.append({
                        'x': leaf['data'][i * 2],
                        'y': leaf['data'][i * 2 + 1]
                    })
                nodes.append({
                    'type': 'leaf',
                    'delta': delta_points
                })

            ensemble['trees'].append({'nodes': nodes})

        result['cascade'].append(ensemble)

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Convert dlib shape_predictor to percent-face format'
    )
    parser.add_argument('input', help='Input .dat file')
    parser.add_argument('output', help='Output .json file')
    parser.add_argument('--debug', action='store_true', help='Print debug info')

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' not found")
        sys.exit(1)

    print(f"Parsing {input_path}...")

    try:
        dlib_data = parse_dlib_binary(str(input_path))
    except Exception as e:
        print(f"Error parsing dlib format: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    if args.debug:
        print(f"Found {dlib_data['num_landmarks']} landmarks")
        print(f"Found {len(dlib_data['cascades'])} cascade levels")
        for i, cascade in enumerate(dlib_data['cascades']):
            print(f"  Cascade {i}: {len(cascade['trees'])} trees")

    print("Converting to percent-face format...")
    result = convert_to_percent_face_format(dlib_data)

    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Wrote {output_path}")


if __name__ == '__main__':
    main()
