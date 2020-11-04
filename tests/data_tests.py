import os as _os
import tempfile as _tempfile


_tmpdir = _tempfile.mkdtemp()


_sample_settings = {
    "N_channels": 3,
    "patch_shape": [50, 50, 50],
    "sample_index_value": 100,
    "sample_label_index_value": 50,
    "dimensionality": {"sample": 3},
    "sample_categories": ["sample"],
}

SINGLE_BINARY_CLASS_TEST = {
    "name": "single binary class",
    "label_file": "test_data/NPZ_Data/labels_single_binary_class.txt",
    "expected_output": {
        "labels": [1, 0],
        "category_labels": {"Label_1": [1, 0]},
        "one_hot_labels": [[0, 1], [1, 0]],
        "one_hot_category_labels": {"Label_1": [[0, 1], [1, 0]]},
        "sample_locations": [
            "./PrognosAIs/tests/test_data/NPZ_Data/Samples/Test_sample_1.npz",
            "./PrognosAIs/tests/test_data/NPZ_Data/Samples/Test_sample_2.npz",
        ],
        "moved_sample_locations": [
            _os.path.join(_tmpdir, "test_location/Test_sample_1.npz"),
            _os.path.join(_tmpdir, "test_location/Test_sample_2.npz"),
        ],
        "class_weights": {"Label_1": {0: 1, 1: 1}},
        "one_hot_class_weights": {"Label_1": {0: 1, 1: 1}},
        "true_missing_value": None,
        "number_of_classes": {"Label_1": 2},
        "N_samples": 2,
        "sample_settings": _sample_settings,
    },
}

MULTI_BINARY_CLASS_TEST = {
    "name": "multi binary class",
    "label_file": "test_data/NPZ_Data/labels_multi_binary_class.txt",
    "expected_output": {
        "labels": [[1, 0], [0, 1], [0, 0], [1, 1]],
        "category_labels": {"Label_1": [1, 0, 0, 1], "Label_2": [0, 1, 0, 1]},
        "one_hot_labels": [[[0, 1], [1, 0]], [[1, 0], [0, 1]], [[1, 0], [1, 0]], [[0, 1], [0, 1]]],
        "one_hot_category_labels": {
            "Label_1": [[0, 1], [1, 0], [1, 0], [0, 1]],
            "Label_2": [[1, 0], [0, 1], [1, 0], [0, 1]],
        },
        "sample_locations": [
            "./PrognosAIs/tests/test_data/NPZ_Data/Samples/Test_sample_1.npz",
            "./PrognosAIs/tests/test_data/NPZ_Data/Samples/Test_sample_2.npz",
            "./PrognosAIs/tests/test_data/NPZ_Data/Samples/Test_sample_3.npz",
            "./PrognosAIs/tests/test_data/NPZ_Data/Samples/Test_sample_4.npz",
        ],
        "moved_sample_locations": [
            _os.path.join(_tmpdir, "test_location/Test_sample_1.npz"),
            _os.path.join(_tmpdir, "test_location/Test_sample_2.npz"),
            _os.path.join(_tmpdir, "test_location/Test_sample_3.npz"),
            _os.path.join(_tmpdir, "test_location/Test_sample_4.npz"),
        ],
        "class_weights": {"Label_1": {0: 1, 1: 1}, "Label_2": {0: 1, 1: 1}},
        "one_hot_class_weights": {"Label_1": {0: 1, 1: 1}, "Label_2": {0: 1, 1: 1}},
        "true_missing_value": None,
        "number_of_classes": {"Label_1": 2, "Label_2": 2},
        "N_samples": 4,
        "sample_settings": _sample_settings,
    },
}

SINGLE_CATEGORICAL_CLASS_TEST = {
    "name": "single categorical class",
    "label_file": "test_data/NPZ_Data/labels_single_categorical_class.txt",
    "expected_output": {
        "labels": [1, 0, 3, 1],
        "category_labels": {"Label_1": [1, 0, 3, 1]},
        "one_hot_labels": [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]],
        "one_hot_category_labels": {
            "Label_1": [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]]
        },
        "sample_locations": [
            "./PrognosAIs/tests/test_data/NPZ_Data/Samples/Test_sample_1.npz",
            "./PrognosAIs/tests/test_data/NPZ_Data/Samples/Test_sample_2.npz",
            "./PrognosAIs/tests/test_data/NPZ_Data/Samples/Test_sample_3.npz",
            "./PrognosAIs/tests/test_data/NPZ_Data/Samples/Test_sample_4.npz",
        ],
        "moved_sample_locations": [
            _os.path.join(_tmpdir, "test_location/Test_sample_1.npz"),
            _os.path.join(_tmpdir, "test_location/Test_sample_2.npz"),
            _os.path.join(_tmpdir, "test_location/Test_sample_3.npz"),
            _os.path.join(_tmpdir, "test_location/Test_sample_4.npz"),
        ],
        "class_weights": {"Label_1": {0: 4 / 3, 1: 2 / 3, 3: 4 / 3}},
        "one_hot_class_weights": {"Label_1": {0: 4 / 3, 1: 2 / 3, 3: 4 / 3}},
        "true_missing_value": None,
        "number_of_classes": {"Label_1": 3},
        "N_samples": 4,
        "sample_settings": _sample_settings,
    },
}

MULTI_BINARY_CLASS_MISSING_TEST = {
    "name": "multi binary class missing",
    "label_file": "test_data/NPZ_Data/labels_multi_binary_class_missing_values.txt",
    "expected_output": {
        "labels": [[1, 0], [-1, 1], [0, -1], [1, 1]],
        "category_labels": {"Label_1": [1, -1, 0, 1], "Label_2": [0, 1, -1, 1]},
        "one_hot_labels": [
            [[0, 0, 1], [0, 1, 0]],
            [[1, 0, 0], [0, 0, 1]],
            [[0, 1, 0], [1, 0, 0]],
            [[0, 0, 1], [0, 0, 1]],
        ],
        "one_hot_labels_filtered": [
            [[0, 1], [1, 0]],
            [[-1, -1], [0, 1]],
            [[1, 0], [-1, -1]],
            [[0, 1], [0, 1]],
        ],
        "one_hot_category_labels": {
            "Label_1": [[0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "Label_2": [[0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 0, 1]],
        },
        "one_hot_category_labels_filtered": {
            "Label_1": [[0, 1], [-1, -1], [1, 0], [0, 1]],
            "Label_2": [[1, 0], [0, 1], [-1, -1], [0, 1]],
        },
        "sample_locations": [
            "./PrognosAIs/tests/test_data/NPZ_Data/Samples/Test_sample_1.npz",
            "./PrognosAIs/tests/test_data/NPZ_Data/Samples/Test_sample_2.npz",
            "./PrognosAIs/tests/test_data/NPZ_Data/Samples/Test_sample_3.npz",
            "./PrognosAIs/tests/test_data/NPZ_Data/Samples/Test_sample_4.npz",
        ],
        "moved_sample_locations": [
            _os.path.join(_tmpdir, "test_location/Test_sample_1.npz"),
            _os.path.join(_tmpdir, "test_location/Test_sample_2.npz"),
            _os.path.join(_tmpdir, "test_location/Test_sample_3.npz"),
            _os.path.join(_tmpdir, "test_location/Test_sample_4.npz"),
        ],
        "class_weights": {
            "Label_1": {-1: 4 / 3, 0: 4 / 3, 1: 2 / 3},
            "Label_2": {-1: 4 / 3, 0: 4 / 3, 1: 2 / 3},
        },
        "class_weights_filtered": {
            "Label_1": {0: 3 / 2, 1: 3 / 4},
            "Label_2": {0: 3 / 2, 1: 3 / 4},
        },
        "one_hot_class_weights": {
            "Label_1": {0: 4 / 3, 1: 4 / 3, 2: 2 / 3},
            "Label_2": {0: 4 / 3, 1: 4 / 3, 2: 2 / 3},
        },
        "one_hot_class_weights_filtered": {
            "Label_1": {0: 3 / 2, 1: 3 / 4},
            "Label_2": {0: 3 / 2, 1: 3 / 4},
        },
        "true_missing_value": -1,
        "number_of_classes": {"Label_1": 3, "Label_2": 3},
        "number_of_classes_filtered": {"Label_1": 2, "Label_2": 2},
        "N_samples": 4,
        "sample_settings": _sample_settings,
    },
}


SINGLE_FILE_CLASS_TEST = {
    "name": "single file class",
    "label_file": "test_data/NPZ_Data/labels_file_single_class.txt",
    "expected_output": {
        "labels": [
            "./PrognosAIs/tests/test_data/NPZ_Data/Samples/Test_sample_1_label.npz",
            "./PrognosAIs/tests/test_data/NPZ_Data/Samples/Test_sample_2_label.npz",
        ],
        "moved_labels": [
            _os.path.join(_tmpdir, "test_location/Test_sample_1_label.npz"),
            _os.path.join(_tmpdir, "test_location/Test_sample_2_label.npz"),
        ],
        "category_labels": {
            "Label_1": [
                "./PrognosAIs/tests/test_data/NPZ_Data/Samples/Test_sample_1_label.npz",
                "./PrognosAIs/tests/test_data/NPZ_Data/Samples/Test_sample_2_label.npz",
            ]
        },
        "moved_category_labels": {
            "Label_1": [
                _os.path.join(_tmpdir, "test_location/Test_sample_1_label.npz"),
                _os.path.join(_tmpdir, "test_location/Test_sample_2_label.npz"),
            ]
        },
        "one_hot_labels": [
            "./PrognosAIs/tests/test_data/NPZ_Data/Samples/Test_sample_1_label.npz",
            "./PrognosAIs/tests/test_data/NPZ_Data/Samples/Test_sample_2_label.npz",
        ],
        "moved_one_hot_labels": [
            _os.path.join(_tmpdir, "test_location/Test_sample_1_label.npz"),
            _os.path.join(_tmpdir, "test_location/Test_sample_2_label.npz"),
        ],
        "one_hot_category_labels": {
            "Label_1": [
                "./PrognosAIs/tests/test_data/NPZ_Data/Samples/Test_sample_1_label.npz",
                "./PrognosAIs/tests/test_data/NPZ_Data/Samples/Test_sample_2_label.npz",
            ]
        },
        "moved_one_hot_category_labels": {
            "Label_1": [
                _os.path.join(_tmpdir, "test_location/Test_sample_1_label.npz"),
                _os.path.join(_tmpdir, "test_location/Test_sample_2_label.npz"),
            ]
        },
        "sample_locations": [
            "./PrognosAIs/tests/test_data/NPZ_Data/Samples/Test_sample_1.npz",
            "./PrognosAIs/tests/test_data/NPZ_Data/Samples/Test_sample_2.npz",
        ],
        "moved_sample_locations": [
            _os.path.join(_tmpdir, "test_location/Test_sample_1.npz"),
            _os.path.join(_tmpdir, "test_location/Test_sample_2.npz"),
        ],
        "class_weights": {"Label_1": None},
        "one_hot_class_weights": {"Label_1": None},
        "true_missing_value": -1,
        "number_of_classes": {"Label_1": -1},
        "N_samples": 2,
        "sample_settings": _sample_settings,
    },
}

SINGLE_FILE_AND_MULTI_CLASS_TEST = {
    "name": "single file and multi categorical class",
    "label_file": "test_data/NPZ_Data/labels_file_and_multi_class.txt",
    "expected_output": {
        "labels": [
            ["./PrognosAIs/tests/test_data/NPZ_Data/Samples/Test_sample_1_label.npz", 0, 1, 0],
            ["./PrognosAIs/tests/test_data/NPZ_Data/Samples/Test_sample_2_label.npz", 1, 0, 3],
            ["./PrognosAIs/tests/test_data/NPZ_Data/Samples/Test_sample_3_label.npz", 1, -1, 3],
            ["./PrognosAIs/tests/test_data/NPZ_Data/Samples/Test_sample_4_label.npz", 0, -1, 1],
            ["./PrognosAIs/tests/test_data/NPZ_Data/Samples/Test_sample_5_label.npz", -1, 0, 1],
            ["./PrognosAIs/tests/test_data/NPZ_Data/Samples/Test_sample_6_label.npz", 1, 1, 1],
        ],
        "moved_labels": [
            [_os.path.join(_tmpdir, "test_location/Test_sample_1_label.npz"), 0, 1, 0],
            [_os.path.join(_tmpdir, "test_location/Test_sample_2_label.npz"), 1, 0, 3],
            [_os.path.join(_tmpdir, "test_location/Test_sample_3_label.npz"), 1, -1, 3],
            [_os.path.join(_tmpdir, "test_location/Test_sample_4_label.npz"), 0, -1, 1],
            [_os.path.join(_tmpdir, "test_location/Test_sample_5_label.npz"), -1, 0, 1],
            [_os.path.join(_tmpdir, "test_location/Test_sample_6_label.npz"), 1, 1, 1],
        ],
        "category_labels": {
            "Label_1": [
                "./PrognosAIs/tests/test_data/NPZ_Data/Samples/Test_sample_1_label.npz",
                "./PrognosAIs/tests/test_data/NPZ_Data/Samples/Test_sample_2_label.npz",
                "./PrognosAIs/tests/test_data/NPZ_Data/Samples/Test_sample_3_label.npz",
                "./PrognosAIs/tests/test_data/NPZ_Data/Samples/Test_sample_4_label.npz",
                "./PrognosAIs/tests/test_data/NPZ_Data/Samples/Test_sample_5_label.npz",
                "./PrognosAIs/tests/test_data/NPZ_Data/Samples/Test_sample_6_label.npz",
            ],
            "Label_2": [0, 1, 1, 0, -1, 1],
            "Label_3": [1, 0, -1, -1, 0, 1],
            "Label_4": [0, 3, 3, 1, 1, 1],
        },
        "moved_category_labels": {
            "Label_1": [
                _os.path.join(_tmpdir, "test_location/Test_sample_1_label.npz"),
                _os.path.join(_tmpdir, "test_location/Test_sample_2_label.npz"),
                _os.path.join(_tmpdir, "test_location/Test_sample_3_label.npz"),
                _os.path.join(_tmpdir, "test_location/Test_sample_4_label.npz"),
                _os.path.join(_tmpdir, "test_location/Test_sample_5_label.npz"),
                _os.path.join(_tmpdir, "test_location/Test_sample_6_label.npz"),
            ],
            "Label_2": [0, 1, 1, 0, -1, 1],
            "Label_3": [1, 0, -1, -1, 0, 1],
            "Label_4": [0, 3, 3, 1, 1, 1],
        },
        "one_hot_labels": [
            [
                "./PrognosAIs/tests/test_data/NPZ_Data/Samples/Test_sample_1_label.npz",
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 0, 0],
            ],
            [
                "./PrognosAIs/tests/test_data/NPZ_Data/Samples/Test_sample_2_label.npz",
                [0, 0, 1],
                [0, 1, 0],
                [0, 0, 0, 1],
            ],
            [
                "./PrognosAIs/tests/test_data/NPZ_Data/Samples/Test_sample_3_label.npz",
                [0, 0, 1],
                [1, 0, 0],
                [0, 0, 0, 1],
            ],
            [
                "./PrognosAIs/tests/test_data/NPZ_Data/Samples/Test_sample_4_label.npz",
                [0, 1, 0],
                [1, 0, 0],
                [0, 1, 0, 0],
            ],
            [
                "./PrognosAIs/tests/test_data/NPZ_Data/Samples/Test_sample_5_label.npz",
                [1, 0, 0],
                [0, 1, 0],
                [0, 1, 0, 0],
            ],
            [
                "./PrognosAIs/tests/test_data/NPZ_Data/Samples/Test_sample_6_label.npz",
                [0, 0, 1],
                [0, 0, 1],
                [0, 1, 0, 0],
            ],
        ],
        "moved_one_hot_labels": [
            [
                _os.path.join(_tmpdir, "test_location/Test_sample_1_label.npz"),
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 0, 0],
            ],
            [
                _os.path.join(_tmpdir, "test_location/Test_sample_2_label.npz"),
                [0, 0, 1],
                [0, 1, 0],
                [0, 0, 0, 1],
            ],
            [
                _os.path.join(_tmpdir, "test_location/Test_sample_3_label.npz"),
                [0, 0, 1],
                [1, 0, 0],
                [0, 0, 0, 1],
            ],
            [
                _os.path.join(_tmpdir, "test_location/Test_sample_4_label.npz"),
                [0, 1, 0],
                [1, 0, 0],
                [0, 1, 0, 0],
            ],
            [
                _os.path.join(_tmpdir, "test_location/Test_sample_5_label.npz"),
                [1, 0, 0],
                [0, 1, 0],
                [0, 1, 0, 0],
            ],
            [
                _os.path.join(_tmpdir, "test_location/Test_sample_6_label.npz"),
                [0, 0, 1],
                [0, 0, 1],
                [0, 1, 0, 0],
            ],
        ],
        "one_hot_labels_filtered": [
            [
                "./PrognosAIs/tests/test_data/NPZ_Data/Samples/Test_sample_1_label.npz",
                [1, 0],
                [0, 1],
                [1, 0, 0, 0],
            ],
            [
                "./PrognosAIs/tests/test_data/NPZ_Data/Samples/Test_sample_2_label.npz",
                [0, 1],
                [1, 0],
                [0, 0, 0, 1],
            ],
            [
                "./PrognosAIs/tests/test_data/NPZ_Data/Samples/Test_sample_3_label.npz",
                [0, 1],
                [-1, -1],
                [0, 0, 0, 1],
            ],
            [
                "./PrognosAIs/tests/test_data/NPZ_Data/Samples/Test_sample_4_label.npz",
                [1, 0],
                [-1, -1],
                [0, 1, 0, 0],
            ],
            [
                "./PrognosAIs/tests/test_data/NPZ_Data/Samples/Test_sample_5_label.npz",
                [-1, -1],
                [1, 0],
                [0, 1, 0, 0],
            ],
            [
                "./PrognosAIs/tests/test_data/NPZ_Data/Samples/Test_sample_6_label.npz",
                [0, 1],
                [0, 1],
                [0, 1, 0, 0],
            ],
        ],
        "moved_one_hot_labels_filtered": [
            [
                _os.path.join(_tmpdir, "test_location/Test_sample_1_label.npz"),
                [1, 0],
                [0, 1],
                [1, 0, 0, 0],
            ],
            [
                _os.path.join(_tmpdir, "test_location/Test_sample_2_label.npz"),
                [0, 1],
                [1, 0],
                [0, 0, 0, 1],
            ],
            [
                _os.path.join(_tmpdir, "test_location/Test_sample_3_label.npz"),
                [0, 1],
                [-1, -1],
                [0, 0, 0, 1],
            ],
            [
                _os.path.join(_tmpdir, "test_location/Test_sample_4_label.npz"),
                [1, 0],
                [-1, -1],
                [0, 1, 0, 0],
            ],
            [
                _os.path.join(_tmpdir, "test_location/Test_sample_5_label.npz"),
                [-1, -1],
                [1, 0],
                [0, 1, 0, 0],
            ],
            [
                _os.path.join(_tmpdir, "test_location/Test_sample_6_label.npz"),
                [0, 1],
                [0, 1],
                [0, 1, 0, 0],
            ],
        ],
        "one_hot_category_labels": {
            "Label_1": [
                "./PrognosAIs/tests/test_data/NPZ_Data/Samples/Test_sample_1_label.npz",
                "./PrognosAIs/tests/test_data/NPZ_Data/Samples/Test_sample_2_label.npz",
                "./PrognosAIs/tests/test_data/NPZ_Data/Samples/Test_sample_3_label.npz",
                "./PrognosAIs/tests/test_data/NPZ_Data/Samples/Test_sample_4_label.npz",
                "./PrognosAIs/tests/test_data/NPZ_Data/Samples/Test_sample_5_label.npz",
                "./PrognosAIs/tests/test_data/NPZ_Data/Samples/Test_sample_6_label.npz",
            ],
            "Label_2": [[0, 1, 0], [0, 0, 1], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 1]],
            "Label_3": [[0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "Label_4": [
                [1, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 0, 1],
                [0, 1, 0, 0],
                [0, 1, 0, 0],
                [0, 1, 0, 0],
            ],
        },
        "moved_one_hot_category_labels": {
            "Label_1": [
                _os.path.join(_tmpdir, "test_location/Test_sample_1_label.npz"),
                _os.path.join(_tmpdir, "test_location/Test_sample_2_label.npz"),
                _os.path.join(_tmpdir, "test_location/Test_sample_3_label.npz"),
                _os.path.join(_tmpdir, "test_location/Test_sample_4_label.npz"),
                _os.path.join(_tmpdir, "test_location/Test_sample_5_label.npz"),
                _os.path.join(_tmpdir, "test_location/Test_sample_6_label.npz"),
            ],
            "Label_2": [[0, 1, 0], [0, 0, 1], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 1]],
            "Label_3": [[0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "Label_4": [
                [1, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 0, 1],
                [0, 1, 0, 0],
                [0, 1, 0, 0],
                [0, 1, 0, 0],
            ],
        },
        "one_hot_category_labels_filtered": {
            "Label_1": [
                "./PrognosAIs/tests/test_data/NPZ_Data/Samples/Test_sample_1_label.npz",
                "./PrognosAIs/tests/test_data/NPZ_Data/Samples/Test_sample_2_label.npz",
                "./PrognosAIs/tests/test_data/NPZ_Data/Samples/Test_sample_3_label.npz",
                "./PrognosAIs/tests/test_data/NPZ_Data/Samples/Test_sample_4_label.npz",
                "./PrognosAIs/tests/test_data/NPZ_Data/Samples/Test_sample_5_label.npz",
                "./PrognosAIs/tests/test_data/NPZ_Data/Samples/Test_sample_6_label.npz",
            ],
            "Label_2": [[1, 0], [0, 1], [0, 1], [1, 0], [-1, -1], [0, 1]],
            "Label_3": [[0, 1], [1, 0], [-1, -1], [-1, -1], [1, 0], [0, 1]],
            "Label_4": [
                [1, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 0, 1],
                [0, 1, 0, 0],
                [0, 1, 0, 0],
                [0, 1, 0, 0],
            ],
        },
        "moved_one_hot_category_labels_filtered": {
            "Label_1": [
                _os.path.join(_tmpdir, "test_location/Test_sample_1_label.npz"),
                _os.path.join(_tmpdir, "test_location/Test_sample_2_label.npz"),
                _os.path.join(_tmpdir, "test_location/Test_sample_3_label.npz"),
                _os.path.join(_tmpdir, "test_location/Test_sample_4_label.npz"),
                _os.path.join(_tmpdir, "test_location/Test_sample_5_label.npz"),
                _os.path.join(_tmpdir, "test_location/Test_sample_6_label.npz"),
            ],
            "Label_2": [[1, 0], [0, 1], [0, 1], [1, 0], [-1, -1], [0, 1]],
            "Label_3": [[0, 1], [1, 0], [-1, -1], [-1, -1], [1, 0], [0, 1]],
            "Label_4": [
                [1, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 0, 1],
                [0, 1, 0, 0],
                [0, 1, 0, 0],
                [0, 1, 0, 0],
            ],
        },
        "sample_locations": [
            "./PrognosAIs/tests/test_data/NPZ_Data/Samples/Test_sample_1.npz",
            "./PrognosAIs/tests/test_data/NPZ_Data/Samples/Test_sample_2.npz",
            "./PrognosAIs/tests/test_data/NPZ_Data/Samples/Test_sample_3.npz",
            "./PrognosAIs/tests/test_data/NPZ_Data/Samples/Test_sample_4.npz",
            "./PrognosAIs/tests/test_data/NPZ_Data/Samples/Test_sample_5.npz",
            "./PrognosAIs/tests/test_data/NPZ_Data/Samples/Test_sample_6.npz",
        ],
        "moved_sample_locations": [
            _os.path.join(_tmpdir, "test_location/Test_sample_1.npz"),
            _os.path.join(_tmpdir, "test_location/Test_sample_2.npz"),
            _os.path.join(_tmpdir, "test_location/Test_sample_3.npz"),
            _os.path.join(_tmpdir, "test_location/Test_sample_4.npz"),
            _os.path.join(_tmpdir, "test_location/Test_sample_5.npz"),
            _os.path.join(_tmpdir, "test_location/Test_sample_6.npz"),
        ],
        "class_weights": {
            "Label_1": None,
            "Label_2": {-1: 2, 0: 1, 1: 2 / 3},
            "Label_3": {-1: 1, 0: 1, 1: 1},
            "Label_4": {0: 2, 1: 2 / 3, 3: 1},
        },
        "class_weights_filtered": {
            "Label_1": None,
            "Label_2": {0: 5 / 4, 1: 5 / 6},
            "Label_3": {0: 1, 1: 1},
            "Label_4": {0: 2, 1: 2 / 3, 3: 1},
        },
        "one_hot_class_weights": {
            "Label_1": None,
            "Label_2": {0: 2, 1: 1, 2: 2 / 3},
            "Label_3": {0: 1, 1: 1, 2: 1},
            "Label_4": {0: 2, 1: 2 / 3, 3: 1},
        },
        "one_hot_class_weights_filtered": {
            "Label_1": None,
            "Label_2": {0: 5 / 4, 1: 5 / 6},
            "Label_3": {0: 1, 1: 1},
            "Label_4": {0: 2, 1: 2 / 3, 3: 1},
        },
        "true_missing_value": -1,
        "number_of_classes": {"Label_1": -1, "Label_2": 3, "Label_3": 3, "Label_4": 3},
        "number_of_classes_filtered": {"Label_1": -1, "Label_2": 2, "Label_3": 2, "Label_4": 3},
        "N_samples": 6,
        "sample_settings": _sample_settings,
    },
}

MULTI_CATEGORICAL_CLASS_MISSING_TEST = {
    "name": "multi categorical class missing",
    "label_file": "test_data/NPZ_Data/labels_multi_categorical_class_missing_values.txt",
    "expected_output": {
        "labels": [[1, 0], [-1, 1], [2, -1], [4, 2], [-1, 2], [1, -1]],
        "category_labels": {"Label_1": [1, -1, 2, 4, -1, 1], "Label_2": [0, 1, -1, 2, 2, -1]},
        "one_hot_labels": [
            [[0, 0, 1, 0, 0, 0], [0, 1, 0, 0]],
            [[1, 0, 0, 0, 0, 0], [0, 0, 1, 0]],
            [[0, 0, 0, 1, 0, 0], [1, 0, 0, 0]],
            [[0, 0, 0, 0, 0, 1], [0, 0, 0, 1]],
            [[1, 0, 0, 0, 0, 0], [0, 0, 0, 1]],
            [[0, 0, 1, 0, 0, 0], [1, 0, 0, 0]],
        ],
        "one_hot_labels_filtered": [
            [[1, 0, 0, 0], [1, 0, 0]],
            [[-1, -1, -1, -1], [0, 1, 0]],
            [[0, 1, 0, 0], [-1, -1, -1]],
            [[0, 0, 0, 1], [0, 0, 1]],
            [[-1, -1, -1, -1], [0, 0, 1]],
            [[1, 0, 0, 0], [-1, -1, -1]],
        ],
        "one_hot_category_labels": {
            "Label_1": [
                [0, 0, 1, 0, 0, 0],
                [1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
            ],
            "Label_2": [
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 0, 1],
                [1, 0, 0, 0],
            ],
        },
        "one_hot_category_labels_filtered": {
            "Label_1": [
                [1, 0, 0, 0],
                [-1, -1, -1, -1],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [-1, -1, -1, -1],
                [1, 0, 0, 0],
            ],
            "Label_2": [[1, 0, 0], [0, 1, 0], [-1, -1, -1], [0, 0, 1], [0, 0, 1], [-1, -1, -1]],
        },
        "sample_locations": [
            "./PrognosAIs/tests/test_data/NPZ_Data/Samples/Test_sample_1.npz",
            "./PrognosAIs/tests/test_data/NPZ_Data/Samples/Test_sample_2.npz",
            "./PrognosAIs/tests/test_data/NPZ_Data/Samples/Test_sample_3.npz",
            "./PrognosAIs/tests/test_data/NPZ_Data/Samples/Test_sample_4.npz",
            "./PrognosAIs/tests/test_data/NPZ_Data/Samples/Test_sample_5.npz",
            "./PrognosAIs/tests/test_data/NPZ_Data/Samples/Test_sample_6.npz",
        ],
        "moved_sample_locations": [
            _os.path.join(_tmpdir, "test_location/Test_sample_1.npz"),
            _os.path.join(_tmpdir, "test_location/Test_sample_2.npz"),
            _os.path.join(_tmpdir, "test_location/Test_sample_3.npz"),
            _os.path.join(_tmpdir, "test_location/Test_sample_4.npz"),
            _os.path.join(_tmpdir, "test_location/Test_sample_5.npz"),
            _os.path.join(_tmpdir, "test_location/Test_sample_6.npz"),
        ],
        "class_weights": {
            "Label_1": {-1: 3 / 4, 1: 3 / 4, 2: 3 / 2, 4: 3 / 2},
            "Label_2": {-1: 3 / 4, 0: 3 / 2, 1: 3 / 2, 2: 3 / 4},
        },
        "class_weights_filtered": {
            "Label_1": {1: 2 / 3, 2: 4 / 3, 4: 4 / 3},
            "Label_2": {0: 4 / 3, 1: 4 / 3, 2: 2 / 3},
        },
        "one_hot_class_weights": {
            "Label_1": {0: 3 / 4, 2: 3 / 4, 3: 3 / 2, 5: 3 / 2},
            "Label_2": {0: 3 / 4, 1: 3 / 2, 2: 3 / 2, 3: 3 / 4},
        },
        "one_hot_class_weights_filtered": {
            "Label_1": {0: 2 / 3, 1: 4 / 3, 3: 4 / 3},
            "Label_2": {0: 4 / 3, 1: 4 / 3, 2: 2 / 3},
        },
        "true_missing_value": -1,
        "number_of_classes": {"Label_1": 4, "Label_2": 4},
        "number_of_classes_filtered": {"Label_1": 3, "Label_2": 3},
        "N_samples": 6,
        "sample_settings": _sample_settings,
    },
}

SINGLE_REGRESSION_CLASS_TEST = {
    "name": "single regression class",
    "label_file": "test_data/NPZ_Data/labels_single_regression_class.txt",
    "expected_output": {
        "labels": [10.0, 112.3994, 15.93013, -1031.00321],
        "category_labels": {"Label_1": [10.0, 112.3994, 15.93013, -1031.00321]},
        "one_hot_labels": [10.0, 112.3994, 15.93013, -1031.00321],
        "one_hot_category_labels": {"Label_1": [10.0, 112.3994, 15.93013, -1031.00321]},
        "sample_locations": [
            "./PrognosAIs/tests/test_data/NPZ_Data/Samples/Test_sample_1.npz",
            "./PrognosAIs/tests/test_data/NPZ_Data/Samples/Test_sample_2.npz",
            "./PrognosAIs/tests/test_data/NPZ_Data/Samples/Test_sample_3.npz",
            "./PrognosAIs/tests/test_data/NPZ_Data/Samples/Test_sample_4.npz",
        ],
        "moved_sample_locations": [
            _os.path.join(_tmpdir, "test_location/Test_sample_1.npz"),
            _os.path.join(_tmpdir, "test_location/Test_sample_2.npz"),
            _os.path.join(_tmpdir, "test_location/Test_sample_3.npz"),
            _os.path.join(_tmpdir, "test_location/Test_sample_4.npz"),
        ],
        "class_weights": {"Label_1": None},
        "one_hot_class_weights": {"Label_1": None},
        "true_missing_value": None,
        "number_of_classes": {"Label_1": -1},
        "N_samples": 4,
        "sample_settings": _sample_settings,
    },
}
