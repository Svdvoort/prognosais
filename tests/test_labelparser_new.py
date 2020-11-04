import pytest
import os
import numpy as np

from PrognosAIs.IO import LabelParser

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_data", "NPZ_Data",)


def test_one_hot_encoding_missing_labels():
    label_file = os.path.join(FIXTURE_DIR, "labels_multi_binary_class_missing_values.txt")
    label_parser = LabelParser.LabelLoader(label_file, filter_missing=True, missing_value=-1)
    label_parser.encode_labels_one_hot()

    result = label_parser.get_labels_from_category("Label_1")

    assert result == pytest.approx(np.asarray([[0, 1], [-1, -1], [1, 0], [0, 1]]))
