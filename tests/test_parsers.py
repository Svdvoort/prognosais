import os

import tensorflow.keras.metrics
import tensorflow as tf

from PrognosAIs.Model.Parsers import LossParser, MetricParser


CUSTOM_DEFINITIONS_FILE = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "test_data", "MyDefinitions.py",
)


def test_metric_parser_multi_metrics():
    metric_config = {
        "accuracy": {"name": "Accuracy", "is_custom": False, "settings": None},
        "auc": {"name": "AUC", "is_custom": False, "settings": None},
    }
    labels = ["Class_1", "Class_2"]
    parser = MetricParser(metric_config, labels)

    metrics = parser.get_metrics()

    assert isinstance(metrics, list)
    assert isinstance(metrics[0], tensorflow.keras.metrics.Accuracy)
    assert isinstance(metrics[1], tensorflow.keras.metrics.AUC)


def test_metric_parser_multi_class():
    metric_config = {
        "Class_1": {"name": "Accuracy", "is_custom": False, "settings": None},
        "Class_2": {"name": "AUC", "is_custom": False, "settings": None},
    }
    labels = ["Class_1", "Class_2"]
    parser = MetricParser(metric_config, labels)

    metrics = parser.get_metrics()

    assert isinstance(metrics, dict)
    assert "Class_1" in metrics and "Class_2" in metrics
    assert isinstance(metrics["Class_1"], tensorflow.keras.metrics.Accuracy)
    assert isinstance(metrics["Class_2"], tensorflow.keras.metrics.AUC)


def test_loss_parsers_custom_loss_from_file():
    loss_config = {"name": "TestLoss", "settings": None}

    definitions_file = CUSTOM_DEFINITIONS_FILE
    parser = LossParser(loss_config, module_paths=[definitions_file])

    result = parser.get_losses()

    assert isinstance(result, tf.keras.losses.Loss)
    assert result.__class__.__name__ == "TestLoss"


def test_loss_parsers_custom_loss_from_file_and_tensorflow_loss():
    # TODO Replace this by a settings without "Settings"
    loss_config = {
        "label_1": {"name": "TestLoss", "settings": None},
        "label_2": {"name": "BinaryCrossentropy", "settings": None},
    }

    definitions_file = CUSTOM_DEFINITIONS_FILE
    parser = LossParser(loss_config, module_paths=[definitions_file])

    result = parser.get_losses()

    assert isinstance(result, dict)
    assert "label_1" in result
    assert isinstance(result["label_1"], tf.keras.losses.Loss)
    assert result["label_1"].__class__.__name__ == "TestLoss"

    assert "label_2" in result
    assert isinstance(result["label_2"], tf.keras.losses.Loss)
    assert result["label_2"].__class__.__name__ == "BinaryCrossentropy"
