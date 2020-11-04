import os
import tempfile

import numpy as np
import pytest
import SimpleITK as sitk
import tensorflow as tf

from PrognosAIs.Model import Losses
from PrognosAIs.Model import Metrics
from PrognosAIs.Model.Architectures import Architecture
from PrognosAIs.Model.Evaluators import Evaluator
import PrognosAIs.Constants


SAMPLES_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "test_data", "HDF5_Data", "Samples",
)

SAMPLES_DIR_PATCHES = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "test_data", "HDF5_Data_mask_patches", "Samples",
)

MODEL_FILE = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "test_data", "Model", "Test_model.hdf5",
)

MODEL_FILE_PATCHES = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "test_data", "Model", "Test_model_patches.hdf5",
)


CONFIG_FILE = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "test_data", "test_config.yaml",
)

CONFIG_FILE_NO_METRICS = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "test_data", "test_config_no_metrics.yaml",
)


class mock_architecture(Architecture.NetworkArchitecture):
    def make_outputs(self, output_info, output_data_type):
        return tf.keras.layers.Dense(2, name=list(output_info.keys())[0], activation="softmax")

    def create_model(self):
        inputs = self.make_inputs(self.input_shapes, self.input_data_type)
        flatten = tf.keras.layers.Flatten()(inputs)
        predictions = self.make_outputs(self.output_info, self.output_data_type)(flatten)
        return tf.keras.Model(inputs=inputs, outputs=predictions)


class mock_multi_input_multi_output(Architecture.ClassificationNetworkArchitecture):
    def create_model(self):
        inputs = self.make_inputs(self.input_shapes, self.input_data_type, squeeze_inputs=False)
        outputs = self.make_outputs(self.output_info, self.output_data_type, squeeze_outputs=False)
        input_branches = []

        for i_input in inputs.values():
            input_branches.append(tf.keras.layers.Conv2D(filters=4, kernel_size=(2, 2))(i_input))

        if len(input_branches) > 1:
            concat_1 = tf.keras.layers.Concatenate()(input_branches)
        else:
            concat_1 = input_branches[0]

        flatten_1 = tf.keras.layers.Flatten()(concat_1)

        predictions = []
        for i_output in outputs.values():
            predictions.append(i_output(flatten_1))

        model = tf.keras.Model(inputs=inputs, outputs=predictions)

        return model


def test_evaluator_init():
    samples_folder = SAMPLES_DIR
    model_file = MODEL_FILE
    tmp = tempfile.mkdtemp()

    result = Evaluator(model_file, samples_folder, CONFIG_FILE, tmp)

    assert isinstance(result, Evaluator)
    assert isinstance(result.model, tf.keras.Model)
    assert result.data_folder == samples_folder
    assert result.output_names == [PrognosAIs.Constants.LABEL_INDEX]
    assert result.input_names == [PrognosAIs.Constants.FEATURE_INDEX]
    assert isinstance(result.data_generator, dict)
    assert "train" in result.data_generator
    assert isinstance(result.label_data_generator, dict)
    assert "train" in result.label_data_generator


def test_init_from_cli():
    tmp = tempfile.mkdtemp()

    result = Evaluator.init_from_sys_args(
        ["--config", CONFIG_FILE, "--input", SAMPLES_DIR, "--output", tmp, "--model", MODEL_FILE]
    )

    assert isinstance(result, Evaluator)
    assert isinstance(result.model, tf.keras.Model)


def test_model_loading_custom_loss():
    model = mock_architecture(
        {PrognosAIs.Constants.FEATURE_INDEX: [10, 10, 1]}, {PrognosAIs.Constants.LABEL_INDEX: 2}
    ).create_model()
    model.compile(loss=Losses.MaskedCategoricalCrossentropy(), metrics=["accuracy"])
    tmp = tempfile.mkdtemp()
    model_file = os.path.join(tmp, "model.hdf5")
    model.save(model_file)

    result = Evaluator.load_model(model_file)

    assert isinstance(result, tf.keras.Model)
    assert isinstance(result.loss, Losses.MaskedCategoricalCrossentropy)


def test_model_loading_custom_metric():
    model = mock_architecture(
        {PrognosAIs.Constants.FEATURE_INDEX: [10, 10, 1]}, {PrognosAIs.Constants.LABEL_INDEX: 2}
    ).create_model()
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[Metrics.MaskedCategoricalAccuracy()],
    )
    tmp = tempfile.mkdtemp()
    model_file = os.path.join(tmp, "model.hdf5")
    model.save(model_file)

    result = Evaluator.load_model(model_file)

    assert isinstance(result, tf.keras.Model)
    assert isinstance(result.loss, tf.keras.losses.CategoricalCrossentropy)
    assert isinstance(result.metrics[1], Metrics.MaskedCategoricalAccuracy)


def test_model_loading_multiple_custom_loss_and_custom_metric():
    model = mock_architecture(
        {PrognosAIs.Constants.FEATURE_INDEX: [10, 10, 1]}, {PrognosAIs.Constants.LABEL_INDEX: 2}
    ).create_model()
    model.compile(
        loss=Losses.MaskedCategoricalCrossentropy(),
        metrics=[Metrics.MaskedCategoricalAccuracy(), Metrics.MaskedAUC()],
    )
    tmp = tempfile.mkdtemp()
    model_file = os.path.join(tmp, "model.hdf5")
    model.save(model_file)

    result = Evaluator.load_model(model_file)

    assert isinstance(result, tf.keras.Model)
    assert isinstance(result.loss, Losses.MaskedCategoricalCrossentropy)
    assert isinstance(result.metrics[1], Metrics.MaskedCategoricalAccuracy)
    assert isinstance(result.metrics[2], Metrics.MaskedAUC)


def test_model_loading_multi_input_multi_output_multiple_custom_loss_and_custom_metric():
    model = mock_multi_input_multi_output(
        {"feature_1": [10, 10, 1], "feature_2": [10, 10, 1]}, {"label_1": 2, "label_2": 2}
    ).create_model()
    model.compile(
        loss=Losses.MaskedCategoricalCrossentropy(),
        metrics=[Metrics.MaskedCategoricalAccuracy(mask_value=-2), Metrics.MaskedAUC()],
    )
    tmp = tempfile.mkdtemp()
    model_file = os.path.join(tmp, "model.hdf5")
    model.save(model_file)

    result = Evaluator.load_model(model_file)

    assert isinstance(result, tf.keras.Model)
    assert isinstance(result.loss, Losses.MaskedCategoricalCrossentropy)


def test_evaluator_real_label_loading():
    tmp = tempfile.mkdtemp()
    evaluator = Evaluator(MODEL_FILE, SAMPLES_DIR, CONFIG_FILE, tmp)

    result = evaluator.get_real_labels_of_sample_subset("train")

    assert isinstance(result, dict)
    assert PrognosAIs.Constants.LABEL_INDEX in result
    assert isinstance(result[PrognosAIs.Constants.LABEL_INDEX], np.ndarray)
    assert result[PrognosAIs.Constants.LABEL_INDEX] == pytest.approx(
        np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]]),
    )


def test_get_metrics():
    model = mock_architecture(
        {PrognosAIs.Constants.FEATURE_INDEX: [10, 10, 1]}, {PrognosAIs.Constants.LABEL_INDEX: 2}
    ).create_model()
    model.compile(
        loss=Losses.MaskedCategoricalCrossentropy(),
        metrics=[Metrics.MaskedCategoricalAccuracy(mask_value=-2), Metrics.MaskedAUC()],
    )
    tmp = tempfile.mkdtemp()
    model_file = os.path.join(tmp, "model.hdf5")
    model.save(model_file)
    evaluator = Evaluator(model_file, SAMPLES_DIR, CONFIG_FILE, tmp)

    result = evaluator.get_to_evaluate_metrics()

    assert isinstance(result, dict)
    assert "label" in result
    assert isinstance(result["label"], list)
    metric_names = [i_metric.name for i_metric in result["label"]]
    assert Metrics.MaskedAUC().name in metric_names
    assert Metrics.MaskedCategoricalAccuracy(mask_value=-2).name in metric_names
    assert tf.keras.metrics.CategoricalAccuracy().name in metric_names
    assert len(metric_names) == 3


def test_get_metrics_multi_output():
    model = mock_multi_input_multi_output(
        {PrognosAIs.Constants.FEATURE_INDEX: [10, 10, 1]}, {"label_1": 2, "label_2": 2}
    ).create_model()
    model.compile(
        loss=Losses.MaskedCategoricalCrossentropy(),
        metrics={
            "label_1": [Metrics.MaskedCategoricalAccuracy(mask_value=-2)],
            "label_2": [Metrics.MaskedAUC()],
        },
    )
    tmp = tempfile.mkdtemp()
    model_file = os.path.join(tmp, "model.hdf5")
    model.save(model_file)
    evaluator = Evaluator(model_file, SAMPLES_DIR, CONFIG_FILE, tmp)

    result = evaluator.get_to_evaluate_metrics()

    assert isinstance(result, dict)
    assert "label_1" in result
    assert isinstance(result["label_1"], list)
    assert "label_2" in result
    assert isinstance(result["label_2"], list)
    label_metric_names = [i_metric.name for i_metric in result["label_1"]]
    label_2_metric_names = [i_metric.name for i_metric in result["label_2"]]
    assert "label_1_" + Metrics.MaskedCategoricalAccuracy(mask_value=-2).name in label_metric_names
    assert tf.keras.metrics.CategoricalAccuracy().name in label_metric_names
    assert len(label_metric_names) == 2
    assert "label_2_" + Metrics.MaskedAUC().name in label_2_metric_names
    assert tf.keras.metrics.CategoricalAccuracy().name in label_2_metric_names
    assert len(label_2_metric_names) == 2


def test_get_metrics_no_metrics():
    model = mock_architecture(
        {PrognosAIs.Constants.FEATURE_INDEX: [10, 10, 1]}, {PrognosAIs.Constants.LABEL_INDEX: 2}
    ).create_model()
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy())
    tmp = tempfile.mkdtemp()
    model_file = os.path.join(tmp, "model.hdf5")
    model.save(model_file)
    evaluator = Evaluator(model_file, SAMPLES_DIR, CONFIG_FILE_NO_METRICS, tmp)

    result = evaluator.get_to_evaluate_metrics()

    assert isinstance(result, dict)
    assert PrognosAIs.Constants.LABEL_INDEX in result
    assert isinstance(result[PrognosAIs.Constants.LABEL_INDEX], list)
    assert result[PrognosAIs.Constants.LABEL_INDEX] == []


def test_evaluator_predictions():
    tmp = tempfile.mkdtemp()
    evaluator = Evaluator(MODEL_FILE, SAMPLES_DIR, CONFIG_FILE, tmp)

    result = evaluator.predict()

    assert isinstance(result, dict)
    assert isinstance(evaluator.predictions, dict)
    assert evaluator.predictions == result
    assert "train" in evaluator.predictions
    assert isinstance(evaluator.predictions["train"], dict)
    assert "label" in evaluator.predictions["train"]
    assert isinstance(evaluator.predictions["train"][PrognosAIs.Constants.LABEL_INDEX], np.ndarray)
    assert evaluator.predictions["train"][PrognosAIs.Constants.LABEL_INDEX].shape == (6, 3)


def test_evaluator_evaluate_metrics_no_metrics():
    model = mock_architecture(
        {PrognosAIs.Constants.FEATURE_INDEX: [30, 30, 30, 4]}, {PrognosAIs.Constants.LABEL_INDEX: 2}
    ).create_model()
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy())
    tmp = tempfile.mkdtemp()
    model_file = os.path.join(tmp, "model.hdf5")
    model.save(model_file)
    evaluator = Evaluator(model_file, SAMPLES_DIR, CONFIG_FILE_NO_METRICS, tmp)

    result = evaluator.evaluate_metrics()

    assert isinstance(result, dict)
    assert result == evaluator.metrics
    assert isinstance(evaluator.metrics, dict)
    assert "train" in evaluator.metrics
    assert isinstance(evaluator.metrics["train"], dict)
    assert "label" in evaluator.metrics["train"]
    assert isinstance(evaluator.metrics["train"][PrognosAIs.Constants.LABEL_INDEX], dict)
    assert evaluator.metrics["train"][PrognosAIs.Constants.LABEL_INDEX] == {}


def test_evaluator_evaluate_metrics():
    tmp = tempfile.mkdtemp()
    evaluator = Evaluator(MODEL_FILE, SAMPLES_DIR, CONFIG_FILE, tmp)

    result = evaluator.evaluate_metrics()

    assert isinstance(result, dict)
    assert result == evaluator.metrics
    assert isinstance(evaluator.metrics, dict)
    assert "train" in evaluator.metrics
    assert isinstance(evaluator.metrics["train"], dict)
    assert "label" in evaluator.metrics["train"]
    assert isinstance(evaluator.metrics["train"][PrognosAIs.Constants.LABEL_INDEX], dict)
    assert "categorical_accuracy" in evaluator.metrics["train"][PrognosAIs.Constants.LABEL_INDEX]
    assert isinstance(
        evaluator.metrics["train"][PrognosAIs.Constants.LABEL_INDEX]["categorical_accuracy"],
        np.float32,
    )


def test_prediction_combination_one_hot_majority():
    predictions = np.asarray([[0.7, 0.3], [0.6, 0.4], [0.8, 0.2], [0.1, 0.9], [1, 0]])
    are_one_hot = True
    combination_type = "vote"

    result = Evaluator.combine_predictions(predictions, are_one_hot, combination_type)

    assert isinstance(result, np.ndarray)
    assert result == pytest.approx(np.asarray([1, 0]))


def test_prediction_combination_one_hot_mean():
    predictions = np.asarray([[0.7, 0.3], [0.6, 0.4], [0.8, 0.2], [0.1, 0.9], [1, 0]])
    are_one_hot = True
    combination_type = "average"

    result = Evaluator.combine_predictions(predictions, are_one_hot, combination_type)

    assert isinstance(result, np.ndarray)
    assert result == pytest.approx(np.asarray([0.64, 0.36]))


def test_prediction_combination_one_hot_multioutput_majority():
    predictions = np.asarray(
        [[0.5, 0.3, 0.2], [0.6, 0.2, 0.2], [0.6, 0.2, 0.2], [0.1, 0.3, 0.6], [1, 0, 0]]
    )
    are_one_hot = True
    combination_type = "vote"

    result = Evaluator.combine_predictions(predictions, are_one_hot, combination_type)

    assert isinstance(result, np.ndarray)
    assert result == pytest.approx(np.asarray([1, 0, 0]))


def test_prediction_combination_majority():
    predictions = np.asarray([1, 2, 3, 4, 1, 1, 2, 3, 1])
    are_one_hot = False
    combination_type = "vote"

    result = Evaluator.combine_predictions(predictions, are_one_hot, combination_type)

    assert result == pytest.approx(np.asarray(1))


def test_evaluator_write_to_file():
    samples_folder = SAMPLES_DIR
    model_file = MODEL_FILE
    tmp = tempfile.mkdtemp()
    evaluator = Evaluator(model_file, samples_folder, CONFIG_FILE, tmp)

    result = evaluator.write_predictions_to_file()

    assert result is None
    assert os.path.exists(os.path.join(tmp, "Results", "train_predictions.csv"))
    assert os.path.exists(os.path.join(tmp, "Results", "train_predictions_combined_patches.csv"))


def test_evaluator_write_metrics_to_file():
    samples_folder = SAMPLES_DIR
    model_file = MODEL_FILE
    tmp = tempfile.mkdtemp()
    evaluator = Evaluator(model_file, samples_folder, CONFIG_FILE, tmp)

    result = evaluator.write_metrics_to_file()

    assert result is None
    assert os.path.exists(os.path.join(tmp, "Results", "metrics.csv"))


def test_evaluator_write_metrics_to_file_no_metrics():
    model = mock_architecture(
        {PrognosAIs.Constants.FEATURE_INDEX: [30, 30, 30, 4]}, {PrognosAIs.Constants.LABEL_INDEX: 2}
    ).create_model()
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy())
    tmp = tempfile.mkdtemp()
    model_file = os.path.join(tmp, "model.hdf5")
    model.save(model_file)
    evaluator = Evaluator(model_file, SAMPLES_DIR, CONFIG_FILE_NO_METRICS, tmp)

    result = evaluator.write_metrics_to_file()

    assert result is None
    assert os.path.exists(os.path.join(tmp, "Results", "metrics.csv"))


def test_patches_to_image_single_patch():
    samples_folder = SAMPLES_DIR
    model_file = MODEL_FILE_PATCHES
    tmp = tempfile.mkdtemp()
    evaluator = Evaluator(model_file, samples_folder, CONFIG_FILE, tmp)
    generator = evaluator.data_generator["train"]
    filenames = [generator.sample_locations[0]]
    output_name = PrognosAIs.Constants.LABEL_INDEX
    predictions = np.zeros([1, 30, 30, 30])
    labels_are_one_hot = False

    result = evaluator.patches_to_sample_image(
        generator, filenames, output_name, predictions, labels_are_one_hot, "vote"
    )

    assert isinstance(result, np.ndarray)


def test_get_image_output_label():
    samples_folder = SAMPLES_DIR_PATCHES
    model_file = MODEL_FILE_PATCHES
    tmp = tempfile.mkdtemp()
    evaluator = Evaluator(model_file, samples_folder, CONFIG_FILE, tmp)

    result = evaluator.get_image_output_labels()

    assert isinstance(result, dict)
    assert PrognosAIs.Constants.LABEL_INDEX in result
    assert result[PrognosAIs.Constants.LABEL_INDEX] == PrognosAIs.Constants.FEATURE_INDEX


def test_patches_to_image_multi_patch():
    samples_folder = SAMPLES_DIR_PATCHES
    model_file = MODEL_FILE_PATCHES
    tmp = tempfile.mkdtemp()
    evaluator = Evaluator(model_file, samples_folder, CONFIG_FILE, tmp)

    (sample_files, sample_predictions,) = evaluator.get_sample_predictions_from_patch_predictions()

    assert isinstance(sample_predictions, dict)
    assert "train" in sample_predictions
    assert isinstance(sample_predictions["train"], dict)
    assert PrognosAIs.Constants.LABEL_INDEX in sample_predictions["train"]
    assert isinstance(sample_predictions["train"][PrognosAIs.Constants.LABEL_INDEX], np.ndarray)
    assert sample_predictions["train"][PrognosAIs.Constants.LABEL_INDEX].shape == (3, 30, 30, 30)


def test_evaluator_write_to_file_patches():
    samples_folder = SAMPLES_DIR_PATCHES
    model_file = MODEL_FILE_PATCHES
    tmp = tempfile.mkdtemp()
    evaluator = Evaluator(model_file, samples_folder, CONFIG_FILE, tmp)

    result = evaluator.write_predictions_to_file()

    assert result is None
    assert os.path.exists(os.path.join(tmp, "Results", "Subject-000_prediction.nii.gz"))
    assert os.path.exists(os.path.join(tmp, "Results", "Subject-000_patch_0_prediction.nii.gz"))
