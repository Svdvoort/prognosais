import pytest
import numpy as np
import tensorflow as tf

from PrognosAIs.Model import Metrics
import sklearn.metrics


def test_dice_init():
    result = Metrics.DICE()

    assert isinstance(result, Metrics.DICE)


def test_dice_init_with_background():
    result = Metrics.DICE(foreground_only=False)

    assert isinstance(result, Metrics.DICE)
    assert not result.foreground_only


def test_dice_single_sample_all_correct_one_hot():
    y_true = np.asarray([1, 1, 1, 1])
    y_pred = np.asarray([1, 1, 1, 1])
    y_true = tf.one_hot(y_true, 2, dtype=tf.int8)
    y_pred = tf.one_hot(y_pred, 2, dtype=tf.float32)
    dice_metric = Metrics.DICE()
    dice_metric.update_state(y_true, y_pred)

    result = dice_metric.result()

    assert isinstance(result, tf.Tensor)
    assert tf.rank(result) == 0
    assert result.numpy() == pytest.approx(1)


def test_dice_single_sample_all_correct_one_hot():
    y_true = np.asarray([[1, 1, 1, 1]])
    y_pred = np.asarray([[1, 1, 1, 1]])
    y_true = tf.one_hot(y_true, 2, dtype=tf.int8)
    y_pred = tf.one_hot(y_pred, 2, dtype=tf.float32)
    dice_metric = Metrics.DICE()
    dice_metric.update_state(y_true, y_pred)

    result = dice_metric.result()

    assert isinstance(result, tf.Tensor)
    assert tf.rank(result) == 0
    assert result.numpy() == pytest.approx(1)
    assert dice_metric.total_samples.numpy() == pytest.approx(1)


def test_dice_single_sample_all_incorrect_one_hot():
    y_true = np.asarray([[1, 1, 1, 1]])
    y_pred = np.asarray([[0, 0, 0, 0]])
    y_true = tf.one_hot(y_true, 2, dtype=tf.int8)
    y_pred = tf.one_hot(y_pred, 2, dtype=tf.float32)
    dice_metric = Metrics.DICE()
    dice_metric.update_state(y_true, y_pred)

    result = dice_metric.result()

    assert isinstance(result, tf.Tensor)
    assert tf.rank(result) == 0
    assert result.numpy() == pytest.approx(0)
    assert dice_metric.total_samples.numpy() == pytest.approx(1)


def test_dice_multi_sample_all_correct_one_hot():
    y_true = np.asarray([[1, 1, 1, 1], [0, 0, 0, 0], [1, 1, 0, 0]])
    y_pred = np.asarray([[1, 1, 1, 1], [0, 0, 0, 0], [1, 1, 0, 0]])
    y_true = tf.one_hot(y_true, 2, dtype=tf.int8)
    y_pred = tf.one_hot(y_pred, 2, dtype=tf.float32)
    dice_metric = Metrics.DICE()
    dice_metric.update_state(y_true, y_pred)

    result = dice_metric.result()

    assert isinstance(result, tf.Tensor)
    assert tf.rank(result) == 0
    assert result.numpy() == pytest.approx(1)
    assert dice_metric.total_samples.numpy() == pytest.approx(3)


def test_dice_multi_sample_all_incorrect_one_hot():
    y_true = np.asarray([[1, 1, 1, 1], [0, 0, 0, 0], [1, 1, 0, 0]])
    y_pred = np.asarray([[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 1, 1]])
    y_true = tf.one_hot(y_true, 2, dtype=tf.int8)
    y_pred = tf.one_hot(y_pred, 2, dtype=tf.float32)
    dice_metric = Metrics.DICE()
    dice_metric.update_state(y_true, y_pred)

    result = dice_metric.result()

    assert isinstance(result, tf.Tensor)
    assert tf.rank(result) == 0
    assert result.numpy() == pytest.approx(0)
    assert dice_metric.total_samples.numpy() == pytest.approx(3)


def test_dice_multi_sample_mixed_correct_one_hot():
    y_true = np.asarray([[1, 1, 1, 1], [0, 0, 0, 0], [1, 1, 0, 0]])
    y_pred = np.asarray([[0, 0, 1, 1], [1, 0, 0, 1], [0, 1, 1, 0]])
    y_true = tf.one_hot(y_true, 2, dtype=tf.int8)
    y_pred = tf.one_hot(y_pred, 2, dtype=tf.float32)
    dice_metric = Metrics.DICE()
    dice_metric.update_state(y_true, y_pred)

    result = dice_metric.result()

    assert isinstance(result, tf.Tensor)
    assert tf.rank(result) == 0
    assert result.numpy() == pytest.approx(7 / 18)
    assert dice_metric.total_samples.numpy() == pytest.approx(3)


def test_dice_multi_sample_multi_call_mixed_correct_one_hot():
    y_true_1 = np.asarray([[1, 1, 1, 1], [0, 0, 0, 0], [1, 1, 0, 0]])
    y_pred_1 = np.asarray([[0, 0, 1, 1], [1, 0, 0, 1], [0, 1, 1, 0]])
    y_true_1 = tf.one_hot(y_true_1, 2, dtype=tf.int8)
    y_pred_1 = tf.one_hot(y_pred_1, 2, dtype=tf.float32)
    y_true_2 = np.asarray([[0, 1, 0, 1], [0, 1, 0, 0], [1, 1, 0, 0]])
    y_pred_2 = np.asarray([[0, 1, 0, 1], [1, 1, 0, 1], [0, 1, 0, 1]])
    y_true_2 = tf.one_hot(y_true_2, 2, dtype=tf.int8)
    y_pred_2 = tf.one_hot(y_pred_2, 2, dtype=tf.float32)
    dice_metric = Metrics.DICE()
    dice_metric.update_state(y_true_1, y_pred_1)
    dice_metric.update_state(y_true_2, y_pred_2)
    dice_1 = 7 / 18
    dice_2 = 2 / 3

    result = dice_metric.result()

    assert isinstance(result, tf.Tensor)
    assert tf.rank(result) == 0
    assert result.numpy() == pytest.approx((dice_1 + dice_2) / 2)
    assert dice_metric.total_samples.numpy() == pytest.approx(6)


def test_dice_multi_sample_multi_call_with_clear_mixed_correct_one_hot():
    y_true_1 = np.asarray([[1, 1, 1, 1], [0, 0, 0, 0], [1, 1, 0, 0]])
    y_pred_1 = np.asarray([[0, 0, 1, 1], [1, 0, 0, 1], [0, 1, 1, 0]])
    y_true_1 = tf.one_hot(y_true_1, 2, dtype=tf.int8)
    y_pred_1 = tf.one_hot(y_pred_1, 2, dtype=tf.float32)
    y_true_2 = np.asarray([[0, 1, 0, 1], [0, 1, 0, 0], [1, 1, 0, 0]])
    y_pred_2 = np.asarray([[0, 1, 0, 1], [1, 1, 0, 1], [0, 1, 0, 1]])
    y_true_2 = tf.one_hot(y_true_2, 2, dtype=tf.int8)
    y_pred_2 = tf.one_hot(y_pred_2, 2, dtype=tf.float32)
    dice_metric = Metrics.DICE()
    dice_1 = 7 / 18
    dice_2 = 2 / 3

    dice_metric.update_state(y_true_1, y_pred_1)
    result_1 = dice_metric.result()

    assert isinstance(result_1, tf.Tensor)
    assert tf.rank(result_1) == 0
    assert result_1.numpy() == pytest.approx(dice_1)
    assert dice_metric.total_samples.numpy() == pytest.approx(3)

    dice_metric.reset_states()
    dice_metric.update_state(y_true_2, y_pred_2)
    result_2 = dice_metric.result()

    assert isinstance(result_2, tf.Tensor)
    assert tf.rank(result_2) == 0
    assert result_2.numpy() == pytest.approx(dice_2)
    assert dice_metric.total_samples.numpy() == pytest.approx(3)


def test_dice_multi_sample_multi_call_mixed_correct_one_hot_pobabilities():
    y_true_1 = np.asarray([[1, 1, 1, 1], [0, 0, 0, 0], [1, 1, 0, 0]])
    y_pred_1 = np.asarray(
        [
            [[0.8, 0.2], [0.7, 0.3], [0.1, 0.9], [0.4, 0.6]],
            [[0.1, 0.9], [1.0, 0.0], [0.75, 0.25], [0.12, 0.88]],
            [[0.7, 0.3], [0.2, 0.8], [0.4, 0.6], [0.90, 0.1]],
        ]
    )
    y_true_1 = tf.one_hot(y_true_1, 2, dtype=tf.int8)
    y_pred_1 = tf.convert_to_tensor(y_pred_1, tf.float32)
    y_true_2 = np.asarray([[0, 1, 0, 1], [0, 1, 0, 0], [1, 1, 0, 0]])
    y_pred_2 = np.asarray(
        [
            [[0.9, 0.1], [0.45, 0.55], [0.56, 0.44], [0.13, 0.87]],
            [[0.1, 0.9], [0.4, 0.6], [0.99, 0.01], [0.05, 0.95]],
            [[1.0, 0.0], [0.15, 0.85], [0.51, 0.49], [0.10, 0.90]],
        ]
    )
    y_true_2 = tf.one_hot(y_true_2, 2, dtype=tf.int8)
    y_pred_2 = tf.convert_to_tensor(y_pred_2, tf.float32)
    dice_metric = Metrics.DICE()
    dice_metric.update_state(y_true_1, y_pred_1)
    dice_metric.update_state(y_true_2, y_pred_2)
    dice_1 = 7 / 18
    dice_2 = 2 / 3

    result = dice_metric.result()

    assert isinstance(result, tf.Tensor)
    assert tf.rank(result) == 0
    assert result.numpy() == pytest.approx((dice_1 + dice_2) / 2)
    assert dice_metric.total_samples.numpy() == pytest.approx(6)


def test_dice_serializable():
    metric_function = Metrics.DICE(foreground_only=True)

    result = tf.keras.losses.serialize(metric_function)

    assert isinstance(result, dict)
    assert result["config"]["foreground_only"]


def test_dice_deserializable():
    metric_function = Metrics.DICE

    result = tf.keras.metrics.deserialize("DICE", custom_objects={"DICE": metric_function},)

    assert isinstance(result, Metrics.DICE)


# ===============================================================
# AUC
# ===============================================================


def test_auc_no_missing_one_hot_perfect():
    y_true = [[0, 1], [1, 0], [1, 0]]
    y_pred = [[0.0, 1.0], [1.0, 0.0], [1.0, 0.0]]
    metric_function = Metrics.MaskedAUC()
    metric_function.update_state(y_true, y_pred)

    result = metric_function.result()

    assert result.numpy() == pytest.approx(1)


def test_auc_no_missing_one_hot():
    y_true = [[1, 0], [1, 0], [0, 1], [0, 1]]
    y_pred = [[0.9, 0.1], [0.6, 0.4], [0.65, 0.35], [0.2, 0.8]]
    # y_true_flat = np.asarray([1, 0, 0])
    # y_pred_flat = np.asarray([0.7, 0.1, 0.6])
    y_true_flat = np.array([0, 0, 1, 1])
    y_pred_flat = np.array([0.1, 0.4, 0.35, 0.8])
    sklearn_result = sklearn.metrics.roc_auc_score(y_true_flat, y_pred_flat)
    metric_function = Metrics.MaskedAUC()
    metric_function.update_state(y_true, y_pred)

    result = metric_function.result().numpy()

    assert result.size == 1
    assert result.size == sklearn_result.size
    assert result == pytest.approx(sklearn_result)


def test_auc_missing_one_hot():
    y_true = [[1, 0], [1, 0], [-1, -1], [0, 1], [0, 1], [-1, -1]]
    y_pred = [[0.9, 0.1], [0.6, 0.4], [1.0, 0.0], [0.65, 0.35], [0.2, 0.8], [0.0, 1.0]]
    # y_true_flat = np.asarray([1, 0, 0])
    # y_pred_flat = np.asarray([0.7, 0.1, 0.6])
    y_true_flat = np.array([0, 0, 1, 1])
    y_pred_flat = np.array([0.1, 0.4, 0.35, 0.8])
    sklearn_result = sklearn.metrics.roc_auc_score(y_true_flat, y_pred_flat)
    metric_function = Metrics.MaskedAUC(mask_value=-1)
    metric_function.update_state(y_true, y_pred)

    result = metric_function.result()
    assert result.numpy() == pytest.approx(sklearn_result)


# ===============================================================
# Specificity
# ===============================================================


def test_masked_specificity_no_missing():
    y_true = [[1, 0], [1, 0], [0, 1], [0, 1], [1, 0]]
    y_pred = [[0.7, 0.3], [0.6, 0.4], [0.3, 0.7], [0.2, 0.8], [0.1, 0.9]]

    metric_function = Metrics.MaskedSpecificity(mask_value=-1)
    metric_function.update_state(y_true, y_pred)

    result = metric_function.result().numpy()

    assert result.size == 1
    assert result == pytest.approx(2 / 3)


# ===============================================================
# Sensitivity
# ===============================================================


def test_mask_sensitivity_no_missing_perfect():
    y_true = [[1, 0], [1, 0], [0, 1], [0, 1], [1, 0]]
    y_pred = [[0.7, 0.3], [0.6, 0.4], [0.3, 0.7], [0.2, 0.8], [0.1, 0.9]]

    metric_function = Metrics.MaskedSensitivity(mask_value=-1)
    metric_function.update_state(y_true, y_pred)

    result = metric_function.result().numpy()

    assert result.size == 1
    assert result == pytest.approx(1)


def test_mask_sensitivity_no_missing():
    y_true = [[1, 0], [0, 1], [0, 1], [0, 1], [1, 0]]
    y_pred = [[0.7, 0.3], [0.6, 0.4], [0.3, 0.7], [0.2, 0.8], [0.1, 0.9]]

    metric_function = Metrics.MaskedSensitivity(mask_value=-1)
    metric_function.update_state(y_true, y_pred)

    result = metric_function.result().numpy()

    assert result.size == 1
    assert result == pytest.approx(2 / 3)


def test_mask_sensitivity_missing_no_TP():
    y_true = [[1, 0], [0, 1], [0, 1], [0, 1], [-1, -1]]
    y_pred = [[0.7, 0.3], [0.6, 0.4], [0.3, 0.7], [0.2, 0.8], [0.1, 0.9]]

    metric_function = Metrics.MaskedSensitivity(mask_value=-1)
    metric_function.update_state(y_true, y_pred)

    result = metric_function.result().numpy()

    assert result.size == 1
    assert result == pytest.approx(2 / 3)


def test_mask_sensitivity_missing_TP():
    y_true = [[1, 0], [0, 1], [-1, -1], [0, 1], [1, 0]]
    y_pred = [[0.7, 0.3], [0.6, 0.4], [0.3, 0.7], [0.2, 0.8], [0.1, 0.9]]

    metric_function = Metrics.MaskedSensitivity(mask_value=-1)
    metric_function.update_state(y_true, y_pred)

    result = metric_function.result().numpy()

    assert result.size == 1
    assert result == pytest.approx(1 / 2)
