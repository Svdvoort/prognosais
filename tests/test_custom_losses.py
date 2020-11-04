import numpy as np
import pytest
import tensorflow as tf
import tensorflow.keras.losses

from PrognosAIs.Model.Losses import DICE_loss
from PrognosAIs.Model.Losses import CoxLoss
from PrognosAIs.Model.Losses import MaskedCategoricalCrossentropy


# ===============================================================
# Masked Categorical Crossentropy
# ===============================================================


def test_maskedcategoricalcrossentropy_no_masks():
    y_true = [[1, 0], [0, 1], [1, 0], [0, 1], [0, 1], [1, 0], [1, 0], [1, 0], [1, 0]]
    y_pred = [
        [0.8, 0.2],
        [0.3, 0.7],
        [0.1, 0.9],
        [0.8, 0.2],
        [0.3, 0.7],
        [0.1, 0.9],
        [0.5, 0.5],
        [0.5, 0.5],
        [0.5, 0.5],
    ]
    true_loss = tensorflow.keras.losses.CategoricalCrossentropy().call(y_true, y_pred)
    loss_function = MaskedCategoricalCrossentropy()

    result = loss_function.call(y_true, y_pred)

    assert isinstance(loss_function, MaskedCategoricalCrossentropy)
    assert tf.rank(result) == 1
    assert result.numpy() == pytest.approx(true_loss.numpy())


def test_maskedcategoricalcrossentropy_no_masks_total_loss():
    y_true = [[1, 0], [0, 1], [1, 0], [0, 1], [0, 1], [1, 0], [1, 0], [1, 0], [1, 0]]
    y_pred = [
        [0.8, 0.2],
        [0.3, 0.7],
        [0.1, 0.9],
        [0.8, 0.2],
        [0.3, 0.7],
        [0.1, 0.9],
        [0.5, 0.5],
        [0.5, 0.5],
        [0.5, 0.5],
    ]
    y_true = tf.convert_to_tensor(y_true)
    y_pred = tf.convert_to_tensor(y_pred)
    true_loss = tensorflow.keras.losses.CategoricalCrossentropy().__call__(y_true, y_pred)
    loss_function = MaskedCategoricalCrossentropy()

    result = loss_function.__call__(y_true, y_pred)

    assert isinstance(loss_function, MaskedCategoricalCrossentropy)
    assert tf.rank(result) == 0
    assert result.numpy() == pytest.approx(true_loss.numpy())


def test_maskedcategoricalcrossentropy_no_masks_multi_dim():
    y_true = [[[1, 0], [0, 1], [1, 0]], [[0, 1], [0, 1], [1, 0]], [[1, 0], [1, 0], [1, 0]]]
    y_pred = [
        [[0.8, 0.2], [0.3, 0.7], [0.1, 0.9]],
        [[0.8, 0.2], [0.3, 0.7], [0.1, 0.9]],
        [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]],
    ]
    true_loss = tensorflow.keras.losses.CategoricalCrossentropy().call(y_true, y_pred)
    loss_function = MaskedCategoricalCrossentropy()

    result = loss_function.call(y_true, y_pred)

    assert isinstance(loss_function, MaskedCategoricalCrossentropy)
    assert tf.rank(result) == 2
    assert result.numpy() == pytest.approx(true_loss.numpy())


def test_maskedcategoricalcrossentropy():
    y_true = [[1, 0], [0, 1], [1, 0], [0, 0], [0, 1], [1, 0], [-1, -1], [-1, -1], [-1, -1]]
    y_pred = [
        [0.8, 0.2],
        [0.3, 0.7],
        [0.1, 0.9],
        [0.8, 0.2],
        [0.3, 0.7],
        [0.1, 0.9],
        [0.5, 0.5],
        [0.5, 0.5],
        [0.5, 0.5],
    ]
    is_masked = [False, False, False, True, False, False, True, True, True]
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    is_masked = np.asarray(is_masked)
    y_true_masked = y_true[np.logical_not(is_masked)]
    y_pred_masked = y_pred[np.logical_not(is_masked)]
    true_loss = tensorflow.keras.losses.CategoricalCrossentropy().call(y_true, y_pred)
    true_masked_loss = tensorflow.keras.losses.CategoricalCrossentropy().call(
        y_true_masked, y_pred_masked,
    )
    loss_function = MaskedCategoricalCrossentropy(mask_value=-1)

    result = loss_function.call(y_true, y_pred)

    assert tf.is_tensor(result)
    assert tf.rank(result) == 1
    assert tf.shape(result) == tf.shape(true_loss)
    assert np.all(result.numpy() >= 0)
    assert result.shape[0] == len(y_true)
    assert np.all(result.numpy()[is_masked] == 0)
    assert result.numpy()[np.logical_not(is_masked)] == pytest.approx(true_masked_loss.numpy())


def test_maskedcategoricalcrossentropy_total_loss():
    y_true = [[1, 0], [0, 1], [1, 0], [-1, -1], [0, 1], [1, 0], [-1, -1], [-1, -1], [-1, -1]]
    y_pred = [
        [0.8, 0.2],
        [0.3, 0.7],
        [0.1, 0.9],
        [0.8, 0.2],
        [0.3, 0.7],
        [0.1, 0.9],
        [0.5, 0.5],
        [0.5, 0.5],
        [0.5, 0.5],
    ]
    is_masked = [False, False, False, True, False, False, True, True, True]
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    is_masked = np.asarray(is_masked)
    y_true_masked = y_true[np.logical_not(is_masked)]
    y_pred_masked = y_pred[np.logical_not(is_masked)]
    true_loss = tensorflow.keras.losses.CategoricalCrossentropy().__call__(
        y_true_masked, y_pred_masked,
    )
    loss_function = MaskedCategoricalCrossentropy()

    result = loss_function.__call__(y_true, y_pred)

    assert tf.is_tensor(result)
    assert tf.rank(result) == 0
    assert np.all(result.numpy() >= 0)
    assert result.numpy() == pytest.approx(true_loss.numpy())


def test_maskedcategoricalcrossentropy_class_weights():
    y_true = [[1, 0], [0, 1], [1, 0], [-1, -1], [0, 1], [1, 0], [-1, -1], [-1, -1], [-1, -1]]
    y_pred = [
        [0.8, 0.2],
        [0.3, 0.7],
        [0.1, 0.9],
        [0.8, 0.2],
        [0.3, 0.7],
        [0.1, 0.9],
        [0.5, 0.5],
        [0.5, 0.5],
        [0.5, 0.5],
    ]
    is_masked = [False, False, False, True, False, False, True, True, True]
    class_weights = {1: 5, 0: 1}
    sample_weights = np.asarray([1, 5, 1, 0, 5, 1, 0, 0, 0])
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    is_masked = np.asarray(is_masked)
    y_true_masked = y_true[np.logical_not(is_masked)]
    y_pred_masked = y_pred[np.logical_not(is_masked)]
    sample_weights_masked = sample_weights[np.logical_not(is_masked)]
    true_loss = tensorflow.keras.losses.CategoricalCrossentropy().call(
        y_true_masked, y_pred_masked,
    )
    loss_function = MaskedCategoricalCrossentropy(class_weight=class_weights)

    result = loss_function.call(y_true, y_pred)

    assert tf.is_tensor(result)
    assert tf.rank(result) == 1
    assert tf.reduce_all(tf.math.greater_equal(result, 0))
    assert result.numpy()[np.logical_not(is_masked)] == pytest.approx(
        true_loss.numpy() * sample_weights_masked,
    )


def test_maskedcategoricalcrossentropy_total_loss_class_weights():
    y_true = [[1, 0], [0, 1], [1, 0], [-1, -1], [0, 1], [1, 0], [-1, -1], [-1, -1], [-1, -1]]
    y_pred = [
        [0.8, 0.2],
        [0.3, 0.7],
        [0.1, 0.9],
        [0.8, 0.2],
        [0.3, 0.7],
        [0.1, 0.9],
        [0.5, 0.5],
        [0.5, 0.5],
        [0.5, 0.5],
    ]
    is_masked = [False, False, False, True, False, False, True, True, True]
    class_weights = {1: 5, 0: 1}
    sample_weights = np.asarray([1, 5, 1, 0, 5, 1, 0, 0, 0])
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    is_masked = np.asarray(is_masked)
    y_true_masked = y_true[np.logical_not(is_masked)]
    y_pred_masked = y_pred[np.logical_not(is_masked)]
    sample_weights_masked = sample_weights[np.logical_not(is_masked)]
    true_loss = tensorflow.keras.losses.CategoricalCrossentropy().__call__(
        y_true_masked, y_pred_masked, sample_weight=sample_weights_masked,
    )
    loss_function = MaskedCategoricalCrossentropy(class_weight=class_weights)

    result = loss_function.__call__(y_true, y_pred)

    assert tf.is_tensor(result)
    assert tf.rank(result) == 0
    assert np.all(result.numpy() >= 0)
    assert result.numpy() == pytest.approx(true_loss.numpy())


def test_maskedcategoricalcrossentropy_total_loss_sample_weights():
    y_true = [[1, 0], [0, 1], [1, 0], [-1, -1], [0, 1], [1, 0], [-1, -1], [-1, -1], [-1, -1]]
    y_pred = [
        [0.8, 0.2],
        [0.3, 0.7],
        [0.1, 0.9],
        [0.8, 0.2],
        [0.3, 0.7],
        [0.1, 0.9],
        [0.5, 0.5],
        [0.5, 0.5],
        [0.5, 0.5],
    ]
    is_masked = [False, False, False, True, False, False, True, True, True]
    sample_weights = np.asarray([1, 5, 1, 0, 5, 1, 0, 0, 0])
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    is_masked = np.asarray(is_masked)
    y_true_masked = y_true[np.logical_not(is_masked)]
    y_pred_masked = y_pred[np.logical_not(is_masked)]
    sample_weights_masked = sample_weights[np.logical_not(is_masked)]
    true_loss = tensorflow.keras.losses.CategoricalCrossentropy().__call__(
        y_true_masked, y_pred_masked, sample_weight=sample_weights_masked,
    )
    loss_function = MaskedCategoricalCrossentropy()

    result = loss_function.__call__(y_true, y_pred, sample_weight=sample_weights)

    assert tf.is_tensor(result)
    assert tf.rank(result) == 0
    assert np.all(result.numpy() >= 0)
    assert result.numpy() == pytest.approx(true_loss.numpy())


def test_maskedcategoricalcrossentropy_serializable():
    loss_function = MaskedCategoricalCrossentropy(mask_value=3, class_weight={0: 0.5, 1: 317})

    result = tf.keras.losses.serialize(loss_function)

    assert isinstance(result, dict)
    assert result["config"]["mask_value"] == 3
    assert result["config"]["class_weight"] == {"0": "0.5", "1": "317"}


def test_maskedcategoricalcrossentropy_deserializable():
    loss_function = MaskedCategoricalCrossentropy

    result = tf.keras.losses.deserialize(
        "MaskedCategoricalCrossentropy",
        custom_objects={"MaskedCategoricalCrossentropy": loss_function},
    )

    assert isinstance(result, MaskedCategoricalCrossentropy)


# ===============================================================
# DICE_loss score
# ===============================================================
def test_DICE():

    loss_function = DICE_loss()
    assert isinstance(loss_function, DICE_loss)

    y_true = [
        [[[1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1]]],
        # Sample 2
        [[[1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1]]],
        # Sample 3
        [[[1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1]]],
    ]

    y_pred = [
        [[[0.0, 0, 0], [0, 0, 0]], [[1.0, 1, 1], [1, 1, 1]]],
        # Sample 2
        [[[1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1]]],
        # Sample 3
        [[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]],
    ]

    # Add a dimension for the channels
    y_true = np.expand_dims(y_true, -1)
    y_pred = np.expand_dims(y_pred, -1)
    true_loss = [1 / 3, 0, 1]
    losses = loss_function.call(y_true, y_pred)
    assert tf.is_tensor(losses)
    assert tf.rank(losses) == 1
    losses = losses.numpy()
    assert losses.shape[0] == len(y_true)
    assert losses == pytest.approx(np.asarray(true_loss))

    total_loss = loss_function.__call__(y_true, y_pred)

    assert tf.is_tensor(total_loss)
    assert tf.rank(total_loss) == 0
    total_loss = total_loss.numpy()

    assert total_loss == pytest.approx(np.mean(losses))


def test_DICE_one_hot():
    loss_function = DICE_loss()
    assert isinstance(loss_function, DICE_loss)

    y_true = [
        [[[1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1]]],
        # Sample 2
        [[[1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1]]],
        # Sample 3
        [[[1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1]]],
    ]

    y_pred = [
        [[[0, 0, 0], [0, 0, 0]], [[1, 1, 1], [1, 1, 1]]],
        # Sample 2
        [[[1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1]]],
        # Sample 3
        [[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]],
    ]

    # Add a dimension for the channels
    y_true = tf.cast(tf.one_hot(y_true, 2), tf.float32)
    y_pred = tf.cast(tf.one_hot(y_pred, 2), tf.float32)
    true_loss_foreground = [1 / 3, 0, 1]
    true_loss_background = [0, 0, 0]
    true_loss = (np.asarray(true_loss_foreground) + np.asarray(true_loss_background)) / 2.0
    losses = loss_function.call(y_true, y_pred)
    assert tf.is_tensor(losses)
    assert tf.rank(losses) == 1
    losses = losses.numpy()
    assert losses.shape[0] == len(y_true)
    assert losses == pytest.approx(np.asarray(true_loss))

    total_loss = loss_function.__call__(y_true, y_pred)

    assert tf.is_tensor(total_loss)
    assert tf.rank(total_loss) == 0
    total_loss = total_loss.numpy()

    assert total_loss == pytest.approx(np.mean(losses))


def test_DICE_one_hot_multi_class():
    loss_function = DICE_loss()
    assert isinstance(loss_function, DICE_loss)

    y_true = [
        [[[1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1]]],
        # Sample 2
        [[[1, 1, 1], [2, 2, 2]], [[1, 1, 1], [1, 1, 1]]],
        # Sample 3
        [[[2, 2, 2], [2, 2, 2]], [[2, 2, 2], [2, 2, 2]]],
    ]

    y_pred = [
        [[[0, 0, 0], [0, 0, 0]], [[1, 1, 1], [1, 1, 1]]],
        # Sample 2
        [[[1, 1, 1], [2, 2, 2]], [[1, 1, 1], [1, 1, 1]]],
        # Sample 3
        [[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]],
    ]

    # Add a dimension for the channels
    # y_true = np.expand_dims(y_true, -1)
    y_true = tf.cast(tf.one_hot(y_true, 3), tf.float32)
    y_pred = tf.cast(tf.one_hot(y_pred, 3), tf.float32)
    true_loss_class0 = [0, 0, 0]
    true_loss_class1 = [1 / 3, 0, 0]
    true_loss_class2 = [0, 0, 1]
    true_loss = (
        1
        / 3
        * (
            np.asarray(true_loss_class0)
            + np.asarray(true_loss_class1)
            + np.asarray(true_loss_class2)
        )
    )
    losses = loss_function.call(y_true, y_pred)
    assert tf.is_tensor(losses)
    assert tf.rank(losses) == 1
    losses = losses.numpy()
    assert losses.shape[0] == len(y_true)
    assert losses == pytest.approx(np.asarray(true_loss))

    total_loss = loss_function.__call__(y_true, y_pred)

    assert tf.is_tensor(total_loss)
    assert tf.rank(total_loss) == 0
    total_loss = total_loss.numpy()

    assert total_loss == pytest.approx(np.mean(losses))


def test_DICE_zeros():

    loss_function = DICE_loss()
    assert isinstance(loss_function, DICE_loss)

    y_true = np.zeros([3, 5, 5, 5, 1])
    y_pred = np.zeros([3, 5, 5, 5, 1])

    true_loss = [0, 0, 0]
    losses = loss_function.call(y_true, y_pred)
    assert tf.is_tensor(losses)
    assert tf.rank(losses) == 1
    losses = losses.numpy()
    assert losses.shape[0] == len(y_true)
    assert losses == pytest.approx(np.asarray(true_loss))

    total_loss = loss_function.__call__(y_true, y_pred)

    assert tf.is_tensor(total_loss)
    assert tf.rank(total_loss) == 0
    total_loss = total_loss.numpy()

    assert total_loss == pytest.approx(np.mean(losses))


def test_dice_serializable():
    loss_function = DICE_loss()

    result = tf.keras.losses.serialize(loss_function)

    assert isinstance(result, dict)


def test_dice_deserializable():
    loss_function = DICE_loss

    result = tf.keras.losses.deserialize("DICE_loss", custom_objects={"DICE_loss": loss_function},)

    assert isinstance(result, DICE_loss)


def test_DICE_one_hot_multi_class_weighted():
    loss_function = DICE_loss(weighted=True)
    assert isinstance(loss_function, DICE_loss)

    # We use only 1 sample, one-dimensional
    y_true = [[2, 1, 1, 0, 0, 2, 0, 1, 0]]
    y_pred = [[1, 1, 0, 0, 0, 2, 1, 1, 0]]

    # Add a dimension for the channels
    # y_true = np.expand_dims(y_true, -1)
    y_true = tf.cast(tf.one_hot(y_true, 3), tf.float32)
    y_pred = tf.cast(tf.one_hot(y_pred, 3), tf.float32)
    true_loss = np.asarray([1 / 4, 3 / 7, 1 / 3])
    weights = np.asarray([9 / 4, 9 / 3, 9 / 2])
    weights = weights / np.sum(weights)
    true_loss = 1 / 3 * np.sum(true_loss * weights)
    losses = loss_function.call(y_true, y_pred)
    assert tf.is_tensor(losses)
    assert tf.rank(losses) == 1
    assert losses.numpy() == pytest.approx(np.asarray([true_loss]))


def test_DICE_one_hot_foreground_only_multi_class_weighted():
    loss_function = DICE_loss(foreground_only=True)
    assert isinstance(loss_function, DICE_loss)

    # We use only 1 sample, one-dimensional
    y_true = [[0, 1, 1, 0, 0, 1, 1, 1, 0]]
    y_pred = [[1, 1, 0, 0, 0, 1, 1, 1, 0]]

    # Add a dimension for the channels
    # y_true = np.expand_dims(y_true, -1)
    y_true = tf.cast(tf.one_hot(y_true, 2), tf.float32)
    y_pred = tf.cast(tf.one_hot(y_pred, 2), tf.float32)
    true_loss = np.asarray([1 / 5])
    losses = loss_function.call(y_true, y_pred)

    assert tf.is_tensor(losses)
    assert tf.rank(losses) == 1
    assert losses.numpy() == pytest.approx(true_loss)


# ===============================================================
# COX loss
# ===============================================================


def test_cox_loss_perfect_prediction():
    y_true = tf.constant([[1, 15], [1, 30], [1, 45], [1, 60]])
    y_pred = tf.constant([25.0, 12.0, 4.0, 0.0])
    true_output = np.zeros([4])
    loss_function = CoxLoss()

    result = loss_function.call(y_true, y_pred)

    assert isinstance(result, tf.Tensor)
    assert tf.rank(result) == 1
    assert result.numpy() == pytest.approx(true_output, abs=1e-1)


def test_cox_loss_overflow():
    y_true = tf.constant([[1, 15], [1, 30], [1, 45], [1, 60]])
    y_pred = tf.constant([999.0, 12.0, 4.0, 0.0])
    true_output = np.zeros([4])
    loss_function = CoxLoss()

    result = loss_function.call(y_true, y_pred).numpy()

    assert not np.any(np.isinf(result))
    assert result == pytest.approx(true_output, abs=1e-1)


def test_cox_loss_wrong_prediction():
    y_true = tf.constant([[1, 15], [1, 30], [1, 45], [1, 60]])
    y_pred = tf.constant([0, 4.0, 12.0, 25.0])
    true_output = np.array([3, 2, 1, 0])
    loss_function = CoxLoss()

    result = loss_function.call(y_true, y_pred)

    assert isinstance(result, tf.Tensor)
    assert tf.rank(result) == 1
    assert np.argsort(result.numpy()) == pytest.approx(true_output)
    assert result.numpy()[3] == 0
    assert np.all(result.numpy()[:3] > 0)


def test_cox_loss_wrong_prediction_missing_events():
    y_true = tf.constant([[1, 15], [0, 30], [1, 45], [1, 60]])
    y_pred = tf.constant([0, 4.0, 12.0, 25.0])
    true_output = np.array([1, 3, 2, 0])
    loss_function = CoxLoss()

    result = loss_function.call(y_true, y_pred)

    assert isinstance(result, tf.Tensor)
    assert tf.rank(result) == 1
    assert np.argsort(result.numpy()) == pytest.approx(true_output)
    assert result.numpy()[1] == 0
    assert result.numpy()[3] == 0
    assert result.numpy()[0] > 0
    assert result.numpy()[2] > 0


def test_cox_loss_wrong_prediction_total():
    y_true = tf.constant([[1, 15], [1, 30], [1, 45], [1, 60]])
    y_pred = tf.constant([0, 4.0, 12.0, 25.0])
    loss_function = CoxLoss()

    result = loss_function.__call__(y_true, y_pred)

    assert isinstance(result, tf.Tensor)
    assert tf.rank(result) == 0
    assert result.numpy() > 0


def test_cox_loss_wrong_prediction_missing_events_total():
    y_true_complete = tf.constant([[1, 15], [1, 30], [1, 45], [1, 60]])
    y_true_missing = tf.constant([[1, 15], [0, 30], [1, 45], [1, 60]])
    y_pred = tf.constant([0, 4.0, 12.0, 25.0])
    loss_function = CoxLoss()
    result_complete = loss_function.__call__(y_true_complete, y_pred)

    result_missing = loss_function.__call__(y_true_missing, y_pred)

    assert isinstance(result_missing, tf.Tensor)
    assert result_missing.numpy() < result_complete.numpy()
    assert result_missing.numpy() > 0


def test_coxloss_is_serializable():
    loss_function = CoxLoss()

    result = tf.keras.losses.serialize(loss_function)

    assert isinstance(result, dict)


def test_coxloss_is_deserializable():
    loss_function = CoxLoss

    result = tf.keras.losses.deserialize("CoxLoss", custom_objects={"CoxLoss": loss_function},)

    assert isinstance(result, CoxLoss)
