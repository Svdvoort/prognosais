import numpy as np
import tensorflow as tf

from tensorflow import keras

# from tensorflow import distribute
# from tensorflow.keras.backend import print_tensor
# from tensorflow.python.ops.losses import util as tf_losses_utils
# from tensorflow.python.framework import dtypes
from tensorflow.keras import layers
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import metrics_utils
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops


def concordance_index(y_true, y_pred):
    """
    This function determines the concordance index given two tensorflow tensors

    y_true contains a label to indicate whether events occurred, and time to events
    (or time to right censored data if no event occurred)

    y_pred is beta*x in the cox model
    """
    events_occurred = y_true[:, 0]
    time_to_events = y_true[:, 1]

    # time_to_events = print_tensor(time_to_events, message='time to events: ')

    y_pred = tf.squeeze(y_pred)
    identity_matrix = tf.cast(y_pred[None, :] < y_pred[:, None], tf.float32)

    risk_set = tf.cast(time_to_events[None, :] > time_to_events[:, None], tf.float32)

    risk_set_idenity_matrix = tf.math.multiply(identity_matrix, risk_set)
    correct_prediction = tf.math.reduce_sum(risk_set_idenity_matrix, axis=1)

    correct_prediction_event_set = tf.math.multiply(correct_prediction, events_occurred)

    concordance_index_unscaled = tf.math.reduce_sum(correct_prediction_event_set)

    concordance_index = concordance_index_unscaled / tf.cast(
        tf.math.count_nonzero(identity_matrix), tf.float32
    )

    return concordance_index


class ConcordanceIndex(keras.metrics.Metric):
    def __init__(self, name="ConcordanceIndex", **kwargs):
        super().__init__(name=name, **kwargs)
        self.summed_concordance = self.add_weight(
            name="summed_corcodance", initializer="zeros", dtype=tf.float32
        )
        self.total_samples = self.add_weight(
            name="total_samples", initializer="zeros", dtype=tf.float32
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.squeeze(y_pred)
        events_occurred = tf.squeeze(y_true[:, 0])
        time_to_events = tf.squeeze(y_true[:, 1])
        identity_matrix = tf.cast(y_pred[None, :] < y_pred[:, None], tf.float32)

        risk_set = tf.cast(events_occurred[None, :] > events_occurred[:, None], tf.float32)

        risk_set_idenity_matrix = tf.math.multiply(identity_matrix, risk_set)
        correct_prediction = tf.math.reduce_sum(risk_set_idenity_matrix, axis=1)

        correct_prediction_event_set = tf.math.multiply(correct_prediction, time_to_events)

        concordance_index_unscaled = tf.math.reduce_sum(correct_prediction_event_set)

        concordance_index = concordance_index_unscaled / tf.cast(
            tf.math.count_nonzero(identity_matrix), tf.float32
        )

        self.summed_concordance.assign_add(concordance_index)
        self.total_samples.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))

    def result(self):
        return tf.math.divide_no_nan(self.summed_concordance, self.total_samples)


class Sensitivity(keras.metrics.Metric):
    def __init__(self, name="Sensitivity_custom", **kwargs):
        super(Sensitivity, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(
            name="true_positives", shape=(1,), initializer="zeros"
        )
        self.true_negatives = self.add_weight(
            name="true_negatives", shape=(1,), initializer="zeros"
        )
        self.false_positives = self.add_weight(
            name="false_positives", shape=(1,), initializer="zeros"
        )
        self.false_negatives = self.add_weight(
            name="false_negatives", shape=(1,), initializer="zeros"
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Get the actual labels
        # y_true = tf.squeeze(y_true)
        # y_pred = tf.argmax(tf.squeeze(y_pred), axis=1)

        metrics_utils.update_confusion_matrix_variables(
            {
                metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self.true_positives,
                metrics_utils.ConfusionMatrix.TRUE_NEGATIVES: self.true_negatives,
                metrics_utils.ConfusionMatrix.FALSE_POSITIVES: self.false_positives,
                metrics_utils.ConfusionMatrix.FALSE_NEGATIVES: self.false_negatives,
            },
            y_true,
            y_pred,
            thresholds=[0.5],
            sample_weight=sample_weight,
        )

    def result(self):
        result = math_ops.div_no_nan(
            self.true_positives, self.true_positives + self.false_negatives
        )
        return result[0]

    def reset_states(self):
        K.batch_set_value([(v, np.zeros((1,))) for v in self.variables])


class Specificity(keras.metrics.Metric):
    def __init__(self, name="Specificity_custom", **kwargs):
        super(Specificity, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(
            name="true_positives", shape=(1,), initializer="zeros"
        )
        self.true_negatives = self.add_weight(
            name="true_negatives", shape=(1,), initializer="zeros"
        )
        self.false_positives = self.add_weight(
            name="false_positives", shape=(1,), initializer="zeros"
        )
        self.false_negatives = self.add_weight(
            name="false_negatives", shape=(1,), initializer="zeros"
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        # y_true = tf.squeeze(y_true)
        # y_pred = tf.argmax(tf.squeeze(y_pred), axis=1)

        metrics_utils.update_confusion_matrix_variables(
            {
                metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self.true_positives,
                metrics_utils.ConfusionMatrix.TRUE_NEGATIVES: self.true_negatives,
                metrics_utils.ConfusionMatrix.FALSE_POSITIVES: self.false_positives,
                metrics_utils.ConfusionMatrix.FALSE_NEGATIVES: self.false_negatives,
            },
            y_true,
            y_pred,
            thresholds=[0.5],
            sample_weight=sample_weight,
        )

    def result(self):
        result = math_ops.div_no_nan(
            self.true_negatives, self.true_negatives + self.false_positives
        )
        return result[0]

    def reset_states(self):
        K.batch_set_value([(v, np.zeros((1,))) for v in self.variables])


class MaskedAUC(keras.metrics.AUC):
    def __init__(self, name="MaskedAUC", mask_value=-1, **kwargs):
        if "multi_label" not in kwargs:
            super().__init__(name=name, multi_label=True, **kwargs)
        else:
            super().__init__(name=name, **kwargs)
        self.mask_value = mask_value

    def update_state(self, y_true, y_pred, sample_weight=None):
        # we remove masked samples from y_true and y_pred
        y_pred = tf.clip_by_value(y_pred, 0, 1)
        y_pred = ops.convert_to_tensor(y_pred)
        y_true = math_ops.cast(y_true, y_pred.dtype)

        unmasked_samples = tf.reduce_all(tf.math.not_equal(y_true, self.mask_value), axis=-1)
        unmasked_samples = tf.cast(unmasked_samples, self.dtype)

        weight_correction = unmasked_samples * tf.math.divide_no_nan(
            tf.cast(tf.gather(tf.shape(y_true), 0), self.dtype),
            tf.math.count_nonzero(unmasked_samples, dtype=self.dtype),
        )

        if sample_weight is not None:
            sample_weight *= weight_correction
        else:
            sample_weight = weight_correction

        super().update_state(y_true, y_pred, sample_weight)

    def get_config(self):
        config = super().get_config()
        config["mask_value"] = self.mask_value
        return config


class MaskedCategoricalAccuracy(keras.metrics.CategoricalAccuracy):
    def __init__(self, name="MaskedCategoricalAccuracy", mask_value=-1, **kwargs):
        super().__init__(name=name, **kwargs)
        self.mask_value = mask_value

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = ops.convert_to_tensor(y_pred)
        y_true = math_ops.cast(y_true, y_pred.dtype)

        masked_samples = tf.reduce_all(tf.math.not_equal(y_true, self.mask_value), axis=-1)
        masked_samples = math_ops.cast(masked_samples, y_true.dtype)
        masked_samples.set_shape([None])

        y_true = tf.boolean_mask(y_true, masked_samples)
        y_pred = tf.boolean_mask(y_pred, masked_samples)

        if sample_weight is not None:
            sample_weight = tf.boolean_mask(sample_weight, masked_samples)

        super().update_state(y_true, y_pred, sample_weight)

    def get_config(self):
        config = super().get_config()
        config["mask_value"] = self.mask_value
        return config


class DICE(keras.metrics.Metric):
    def __init__(self, name="dice_coefficient", foreground_only=True, **kwargs):
        super().__init__(name=name, **kwargs)
        self.summed_dice = self.add_weight(
            name="summed_dice", initializer="zeros", dtype=tf.float32
        )
        self.total_samples = self.add_weight(
            name="total_samples", initializer="zeros", dtype=tf.float32
        )
        self.foreground_only = foreground_only

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = ops.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        y_pred = tf.math.round(y_pred)

        to_reduce_axis = tf.range(1, tf.rank(y_true) - 1, 1)
        numerator = tf.math.reduce_sum(y_true * y_pred, axis=to_reduce_axis)
        denominator = tf.math.reduce_sum(y_true + y_pred, axis=to_reduce_axis)

        # We first calculate the loss (1 -dice, which adds he denominator in te numerato)
        # Because thn using divide_no_nan give 0 for the loss if no
        dice_per_class = 1.0 - tf.math.divide_no_nan(denominator - 2.0 * numerator, denominator)

        if self.foreground_only:
            dice = tf.math.reduce_sum(dice_per_class[..., 1:])
        else:
            dice = tf.math.reduce_sum(tf.math.reduce_mean(dice_per_class))

        self.summed_dice.assign_add(dice)
        self.total_samples.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))

    def result(self):
        return tf.math.divide_no_nan(self.summed_dice, self.total_samples)

    def get_config(self):
        config = super().get_config()
        config["foreground_only"] = self.foreground_only

        return config


class MaskedSpecificity(keras.metrics.Metric):
    def __init__(self, name="masked_specificity", mask_value=-1, **kwargs):
        super().__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(
            name="true_positives", shape=(1,), initializer="zeros"
        )
        self.true_negatives = self.add_weight(
            name="true_negatives", shape=(1,), initializer="zeros"
        )
        self.false_positives = self.add_weight(
            name="false_positives", shape=(1,), initializer="zeros"
        )
        self.false_negatives = self.add_weight(
            name="false_negatives", shape=(1,), initializer="zeros"
        )
        self.mask_value = mask_value

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = ops.convert_to_tensor(y_pred)
        y_true = math_ops.cast(y_true, y_pred.dtype)

        masked_samples = tf.reduce_all(tf.math.not_equal(y_true, self.mask_value), axis=-1)
        masked_samples = math_ops.cast(masked_samples, y_true.dtype)
        masked_samples.set_shape([None])

        y_true = tf.boolean_mask(y_true, masked_samples)
        y_pred = tf.boolean_mask(y_pred, masked_samples)

        y_true = tf.math.argmax(y_true, axis=-1)
        y_pred = tf.math.argmax(y_pred, axis=-1)

        metrics_utils.update_confusion_matrix_variables(
            {
                metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self.true_positives,
                metrics_utils.ConfusionMatrix.TRUE_NEGATIVES: self.true_negatives,
                metrics_utils.ConfusionMatrix.FALSE_POSITIVES: self.false_positives,
                metrics_utils.ConfusionMatrix.FALSE_NEGATIVES: self.false_negatives,
            },
            y_true,
            y_pred,
            thresholds=[0.5],
            sample_weight=sample_weight,
        )

    def result(self):
        result = math_ops.div_no_nan(
            self.true_negatives, self.true_negatives + self.false_positives
        )
        return result[0]

    def reset_states(self):
        K.batch_set_value([(v, np.zeros((1,))) for v in self.variables])


class MaskedSensitivity(keras.metrics.Metric):
    def __init__(self, name="masked_sensitivity", mask_value=-1, **kwargs):
        super().__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(
            name="true_positives", shape=(1,), initializer="zeros"
        )
        self.true_negatives = self.add_weight(
            name="true_negatives", shape=(1,), initializer="zeros"
        )
        self.false_positives = self.add_weight(
            name="false_positives", shape=(1,), initializer="zeros"
        )
        self.false_negatives = self.add_weight(
            name="false_negatives", shape=(1,), initializer="zeros"
        )
        self.mask_value = mask_value

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = ops.convert_to_tensor(y_pred)
        y_true = math_ops.cast(y_true, y_pred.dtype)

        masked_samples = tf.reduce_all(tf.math.not_equal(y_true, self.mask_value), axis=-1)
        masked_samples = math_ops.cast(masked_samples, y_true.dtype)
        masked_samples.set_shape([None])

        y_true = tf.boolean_mask(y_true, masked_samples)
        y_pred = tf.boolean_mask(y_pred, masked_samples)

        y_true = tf.math.argmax(y_true, axis=-1)
        y_pred = tf.math.argmax(y_pred, axis=-1)

        if sample_weight is not None:
            sample_weight = tf.boolean_mask(sample_weight, masked_samples)

        metrics_utils.update_confusion_matrix_variables(
            {
                metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self.true_positives,
                metrics_utils.ConfusionMatrix.TRUE_NEGATIVES: self.true_negatives,
                metrics_utils.ConfusionMatrix.FALSE_POSITIVES: self.false_positives,
                metrics_utils.ConfusionMatrix.FALSE_NEGATIVES: self.false_negatives,
            },
            y_true,
            y_pred,
            thresholds=[0.5],
            sample_weight=sample_weight,
        )

    def result(self):
        result = math_ops.div_no_nan(
            self.true_positives, self.true_positives + self.false_negatives
        )
        return result[0]

    def reset_states(self):
        K.batch_set_value([(v, np.zeros((1,))) for v in self.variables])
