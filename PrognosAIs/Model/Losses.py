import tensorflow as tf

from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.losses import Loss
from tensorflow.python.framework import ops
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.ops import math_ops


class MaskedCategoricalCrossentropy(CategoricalCrossentropy):
    """Caterogical crossentropy loss which takes into account missing values."""

    def __init__(
        self,
        name: str = "masked_categorical_crossentropy",
        class_weight: dict = None,
        mask_value: int = -1,
        **kwargs,
    ) -> None:
        r"""
        Caterogical crossentropy loss which takes into account missing values.

        For the samples with masked values a cross entropy of 0 will be used, for the other samples
        the standard cross entropy loss will be calculated

        Args:
            name (str): Optional name for the op
            class_weight (dict): Weights for each class
            mask_value (int): The value that indicates that a sample is missing
            **kwargs:  arguments to pass the default CategoricalCrossentropy loss
        """
        super().__init__(name=name, **kwargs)
        self.dtype = tf.dtypes.as_dtype(tf.keras.backend.floatx())

        self.mask_value = mask_value
        self.original_class_weight = class_weight
        if class_weight is not None:
            # Got a dict with class weights for different classes
            # Need to prepare this for tensorflow computation
            self.class_weight = tf.constant(
                [
                    float(v)
                    for k, v in sorted(class_weight.items(), key=lambda item: float(item[0]))
                ],
            )
            print(self.class_weight)
        else:
            self.class_weight = None

    def is_unmasked_sample(self, y_true: tf.Tensor) -> tf.Tensor:
        """
        Get whether the samples are unmasked (i.e. have real label data).

        Args:
            y_true (tf.Tensor): Tensor of the true labels

        Returns:
            tf.Tensor: Tensor of 0s and 1s indicating whether that sample is unmasked.
        """
        unmasked_samples = tf.reduce_all(tf.math.not_equal(y_true, self.mask_value), axis=-1)
        return math_ops.cast(unmasked_samples, self.dtype)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Obtain the masked categorical crossentropy loss for each sample.

        Args:
            y_true (tf.Tensor): Ground-truth labels, one-hot encoded
                (batch_size, N_1, N_2, .... N_d) tensor, with N_d the number of outputs
            y_pred (tf.Tensor): Predictions one-hot encoded, for example from softmax,
                (batch_size, N_1, N_2, .... N_d) tensor, with N_d the number of outputs

        Returns:
            tf.Tensor: The masked categorial crossentropy loss for each sample, has rank
                one less than the inputs tensors
        """
        y_pred = ops.convert_to_tensor(y_pred)
        y_pred = tf.cast(y_pred, self.dtype)
        y_true = math_ops.cast(y_true, y_pred.dtype)
        loss = super().call(y_true, y_pred) * self.is_unmasked_sample(y_true)
        if self.class_weight is not None:
            # Correction factor is need to keep loss total the same as
            # for case without class weights
            y_true_indexes = tf.math.argmax(y_true, axis=-1)
            weights = math_ops.cast(tf.gather(self.class_weight, y_true_indexes), loss.dtype)
            loss *= weights
        return loss

    def __call__(
        self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight: tf.Tensor = None,
    ) -> tf.Tensor:
        """
        Obtain the total masked categorical crossentropy loss for the batch.

        Args:
            y_true (tf.Tensor): Ground-truth labels, one-hot encoded
                (batch_size, N_1, N_2, .... N_d) tensor, with N_d the number of outputs
            y_pred (tf.Tensor): Predictions one-hot encoded, for example from softmax,
                (batch_size, N_1, N_2, .... N_d) tensor, with N_d the number of outputs
            sample_weight (tf.Tensor): Sample weight for each indidvidual label
                to be used in reduction of sample loss to overal batch loss

        Returns:
            tf.Tensor: The total masked categorial crossentropy loss, scalar tensor with rank 0
        """
        losses = self.call(y_true, y_pred)
        unmasked_samples = self.is_unmasked_sample(y_true)
        weight_correction = unmasked_samples * tf.math.divide_no_nan(
            tf.cast(tf.gather(tf.shape(y_true), 0), self.dtype),
            tf.math.count_nonzero(unmasked_samples, dtype=self.dtype),
        )

        if sample_weight is not None:
            sample_weight *= weight_correction
        else:
            sample_weight = weight_correction

        return losses_utils.compute_weighted_loss(
            losses, sample_weight, reduction=self._get_reduction(),
        )

    def get_config(self) -> dict:
        """
        Get the configuration of the loss.

        Returns:
            dict: Configuration parameters of the loss
        """
        config = super().get_config()
        config["mask_value"] = self.mask_value
        if self.original_class_weight is not None:
            to_write_class_weight = {}
            for key in self.original_class_weight.keys():
                to_write_class_weight[str(key)] = str(self.original_class_weight[key])
            config["class_weight"] = to_write_class_weight

        return config


class DICE_loss(Loss):
    """Loss class for the Sørensen–Dice coefficient."""

    def __init__(
        self,
        name: str = "dice_loss",
        weighted: bool = False,
        foreground_only: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(name=name, **kwargs)

        self.weighted = weighted
        self.foreground_only = foreground_only

    # TODO: Add masked DICE loss as well
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        r"""
        Calculate the DICE loss.

        This functions calculates the DICE loss defined as:

        .. math::
            1 - 2 * \frac{|A \cap B|}{|A| + |B|}

        When no positive labels are found in both A and B the loss
        returns 0 by default.
        The loss works both for one-hot predicted labels and binary labels.

        Args:
            y_true (tf.Tensor): The ground truth labels, shape: (batch_size, N_1, N_2 ... N_d)
                where N_d is the number of channels (can be 1). For a 3D tensor with 1 channel (binary class)
                and batch size of 1 it will have a shape of (1, N_1, N_2, N_3, 1)
            y_pred (tf.Tensor): The predicted labels. shape: (batch_size, N_1, N_2 ... N_d)
                where N_d is the number of channels. When a binary prediction is done (last activation function
                is sigmoid), N_d = 1. When one-hot prediction are done (last activation function is softmax)
                N_d = number of classes

        Returns:
            tf.Tensor: Tensor of length batch_size with the DICE loss for each sample
        """
        y_pred = ops.convert_to_tensor(y_pred)
        # TODO FIX THIS CLIPPING< TO AVOID NAN IN FLOAT16
        y_pred = tf.clip_by_value(y_pred, 0, 1)
        y_true = math_ops.cast(y_true, y_pred.dtype)

        to_reduce_axis = tf.range(1, tf.rank(y_true) - 1, 1)
        numerator = tf.math.reduce_sum(y_true * y_pred, axis=to_reduce_axis)
        denominator = tf.math.reduce_sum(y_true + y_pred, axis=to_reduce_axis)
        # We take 1 minus the dice to get the loss instead of the score
        # We multiply by product of y_true to get loss of 0 when class is not present
        dice_loss_per_class = (
            1.0 - 2.0 * tf.math.divide_no_nan(numerator, denominator)
        ) * tf.math.reduce_max(y_true, axis=to_reduce_axis)

        if self.weighted:
            samples_per_class = tf.math.reduce_sum(y_true, axis=to_reduce_axis)
            total_samples = tf.math.reduce_sum(samples_per_class, axis=-1, keepdims=True)
            # total_samples = tf.broadcast_to(total_samples, tf.shape(samples_per_class))
            weight_per_class = tf.math.divide_no_nan(total_samples, samples_per_class)
            # Apply normalization factor
            weight_per_class = tf.math.divide(
                weight_per_class, tf.math.reduce_sum(weight_per_class, axis=-1, keepdims=True)
            )

            dice_loss_per_class = dice_loss_per_class * weight_per_class

        if self.foreground_only:
            dice_loss_per_class = dice_loss_per_class[..., 1:]

        # Finally we get mean over classes
        return tf.math.reduce_mean(dice_loss_per_class, axis=-1)

    def get_config(self) -> dict:
        """
        Get the configuration of the loss.

        Returns:
            dict: configuration of the loss
        """
        config = super().get_config()
        config["weighted"] = self.weighted
        config["foreground_only"] = self.foreground_only
        return config


class CoxLoss(Loss):
    """Cox loss as defined in https://arxiv.org/pdf/1606.00931.pdf."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.dtype = tf.dtypes.as_dtype(tf.keras.backend.floatx())

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        r"""
        Calculate the cox loss.

        Args:
            y_true (tf.Tensor): Tensor of shape (batch_size, 2), with the first index
                containing whether and event occurred for each sample, and the second
                index containing the time to event, or follow-up time if no event
                has occurred
            y_pred (tf.Tensor): The :math:`\hat{h}_{\sigma\}` as predicted by the network

        Returns:
            tf.Tensor: The cox loss for each sample in the batch
        """
        # We clip the predictions to make sure we dont get overflow problems
        y_pred = tf.clip_by_value(ops.convert_to_tensor(y_pred), 0, tf.math.log(self.dtype.max) - 1)
        y_true = math_ops.cast(y_true, y_pred.dtype)
        events_occurred = y_true[:, 0]
        time_to_events = y_true[:, 1]
        epsilon = 1e-6

        # This creates a matrix comparing each time to event with all others
        risk_set = tf.cast(
            tf.math.greater_equal(time_to_events[None, :], time_to_events[:, None]), self.dtype,
        )

        # Get the exponential hazard for all sample_locations
        # We clip in case of very large predicted values to ensure that
        # We will never get an overflow
        exp_hazard = tf.math.exp(y_pred)

        # Get risk only for samples for which event did not occur yet
        # Calculate the sum over the risk sets
        exp_hazard_risk_set = tf.squeeze(tf.math.reduce_sum(exp_hazard[None, :] * risk_set, axis=1))

        # Get log of sum over risk set, with epsilon in case it is 0
        # Get difference between prediction and log of risk set
        hazard_difference = tf.squeeze(y_pred) - tf.math.log(exp_hazard_risk_set + epsilon)

        # Only values for patients with events
        # We want to minimize a function, so need to make loss positive
        return tf.math.negative(hazard_difference * events_occurred)

    def get_config(self) -> dict:
        """
        Get the configuration of the loss.

        Returns:
            dict: configuration of the loss
        """
        return super().get_config()
