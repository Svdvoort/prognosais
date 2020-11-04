from PrognosAIs.Model.Architectures.Architecture import ClassificationNetworkArchitecture
from PrognosAIs.Model.Architectures.Architecture import NetworkArchitecture
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv3D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import ReLU
from tensorflow.keras.models import Model
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
import tensorflow as tf
from tensorflow.keras.losses import Loss


class TestNet_2D(ClassificationNetworkArchitecture):
    def create_model(self):
        conv_1 = Conv2D(filters=4, kernel_size=(2, 2))(self.inputs)

        relu_1 = ReLU()(conv_1)

        flatten_1 = Flatten()(relu_1)

        dense_1 = Dense(units=50)(flatten_1)

        predictions = self.outputs(dense_1)

        model = Model(inputs=self.inputs, outputs=predictions)

        return model


class TestNet_multiinput_multioutput_2D(ClassificationNetworkArchitecture):
    def create_model(self):
        inputs = self.make_inputs(self.input_shapes, self.input_data_type, squeeze_inputs=False)
        outputs = self.make_outputs(self.output_info, self.output_data_type, squeeze_outputs=False)
        input_branches = []

        for i_input in inputs.values():
            input_branches.append(Conv2D(filters=4, kernel_size=(2, 2))(i_input))

        if len(input_branches) > 1:
            concat_1 = Concatenate()(input_branches)
        else:
            concat_1 = input_branches[0]

        relu_1 = ReLU()(concat_1)

        flatten_1 = Flatten()(relu_1)

        dense_1 = Dense(units=50)(flatten_1)

        predictions = []
        for i_output in outputs.values():
            predictions.append(i_output(dense_1))

        model = Model(inputs=inputs, outputs=predictions)

        return model


class TestNet_3D(ClassificationNetworkArchitecture):
    def create_model(self):
        conv_1 = Conv3D(filters=4, kernel_size=(2, 2, 2))(self.inputs)

        relu_1 = ReLU()(conv_1)

        flatten_1 = Flatten()(relu_1)

        dense_1 = Dense(units=50)(flatten_1)

        predictions = self.outputs(dense_1)

        model = Model(inputs=self.inputs, outputs=predictions)

        return model


class TestNet_MASK_3D(NetworkArchitecture):
    def make_outputs(
        self, output_info: dict, output_data_type: str, activation_type: str = "softmax"
    ):
        outputs = []
        if self.model_config is not None and "one_hot_output" in self.model_config:
            self.one_hot_output = self.model_config["one_hot_output"]
        else:
            self.one_hot_output = False

        for i_output_name, i_output_classes in output_info.items():
            if i_output_classes == 2 and not self.one_hot_output:
                temp_output = Conv3D(
                    filters=1,
                    kernel_size=(1, 1, 1),
                    padding="same",
                    dtype="float32",
                    activation="sigmoid",
                    name=i_output_name,
                )
            else:
                temp_output = Conv3D(
                    filters=i_output_classes,
                    kernel_size=(1, 1, 1),
                    padding="same",
                    dtype="float32",
                    activation="softmax",
                    name=i_output_name,
                )

            outputs.append(temp_output)
        if len(outputs) == 1:
            outputs = outputs[0]

        return outputs

    def create_model(self):
        self.inputs = self.make_inputs(self.input_shapes, self.input_data_type)
        conv_1 = Conv3D(filters=4, kernel_size=(2, 2, 2), padding="same")(self.inputs)

        relu_1 = ReLU()(conv_1)

        outputs = self.make_outputs(self.output_info, self.output_data_type)
        predictions = outputs(relu_1)
        model = Model(inputs=self.inputs, outputs=predictions)

        return model


class TestLoss(Loss):
    def call(self, y_true, y_pred):
        y_pred = ops.convert_to_tensor_v2(y_pred)
        y_true = math_ops.cast(y_true, y_pred.dtype)
        return tf.math.reduce_mean(math_ops.square(y_pred - y_true), axis=-1)


class TestMetric(tf.keras.metrics.Metric):
    def __init__(self, name="test_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name="tp", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(y_pred, tf.bool)

        values = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
        values = tf.cast(values, self.dtype)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            sample_weight = tf.broadcast_weights(sample_weight, values)
            values = tf.multiply(values, sample_weight)
        self.true_positives.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.true_positives
