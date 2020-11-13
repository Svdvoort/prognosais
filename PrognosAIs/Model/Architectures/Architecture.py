from abc import ABC
from abc import abstractmethod
from typing import Union

import numpy as np
import tensorflow

from tensorflow.keras import Input
from tensorflow.keras.backend import int_shape
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout


class NetworkArchitecture(ABC):
    def __init__(
        self,
        input_shapes: dict,
        output_info: dict,
        input_data_type="float32",
        output_data_type="float32",
        model_config: dict = {},
    ):
        self.input_data_type = input_data_type
        self.output_data_type = output_data_type
        self.input_shapes = input_shapes
        self.output_info = output_info
        self.model_config = model_config
        # self.inputs = self.make_inputs(input_shapes, input_data_type)
        # self.outputs = self.make_outputs(output_info, output_data_type)

    def make_inputs(
        self, input_shapes: dict, input_dtype: str, squeeze_inputs: bool = True
    ) -> Union[dict, Input]:
        inputs = {}
        for i_input_name, i_input_shape in input_shapes.items():
            # TODO pass correct input type according to policy
            inputs[i_input_name] = Input(shape=i_input_shape, name=i_input_name)

        if squeeze_inputs and len(inputs) == 1:
            inputs = list(inputs.values())[0]

        return inputs

    @staticmethod
    def check_minimum_input_size(input_layer: Input, minimum_input_size: np.ndarray):
        input_shape = int_shape(input_layer)
        if len(input_shape) - 1 == len(minimum_input_size):
            err_msg = (
                "It seems like you have forgotten to include a"
                " (potentially empty) channel dimension as the last dimension.\n"
                "Please fix this and run the model again."
            )
            raise ValueError(err_msg)
        input_shape = input_shape[1:-1]
        minimum_input_size = np.asarray(minimum_input_size)

        if any(input_shape < minimum_input_size):
            min_inputs = [str(i_input_size) for i_input_size in minimum_input_size]
            min_input_format = " x ".join(min_inputs)

            cur_inputs = [str(i_input_shape) for i_input_shape in input_shape]
            cur_input_format = " x ".join(cur_inputs)

            err_msg = (
                "Minimum input size for this model is: {}\n"
                "Your input size is: {}\n"
                "Please fix your input"
            ).format(min_input_format, cur_input_format,)
            raise ValueError(err_msg)

    @staticmethod
    def get_corrected_stride_size(
        layer: tensorflow.keras.layers, stride_size: list, conv_size: list
    ):
        """
        Ensure that the stride is never bigger than the actual input
        In this way any network can keep working, indepedent of size
        """

        input_shape = int_shape(layer)[1:-1]
        stride_size = np.asarray(stride_size)
        conv_size = np.asarray(conv_size)

        stride_total_size = stride_size * conv_size

        stride_size = np.floor(np.minimum(input_shape, stride_total_size) / conv_size)

        # Ensure that stride_size is at least one
        stride_size = np.maximum(stride_size, np.ones(len(stride_size)))
        stride_size = stride_size.astype(np.int).tolist()

        return stride_size

    def make_dropout_layer(self, layer):
        if "dropout" in self.model_config and self.model_config["dropout"] > 0:
            out_layer = Dropout(self.model_config["dropout"])(layer)
        else:
            out_layer = layer
        return out_layer

    @abstractmethod
    def make_outputs(self, output_info: dict, output_data_type: str) -> tensorflow.keras.layers:
        """Make the outputs"""

    @abstractmethod
    def create_model(self):
        """Here the code to create the actual model"""


class ClassificationNetworkArchitecture(NetworkArchitecture):
    def __init__(
        self,
        input_shapes: dict,
        output_info: dict,
        input_data_type="float32",
        output_data_type="float32",
        model_config={},
    ):

        super().__init__(input_shapes, output_info, input_data_type, output_data_type, model_config)
        self.inputs = self.make_inputs(input_shapes, input_data_type)
        self.outputs = self.make_outputs(output_info, output_data_type)

    @staticmethod
    def make_outputs(
        output_info: dict,
        output_data_type: str,
        activation_type: str = "softmax",
        squeeze_outputs: bool = True,
    ) -> dict:
        outputs = {}
        for i_output_name, i_output_classes in output_info.items():
            outputs[i_output_name] = Dense(
                i_output_classes, name=i_output_name, activation=activation_type, dtype="float32",
            )

        if squeeze_outputs and len(outputs) == 1:
            outputs = list(outputs.values())[0]

        return outputs
