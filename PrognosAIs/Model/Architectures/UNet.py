import numpy as np

from PrognosAIs.Model.Architectures.Architecture import NetworkArchitecture
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv3D
from tensorflow.keras.layers import Cropping2D
from tensorflow.keras.layers import Cropping3D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import MaxPooling3D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import UpSampling3D
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import ZeroPadding3D
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers


class Unet(NetworkArchitecture):
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
                temp_output = self.conv_func(
                    filters=1,
                    kernel_size=1,
                    padding="same",
                    dtype="float32",
                    activation="sigmoid",
                    name=i_output_name,
                )
            else:
                temp_output = self.conv_func(
                    filters=i_output_classes,
                    kernel_size=1,
                    padding="same",
                    dtype="float32",
                    activation="softmax",
                    name=i_output_name,
                )

            outputs.append(temp_output)
        if len(outputs) == 1:
            outputs = outputs[0]

        return outputs

    def get_number_of_filters(self):
        if self.model_config is not None and "number_of_filters" in self.model_config:
            return self.model_config["number_of_filters"]
        else:
            return 64

    def get_depth(self):
        if self.model_config is not None and "depth" in self.model_config:
            return self.model_config["depth"]
        else:
            return 5

    def init_dimensionality(self, N_dimension):
        if N_dimension == 2:
            self.dims = 2
            self.conv_func = Conv2D
            self.pool_func = MaxPooling2D
            self.upsample_func = UpSampling2D
            self.padding_func = ZeroPadding2D
            self.cropping_func = Cropping2D
        elif N_dimension == 3:
            self.dims = 3
            self.conv_func = Conv3D
            self.pool_func = MaxPooling3D
            self.upsample_func = UpSampling3D
            self.padding_func = ZeroPadding3D
            self.cropping_func = Cropping3D

    def get_conv_block(
        self, layer, N_filters, kernel_size=3, activation="relu", kernel_regularizer=None
    ):
        return self.conv_func(
            filters=N_filters,
            kernel_size=kernel_size,
            padding="same",
            activation=activation,
            kernel_regularizer=kernel_regularizer,
        )(layer)

    def get_pool_block(self, layer):
        return self.pool_func(pool_size=2, padding="same")(layer)

    def get_upsampling_block(self, layer, N_filters, activation="relu", kernel_regularizer=None):
        upsampling_layer = self.upsample_func(size=2)(layer)

        conv_layer = self.conv_func(
            filters=N_filters,
            kernel_size=2,
            padding="same",
            activation=activation,
            kernel_regularizer=kernel_regularizer,
        )(upsampling_layer)
        return conv_layer

    def get_padding_block(self, layer):
        input_size = layer.get_shape()
        input_size = np.asarray(input_size[1:-1])

        is_odd = []
        required_padding = []
        for i_input_size in input_size:
            if i_input_size % 2 != 0:
                is_odd.append(True)
                required_padding.append((1, 0))
            else:
                is_odd.append(False)
                required_padding.append((0, 0))

        if any(is_odd):
            required_padding = tuple(required_padding)
            padding_layer = self.padding_func(required_padding)(layer)
        else:
            padding_layer = layer
        return padding_layer

    def get_cropping_block(self, conv_layer, upsampling_layer):
        conv_size = conv_layer.get_shape()
        upsampling_size = upsampling_layer.get_shape()

        conv_size = np.asarray(conv_size[1:-1])
        upsampling_size = np.asarray(upsampling_size[1:-1])

        size_difference = upsampling_size - conv_size

        need_cropping = []
        required_crop = []
        for i_size_difference in size_difference:
            if i_size_difference != 0:
                need_cropping.append(True)
                required_crop.append((i_size_difference, 0))
            else:
                need_cropping.append(False)
                required_crop.append((0, 0))

        if any(need_cropping):
            required_crop = tuple(required_crop)
            cropping_layer = self.cropping_func(required_crop)(upsampling_layer)
        else:
            cropping_layer = upsampling_layer
        return cropping_layer


class UNet_2D(Unet):
    dims = 2

    def create_model(self):
        self.init_dimensionality(self.dims)
        self.inputs = self.make_inputs(self.input_shapes, self.input_data_type)
        self.N_filters = self.get_number_of_filters()
        self.depth = self.get_depth()

        head = self.inputs

        skip_layers = []
        for i_depth in range(self.depth - 1):
            head = self.get_conv_block(head, self.N_filters * (2 ** i_depth))
            head = self.get_conv_block(head, self.N_filters * (2 ** i_depth))
            head = self.make_dropout_layer(head)
            skip_layers.append(head)
            head = self.get_padding_block(head)
            head = self.get_pool_block(head)

        head = self.get_conv_block(head, self.N_filters * (2 ** self.depth))
        head = self.get_conv_block(head, self.N_filters * (2 ** self.depth))
        head = self.make_dropout_layer(head)

        for i_depth in range(self.depth - 2, -1, -1):
            head = self.get_upsampling_block(head, self.N_filters * (2 ** i_depth))
            head = self.get_cropping_block(skip_layers[i_depth], head)
            head = Concatenate(axis=-1)([skip_layers[i_depth], head])
            head = self.make_dropout_layer(head)

            head = self.get_conv_block(head, self.N_filters * (2 ** i_depth))
            head = self.get_conv_block(head, self.N_filters * (2 ** i_depth))

        outputs = self.make_outputs(self.output_info, self.output_data_type)
        predictions = outputs(head)

        model = Model(inputs=self.inputs, outputs=predictions)

        return model


class UNet_3D(Unet):
    dims = 3

    def create_model(self):
        self.init_dimensionality(self.dims)
        self.inputs = self.make_inputs(self.input_shapes, self.input_data_type)
        self.N_filters = self.get_number_of_filters()
        self.depth = self.get_depth()

        head = self.inputs

        skip_layers = []
        for i_depth in range(self.depth - 1):
            head = self.get_conv_block(head, self.N_filters * (2 ** i_depth))
            head = self.get_conv_block(head, self.N_filters * (2 ** i_depth))
            head = self.make_dropout_layer(head)
            skip_layers.append(head)
            head = self.get_padding_block(head)
            head = self.get_pool_block(head)

        head = self.get_conv_block(head, self.N_filters * (2 ** self.depth))
        head = self.get_conv_block(head, self.N_filters * (2 ** self.depth))
        head = self.make_dropout_layer(head)

        for i_depth in range(self.depth - 2, -1, -1):
            head = self.get_upsampling_block(head, self.N_filters * (2 ** i_depth))
            head = self.get_cropping_block(skip_layers[i_depth], head)
            head = Concatenate(axis=-1)([skip_layers[i_depth], head])

            head = self.get_conv_block(head, self.N_filters * (2 ** i_depth))
            head = self.get_conv_block(head, self.N_filters * (2 ** i_depth))
            head = self.make_dropout_layer(head)

        outputs = self.make_outputs(self.output_info, self.output_data_type)
        predictions = outputs(head)

        model = Model(inputs=self.inputs, outputs=predictions)

        return model
