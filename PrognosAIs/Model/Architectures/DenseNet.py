# TF general functions
from PrognosAIs.Model.Architectures.Architecture import ClassificationNetworkArchitecture
from tensorflow.keras.backend import int_shape

# TF 3D functions
# TF 2D functions
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv3D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import GlobalAveragePooling3D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import MaxPooling3D
from tensorflow.keras.layers import ReLU
from tensorflow.keras.models import Model


# Other


class DenseNet(ClassificationNetworkArchitecture):
    def init_dimensionality(self, N_dimension):
        if N_dimension == 2:
            self.dims = 2
            self.conv_func = Conv2D
            self.pool_func = MaxPooling2D
        elif N_dimension == 3:
            self.dims = 3
            self.conv_func = Conv3D
            self.pool_func = MaxPooling3D

    def get_dense_stem(self, layer, N_filters):
        stride_size = self.get_corrected_stride_size(layer, [2] * self.dims, [7] * self.dims)

        conv_1 = self.conv_func(
            filters=N_filters,
            kernel_size=[7] * self.dims,
            strides=stride_size,
            padding="same",
            data_format="channels_last",
            activation="relu",
        )(layer)

        stride_size = self.get_corrected_stride_size(conv_1, [2] * self.dims, [3] * self.dims)
        pooling_1 = self.pool_func(
            pool_size=[3] * self.dims,
            strides=stride_size,
            padding="same",
            data_format="channels_last",
        )(conv_1)

        return pooling_1

    def get_dense_block(self, layer, N_filters, N_conv_layers):
        for i_conv_layer in range(N_conv_layers):
            head = BatchNormalization(axis=-1)(layer)
            head = ReLU()(head)
            head = self.conv_func(
                filters=N_filters * 4,
                kernel_size=[1] * self.dims,
                strides=[1] * self.dims,
                padding="same",
                data_format="channels_last",
                activation="linear",
            )(head)

            head = BatchNormalization(axis=-1)(head)
            head = ReLU()(head)
            head = self.conv_func(
                filters=N_filters,
                kernel_size=[3] * self.dims,
                strides=[1] * self.dims,
                padding="same",
                data_format="channels_last",
                activation="linear",
            )(head)

            head = Concatenate(axis=-1)([layer, head])
            layer = head

        return head

    def get_transition_block(self, layer, N_filters, theta):
        conv_1 = self.conv_func(
            filters=int(N_filters * theta),
            kernel_size=[1] * self.dims,
            strides=[1] * self.dims,
            padding="same",
            data_format="channels_last",
            activation="linear",
        )(layer)

        stride_size = self.get_corrected_stride_size(conv_1, [2] * self.dims, [2] * self.dims)

        pooling_1 = self.pool_func(
            pool_size=[2] * self.dims,
            strides=stride_size,
            padding="valid",
            data_format="channels_last",
        )(conv_1)

        return pooling_1


class DenseNet_121_2D(DenseNet):
    dims = 2
    GROWTH_RATE = 32
    INITIAL_FILTERS = 2 * GROWTH_RATE
    # Settings theta to 1 will create a DenseNet-B
    THETA = 0.5

    def create_model(self):
        self.init_dimensionality(self.dims)
        self.check_minimum_input_size(self.inputs, [21, 21])

        dense_stem = self.get_dense_stem(self.inputs, self.INITIAL_FILTERS)
        dense_block_1 = self.get_dense_block(dense_stem, self.GROWTH_RATE, 6)
        N_filters = int_shape(dense_block_1)[-1]

        dense_block_1 = self.make_dropout_layer(dense_block_1)
        trans_block_1 = self.get_transition_block(dense_block_1, N_filters, self.THETA)
        dense_block_2 = self.get_dense_block(trans_block_1, self.GROWTH_RATE, 12)
        N_filters = int_shape(dense_block_2)[-1]

        dense_block_2 = self.make_dropout_layer(dense_block_2)
        trans_block_2 = self.get_transition_block(dense_block_2, N_filters, self.THETA)
        dense_block_3 = self.get_dense_block(trans_block_2, self.GROWTH_RATE, 24)
        N_filters = int_shape(dense_block_3)[-1]

        dense_block_3 = self.make_dropout_layer(dense_block_3)
        trans_block_3 = self.get_transition_block(dense_block_3, N_filters, self.THETA)
        dense_block_4 = self.get_dense_block(trans_block_3, self.GROWTH_RATE, 16)

        gap_1 = GlobalAveragePooling2D(data_format="channels_last")(dense_block_4)

        predictions = self.outputs(gap_1)

        model = Model(inputs=self.inputs, outputs=predictions)

        return model


class DenseNet_121_3D(DenseNet):
    dims = 3
    GROWTH_RATE = 32
    INITIAL_FILTERS = 2 * GROWTH_RATE
    # Settings theta to 1 will create a DenseNet-B
    THETA = 0.5

    def create_model(self):
        self.init_dimensionality(self.dims)
        self.check_minimum_input_size(self.inputs, [21, 21, 21])

        dense_stem = self.get_dense_stem(self.inputs, self.INITIAL_FILTERS)
        dense_block_1 = self.get_dense_block(dense_stem, self.GROWTH_RATE, 6)
        N_filters = int_shape(dense_block_1)[-1]

        dense_block_1 = self.make_dropout_layer(dense_block_1)
        trans_block_1 = self.get_transition_block(dense_block_1, N_filters, self.THETA)
        dense_block_2 = self.get_dense_block(trans_block_1, self.GROWTH_RATE, 12)
        N_filters = int_shape(dense_block_2)[-1]

        dense_block_2 = self.make_dropout_layer(dense_block_2)
        trans_block_2 = self.get_transition_block(dense_block_2, N_filters, self.THETA)
        dense_block_3 = self.get_dense_block(trans_block_2, self.GROWTH_RATE, 24)
        N_filters = int_shape(dense_block_3)[-1]

        dense_block_3 = self.make_dropout_layer(dense_block_3)
        trans_block_3 = self.get_transition_block(dense_block_3, N_filters, self.THETA)
        dense_block_4 = self.get_dense_block(trans_block_3, self.GROWTH_RATE, 16)

        gap_1 = GlobalAveragePooling3D(data_format="channels_last")(dense_block_4)

        predictions = self.outputs(gap_1)

        model = Model(inputs=self.inputs, outputs=predictions)

        return model


class DenseNet_169_2D(DenseNet):
    dims = 2
    GROWTH_RATE = 32
    INITIAL_FILTERS = 2 * GROWTH_RATE
    # Settings theta to 1 will create a DenseNet-B
    THETA = 0.5

    def create_model(self):
        self.init_dimensionality(self.dims)
        self.check_minimum_input_size(self.inputs, [21, 21])

        dense_stem = self.get_dense_stem(self.inputs, self.INITIAL_FILTERS)
        dense_block_1 = self.get_dense_block(dense_stem, self.GROWTH_RATE, 6)
        N_filters = int_shape(dense_block_1)[-1]

        dense_block_1 = self.make_dropout_layer(dense_block_1)
        trans_block_1 = self.get_transition_block(dense_block_1, N_filters, self.THETA)
        dense_block_2 = self.get_dense_block(trans_block_1, self.GROWTH_RATE, 12)
        N_filters = int_shape(dense_block_2)[-1]

        dense_block_2 = self.make_dropout_layer(dense_block_2)
        trans_block_2 = self.get_transition_block(dense_block_2, N_filters, self.THETA)
        dense_block_3 = self.get_dense_block(trans_block_2, self.GROWTH_RATE, 32)
        N_filters = int_shape(dense_block_3)[-1]

        dense_block_3 = self.make_dropout_layer(dense_block_3)
        trans_block_3 = self.get_transition_block(dense_block_3, N_filters, self.THETA)
        dense_block_4 = self.get_dense_block(trans_block_3, self.GROWTH_RATE, 32)

        gap_1 = GlobalAveragePooling2D(data_format="channels_last")(dense_block_4)

        predictions = self.outputs(gap_1)

        model = Model(inputs=self.inputs, outputs=predictions)

        return model


class DenseNet_169_3D(DenseNet):
    dims = 3
    GROWTH_RATE = 32
    INITIAL_FILTERS = 2 * GROWTH_RATE
    # Settings theta to 1 will create a DenseNet-B
    THETA = 0.5

    def create_model(self):
        self.init_dimensionality(self.dims)
        self.check_minimum_input_size(self.inputs, [21, 21, 21])

        dense_stem = self.get_dense_stem(self.inputs, self.INITIAL_FILTERS)
        dense_block_1 = self.get_dense_block(dense_stem, self.GROWTH_RATE, 6)
        N_filters = int_shape(dense_block_1)[-1]

        dense_block_1 = self.make_dropout_layer(dense_block_1)
        trans_block_1 = self.get_transition_block(dense_block_1, N_filters, self.THETA)
        dense_block_2 = self.get_dense_block(trans_block_1, self.GROWTH_RATE, 12)
        N_filters = int_shape(dense_block_2)[-1]

        dense_block_2 = self.make_dropout_layer(dense_block_2)
        trans_block_2 = self.get_transition_block(dense_block_2, N_filters, self.THETA)
        dense_block_3 = self.get_dense_block(trans_block_2, self.GROWTH_RATE, 32)
        N_filters = int_shape(dense_block_3)[-1]

        dense_block_2 = self.make_dropout_layer(dense_block_2)
        trans_block_3 = self.get_transition_block(dense_block_3, N_filters, self.THETA)
        dense_block_4 = self.get_dense_block(trans_block_3, self.GROWTH_RATE, 32)

        gap_1 = GlobalAveragePooling3D(data_format="channels_last")(dense_block_4)

        predictions = self.outputs(gap_1)

        model = Model(inputs=self.inputs, outputs=predictions)

        return model


class DenseNet_201_2D(DenseNet):
    dims = 2
    GROWTH_RATE = 32
    INITIAL_FILTERS = 2 * GROWTH_RATE
    # Settings theta to 1 will create a DenseNet-B
    THETA = 0.5

    def create_model(self):
        self.init_dimensionality(self.dims)
        self.check_minimum_input_size(self.inputs, [21, 21])

        dense_stem = self.get_dense_stem(self.inputs, self.INITIAL_FILTERS)
        dense_block_1 = self.get_dense_block(dense_stem, self.GROWTH_RATE, 6)
        N_filters = int_shape(dense_block_1)[-1]

        dense_block_1 = self.make_dropout_layer(dense_block_1)
        trans_block_1 = self.get_transition_block(dense_block_1, N_filters, self.THETA)
        dense_block_2 = self.get_dense_block(trans_block_1, self.GROWTH_RATE, 12)
        N_filters = int_shape(dense_block_2)[-1]

        dense_block_2 = self.make_dropout_layer(dense_block_2)
        trans_block_2 = self.get_transition_block(dense_block_2, N_filters, self.THETA)
        dense_block_3 = self.get_dense_block(trans_block_2, self.GROWTH_RATE, 48)
        N_filters = int_shape(dense_block_3)[-1]

        dense_block_3 = self.make_dropout_layer(dense_block_3)
        trans_block_3 = self.get_transition_block(dense_block_3, N_filters, self.THETA)
        dense_block_4 = self.get_dense_block(trans_block_3, self.GROWTH_RATE, 48)

        gap_1 = GlobalAveragePooling2D(data_format="channels_last")(dense_block_4)

        predictions = self.outputs(gap_1)

        model = Model(inputs=self.inputs, outputs=predictions)

        return model


class DenseNet_201_3D(DenseNet):
    dims = 3
    GROWTH_RATE = 32
    INITIAL_FILTERS = 2 * GROWTH_RATE
    # Settings theta to 1 will create a DenseNet-B
    THETA = 0.5

    def create_model(self):
        self.init_dimensionality(self.dims)
        self.check_minimum_input_size(self.inputs, [21, 21, 21])

        dense_stem = self.get_dense_stem(self.inputs, self.INITIAL_FILTERS)
        dense_block_1 = self.get_dense_block(dense_stem, self.GROWTH_RATE, 6)
        N_filters = int_shape(dense_block_1)[-1]

        dense_block_1 = self.make_dropout_layer(dense_block_1)
        trans_block_1 = self.get_transition_block(dense_block_1, N_filters, self.THETA)
        dense_block_2 = self.get_dense_block(trans_block_1, self.GROWTH_RATE, 12)
        N_filters = int_shape(dense_block_2)[-1]

        dense_block_2 = self.make_dropout_layer(dense_block_2)
        trans_block_2 = self.get_transition_block(dense_block_2, N_filters, self.THETA)
        dense_block_3 = self.get_dense_block(trans_block_2, self.GROWTH_RATE, 48)
        N_filters = int_shape(dense_block_3)[-1]

        dense_block_3 = self.make_dropout_layer(dense_block_3)
        trans_block_3 = self.get_transition_block(dense_block_3, N_filters, self.THETA)
        dense_block_4 = self.get_dense_block(trans_block_3, self.GROWTH_RATE, 48)

        gap_1 = GlobalAveragePooling3D(data_format="channels_last")(dense_block_4)

        predictions = self.outputs(gap_1)

        model = Model(inputs=self.inputs, outputs=predictions)

        return model


class DenseNet_264_2D(DenseNet):
    dims = 2
    GROWTH_RATE = 32
    INITIAL_FILTERS = 2 * GROWTH_RATE
    # Settings theta to 1 will create a DenseNet-B
    THETA = 0.5

    def create_model(self):
        self.init_dimensionality(self.dims)
        self.check_minimum_input_size(self.inputs, [21, 21])

        dense_stem = self.get_dense_stem(self.inputs, self.INITIAL_FILTERS)
        dense_block_1 = self.get_dense_block(dense_stem, self.GROWTH_RATE, 6)
        N_filters = int_shape(dense_block_1)[-1]

        dense_block_1 = self.make_dropout_layer(dense_block_1)
        trans_block_1 = self.get_transition_block(dense_block_1, N_filters, self.THETA)
        dense_block_2 = self.get_dense_block(trans_block_1, self.GROWTH_RATE, 12)
        N_filters = int_shape(dense_block_2)[-1]

        dense_block_2 = self.make_dropout_layer(dense_block_2)
        trans_block_2 = self.get_transition_block(dense_block_2, N_filters, self.THETA)
        dense_block_3 = self.get_dense_block(trans_block_2, self.GROWTH_RATE, 64)
        N_filters = int_shape(dense_block_3)[-1]

        dense_block_3 = self.make_dropout_layer(dense_block_3)
        trans_block_3 = self.get_transition_block(dense_block_3, N_filters, self.THETA)
        dense_block_4 = self.get_dense_block(trans_block_3, self.GROWTH_RATE, 48)

        gap_1 = GlobalAveragePooling2D(data_format="channels_last")(dense_block_4)

        predictions = self.outputs(gap_1)

        model = Model(inputs=self.inputs, outputs=predictions)

        return model


class DenseNet_264_3D(DenseNet):
    dims = 3
    GROWTH_RATE = 32
    INITIAL_FILTERS = 2 * GROWTH_RATE
    # Settings theta to 1 will create a DenseNet-B
    THETA = 0.5

    def create_model(self):
        self.init_dimensionality(self.dims)
        self.check_minimum_input_size(self.inputs, [21, 21, 21])

        dense_stem = self.get_dense_stem(self.inputs, self.INITIAL_FILTERS)
        dense_block_1 = self.get_dense_block(dense_stem, self.GROWTH_RATE, 6)
        N_filters = int_shape(dense_block_1)[-1]

        dense_block_1 = self.make_dropout_layer(dense_block_1)
        trans_block_1 = self.get_transition_block(dense_block_1, N_filters, self.THETA)
        dense_block_2 = self.get_dense_block(trans_block_1, self.GROWTH_RATE, 12)
        N_filters = int_shape(dense_block_2)[-1]

        dense_block_2 = self.make_dropout_layer(dense_block_2)
        trans_block_2 = self.get_transition_block(dense_block_2, N_filters, self.THETA)
        dense_block_3 = self.get_dense_block(trans_block_2, self.GROWTH_RATE, 64)
        N_filters = int_shape(dense_block_3)[-1]

        dense_block_3 = self.make_dropout_layer(dense_block_3)
        trans_block_3 = self.get_transition_block(dense_block_3, N_filters, self.THETA)
        dense_block_4 = self.get_dense_block(trans_block_3, self.GROWTH_RATE, 48)

        gap_1 = GlobalAveragePooling3D(data_format="channels_last")(dense_block_4)

        predictions = self.outputs(gap_1)

        model = Model(inputs=self.inputs, outputs=predictions)

        return model
