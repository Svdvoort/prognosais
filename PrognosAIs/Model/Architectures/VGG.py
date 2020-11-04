from PrognosAIs.Model.Architectures.Architecture import ClassificationNetworkArchitecture
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv3D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import MaxPooling3D
from tensorflow.keras.models import Model


class VGG(ClassificationNetworkArchitecture):
    def init_dimensionality(self, N_dimension):
        if N_dimension == 2:
            self.dims = 2
            self.conv_func = Conv2D
            self.pool_func = MaxPooling2D
        elif N_dimension == 3:
            self.dims = 3
            self.conv_func = Conv3D
            self.pool_func = MaxPooling3D

    def get_VGG_block(self, layer, N_filters, N_conv_layer):
        stride_size = self.get_corrected_stride_size(layer, [2] * self.dims, [2] * self.dims)

        pooling = self.pool_func(pool_size=[2] * self.dims, strides=stride_size, padding="valid")(
            layer
        )

        conv = self.conv_func(
            filters=N_filters,
            kernel_size=[3] * self.dims,
            strides=[1] * self.dims,
            padding="same",
            activation="relu",
        )(pooling)

        for i_conv_layer in range(N_conv_layer - 1):
            conv = self.conv_func(
                filters=N_filters,
                kernel_size=[3] * self.dims,
                strides=[1] * self.dims,
                padding="same",
                activation="relu",
            )(conv)

        return conv


class VGG_16_2D(VGG):
    dims = 2

    def create_model(self):
        self.init_dimensionality(self.dims)
        self.check_minimum_input_size(self.inputs, [24, 24])

        conv_1 = Conv2D(
            filters=64, kernel_size=[3, 3], strides=[1, 1], padding="same", activation="relu",
        )(self.inputs)

        conv_2 = Conv2D(
            filters=64, kernel_size=[3, 3], strides=[1, 1], padding="same", activation="relu",
        )(conv_1)

        vgg_block_1 = self.get_VGG_block(conv_2, 128, 2)
        vgg_block_1 = self.make_dropout_layer(vgg_block_1)
        vgg_block_2 = self.get_VGG_block(vgg_block_1, 256, 3)
        vgg_block_2 = self.make_dropout_layer(vgg_block_2)
        vgg_block_3 = self.get_VGG_block(vgg_block_2, 512, 3)
        vgg_block_3 = self.make_dropout_layer(vgg_block_3)
        vgg_block_4 = self.get_VGG_block(vgg_block_3, 512, 3)

        stride_size = self.get_corrected_stride_size(vgg_block_4, [2, 2], [2, 2])
        pooling_1 = MaxPooling2D(pool_size=[2, 2], strides=stride_size, padding="valid",)(
            vgg_block_4
        )
        flatten_1 = Flatten()(pooling_1)
        flatten_1 = self.make_dropout_layer(flatten_1)
        dense_1 = Dense(4096, activation="relu")(flatten_1)
        dense_2 = Dense(4096, activation="relu")(dense_1)
        predictions = self.outputs(dense_2)

        model = Model(inputs=self.inputs, outputs=predictions)

        return model


class VGG_16_3D(VGG):
    dims = 3

    def create_model(self):
        self.init_dimensionality(self.dims)
        self.check_minimum_input_size(self.inputs, [24, 24, 24])

        conv_1 = Conv3D(
            filters=64, kernel_size=[3, 3, 3], strides=[1, 1, 1], padding="same", activation="relu",
        )(self.inputs)

        conv_2 = Conv3D(
            filters=64, kernel_size=[3, 3, 3], strides=[1, 1, 1], padding="same", activation="relu",
        )(conv_1)

        vgg_block_1 = self.get_VGG_block(conv_2, 128, 2)
        vgg_block_1 = self.make_dropout_layer(vgg_block_1)
        vgg_block_2 = self.get_VGG_block(vgg_block_1, 256, 3)
        vgg_block_2 = self.make_dropout_layer(vgg_block_2)
        vgg_block_3 = self.get_VGG_block(vgg_block_2, 512, 3)
        vgg_block_3 = self.make_dropout_layer(vgg_block_3)
        vgg_block_4 = self.get_VGG_block(vgg_block_3, 512, 3)

        stride_size = self.get_corrected_stride_size(vgg_block_4, [2, 2, 2], [2, 2, 2])
        pooling_1 = MaxPooling3D(pool_size=[2, 2, 2], strides=stride_size, padding="valid",)(
            vgg_block_4
        )
        flatten_1 = Flatten()(pooling_1)
        flatten_1 = self.make_dropout_layer(flatten_1)
        dense_1 = Dense(4096, activation="relu")(flatten_1)
        dense_2 = Dense(4096, activation="relu")(dense_1)
        predictions = self.outputs(dense_2)

        model = Model(inputs=self.inputs, outputs=predictions)

        return model


class VGG_19_2D(VGG):
    dims = 2

    def create_model(self):
        self.init_dimensionality(self.dims)
        self.check_minimum_input_size(self.inputs, [24, 24])

        conv_1 = Conv2D(
            filters=64, kernel_size=[3, 3], strides=[1, 1], padding="same", activation="relu",
        )(self.inputs)

        conv_2 = Conv2D(
            filters=64, kernel_size=[3, 3], strides=[1, 1], padding="same", activation="relu",
        )(conv_1)

        vgg_block_1 = self.get_VGG_block(conv_2, 128, 2)
        vgg_block_1 = self.make_dropout_layer(vgg_block_1)
        vgg_block_2 = self.get_VGG_block(vgg_block_1, 256, 4)
        vgg_block_2 = self.make_dropout_layer(vgg_block_2)
        vgg_block_3 = self.get_VGG_block(vgg_block_2, 512, 4)
        vgg_block_3 = self.make_dropout_layer(vgg_block_3)
        vgg_block_4 = self.get_VGG_block(vgg_block_3, 512, 4)

        stride_size = self.get_corrected_stride_size(vgg_block_4, [2, 2], [2, 2])
        pooling_1 = MaxPooling2D(pool_size=[2, 2], strides=stride_size, padding="valid",)(
            vgg_block_4
        )
        flatten_1 = Flatten()(pooling_1)
        flatten_1 = self.make_dropout_layer(flatten_1)
        dense_1 = Dense(4096, activation="relu")(flatten_1)
        dense_2 = Dense(4096, activation="relu")(dense_1)
        predictions = self.outputs(dense_2)

        model = Model(inputs=self.inputs, outputs=predictions)

        return model


class VGG_19_3D(VGG):
    dims = 3

    def create_model(self):
        self.init_dimensionality(self.dims)
        self.check_minimum_input_size(self.inputs, [24, 24, 24])

        conv_1 = Conv3D(
            filters=64, kernel_size=[3, 3, 3], strides=[1, 1, 1], padding="same", activation="relu",
        )(self.inputs)

        conv_2 = Conv3D(
            filters=64, kernel_size=[3, 3, 3], strides=[1, 1, 1], padding="same", activation="relu",
        )(conv_1)

        vgg_block_1 = self.get_VGG_block(conv_2, 128, 2)
        vgg_block_1 = self.make_dropout_layer(vgg_block_1)
        vgg_block_2 = self.get_VGG_block(vgg_block_1, 256, 4)
        vgg_block_2 = self.make_dropout_layer(vgg_block_2)
        vgg_block_3 = self.get_VGG_block(vgg_block_2, 512, 4)
        vgg_block_3 = self.make_dropout_layer(vgg_block_3)
        vgg_block_4 = self.get_VGG_block(vgg_block_3, 512, 4)

        stride_size = self.get_corrected_stride_size(vgg_block_4, [2, 2, 2], [2, 2, 2])
        pooling_1 = MaxPooling3D(pool_size=[2, 2, 2], strides=stride_size, padding="valid",)(
            vgg_block_4
        )
        flatten_1 = Flatten()(pooling_1)
        flatten_1 = self.make_dropout_layer(flatten_1)
        dense_1 = Dense(4096, activation="relu")(flatten_1)
        dense_2 = Dense(4096, activation="relu")(dense_1)
        predictions = self.outputs(dense_2)

        model = Model(inputs=self.inputs, outputs=predictions)

        return model
