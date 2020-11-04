from PrognosAIs.Model.Architectures.Architecture import ClassificationNetworkArchitecture
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv3D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import MaxPooling3D
from tensorflow.keras.layers import ReLU
from tensorflow.keras.models import Model


class AlexNet_2D(ClassificationNetworkArchitecture):
    padding_type = "valid"

    def create_model(self):
        self.check_minimum_input_size(self.inputs, [67, 67])
        stride_size = self.get_corrected_stride_size(self.inputs, [4, 4], [11, 11])

        conv_1 = Conv2D(filters=96, kernel_size=(11, 11), strides=stride_size)(self.inputs)

        relu_1 = ReLU()(conv_1)

        stride_size = self.get_corrected_stride_size(relu_1, [2, 2], [3, 3])

        pooling_1 = MaxPooling2D(
            pool_size=(3, 3),
            strides=stride_size,
            padding=self.padding_type,
            data_format="channels_last",
        )(relu_1)

        conv_2 = Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding="same")(pooling_1)

        relu_2 = ReLU()(conv_2)

        stride_size = self.get_corrected_stride_size(relu_2, [2, 2], [3, 3])

        pooling_2 = MaxPooling2D(
            pool_size=(3, 3),
            strides=stride_size,
            padding=self.padding_type,
            data_format="channels_last",
        )(relu_2)

        conv_3 = Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding="same")(pooling_2)

        relu_3 = ReLU()(conv_3)

        conv_4 = Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding="same")(relu_3)

        relu_4 = ReLU()(conv_4)

        conv_5 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same")(relu_4)

        relu_5 = ReLU()(conv_5)

        stride_size = self.get_corrected_stride_size(relu_5, [2, 2], [3, 3])

        pooling_5 = MaxPooling2D(
            pool_size=(3, 3),
            strides=stride_size,
            padding=self.padding_type,
            data_format="channels_last",
        )(relu_5)

        flatten_1 = Flatten()(pooling_5)

        dense_1 = Dense(
            units=4096,
            activation=None,
            use_bias=True,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
        )(flatten_1)

        dropout_1 = Dropout(0.5)(dense_1)

        dense_2 = Dense(
            units=4096,
            activation=None,
            use_bias=True,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
        )(dropout_1)

        dropout_2 = Dropout(0.5)(dense_2)

        predictions = self.outputs(dropout_2)

        model = Model(inputs=self.inputs, outputs=predictions)

        return model


class AlexNet_3D(ClassificationNetworkArchitecture):
    padding_type = "valid"

    def create_model(self):
        self.check_minimum_input_size(self.inputs, [67, 67, 67])
        stride_size = self.get_corrected_stride_size(self.inputs, [4, 4, 4], [11, 11, 11])

        conv_1 = Conv3D(filters=96, kernel_size=(11, 11, 11), strides=stride_size)(self.inputs)

        relu_1 = ReLU()(conv_1)

        stride_size = self.get_corrected_stride_size(relu_1, [2, 2, 2], [3, 3, 3])

        pooling_1 = MaxPooling3D(
            pool_size=(3, 3, 3),
            strides=stride_size,
            padding=self.padding_type,
            data_format="channels_last",
        )(relu_1)

        conv_2 = Conv3D(filters=256, kernel_size=(5, 5, 5), strides=(1, 1, 1), padding="same")(
            pooling_1
        )

        relu_2 = ReLU()(conv_2)

        stride_size = self.get_corrected_stride_size(relu_2, [2, 2, 2], [3, 3, 3])

        pooling_2 = MaxPooling3D(
            pool_size=(3, 3, 3),
            strides=stride_size,
            padding=self.padding_type,
            data_format="channels_last",
        )(relu_2)

        conv_3 = Conv3D(filters=384, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same")(
            pooling_2
        )

        relu_3 = ReLU()(conv_3)

        conv_4 = Conv3D(filters=384, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same")(
            relu_3
        )

        relu_4 = ReLU()(conv_4)

        conv_5 = Conv3D(filters=256, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same")(
            relu_4
        )

        relu_5 = ReLU()(conv_5)

        stride_size = self.get_corrected_stride_size(relu_5, [2, 2, 2], [3, 3, 3])

        pooling_5 = MaxPooling3D(
            pool_size=(3, 3, 3),
            strides=stride_size,
            padding=self.padding_type,
            data_format="channels_last",
        )(relu_5)

        flatten_1 = Flatten()(pooling_5)

        dense_1 = Dense(
            units=4096,
            activation=None,
            use_bias=True,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
        )(flatten_1)

        dropout_1 = Dropout(0.5)(dense_1)

        dense_2 = Dense(
            units=4096,
            activation=None,
            use_bias=True,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
        )(dropout_1)

        dropout_2 = Dropout(0.5)(dense_2)

        predictions = self.outputs(dropout_2)

        model = Model(inputs=self.inputs, outputs=predictions)

        return model
