import tensorflow

from PrognosAIs.Model.Architectures.Architecture import ClassificationNetworkArchitecture
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv3D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import GlobalAveragePooling3D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import MaxPooling3D
from tensorflow.keras.layers import ReLU
from tensorflow.keras.models import Model


class ResNet(ClassificationNetworkArchitecture):
    def get_residual_conv_block(
        self, layer: tensorflow.keras.layers, N_filters: int, kernel_size: list
    ):
        N_dims = len(kernel_size)

        if N_dims == 2:
            conv_func = Conv2D
        elif N_dims == 3:
            conv_func = Conv3D

        stride_size = self.get_corrected_stride_size(layer, [2] * N_dims, kernel_size)

        conv_1 = conv_func(
            filters=N_filters, kernel_size=kernel_size, strides=stride_size, padding="same",
        )(layer)
        batchnorm_1 = BatchNormalization(axis=-1)(conv_1)
        relu_1 = ReLU()(batchnorm_1)

        conv_2 = conv_func(
            filters=N_filters, kernel_size=kernel_size, strides=[1] * N_dims, padding="same"
        )(relu_1)
        batchnorm_2 = BatchNormalization(axis=-1)(conv_2)

        conv_skip = conv_func(
            filters=N_filters, kernel_size=[1] * N_dims, strides=stride_size, padding="valid"
        )(layer)
        batch_skip = BatchNormalization(axis=-1)(conv_skip)

        res_output = Add()([batch_skip, batchnorm_2])

        relu_output = ReLU()(res_output)

        return relu_output

    def get_residual_identity_block(self, layer, N_filters, kernel_size):
        N_dims = len(kernel_size)
        if N_dims == 2:
            conv_func = Conv2D
        elif N_dims == 3:
            conv_func = Conv3D

        conv_1 = conv_func(
            filters=N_filters, kernel_size=kernel_size, strides=[1] * N_dims, padding="same"
        )(layer)
        batchnorm_1 = BatchNormalization(axis=-1)(conv_1)
        relu_1 = ReLU()(batchnorm_1)

        conv_2 = conv_func(
            filters=N_filters, kernel_size=kernel_size, strides=[1] * N_dims, padding="same"
        )(relu_1)
        batchnorm_2 = BatchNormalization(axis=-1)(conv_2)

        res_output = Add()([batchnorm_2, layer])
        relu_output = ReLU()(res_output)

        return relu_output


class ResNet_18_2D(ResNet):
    def create_model(self):
        self.check_minimum_input_size(self.inputs, [9, 9])
        stride_size = self.get_corrected_stride_size(self.inputs, [2, 2], [7, 7])

        conv_1 = Conv2D(
            filters=64, kernel_size=[7, 7], strides=stride_size, padding="valid", activation="relu",
        )(self.inputs)

        stride_size = self.get_corrected_stride_size(conv_1, [2, 2], [3, 3])

        pooling_1 = MaxPooling2D(pool_size=[3, 3], strides=stride_size, padding="valid",)(conv_1)

        res_block_1 = self.get_residual_identity_block(pooling_1, 64, [3, 3])
        res_block_2 = self.get_residual_identity_block(res_block_1, 64, [3, 3])

        res_block_2 = self.make_dropout_layer(res_block_2)

        res_block_3 = self.get_residual_conv_block(res_block_2, 128, [3, 3])
        res_block_4 = self.get_residual_identity_block(res_block_3, 128, [3, 3])

        res_block_4 = self.make_dropout_layer(res_block_4)

        res_block_5 = self.get_residual_conv_block(res_block_4, 256, [3, 3])
        res_block_6 = self.get_residual_identity_block(res_block_5, 256, [3, 3])

        res_block_6 = self.make_dropout_layer(res_block_6)

        res_block_7 = self.get_residual_conv_block(res_block_6, 512, [3, 3])
        res_block_8 = self.get_residual_identity_block(res_block_7, 512, [3, 3])

        gap_1 = GlobalAveragePooling2D()(res_block_8)

        gap_1 = self.make_dropout_layer(gap_1)

        predictions = self.outputs(gap_1)

        model = Model(inputs=self.inputs, outputs=predictions)

        return model


class ResNet_34_2D(ResNet):
    def create_model(self):
        self.check_minimum_input_size(self.inputs, [9, 9])
        stride_size = self.get_corrected_stride_size(self.inputs, [2, 2], [7, 7])

        conv_1 = Conv2D(
            filters=64, kernel_size=[7, 7], strides=stride_size, padding="valid", activation="relu",
        )(self.inputs)

        stride_size = self.get_corrected_stride_size(conv_1, [2, 2], [3, 3])

        pooling_1 = MaxPooling2D(pool_size=[3, 3], strides=stride_size, padding="valid",)(conv_1)

        res_block_1 = self.get_residual_identity_block(pooling_1, 64, [3, 3])
        res_block_2 = self.get_residual_identity_block(res_block_1, 64, [3, 3])
        res_block_3 = self.get_residual_identity_block(res_block_2, 64, [3, 3])

        res_block_3 = self.make_dropout_layer(res_block_3)

        res_block_4 = self.get_residual_conv_block(res_block_3, 128, [3, 3])
        res_block_5 = self.get_residual_identity_block(res_block_4, 128, [3, 3])
        res_block_6 = self.get_residual_identity_block(res_block_5, 128, [3, 3])
        res_block_7 = self.get_residual_identity_block(res_block_6, 128, [3, 3])

        res_block_7 = self.make_dropout_layer(res_block_7)

        res_block_8 = self.get_residual_conv_block(res_block_7, 256, [3, 3])
        res_block_9 = self.get_residual_identity_block(res_block_8, 256, [3, 3])
        res_block_10 = self.get_residual_identity_block(res_block_9, 256, [3, 3])
        res_block_11 = self.get_residual_identity_block(res_block_10, 256, [3, 3])
        res_block_12 = self.get_residual_identity_block(res_block_11, 256, [3, 3])
        res_block_13 = self.get_residual_identity_block(res_block_12, 256, [3, 3])

        res_block_13 = self.make_dropout_layer(res_block_13)

        res_block_14 = self.get_residual_conv_block(res_block_13, 512, [3, 3])
        res_block_15 = self.get_residual_identity_block(res_block_14, 512, [3, 3])
        res_block_16 = self.get_residual_identity_block(res_block_15, 512, [3, 3])

        gap_1 = GlobalAveragePooling2D()(res_block_16)

        gap_1 = self.make_dropout_layer(gap_1)

        predictions = self.outputs(gap_1)

        model = Model(inputs=self.inputs, outputs=predictions)

        return model


class ResNet_18_3D(ResNet):
    def create_model(self):
        self.check_minimum_input_size(self.inputs, [9, 9, 9])

        stride_size = self.get_corrected_stride_size(self.inputs, [2, 2, 2], [7, 7, 7])

        conv_1 = Conv3D(
            filters=64,
            kernel_size=[7, 7, 7],
            strides=stride_size,
            padding="valid",
            activation="relu",
        )(self.inputs)

        stride_size = self.get_corrected_stride_size(conv_1, [2, 2, 2], [3, 3, 3])

        pooling_1 = MaxPooling3D(pool_size=[3, 3, 3], strides=stride_size, padding="valid",)(conv_1)

        res_block_1 = self.get_residual_identity_block(pooling_1, 64, [3, 3, 3])
        res_block_2 = self.get_residual_identity_block(res_block_1, 64, [3, 3, 3])

        res_block_2 = self.make_dropout_layer(res_block_2)

        res_block_3 = self.get_residual_conv_block(res_block_2, 128, [3, 3, 3])
        res_block_4 = self.get_residual_identity_block(res_block_3, 128, [3, 3, 3])

        res_block_4 = self.make_dropout_layer(res_block_4)

        res_block_5 = self.get_residual_conv_block(res_block_4, 256, [3, 3, 3])
        res_block_6 = self.get_residual_identity_block(res_block_5, 256, [3, 3, 3])

        res_block_6 = self.make_dropout_layer(res_block_6)

        res_block_7 = self.get_residual_conv_block(res_block_6, 512, [3, 3, 3])
        res_block_8 = self.get_residual_identity_block(res_block_7, 512, [3, 3, 3])

        gap_1 = GlobalAveragePooling3D()(res_block_8)

        gap_1 = self.make_dropout_layer(gap_1)

        predictions = self.outputs(gap_1)

        model = Model(inputs=self.inputs, outputs=predictions)
        return model


class ResNet_18_multioutput_3D(ResNet):
    def create_model(self):
        self.check_minimum_input_size(self.inputs, [9, 9, 9])

        stride_size = self.get_corrected_stride_size(self.inputs, [2, 2, 2], [7, 7, 7])

        conv_1 = Conv3D(
            filters=64,
            kernel_size=[7, 7, 7],
            strides=stride_size,
            padding="valid",
            activation="relu",
        )(self.inputs)

        stride_size = self.get_corrected_stride_size(conv_1, [2, 2, 2], [3, 3, 3])

        pooling_1 = MaxPooling3D(pool_size=[3, 3, 3], strides=stride_size, padding="valid",)(conv_1)

        res_block_1 = self.get_residual_identity_block(pooling_1, 64, [3, 3, 3])
        res_block_2 = self.get_residual_identity_block(res_block_1, 64, [3, 3, 3])

        res_block_2 = self.make_dropout_layer(res_block_2)

        res_block_3 = self.get_residual_conv_block(res_block_2, 128, [3, 3, 3])
        res_block_4 = self.get_residual_identity_block(res_block_3, 128, [3, 3, 3])

        res_block_4 = self.make_dropout_layer(res_block_4)

        res_block_5 = self.get_residual_conv_block(res_block_4, 256, [3, 3, 3])
        res_block_6 = self.get_residual_identity_block(res_block_5, 256, [3, 3, 3])

        res_block_6 = self.make_dropout_layer(res_block_6)

        predictions = []
        for i_output in self.outputs.values():
            res_block_7_output_branch = self.get_residual_conv_block(res_block_6, 512, [3, 3, 3])
            res_block_8_output_branch = self.get_residual_identity_block(
                res_block_7_output_branch, 512, [3, 3, 3]
            )

            gap_1_output_branch = GlobalAveragePooling3D()(res_block_8_output_branch)

            gap_1_output_branch = self.make_dropout_layer(gap_1_output_branch)

            predictions.append(i_output(gap_1_output_branch))

        model = Model(inputs=self.inputs, outputs=predictions)
        return model


class ResNet_34_3D(ResNet):
    def create_model(self):
        self.check_minimum_input_size(self.inputs, [9, 9, 9])
        stride_size = self.get_corrected_stride_size(self.inputs, [2, 2, 2], [7, 7, 7])

        conv_1 = Conv3D(
            filters=64,
            kernel_size=[7, 7, 7],
            strides=stride_size,
            padding="valid",
            activation="relu",
        )(self.inputs)

        stride_size = self.get_corrected_stride_size(conv_1, [2, 2, 2], [3, 3, 3])

        pooling_1 = MaxPooling3D(pool_size=[3, 3, 3], strides=stride_size, padding="valid",)(conv_1)

        res_block_1 = self.get_residual_identity_block(pooling_1, 64, [3, 3, 3])
        res_block_2 = self.get_residual_identity_block(res_block_1, 64, [3, 3, 3])
        res_block_3 = self.get_residual_identity_block(res_block_2, 64, [3, 3, 3])

        res_block_3 = self.make_dropout_layer(res_block_3)

        res_block_4 = self.get_residual_conv_block(res_block_3, 128, [3, 3, 3])
        res_block_5 = self.get_residual_identity_block(res_block_4, 128, [3, 3, 3])
        res_block_6 = self.get_residual_identity_block(res_block_5, 128, [3, 3, 3])
        res_block_7 = self.get_residual_identity_block(res_block_6, 128, [3, 3, 3])

        res_block_7 = self.make_dropout_layer(res_block_7)

        res_block_8 = self.get_residual_conv_block(res_block_7, 256, [3, 3, 3])
        res_block_9 = self.get_residual_identity_block(res_block_8, 256, [3, 3, 3])
        res_block_10 = self.get_residual_identity_block(res_block_9, 256, [3, 3, 3])
        res_block_11 = self.get_residual_identity_block(res_block_10, 256, [3, 3, 3])
        res_block_12 = self.get_residual_identity_block(res_block_11, 256, [3, 3, 3])
        res_block_13 = self.get_residual_identity_block(res_block_12, 256, [3, 3, 3])

        res_block_13 = self.make_dropout_layer(res_block_13)

        res_block_14 = self.get_residual_conv_block(res_block_13, 512, [3, 3, 3])
        res_block_15 = self.get_residual_identity_block(res_block_14, 512, [3, 3, 3])
        res_block_16 = self.get_residual_identity_block(res_block_15, 512, [3, 3, 3])

        gap_1 = GlobalAveragePooling3D()(res_block_16)

        gap_1 = self.make_dropout_layer(gap_1)

        predictions = self.outputs(gap_1)

        model = Model(inputs=self.inputs, outputs=predictions)

        return model
