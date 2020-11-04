import numpy as np

from PrognosAIs.Model.Architectures.Architecture import ClassificationNetworkArchitecture
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv3D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import GlobalAveragePooling3D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import MaxPooling3D
from tensorflow.keras.layers import ReLU
from tensorflow.keras.models import Model


class InceptionResNet(ClassificationNetworkArchitecture):
    def init_dimensionality(self, N_dimension):
        if N_dimension == 2:
            self.dims = 2
            self.conv_func = Conv2D
            self.pool_func = MaxPooling2D
        elif N_dimension == 3:
            self.dims = 3
            self.conv_func = Conv3D
            self.pool_func = MaxPooling3D

    def get_inception_stem(self, layer):
        stride_size = self.get_corrected_stride_size(layer, [2] * self.dims, [3] * self.dims)
        conv_1 = self.conv_func(
            filters=32,
            kernel_size=[3] * self.dims,
            strides=stride_size,
            padding="valid",
            data_format="channels_last",
            activation="relu",
        )(layer)
        bn_1 = BatchNormalization(axis=-1, scale=False)(conv_1)

        conv_2 = self.conv_func(
            filters=32,
            kernel_size=[3] * self.dims,
            strides=[1] * self.dims,
            padding="valid",
            data_format="channels_last",
            activation="relu",
        )(bn_1)

        bn_2 = BatchNormalization(axis=-1, scale=False)(conv_2)

        conv_3 = self.conv_func(
            filters=64,
            kernel_size=[3] * self.dims,
            strides=[1] * self.dims,
            padding="same",
            data_format="channels_last",
            activation="relu",
        )(bn_2)

        bn_3 = BatchNormalization(axis=-1, scale=False)(conv_3)

        stride_size = self.get_corrected_stride_size(bn_3, [2] * self.dims, [3] * self.dims)
        pooling_1 = self.pool_func(
            pool_size=[3] * self.dims,
            strides=stride_size,
            padding="valid",
            data_format="channels_last",
        )(bn_3)

        stride_size = self.get_corrected_stride_size(conv_3, [2] * self.dims, [3] * self.dims)

        conv_4 = self.conv_func(
            filters=96,
            kernel_size=[3] * self.dims,
            strides=stride_size,
            padding="valid",
            data_format="channels_last",
            activation="relu",
        )(conv_3)

        bn_4 = BatchNormalization(axis=-1, scale=False)(conv_4)

        concat_1 = Concatenate(axis=-1)([pooling_1, bn_4])

        conv_5_a_1 = self.conv_func(
            filters=64,
            kernel_size=[1] * self.dims,
            strides=[1] * self.dims,
            padding="same",
            data_format="channels_last",
            activation="relu",
        )(concat_1)
        bn_5_a_1 = BatchNormalization(axis=-1, scale=False)(conv_5_a_1)

        conv_5_a_2 = self.conv_func(
            filters=96,
            kernel_size=[3] * self.dims,
            strides=[1] * self.dims,
            padding="valid",
            data_format="channels_last",
            activation="relu",
        )(bn_5_a_1)
        bn_5_a_2 = BatchNormalization(axis=-1, scale=False)(conv_5_a_2)

        conv_5_b_1 = self.conv_func(
            filters=64,
            kernel_size=[1] * self.dims,
            strides=[1] * self.dims,
            padding="same",
            data_format="channels_last",
            activation="relu",
        )(concat_1)
        bn_5_b_1 = BatchNormalization(axis=-1, scale=False)(conv_5_b_1)

        head = bn_5_b_1
        for i_dim in range(self.dims):
            kernel_size = np.ones(self.dims, dtype=np.int)
            kernel_size[i_dim] = 7
            head = self.conv_func(
                filters=64,
                kernel_size=kernel_size.tolist(),
                strides=[1] * self.dims,
                padding="same",
                data_format="channels_last",
                activation="relu",
            )(head)
            head = BatchNormalization(axis=-1, scale=False)(head)

        conv_5_b_5 = self.conv_func(
            filters=96,
            kernel_size=[3] * self.dims,
            strides=[1] * self.dims,
            padding="valid",
            data_format="channels_last",
            activation="relu",
        )(head)
        bn_5_b_5 = BatchNormalization(axis=-1, scale=False)(conv_5_b_5)

        concat_2 = Concatenate(axis=-1)([bn_5_a_2, bn_5_b_5])

        stride_size = self.get_corrected_stride_size(concat_2, [2] * self.dims, [3] * self.dims)
        conv_6 = self.conv_func(
            filters=192,
            kernel_size=[3] * self.dims,
            strides=stride_size,
            padding="valid",
            data_format="channels_last",
            activation="relu",
        )(concat_2)
        bn_6 = BatchNormalization(axis=-1, scale=False)(conv_6)

        pooling_2 = self.pool_func(
            pool_size=[3] * self.dims,
            strides=stride_size,
            padding="valid",
            data_format="channels_last",
        )(concat_2)

        concat_3 = Concatenate(axis=-1)([bn_6, pooling_2])

        return concat_3

    def get_inception_resnet_A(self, layer):
        relu_1 = ReLU()(layer)

        conv_1_a_1 = self.conv_func(
            filters=32,
            kernel_size=[1] * self.dims,
            strides=[1] * self.dims,
            padding="same",
            data_format="channels_last",
            activation="relu",
        )(relu_1)
        bn_1_a_1 = BatchNormalization(axis=-1, scale=False)(conv_1_a_1)

        conv_1_b_1 = self.conv_func(
            filters=32,
            kernel_size=[1] * self.dims,
            strides=[1] * self.dims,
            padding="same",
            data_format="channels_last",
            activation="relu",
        )(relu_1)
        bn_1_b_1 = BatchNormalization(axis=-1, scale=False)(conv_1_b_1)

        conv_1_b_2 = self.conv_func(
            filters=32,
            kernel_size=[3] * self.dims,
            strides=[1] * self.dims,
            padding="same",
            data_format="channels_last",
            activation="relu",
        )(bn_1_b_1)
        bn_1_b_2 = BatchNormalization(axis=-1, scale=False)(conv_1_b_2)

        conv_1_c_1 = self.conv_func(
            filters=32,
            kernel_size=[1] * self.dims,
            strides=[1] * self.dims,
            padding="same",
            data_format="channels_last",
            activation="relu",
        )(relu_1)
        bn_1_c_1 = BatchNormalization(axis=-1, scale=False)(conv_1_c_1)

        conv_1_c_2 = self.conv_func(
            filters=48,
            kernel_size=[3] * self.dims,
            strides=[1] * self.dims,
            padding="same",
            data_format="channels_last",
            activation="relu",
        )(bn_1_c_1)
        bn_1_c_2 = BatchNormalization(axis=-1, scale=False)(conv_1_c_2)

        conv_1_c_3 = self.conv_func(
            filters=64,
            kernel_size=[3] * self.dims,
            strides=[1] * self.dims,
            padding="same",
            data_format="channels_last",
            activation="relu",
        )(bn_1_c_2)
        bn_1_c_3 = BatchNormalization(axis=-1, scale=False)(conv_1_c_3)

        concat_1 = Concatenate(axis=-1)([bn_1_a_1, bn_1_b_2, bn_1_c_3])

        conv_2 = self.conv_func(
            filters=384,
            kernel_size=[1] * self.dims,
            strides=[1] * self.dims,
            padding="same",
            data_format="channels_last",
            activation="linear",
        )(concat_1)

        add_1 = Add()([conv_2, relu_1])
        relu_2 = ReLU()(add_1)

        bn_relu = BatchNormalization(axis=-1, scale=False)(relu_2)

        return bn_relu

    def get_inception_resnet_reduction_A(self, layer):
        stride_size = self.get_corrected_stride_size(layer, [2] * self.dims, [3] * self.dims)

        pooling_1 = self.pool_func(
            pool_size=[3] * self.dims,
            strides=stride_size,
            padding="valid",
            data_format="channels_last",
        )(layer)

        stride_size = self.get_corrected_stride_size(layer, [2] * self.dims, [3] * self.dims)
        conv_1_a_1 = self.conv_func(
            filters=384,
            kernel_size=[3] * self.dims,
            strides=stride_size,
            padding="valid",
            data_format="channels_last",
            activation="relu",
        )(layer)
        bn_1_a_1 = BatchNormalization(axis=-1, scale=False)(conv_1_a_1)

        conv_1_b_1 = self.conv_func(
            filters=256,
            kernel_size=[1] * self.dims,
            strides=[1] * self.dims,
            padding="same",
            data_format="channels_last",
            activation="relu",
        )(layer)
        bn_1_b_1 = BatchNormalization(axis=-1, scale=False)(conv_1_b_1)

        conv_1_b_2 = self.conv_func(
            filters=256,
            kernel_size=[3] * self.dims,
            strides=[1] * self.dims,
            padding="same",
            data_format="channels_last",
            activation="relu",
        )(bn_1_b_1)
        bn_1_b_2 = BatchNormalization(axis=-1, scale=False)(conv_1_b_2)
        stride_size = self.get_corrected_stride_size(bn_1_b_2, [2] * self.dims, [3] * self.dims)
        conv_1_b_3 = self.conv_func(
            filters=384,
            kernel_size=[3] * self.dims,
            strides=stride_size,
            padding="valid",
            data_format="channels_last",
            activation="relu",
        )(bn_1_b_2)
        bn_1_b_3 = BatchNormalization(axis=-1, scale=False)(conv_1_b_3)

        concat_1 = Concatenate()([pooling_1, bn_1_a_1, bn_1_b_3])

        return concat_1

    def get_inception_resnet_B(self, layer):
        relu_1 = ReLU()(layer)

        conv_1_a_1 = self.conv_func(
            filters=192,
            kernel_size=[1] * self.dims,
            strides=[1] * self.dims,
            padding="same",
            data_format="channels_last",
            activation="relu",
        )(relu_1)
        bn_1_a_1 = BatchNormalization(axis=-1, scale=False)(conv_1_a_1)

        conv_1_b_1 = self.conv_func(
            filters=128,
            kernel_size=[1] * self.dims,
            strides=[1] * self.dims,
            padding="same",
            data_format="channels_last",
            activation="relu",
        )(relu_1)
        bn_1_b_1 = BatchNormalization(axis=-1, scale=False)(conv_1_b_1)

        head = bn_1_b_1
        start_filters = 160
        for i_dim in range(self.dims):
            kernel_size = np.ones(self.dims, dtype=np.int)
            kernel_size[i_dim] = 7
            head = self.conv_func(
                filters=start_filters + 32 * i_dim,
                kernel_size=kernel_size.tolist(),
                strides=[1] * self.dims,
                padding="same",
                data_format="channels_last",
                activation="relu",
            )(head)
            head = BatchNormalization(axis=-1, scale=False)(head)

        concat_1 = Concatenate(axis=-1)([bn_1_a_1, head])

        conv_2 = self.conv_func(
            filters=1152,
            kernel_size=[1] * self.dims,
            strides=[1] * self.dims,
            padding="same",
            data_format="channels_last",
            activation="linear",
        )(concat_1)

        add_1 = Add()([conv_2, relu_1])
        relu_2 = ReLU()(add_1)
        bn_relu = BatchNormalization(axis=-1, scale=False)(relu_2)

        return bn_relu

    def get_inception_resnet_reduction_B(self, layer):
        stride_size = self.get_corrected_stride_size(layer, [2] * self.dims, [3] * self.dims)

        pooling_1 = self.pool_func(
            pool_size=[3] * self.dims,
            strides=stride_size,
            padding="valid",
            data_format="channels_last",
        )(layer)

        conv_1_a_1 = self.conv_func(
            filters=256,
            kernel_size=[1] * self.dims,
            strides=[1] * self.dims,
            padding="same",
            data_format="channels_last",
            activation="relu",
        )(layer)
        bn_1_a_1 = BatchNormalization(axis=-1, scale=False)(conv_1_a_1)

        stride_size = self.get_corrected_stride_size(bn_1_a_1, [2] * self.dims, [3] * self.dims)
        conv_1_a_2 = self.conv_func(
            filters=384,
            kernel_size=[3] * self.dims,
            strides=stride_size,
            padding="valid",
            data_format="channels_last",
            activation="relu",
        )(bn_1_a_1)
        bn_1_a_2 = BatchNormalization(axis=-1, scale=False)(conv_1_a_2)

        conv_1_b_1 = self.conv_func(
            filters=256,
            kernel_size=[1] * self.dims,
            strides=[1] * self.dims,
            padding="same",
            data_format="channels_last",
            activation="relu",
        )(layer)
        bn_1_b_1 = BatchNormalization(axis=-1, scale=False)(conv_1_b_1)

        stride_size = self.get_corrected_stride_size(bn_1_b_1, [2] * self.dims, [3] * self.dims)
        conv_1_b_2 = self.conv_func(
            filters=288,
            kernel_size=[3] * self.dims,
            strides=stride_size,
            padding="valid",
            data_format="channels_last",
            activation="relu",
        )(bn_1_b_1)
        bn_1_b_2 = BatchNormalization(axis=-1, scale=False)(conv_1_b_2)

        conv_1_c_1 = self.conv_func(
            filters=256,
            kernel_size=[1] * self.dims,
            strides=[1] * self.dims,
            padding="same",
            data_format="channels_last",
            activation="relu",
        )(layer)
        bn_1_c_1 = BatchNormalization(axis=-1, scale=False)(conv_1_c_1)

        conv_1_c_2 = self.conv_func(
            filters=288,
            kernel_size=[3] * self.dims,
            strides=[1] * self.dims,
            padding="same",
            data_format="channels_last",
            activation="relu",
        )(bn_1_c_1)
        bn_1_c_2 = BatchNormalization(axis=-1, scale=False)(conv_1_c_2)
        stride_size = self.get_corrected_stride_size(bn_1_c_2, [2] * self.dims, [3] * self.dims)

        conv_1_c_3 = self.conv_func(
            filters=320,
            kernel_size=[3] * self.dims,
            strides=stride_size,
            padding="valid",
            data_format="channels_last",
            activation="relu",
        )(bn_1_c_2)
        bn_1_c_3 = BatchNormalization(axis=-1, scale=False)(conv_1_c_3)

        concat_1 = Concatenate(axis=-1)([pooling_1, bn_1_a_2, bn_1_b_2, bn_1_c_3])

        return concat_1

    def get_inception_resnet_C(self, layer):
        relu_1 = ReLU()(layer)

        conv_1_a_1 = self.conv_func(
            filters=192,
            kernel_size=[1] * self.dims,
            strides=[1] * self.dims,
            padding="same",
            data_format="channels_last",
            activation="relu",
        )(relu_1)
        bn_1_a_1 = BatchNormalization(axis=-1, scale=False)(conv_1_a_1)

        conv_1_b_1 = self.conv_func(
            filters=192,
            kernel_size=[1] * self.dims,
            strides=[1] * self.dims,
            padding="same",
            data_format="channels_last",
            activation="relu",
        )(relu_1)
        bn_1_b_1 = BatchNormalization(axis=-1, scale=False)(conv_1_b_1)

        head = bn_1_b_1
        start_filters = 224
        for i_dim in range(self.dims):
            kernel_size = np.ones(self.dims, dtype=np.int)
            kernel_size[i_dim] = 3
            head = self.conv_func(
                filters=start_filters + 32 * i_dim,
                kernel_size=kernel_size.tolist(),
                strides=[1] * self.dims,
                padding="same",
                data_format="channels_last",
                activation="relu",
            )(head)
            head = BatchNormalization(axis=-1, scale=False)(head)

        concat_1 = Concatenate(axis=-1)([bn_1_a_1, head])

        conv_2 = self.conv_func(
            filters=2144,
            kernel_size=[1] * self.dims,
            strides=[1] * self.dims,
            padding="same",
            data_format="channels_last",
            activation="linear",
        )(concat_1)

        add_1 = Add()([conv_2, relu_1])
        relu_2 = ReLU()(add_1)
        bn_relu = BatchNormalization(axis=-1, scale=False)(relu_2)

        return bn_relu


class InceptionNet_InceptionResNetV2_2D(InceptionResNet):
    def create_model(self):
        self.check_minimum_input_size(self.inputs, [59, 59])
        self.init_dimensionality(2)
        stem = self.get_inception_stem(self.inputs)

        incep_res_A_1 = self.get_inception_resnet_A(stem)
        incep_res_A_2 = self.get_inception_resnet_A(incep_res_A_1)
        incep_res_A_3 = self.get_inception_resnet_A(incep_res_A_2)
        incep_res_A_4 = self.get_inception_resnet_A(incep_res_A_3)
        incep_res_A_5 = self.get_inception_resnet_A(incep_res_A_4)

        incep_res_A_reduc = self.get_inception_resnet_reduction_A(incep_res_A_5)

        incep_res_B_1 = self.get_inception_resnet_B(incep_res_A_reduc)
        incep_res_B_2 = self.get_inception_resnet_B(incep_res_B_1)
        incep_res_B_3 = self.get_inception_resnet_B(incep_res_B_2)
        incep_res_B_4 = self.get_inception_resnet_B(incep_res_B_3)
        incep_res_B_5 = self.get_inception_resnet_B(incep_res_B_4)
        incep_res_B_6 = self.get_inception_resnet_B(incep_res_B_5)
        incep_res_B_7 = self.get_inception_resnet_B(incep_res_B_6)
        incep_res_B_8 = self.get_inception_resnet_B(incep_res_B_7)
        incep_res_B_9 = self.get_inception_resnet_B(incep_res_B_8)
        incep_res_B_10 = self.get_inception_resnet_B(incep_res_B_9)

        incep_res_B_reduc = self.get_inception_resnet_reduction_B(incep_res_B_10)

        incep_res_C_1 = self.get_inception_resnet_C(incep_res_B_reduc)
        incep_res_C_2 = self.get_inception_resnet_C(incep_res_C_1)
        incep_res_C_3 = self.get_inception_resnet_C(incep_res_C_2)
        incep_res_C_4 = self.get_inception_resnet_C(incep_res_C_3)
        incep_res_C_5 = self.get_inception_resnet_C(incep_res_C_4)

        gap_1 = GlobalAveragePooling2D()(incep_res_C_5)
        dropout_1 = Dropout(0.2)(gap_1)

        predictions = self.outputs(dropout_1)

        model = Model(inputs=self.inputs, outputs=predictions)

        return model


class InceptionNet_InceptionResNetV2_3D(InceptionResNet):
    def create_model(self):
        self.check_minimum_input_size(self.inputs, [59, 59, 59])
        self.init_dimensionality(3)
        stem = self.get_inception_stem(self.inputs)

        incep_res_A_1 = self.get_inception_resnet_A(stem)
        incep_res_A_2 = self.get_inception_resnet_A(incep_res_A_1)
        incep_res_A_3 = self.get_inception_resnet_A(incep_res_A_2)
        incep_res_A_4 = self.get_inception_resnet_A(incep_res_A_3)
        incep_res_A_5 = self.get_inception_resnet_A(incep_res_A_4)

        incep_res_A_reduc = self.get_inception_resnet_reduction_A(incep_res_A_5)

        incep_res_B_1 = self.get_inception_resnet_B(incep_res_A_reduc)
        incep_res_B_2 = self.get_inception_resnet_B(incep_res_B_1)
        incep_res_B_3 = self.get_inception_resnet_B(incep_res_B_2)
        incep_res_B_4 = self.get_inception_resnet_B(incep_res_B_3)
        incep_res_B_5 = self.get_inception_resnet_B(incep_res_B_4)
        incep_res_B_6 = self.get_inception_resnet_B(incep_res_B_5)
        incep_res_B_7 = self.get_inception_resnet_B(incep_res_B_6)
        incep_res_B_8 = self.get_inception_resnet_B(incep_res_B_7)
        incep_res_B_9 = self.get_inception_resnet_B(incep_res_B_8)
        incep_res_B_10 = self.get_inception_resnet_B(incep_res_B_9)

        incep_res_B_reduc = self.get_inception_resnet_reduction_B(incep_res_B_10)

        incep_res_C_1 = self.get_inception_resnet_C(incep_res_B_reduc)
        incep_res_C_2 = self.get_inception_resnet_C(incep_res_C_1)
        incep_res_C_3 = self.get_inception_resnet_C(incep_res_C_2)
        incep_res_C_4 = self.get_inception_resnet_C(incep_res_C_3)
        incep_res_C_5 = self.get_inception_resnet_C(incep_res_C_4)

        gap_1 = GlobalAveragePooling3D()(incep_res_C_5)
        dropout_1 = Dropout(0.2)(gap_1)

        predictions = self.outputs(dropout_1)

        model = Model(inputs=self.inputs, outputs=predictions)

        return model
