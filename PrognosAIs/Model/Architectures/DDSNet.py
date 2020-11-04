from PrognosAIs.Model.Architectures.Architecture import ClassificationNetworkArchitecture
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv3D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import MaxPooling3D
from tensorflow.keras.layers import PReLU
from tensorflow.keras.models import Model


class DDSNet(ClassificationNetworkArchitecture):
    def init_dimensionality(self, N_dimension):
        if N_dimension == 2:
            self.dims = 2
            self.conv_func = Conv2D
            self.pool_func = MaxPooling2D
        elif N_dimension == 3:
            self.dims = 3
            self.conv_func = Conv3D
            self.pool_func = MaxPooling3D

    def get_DDS_block(self, layer, N_filters):
        conv1 = self.conv_func(N_filters, [5] * self.dims)(layer)
        batch1 = BatchNormalization(axis=-1)(conv1)
        relu1 = PReLU()(batch1)
        conv2 = self.conv_func(N_filters, [5] * self.dims)(relu1)
        batch2 = BatchNormalization(axis=-1)(conv2)
        relu2 = PReLU()(batch2)
        pooling1 = self.pool_func(pool_size=[3] * self.dims)(relu2)

        return pooling1


class DDSNet_2D(DDSNet):
    dims = 2

    def create_model(self):
        self.check_minimum_input_size(self.inputs, [131, 131])
        self.init_dimensionality(self.dims)
        block_1 = self.get_DDS_block(self.inputs, 32)

        block_2 = self.get_DDS_block(block_1, 64)

        block_3 = self.get_DDS_block(block_2, 64)

        flattened_CNN_output = Flatten()(block_3)

        flattened_CNN_output = self.make_dropout_layer(flattened_CNN_output)

        dense_1 = Dense(1024, activation="relu")(flattened_CNN_output)

        dense_1 = self.make_dropout_layer(dense_1)

        predictions = self.outputs(dense_1)

        model = Model(inputs=self.inputs, outputs=predictions)

        return model


class DDSNet_3D(DDSNet):
    dims = 3

    def create_model(self):
        self.check_minimum_input_size(self.inputs, [131, 131, 131])
        self.init_dimensionality(self.dims)
        block_1 = self.get_DDS_block(self.inputs, 32)

        block_2 = self.get_DDS_block(block_1, 64)

        block_3 = self.get_DDS_block(block_2, 64)

        flattened_CNN_output = Flatten()(block_3)

        flattened_CNN_output = self.make_dropout_layer(flattened_CNN_output)

        dense_1 = Dense(1024, activation="relu")(flattened_CNN_output)

        dense_1 = self.make_dropout_layer(dense_1)

        predictions = self.outputs(dense_1)

        model = Model(inputs=self.inputs, outputs=predictions)

        return model
