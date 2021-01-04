import numpy as np
import pytest
import tensorflow as tf

from PrognosAIs.Model.Architectures import VGG
from PrognosAIs.Model.Architectures import AlexNet
from PrognosAIs.Model.Architectures import Architecture
from PrognosAIs.Model.Architectures import DDSNet
from PrognosAIs.Model.Architectures import DenseNet
from PrognosAIs.Model.Architectures import InceptionNet
from PrognosAIs.Model.Architectures import ResNet
from PrognosAIs.Model.Architectures import UNet


loss_fuction = "categorical_crossentropy"


class mock_architecture(Architecture.NetworkArchitecture):
    def make_outputs(self, output_info, output_data_type):
        pass

    def create_model(self):
        pass


# ===============================================================
# Base architecture classes
# ===============================================================


def test_minimum_input_size_pass():
    input_layer = tf.keras.Input(shape=[50, 50, 50, 1])
    min_input_size = np.asarray([25, 25, 25])

    result = Architecture.NetworkArchitecture.check_minimum_input_size(input_layer, min_input_size)

    assert result is None


def test_minimum_input_size_not_pass():
    input_layer = tf.keras.Input(shape=[20, 50, 50, 1])
    min_input_size = np.asarray([25, 25, 25])
    error_text = (
        "Minimum input size for this model is: 25 x 25 x 25\n"
        "Your input size is: 20 x 50 x 50\n"
        "Please fix your input"
    )

    with pytest.raises(ValueError, match=error_text):
        Architecture.NetworkArchitecture.check_minimum_input_size(input_layer, min_input_size)


def test_missing_channel_error_throwing():
    input_layer = tf.keras.Input(shape=[50, 50, 50])
    min_input_size = np.asarray([25, 25, 25])
    # The error text is a regex expression so we need to
    # escape the braces
    error_text = (
        "It seems like you have forgotten to include a"
        " \\(potentially empty\\) channel dimension as the last dimension.\n"
        "Please fix this and run the model again."
    )

    with pytest.raises(ValueError, match=error_text):
        Architecture.NetworkArchitecture.check_minimum_input_size(input_layer, min_input_size)


def test_get_dropout_layer():
    architecture = mock_architecture(
        {"input_1": [50, 50, 1]}, {"output_1": 10}, model_config={"dropout": 0.314}
    )
    inputs = architecture.make_inputs({"input_1": [50, 50, 1]}, "float32")
    first_layer = tf.keras.layers.Dense(20)(inputs)

    result = architecture.make_dropout_layer(first_layer).__dict__["_keras_history"][0]

    assert isinstance(result, tf.keras.layers.Dropout)
    assert result.rate == 0.314
    assert result.input._name == first_layer._name


def test_dont_get_dropout_layer():
    architecture = mock_architecture({"input_1": [50, 50, 1]}, {"output_1": 10})
    inputs = architecture.make_inputs({"input_1": [50, 50, 1]}, "float32")
    first_layer = tf.keras.layers.Dense(20)(inputs)

    result = architecture.make_dropout_layer(first_layer)

    assert isinstance(result.__dict__["_keras_history"][0], tf.keras.layers.Dense)
    assert result._name == first_layer._name


def test_get_dropout_layer_0_droput():
    architecture = mock_architecture(
        {"input_1": [50, 50, 1]}, {"output_1": 10}, model_config={"dropout": 0}
    )
    inputs = architecture.make_inputs({"input_1": [50, 50, 1]}, "float32")
    first_layer = tf.keras.layers.Dense(20)(inputs)

    result = architecture.make_dropout_layer(first_layer)

    assert isinstance(result.__dict__["_keras_history"][0], tf.keras.layers.Dense)
    assert result._name == first_layer._name


# ===============================================================
# ResNet
# ===============================================================


def test_ResNet_18_2D_architecture():
    # Last parameter is the channel, so it is a 2D image, with 1 channel
    architecture = ResNet.ResNet_18_2D({"input_1": [50, 50, 1]}, {"output_1": 10})
    model = architecture.create_model()
    # Make sure we can compile the model
    model.compile(loss=loss_fuction)

    model_config = model.get_config()
    model_layers = model_config["layers"]
    assert len(model_layers) == 67

    assert model_layers[0]["class_name"] == "InputLayer"
    assert model_layers[0]["config"]["batch_input_shape"] == (None, 50, 50, 1)
    assert model_layers[-1]["class_name"] == "Dense"
    assert model_layers[-1]["config"]["units"] == 10
    assert model_layers[-1]["config"]["activation"] == "softmax"
    assert model_layers[-1]["config"]["dtype"] == "float32"

    layer_names = [i_layer["class_name"] for i_layer in model_layers]

    assert sorted(np.unique(layer_names).tolist()) == [
        "Add",
        "BatchNormalization",
        "Conv2D",
        "Dense",
        "GlobalAveragePooling2D",
        "InputLayer",
        "MaxPooling2D",
        "ReLU",
    ]


def test_ResNet_34_2D_architecture():
    # Last parameter is the channel, so it is a 2D image, with 1 channel
    architecture = ResNet.ResNet_34_2D({"input_1": [50, 50, 1]}, {"output_1": 10})
    model = architecture.create_model()
    # Make sure we can compile the model
    model.compile(loss=loss_fuction)

    model_config = model.get_config()
    model_layers = model_config["layers"]
    assert len(model_layers) == 123

    assert model_layers[0]["class_name"] == "InputLayer"
    assert model_layers[0]["config"]["batch_input_shape"] == (None, 50, 50, 1)
    assert model_layers[-1]["class_name"] == "Dense"
    assert model_layers[-1]["config"]["units"] == 10
    assert model_layers[-1]["config"]["activation"] == "softmax"

    layer_names = [i_layer["class_name"] for i_layer in model_layers]

    assert sorted(np.unique(layer_names).tolist()) == [
        "Add",
        "BatchNormalization",
        "Conv2D",
        "Dense",
        "GlobalAveragePooling2D",
        "InputLayer",
        "MaxPooling2D",
        "ReLU",
    ]


def test_ResNet_18_3D_architecture():
    # Last parameter is the channel, so it is a 2D image, with 1 channel
    architecture = ResNet.ResNet_18_3D({"input_1": [50, 50, 50, 1]}, {"output_1": 10})
    model = architecture.create_model()
    # Make sure we can compile the model
    model.compile(loss=loss_fuction)

    model_config = model.get_config()
    model_layers = model_config["layers"]
    assert len(model_layers) == 67

    assert model_layers[0]["class_name"] == "InputLayer"
    assert model_layers[0]["config"]["batch_input_shape"] == (None, 50, 50, 50, 1)
    assert model_layers[-1]["class_name"] == "Dense"
    assert model_layers[-1]["config"]["units"] == 10
    assert model_layers[-1]["config"]["activation"] == "softmax"

    layer_names = [i_layer["class_name"] for i_layer in model_layers]

    assert sorted(np.unique(layer_names).tolist()) == [
        "Add",
        "BatchNormalization",
        "Conv3D",
        "Dense",
        "GlobalAveragePooling3D",
        "InputLayer",
        "MaxPooling3D",
        "ReLU",
    ]


def test_ResNet_34_3D_architecture():
    # Last parameter is the channel, so it is a 2D image, with 1 channel
    architecture = ResNet.ResNet_34_3D({"input_1": [50, 50, 50, 1]}, {"output_1": 10})
    model = architecture.create_model()
    # Make sure we can compile the model
    model.compile(loss=loss_fuction)

    model_config = model.get_config()
    model_layers = model_config["layers"]
    assert len(model_layers) == 123

    assert model_layers[0]["class_name"] == "InputLayer"
    assert model_layers[0]["config"]["batch_input_shape"] == (None, 50, 50, 50, 1)
    assert model_layers[-1]["class_name"] == "Dense"
    assert model_layers[-1]["config"]["units"] == 10
    assert model_layers[-1]["config"]["activation"] == "softmax"

    layer_names = [i_layer["class_name"] for i_layer in model_layers]

    assert sorted(np.unique(layer_names).tolist()) == [
        "Add",
        "BatchNormalization",
        "Conv3D",
        "Dense",
        "GlobalAveragePooling3D",
        "InputLayer",
        "MaxPooling3D",
        "ReLU",
    ]


def test_ResNet_18_2D_dropout_architecture():
    # Last parameter is the channel, so it is a 2D image, with 1 channel
    architecture = ResNet.ResNet_18_2D(
        {"input_1": [50, 50, 1]}, {"output_1": 10}, model_config={"dropout": 0.5}
    )
    model = architecture.create_model()
    # Make sure we can compile the model
    model.compile(loss=loss_fuction)

    model_config = model.get_config()
    model_layers = model_config["layers"]
    assert len(model_layers) == 71

    assert model_layers[0]["class_name"] == "InputLayer"
    assert model_layers[0]["config"]["batch_input_shape"] == (None, 50, 50, 1)
    assert model_layers[-1]["class_name"] == "Dense"
    assert model_layers[-1]["config"]["units"] == 10
    assert model_layers[-1]["config"]["activation"] == "softmax"

    layer_names = [i_layer["class_name"] for i_layer in model_layers]
    dropout_layer_index = np.squeeze(np.argwhere(np.asarray(layer_names) == "Dropout"))[0]
    assert model_layers[dropout_layer_index]["config"]["rate"] == 0.5

    assert sorted(np.unique(layer_names).tolist()) == [
        "Add",
        "BatchNormalization",
        "Conv2D",
        "Dense",
        "Dropout",
        "GlobalAveragePooling2D",
        "InputLayer",
        "MaxPooling2D",
        "ReLU",
    ]


def test_ResNet_34_2D_dropout_architecture():
    # Last parameter is the channel, so it is a 2D image, with 1 channel
    architecture = ResNet.ResNet_34_2D(
        {"input_1": [50, 50, 1]}, {"output_1": 10}, model_config={"dropout": 0.5}
    )
    model = architecture.create_model()
    # Make sure we can compile the model
    model.compile(loss=loss_fuction)

    model_config = model.get_config()
    model_layers = model_config["layers"]
    assert len(model_layers) == 127

    assert model_layers[0]["class_name"] == "InputLayer"
    assert model_layers[0]["config"]["batch_input_shape"] == (None, 50, 50, 1)
    assert model_layers[-1]["class_name"] == "Dense"
    assert model_layers[-1]["config"]["units"] == 10
    assert model_layers[-1]["config"]["activation"] == "softmax"

    layer_names = [i_layer["class_name"] for i_layer in model_layers]
    dropout_layer_index = np.squeeze(np.argwhere(np.asarray(layer_names) == "Dropout"))[0]
    assert model_layers[dropout_layer_index]["config"]["rate"] == 0.5

    assert sorted(np.unique(layer_names).tolist()) == [
        "Add",
        "BatchNormalization",
        "Conv2D",
        "Dense",
        "Dropout",
        "GlobalAveragePooling2D",
        "InputLayer",
        "MaxPooling2D",
        "ReLU",
    ]


def test_ResNet_18_3D_dropout_architecture():
    # Last parameter is the channel, so it is a 2D image, with 1 channel
    architecture = ResNet.ResNet_18_3D(
        {"input_1": [50, 50, 50, 1]}, {"output_1": 10}, model_config={"dropout": 0.5}
    )
    model = architecture.create_model()
    # Make sure we can compile the model
    model.compile(loss=loss_fuction)

    model_config = model.get_config()
    model_layers = model_config["layers"]
    assert len(model_layers) == 71

    assert model_layers[0]["class_name"] == "InputLayer"
    assert model_layers[0]["config"]["batch_input_shape"] == (None, 50, 50, 50, 1)
    assert model_layers[-1]["class_name"] == "Dense"
    assert model_layers[-1]["config"]["units"] == 10
    assert model_layers[-1]["config"]["activation"] == "softmax"

    layer_names = [i_layer["class_name"] for i_layer in model_layers]
    dropout_layer_index = np.squeeze(np.argwhere(np.asarray(layer_names) == "Dropout"))[0]
    assert model_layers[dropout_layer_index]["config"]["rate"] == 0.5

    assert sorted(np.unique(layer_names).tolist()) == [
        "Add",
        "BatchNormalization",
        "Conv3D",
        "Dense",
        "Dropout",
        "GlobalAveragePooling3D",
        "InputLayer",
        "MaxPooling3D",
        "ReLU",
    ]


def test_ResNet_34_3D_dropout_architecture():
    # Last parameter is the channel, so it is a 2D image, with 1 channel
    architecture = ResNet.ResNet_34_3D(
        {"input_1": [50, 50, 50, 1]}, {"output_1": 10}, model_config={"dropout": 0.5}
    )
    model = architecture.create_model()
    # Make sure we can compile the model
    model.compile(loss=loss_fuction)

    model_config = model.get_config()
    model_layers = model_config["layers"]
    assert len(model_layers) == 127

    assert model_layers[0]["class_name"] == "InputLayer"
    assert model_layers[0]["config"]["batch_input_shape"] == (None, 50, 50, 50, 1)
    assert model_layers[-1]["class_name"] == "Dense"
    assert model_layers[-1]["config"]["units"] == 10
    assert model_layers[-1]["config"]["activation"] == "softmax"

    layer_names = [i_layer["class_name"] for i_layer in model_layers]
    dropout_layer_index = np.squeeze(np.argwhere(np.asarray(layer_names) == "Dropout"))[0]
    assert model_layers[dropout_layer_index]["config"]["rate"] == 0.5

    assert sorted(np.unique(layer_names).tolist()) == [
        "Add",
        "BatchNormalization",
        "Conv3D",
        "Dense",
        "Dropout",
        "GlobalAveragePooling3D",
        "InputLayer",
        "MaxPooling3D",
        "ReLU",
    ]


def test_ResNet_18_multioutput_3D_architecture():
    # Last parameter is the channel, so it is a 2D image, with 1 channel
    outputs = {"output_1": 10, "output_2": 5, "output_3": 2}
    architecture = ResNet.ResNet_18_multioutput_3D({"input_1": [50, 50, 50, 1]}, outputs)
    model = architecture.create_model()
    # Make sure we can compile the model
    model.compile(loss=loss_fuction)

    model_config = model.get_config()
    output_layers = model_config["output_layers"]
    assert len(output_layers) == len(outputs)

    output_layer_names = sorted([i_output[0] for i_output in model_config["output_layers"]])
    assert output_layer_names == sorted(list(outputs.keys()))

    for i_output_layer_name in output_layer_names:
        i_output_layer = model.get_layer(i_output_layer_name)
        i_config = i_output_layer.get_config()
        assert i_config["units"] == outputs[i_output_layer_name]
        assert isinstance(i_output_layer, tf.keras.layers.Dense)
        assert i_config["activation"] == "softmax"

    model_layers = model_config["layers"]
    expected_layers = 49 + 18 * len(outputs)
    assert len(model_layers) == expected_layers

    assert model_layers[0]["class_name"] == "InputLayer"
    assert model_layers[0]["config"]["batch_input_shape"] == (None, 50, 50, 50, 1)
    layer_names = [i_layer["class_name"] for i_layer in model_layers]

    assert sorted(np.unique(layer_names).tolist()) == [
        "Add",
        "BatchNormalization",
        "Conv3D",
        "Dense",
        "GlobalAveragePooling3D",
        "InputLayer",
        "MaxPooling3D",
        "ReLU",
    ]


# ===============================================================
#  DenseNet
# ===============================================================


def test_DenseNet_121_2D():
    # Last parameter is the channel, so it is a 2D image, with 1 channel
    architecture = DenseNet.DenseNet_121_2D({"input_1": [50, 50, 1]}, {"output_1": 10})
    model = architecture.create_model()
    # Make sure we can compile the model
    model.compile(loss=loss_fuction)

    model_config = model.get_config()
    model_layers = model_config["layers"]
    assert len(model_layers) == 417

    assert model_layers[0]["class_name"] == "InputLayer"
    assert model_layers[0]["config"]["batch_input_shape"] == (None, 50, 50, 1)
    assert model_layers[-1]["class_name"] == "Dense"
    assert model_layers[-1]["config"]["units"] == 10
    assert model_layers[-1]["config"]["activation"] == "softmax"

    layer_names = [i_layer["class_name"] for i_layer in model_layers]

    assert sorted(np.unique(layer_names).tolist()) == [
        "BatchNormalization",
        "Concatenate",
        "Conv2D",
        "Dense",
        "GlobalAveragePooling2D",
        "InputLayer",
        "MaxPooling2D",
        "ReLU",
    ]


def test_DenseNet_121_3D():
    # Last parameter is the channel, so it is a 2D image, with 1 channel
    architecture = DenseNet.DenseNet_121_3D({"input_1": [50, 50, 50, 1]}, {"output_1": 10})
    model = architecture.create_model()
    # Make sure we can compile the model
    model.compile(loss=loss_fuction)

    model_config = model.get_config()
    model_layers = model_config["layers"]
    assert len(model_layers) == 417

    assert model_layers[0]["class_name"] == "InputLayer"
    assert model_layers[0]["config"]["batch_input_shape"] == (None, 50, 50, 50, 1)
    assert model_layers[-1]["class_name"] == "Dense"
    assert model_layers[-1]["config"]["units"] == 10
    assert model_layers[-1]["config"]["activation"] == "softmax"

    layer_names = [i_layer["class_name"] for i_layer in model_layers]

    assert sorted(np.unique(layer_names).tolist()) == [
        "BatchNormalization",
        "Concatenate",
        "Conv3D",
        "Dense",
        "GlobalAveragePooling3D",
        "InputLayer",
        "MaxPooling3D",
        "ReLU",
    ]


def test_DenseNet_169_2D():
    # Last parameter is the channel, so it is a 2D image, with 1 channel
    architecture = DenseNet.DenseNet_169_2D({"input_1": [50, 50, 1]}, {"output_1": 10})
    model = architecture.create_model()
    # Make sure we can compile the model
    model.compile(loss=loss_fuction)

    model_config = model.get_config()
    model_layers = model_config["layers"]
    assert len(model_layers) == 585

    assert model_layers[0]["class_name"] == "InputLayer"
    assert model_layers[0]["config"]["batch_input_shape"] == (None, 50, 50, 1)
    assert model_layers[-1]["class_name"] == "Dense"
    assert model_layers[-1]["config"]["units"] == 10
    assert model_layers[-1]["config"]["activation"] == "softmax"

    layer_names = [i_layer["class_name"] for i_layer in model_layers]

    assert sorted(np.unique(layer_names).tolist()) == [
        "BatchNormalization",
        "Concatenate",
        "Conv2D",
        "Dense",
        "GlobalAveragePooling2D",
        "InputLayer",
        "MaxPooling2D",
        "ReLU",
    ]


def test_DenseNet_169_3D():
    architecture = DenseNet.DenseNet_169_3D({"input_1": [50, 50, 50, 1]}, {"output_1": 10})
    model = architecture.create_model()
    # Make sure we can compile the model
    model.compile(loss=loss_fuction)

    model_config = model.get_config()
    model_layers = model_config["layers"]
    assert len(model_layers) == 585

    assert model_layers[0]["class_name"] == "InputLayer"
    assert model_layers[0]["config"]["batch_input_shape"] == (None, 50, 50, 50, 1)
    assert model_layers[-1]["class_name"] == "Dense"
    assert model_layers[-1]["config"]["units"] == 10
    assert model_layers[-1]["config"]["activation"] == "softmax"

    layer_names = [i_layer["class_name"] for i_layer in model_layers]

    assert sorted(np.unique(layer_names).tolist()) == [
        "BatchNormalization",
        "Concatenate",
        "Conv3D",
        "Dense",
        "GlobalAveragePooling3D",
        "InputLayer",
        "MaxPooling3D",
        "ReLU",
    ]


def test_DenseNet_201_2D():
    architecture = DenseNet.DenseNet_201_2D({"input_1": [50, 50, 1]}, {"output_1": 10})
    model = architecture.create_model()
    # Make sure we can compile the model
    model.compile(loss=loss_fuction)

    model_config = model.get_config()
    model_layers = model_config["layers"]
    assert len(model_layers) == 809

    assert model_layers[0]["class_name"] == "InputLayer"
    assert model_layers[0]["config"]["batch_input_shape"] == (None, 50, 50, 1)
    assert model_layers[-1]["class_name"] == "Dense"
    assert model_layers[-1]["config"]["units"] == 10
    assert model_layers[-1]["config"]["activation"] == "softmax"

    layer_names = [i_layer["class_name"] for i_layer in model_layers]

    assert sorted(np.unique(layer_names).tolist()) == [
        "BatchNormalization",
        "Concatenate",
        "Conv2D",
        "Dense",
        "GlobalAveragePooling2D",
        "InputLayer",
        "MaxPooling2D",
        "ReLU",
    ]


def test_DenseNet_201_3D():
    architecture = DenseNet.DenseNet_201_3D({"input_1": [50, 50, 50, 1]}, {"output_1": 10})
    model = architecture.create_model()
    # Make sure we can compile the model
    model.compile(loss=loss_fuction)

    model_config = model.get_config()
    model_layers = model_config["layers"]
    assert len(model_layers) == 809

    assert model_layers[0]["class_name"] == "InputLayer"
    assert model_layers[0]["config"]["batch_input_shape"] == (None, 50, 50, 50, 1)
    assert model_layers[-1]["class_name"] == "Dense"
    assert model_layers[-1]["config"]["units"] == 10
    assert model_layers[-1]["config"]["activation"] == "softmax"

    layer_names = [i_layer["class_name"] for i_layer in model_layers]

    assert sorted(np.unique(layer_names).tolist()) == [
        "BatchNormalization",
        "Concatenate",
        "Conv3D",
        "Dense",
        "GlobalAveragePooling3D",
        "InputLayer",
        "MaxPooling3D",
        "ReLU",
    ]


def test_DenseNet_264_2D():
    architecture = DenseNet.DenseNet_264_2D({"input_1": [50, 50, 1]}, {"output_1": 10})
    model = architecture.create_model()
    # Make sure we can compile the model
    model.compile(loss=loss_fuction)

    model_config = model.get_config()
    model_layers = model_config["layers"]
    assert len(model_layers) == 921

    assert model_layers[0]["class_name"] == "InputLayer"
    assert model_layers[0]["config"]["batch_input_shape"] == (None, 50, 50, 1)
    assert model_layers[-1]["class_name"] == "Dense"
    assert model_layers[-1]["config"]["units"] == 10
    assert model_layers[-1]["config"]["activation"] == "softmax"

    layer_names = [i_layer["class_name"] for i_layer in model_layers]

    assert sorted(np.unique(layer_names).tolist()) == [
        "BatchNormalization",
        "Concatenate",
        "Conv2D",
        "Dense",
        "GlobalAveragePooling2D",
        "InputLayer",
        "MaxPooling2D",
        "ReLU",
    ]


def test_DenseNet_264_3D():
    architecture = DenseNet.DenseNet_264_3D({"input_1": [50, 50, 50, 1]}, {"output_1": 10})
    model = architecture.create_model()
    # Make sure we can compile the model
    model.compile(loss=loss_fuction)

    model_config = model.get_config()
    model_layers = model_config["layers"]
    assert len(model_layers) == 921

    assert model_layers[0]["class_name"] == "InputLayer"
    assert model_layers[0]["config"]["batch_input_shape"] == (None, 50, 50, 50, 1)
    assert model_layers[-1]["class_name"] == "Dense"
    assert model_layers[-1]["config"]["units"] == 10
    assert model_layers[-1]["config"]["activation"] == "softmax"

    layer_names = [i_layer["class_name"] for i_layer in model_layers]

    assert sorted(np.unique(layer_names).tolist()) == [
        "BatchNormalization",
        "Concatenate",
        "Conv3D",
        "Dense",
        "GlobalAveragePooling3D",
        "InputLayer",
        "MaxPooling3D",
        "ReLU",
    ]


# ===============================================================
#  AlexNet
# ===============================================================


def test_AlexNet_2D():
    architecture = AlexNet.AlexNet_2D({"input_1": [75, 75, 1]}, {"output_1": 10})
    model = architecture.create_model()
    # Make sure we can compile the model
    model.compile(loss=loss_fuction)

    model_config = model.get_config()
    model_layers = model_config["layers"]
    assert len(model_layers) == 20

    assert model_layers[0]["class_name"] == "InputLayer"
    assert model_layers[0]["config"]["batch_input_shape"] == (None, 75, 75, 1)
    assert model_layers[-1]["class_name"] == "Dense"
    assert model_layers[-1]["config"]["units"] == 10
    assert model_layers[-1]["config"]["activation"] == "softmax"

    layer_names = [i_layer["class_name"] for i_layer in model_layers]

    assert sorted(np.unique(layer_names).tolist()) == [
        "Conv2D",
        "Dense",
        "Dropout",
        "Flatten",
        "InputLayer",
        "MaxPooling2D",
        "ReLU",
    ]


def test_AlexNet_3D():
    architecture = AlexNet.AlexNet_3D({"input_1": [75, 75, 75, 1]}, {"output_1": 10})
    model = architecture.create_model()
    # Make sure we can compile the model
    model.compile(loss=loss_fuction)

    model_config = model.get_config()
    model_layers = model_config["layers"]
    assert len(model_layers) == 20

    assert model_layers[0]["class_name"] == "InputLayer"
    assert model_layers[0]["config"]["batch_input_shape"] == (None, 75, 75, 75, 1)
    assert model_layers[-1]["class_name"] == "Dense"
    assert model_layers[-1]["config"]["units"] == 10
    assert model_layers[-1]["config"]["activation"] == "softmax"

    layer_names = [i_layer["class_name"] for i_layer in model_layers]

    assert sorted(np.unique(layer_names).tolist()) == [
        "Conv3D",
        "Dense",
        "Dropout",
        "Flatten",
        "InputLayer",
        "MaxPooling3D",
        "ReLU",
    ]


# ===============================================================
#  InceptionNet
# ===============================================================


def test_InceptionResNet_V2_2D():
    architecture = InceptionNet.InceptionNet_InceptionResNetV2_2D(
        {"input_1": [65, 65, 1]}, {"output_1": 10}
    )
    model = architecture.create_model()
    # Make sure we can compile the model
    model.compile(loss=loss_fuction)

    model_config = model.get_config()
    model_layers = model_config["layers"]
    assert len(model_layers) == 357

    assert model_layers[0]["class_name"] == "InputLayer"
    assert model_layers[0]["config"]["batch_input_shape"] == (None, 65, 65, 1)
    assert model_layers[-1]["class_name"] == "Dense"
    assert model_layers[-1]["config"]["units"] == 10
    assert model_layers[-1]["config"]["activation"] == "softmax"

    layer_names = [i_layer["class_name"] for i_layer in model_layers]

    assert sorted(np.unique(layer_names).tolist()) == [
        "Add",
        "BatchNormalization",
        "Concatenate",
        "Conv2D",
        "Dense",
        "Dropout",
        "GlobalAveragePooling2D",
        "InputLayer",
        "MaxPooling2D",
        "ReLU",
    ]


def test_InceptionResNet_V2_3D():
    architecture = InceptionNet.InceptionNet_InceptionResNetV2_3D(
        {"input_1": [65, 65, 65, 1]}, {"output_1": 10}
    )
    model = architecture.create_model()
    # Make sure we can compile the model
    model.compile(loss=loss_fuction)

    model_config = model.get_config()
    model_layers = model_config["layers"]
    assert len(model_layers) == 389

    assert model_layers[0]["class_name"] == "InputLayer"
    assert model_layers[0]["config"]["batch_input_shape"] == (None, 65, 65, 65, 1)
    assert model_layers[-1]["class_name"] == "Dense"
    assert model_layers[-1]["config"]["units"] == 10
    assert model_layers[-1]["config"]["activation"] == "softmax"

    layer_names = [i_layer["class_name"] for i_layer in model_layers]

    assert sorted(np.unique(layer_names).tolist()) == [
        "Add",
        "BatchNormalization",
        "Concatenate",
        "Conv3D",
        "Dense",
        "Dropout",
        "GlobalAveragePooling3D",
        "InputLayer",
        "MaxPooling3D",
        "ReLU",
    ]


# ===============================================================
#  VGG
# ===============================================================


def test_VGG_16_2D():
    # Last parameter is the channel, so it is a 2D image, with 1 channel
    architecture = VGG.VGG_16_2D({"input_1": [50, 50, 1]}, {"output_1": 10})
    model = architecture.create_model()
    # Make sure we can compile the model
    model.compile(loss=loss_fuction)

    model_config = model.get_config()
    model_layers = model_config["layers"]
    assert len(model_layers) == 23

    assert model_layers[0]["class_name"] == "InputLayer"
    assert model_layers[0]["config"]["batch_input_shape"] == (None, 50, 50, 1)
    assert model_layers[-1]["class_name"] == "Dense"
    assert model_layers[-1]["config"]["units"] == 10
    assert model_layers[-1]["config"]["activation"] == "softmax"

    layer_names = [i_layer["class_name"] for i_layer in model_layers]

    assert sorted(np.unique(layer_names).tolist()) == [
        "Conv2D",
        "Dense",
        "Flatten",
        "InputLayer",
        "MaxPooling2D",
    ]


def test_VGG_19_2D():
    # Last parameter is the channel, so it is a 2D image, with 1 channel
    architecture = VGG.VGG_19_2D({"input_1": [50, 50, 1]}, {"output_1": 10})
    model = architecture.create_model()
    # Make sure we can compile the model
    model.compile(loss=loss_fuction)

    model_config = model.get_config()
    model_layers = model_config["layers"]
    assert len(model_layers) == 26

    assert model_layers[0]["class_name"] == "InputLayer"
    assert model_layers[0]["config"]["batch_input_shape"] == (None, 50, 50, 1)
    assert model_layers[-1]["class_name"] == "Dense"
    assert model_layers[-1]["config"]["units"] == 10
    assert model_layers[-1]["config"]["activation"] == "softmax"

    layer_names = [i_layer["class_name"] for i_layer in model_layers]

    assert sorted(np.unique(layer_names).tolist()) == [
        "Conv2D",
        "Dense",
        "Flatten",
        "InputLayer",
        "MaxPooling2D",
    ]


def test_VGG_16_3D():
    # Last parameter is the channel, so it is a 2D image, with 1 channel
    architecture = VGG.VGG_16_3D({"input_1": [50, 50, 50, 1]}, {"output_1": 10})
    model = architecture.create_model()
    # Make sure we can compile the model
    model.compile(loss=loss_fuction)

    model_config = model.get_config()
    model_layers = model_config["layers"]
    assert len(model_layers) == 23

    assert model_layers[0]["class_name"] == "InputLayer"
    assert model_layers[0]["config"]["batch_input_shape"] == (None, 50, 50, 50, 1)
    assert model_layers[-1]["class_name"] == "Dense"
    assert model_layers[-1]["config"]["units"] == 10
    assert model_layers[-1]["config"]["activation"] == "softmax"

    layer_names = [i_layer["class_name"] for i_layer in model_layers]

    assert sorted(np.unique(layer_names).tolist()) == [
        "Conv3D",
        "Dense",
        "Flatten",
        "InputLayer",
        "MaxPooling3D",
    ]


def test_VGG_19_3D():
    # Last parameter is the channel, so it is a 2D image, with 1 channel
    architecture = VGG.VGG_19_3D({"input_1": [50, 50, 50, 1]}, {"output_1": 10})
    model = architecture.create_model()
    # Make sure we can compile the model
    model.compile(loss=loss_fuction)

    model_config = model.get_config()
    model_layers = model_config["layers"]
    assert len(model_layers) == 26

    assert model_layers[0]["class_name"] == "InputLayer"
    assert model_layers[0]["config"]["batch_input_shape"] == (None, 50, 50, 50, 1)
    assert model_layers[-1]["class_name"] == "Dense"
    assert model_layers[-1]["config"]["units"] == 10
    assert model_layers[-1]["config"]["activation"] == "softmax"

    layer_names = [i_layer["class_name"] for i_layer in model_layers]

    assert sorted(np.unique(layer_names).tolist()) == [
        "Conv3D",
        "Dense",
        "Flatten",
        "InputLayer",
        "MaxPooling3D",
    ]


# ===============================================================
#  UNet
# ===============================================================


def test_UNet_2D():
    # Last parameter is the channel, so it is a 2D image, with 1 channel
    architecture = UNet.UNet_2D({"input_1": [25, 25, 1]}, {"output_1": 2})
    model = architecture.create_model()
    # Make sure we can compile the model
    model.compile(loss=loss_fuction)

    model_config = model.get_config()
    model_layers = model_config["layers"]
    assert len(model_layers) == 42

    assert model_layers[0]["class_name"] == "InputLayer"
    assert model_layers[0]["config"]["batch_input_shape"] == (None, 25, 25, 1)
    assert model_layers[-1]["class_name"] == "Conv2D"
    assert model_layers[-1]["config"]["filters"] == 1
    assert model_layers[-1]["config"]["activation"] == "sigmoid"

    layer_names = [i_layer["class_name"] for i_layer in model_layers]

    assert sorted(np.unique(layer_names).tolist()) == [
        "Concatenate",
        "Conv2D",
        "Cropping2D",
        "InputLayer",
        "MaxPooling2D",
        "UpSampling2D",
        "ZeroPadding2D",
    ]


def test_UNet_2D_one_hot():
    # Last parameter is the channel, so it is a 2D image, with 1 channel
    architecture = UNet.UNet_2D(
        {"input_1": [25, 25, 1]}, {"output_1": 2}, model_config={"one_hot_output": True}
    )
    model = architecture.create_model()
    # Make sure we can compile the model
    model.compile(loss=loss_fuction)

    model_config = model.get_config()
    model_layers = model_config["layers"]
    assert len(model_layers) == 42

    assert model_layers[0]["class_name"] == "InputLayer"
    assert model_layers[0]["config"]["batch_input_shape"] == (None, 25, 25, 1)
    assert model_layers[-1]["class_name"] == "Conv2D"
    assert model_layers[-1]["config"]["filters"] == 2
    assert model_layers[-1]["config"]["activation"] == "softmax"

    layer_names = [i_layer["class_name"] for i_layer in model_layers]

    assert sorted(np.unique(layer_names).tolist()) == [
        "Concatenate",
        "Conv2D",
        "Cropping2D",
        "InputLayer",
        "MaxPooling2D",
        "UpSampling2D",
        "ZeroPadding2D",
    ]


def test_UNet_2D_multi_class():
    # Last parameter is the channel, so it is a 2D image, with 1 channel
    architecture = UNet.UNet_2D({"input_1": [25, 25, 1]}, {"output_1": 4})
    model = architecture.create_model()
    # Make sure we can compile the model
    model.compile(loss=loss_fuction)

    model_config = model.get_config()
    model_layers = model_config["layers"]
    assert len(model_layers) == 42

    assert model_layers[0]["class_name"] == "InputLayer"
    assert model_layers[0]["config"]["batch_input_shape"] == (None, 25, 25, 1)
    assert model_layers[-1]["class_name"] == "Conv2D"
    assert model_layers[-1]["config"]["filters"] == 4
    assert model_layers[-1]["config"]["activation"] == "softmax"

    layer_names = [i_layer["class_name"] for i_layer in model_layers]

    assert sorted(np.unique(layer_names).tolist()) == [
        "Concatenate",
        "Conv2D",
        "Cropping2D",
        "InputLayer",
        "MaxPooling2D",
        "UpSampling2D",
        "ZeroPadding2D",
    ]


def test_UNet_depth_4_filters_16_2D():
    # Last parameter is the channel, so it is a 2D image, with 1 channel
    architecture = UNet.UNet_2D(
        {"input_1": [25, 25, 1]},
        {"output_1": 2},
        model_config={"number_of_filters": 16, "depth": 4},
    )
    model = architecture.create_model()
    # Make sure we can compile the model
    model.compile(loss=loss_fuction)

    model_config = model.get_config()
    model_layers = model_config["layers"]
    assert len(model_layers) == 34

    assert model_layers[0]["class_name"] == "InputLayer"
    assert model_layers[0]["config"]["batch_input_shape"] == (None, 25, 25, 1)
    assert model_layers[1]["class_name"] == "Conv2D"
    assert model_layers[1]["config"]["filters"] == 16
    assert model_layers[-1]["class_name"] == "Conv2D"
    assert model_layers[-1]["config"]["filters"] == 1
    assert model_layers[-1]["config"]["activation"] == "sigmoid"

    layer_names = [i_layer["class_name"] for i_layer in model_layers]

    assert sorted(np.unique(layer_names).tolist()) == [
        "Concatenate",
        "Conv2D",
        "Cropping2D",
        "InputLayer",
        "MaxPooling2D",
        "UpSampling2D",
        "ZeroPadding2D",
    ]


def test_UNet_3D():
    architecture = UNet.UNet_3D({"input_1": [30, 30, 30, 1]}, {"output_1": 2})
    model = architecture.create_model()
    # Make sure we can compile the model
    model.compile(loss=loss_fuction)

    model_config = model.get_config()
    model_layers = model_config["layers"]
    assert len(model_layers) == 38

    assert model_layers[0]["class_name"] == "InputLayer"
    assert model_layers[0]["config"]["batch_input_shape"] == (None, 30, 30, 30, 1)
    assert model_layers[-1]["class_name"] == "Conv3D"
    assert model_layers[-1]["config"]["filters"] == 1
    assert model_layers[-1]["config"]["activation"] == "sigmoid"

    layer_names = [i_layer["class_name"] for i_layer in model_layers]

    assert sorted(np.unique(layer_names).tolist()) == [
        "Concatenate",
        "Conv3D",
        "Cropping3D",
        "InputLayer",
        "MaxPooling3D",
        "UpSampling3D",
        "ZeroPadding3D",
    ]


def test_UNet_3D_one_hot():
    architecture = UNet.UNet_3D(
        {"input_1": [30, 30, 30, 1]}, {"output_1": 2}, model_config={"one_hot_output": True}
    )
    model = architecture.create_model()
    # Make sure we can compile the model
    model.compile(loss=loss_fuction)

    model_config = model.get_config()
    model_layers = model_config["layers"]
    assert len(model_layers) == 38

    assert model_layers[0]["class_name"] == "InputLayer"
    assert model_layers[0]["config"]["batch_input_shape"] == (None, 30, 30, 30, 1)
    assert model_layers[-1]["class_name"] == "Conv3D"
    assert model_layers[-1]["config"]["filters"] == 2
    assert model_layers[-1]["config"]["activation"] == "softmax"

    layer_names = [i_layer["class_name"] for i_layer in model_layers]

    assert sorted(np.unique(layer_names).tolist()) == [
        "Concatenate",
        "Conv3D",
        "Cropping3D",
        "InputLayer",
        "MaxPooling3D",
        "UpSampling3D",
        "ZeroPadding3D",
    ]


def test_UNet_3D_multi_class():
    architecture = UNet.UNet_3D({"input_1": [30, 30, 30, 1]}, {"output_1": 4})
    model = architecture.create_model()
    # Make sure we can compile the model
    model.compile(loss=loss_fuction)

    model_config = model.get_config()
    model_layers = model_config["layers"]
    assert len(model_layers) == 38

    assert model_layers[0]["class_name"] == "InputLayer"
    assert model_layers[0]["config"]["batch_input_shape"] == (None, 30, 30, 30, 1)
    assert model_layers[-1]["class_name"] == "Conv3D"
    assert model_layers[-1]["config"]["filters"] == 4
    assert model_layers[-1]["config"]["activation"] == "softmax"

    layer_names = [i_layer["class_name"] for i_layer in model_layers]

    assert sorted(np.unique(layer_names).tolist()) == [
        "Concatenate",
        "Conv3D",
        "Cropping3D",
        "InputLayer",
        "MaxPooling3D",
        "UpSampling3D",
        "ZeroPadding3D",
    ]


# ===============================================================
#  DDSNet
# ===============================================================


def test_DDS_2D():
    architecture = DDSNet.DDSNet_2D({"input_1": [131, 131, 1]}, {"output_1": 10})
    model = architecture.create_model()
    # Make sure we can compile the model
    model.compile(loss=loss_fuction)

    model_config = model.get_config()
    model_layers = model_config["layers"]
    assert len(model_layers) == 25

    assert model_layers[0]["class_name"] == "InputLayer"
    assert model_layers[0]["config"]["batch_input_shape"] == (None, 131, 131, 1)
    assert model_layers[-1]["class_name"] == "Dense"
    assert model_layers[-1]["config"]["units"] == 10
    assert model_layers[-1]["config"]["activation"] == "softmax"

    layer_names = [i_layer["class_name"] for i_layer in model_layers]

    assert sorted(np.unique(layer_names).tolist()) == [
        "BatchNormalization",
        "Conv2D",
        "Dense",
        "Flatten",
        "InputLayer",
        "MaxPooling2D",
        "PReLU",
    ]


def test_DDS_3D():
    architecture = DDSNet.DDSNet_3D({"input_1": [131, 131, 131, 1]}, {"output_1": 10})
    model = architecture.create_model()
    # Make sure we can compile the model
    model.compile(loss=loss_fuction)

    model_config = model.get_config()
    model_layers = model_config["layers"]
    assert len(model_layers) == 25

    assert model_layers[0]["class_name"] == "InputLayer"
    assert model_layers[0]["config"]["batch_input_shape"] == (None, 131, 131, 131, 1)
    assert model_layers[-1]["class_name"] == "Dense"
    assert model_layers[-1]["config"]["units"] == 10
    assert model_layers[-1]["config"]["activation"] == "softmax"

    layer_names = [i_layer["class_name"] for i_layer in model_layers]

    assert sorted(np.unique(layer_names).tolist()) == [
        "BatchNormalization",
        "Conv3D",
        "Dense",
        "Flatten",
        "InputLayer",
        "MaxPooling3D",
        "PReLU",
    ]
