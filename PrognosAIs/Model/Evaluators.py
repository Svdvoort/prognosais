import argparse
import os
import sys

from types import ModuleType
from typing import Dict
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
import SimpleITK as sitk
import tensorflow as tf

from tensorflow.keras.models import load_model

import PrognosAIs.Constants
import PrognosAIs.IO.utils as IO_utils
import PrognosAIs.Model.Losses
import PrognosAIs.Model.Parsers as ModelParsers

from PrognosAIs.IO import ConfigLoader
from PrognosAIs.IO import DataGenerator


class Evaluator:
    def __init__(self, model_file, data_folder, config_file, output_folder) -> None:
        self.EVALUATION_FOLDER = "Results"
        self.data_folder = data_folder
        self.model_file = model_file
        self.predictions = None
        self.metrics = {}
        self.metric_functions = None
        self.sample_predictions = None
        self.sample_labels = None
        self.sample_names = None
        self.sample_metrics = None
        self.output_folder = os.path.join(output_folder, self.EVALUATION_FOLDER)
        IO_utils.create_directory(self.output_folder)

        self.config = ConfigLoader.ConfigLoader(config_file)
        self.batch_size = self.config.get_batch_size()

        self.custom_module = IO_utils.load_module_from_file(
            self.config.get_custom_definitions_file(),
        )
        self.model = self.load_model(self.model_file, self.custom_module)

        self.init_model_parameters()

        self.data_generator = self.init_data_generators()
        self.dataset_names = list(self.data_generator.keys())
        self.sample_metadata = self.data_generator[self.dataset_names[0]].get_feature_metadata()
        self.label_data_generator = self.init_label_generators()

        self.metric_functions = self.get_to_evaluate_metrics()

        self.image_output_labels = self.get_image_output_labels()

    @staticmethod
    def _load_model(model_file: str, custom_objects: dict) -> Tuple[tf.keras.Model, ValueError]:
        """
        Try to load a model, if it doesnt work parse the error.

        Args:
            model_file (str): Location of the model file
            custom_objects (dict): Potential custom objects to use during model loading

        Returns:
            Tuple[tf.keras.Model, ValueError]: The model if successfully loaded, otherwise the error
        """
        try:
            model = load_model(model_file, compile=True, custom_objects=custom_objects)
            out_error = None
        except ValueError as the_error:
            out_error = the_error
            model = None

        return model, out_error

    @staticmethod
    def _fake_fit(model: tf.keras.Model) -> tf.keras.Model:
        """
        Fit of the model on fake date to properly initialize the model.

        Args:
            model (tf.keras.Model): Model to initialize

        Returns:
            tf.keras.Model: Initalized model.
        """
        input_shapes = model.input_shape
        output_shapes = model.output_shape

        if isinstance(input_shapes, dict):
            fake_input = {}
            for i_key, i_value in input_shapes.items():
                fake_input[i_key] = tf.random.uniform((1,) + i_value[1:])
        else:
            fake_input = tf.random.uniform((1,) + input_shapes[1:])

        if isinstance(output_shapes, list):
            fake_output = []
            for i_value in output_shapes:
                fake_output.append(tf.random.uniform((1,) + i_value[1:]))
        else:
            fake_output = tf.random.uniform((1,) + output_shapes[1:])

        # We are not currently doing it as number of epochs is not very important in evaluation
        # But you could get the run number of epochs from the loaded model via:
        # model.optimizer.iterations and divided this by the number of steps
        # in each epoch (which is #samples/batch size)

        model.fit(fake_input, fake_output, verbose=0)
        return model

    @staticmethod
    def load_model(model_file: str, custom_module: ModuleType = None) -> tf.keras.Model:
        """
        Load the model, including potential custom losses.

        Args:
            model_file (str): Location of the model file
            custom_module (ModuleType): Custom module from which to load losses or metrics

        Raises:
            error: If the model could not be loaded
                and the problem is not due to a missing loss or metric function.

        Returns:
            tf.keras.Model: The loaded model
        """
        # type hint for mypy
        custom_objects: Dict[str, Union[tf.keras.losses.Loss.tf.keras.metrics.Metric]] = {}
        model_is_loaded = False
        model = None
        while not model_is_loaded:
            model, error = Evaluator._load_model(model_file, custom_objects)
            if model is not None:
                model_is_loaded = True
                break

            err_msg = error.args[0]
            custom_object_name = err_msg.split(":")[-1].strip()

            if "Unknown loss function" in str(err_msg):
                custom_object = getattr(PrognosAIs.Model.Losses, custom_object_name, None)
                if custom_object is None:
                    custom_object = getattr(custom_module, custom_object_name)

            elif "Unknown metric function" in str(err_msg):
                custom_object = getattr(PrognosAIs.Model.Metrics, custom_object_name, None)
                if custom_object is None:
                    custom_object = getattr(custom_module, custom_object_name)
            elif "Unknown optimizer" in str(err_msg):
                custom_object = getattr(custom_module, custom_object_name)
            else:
                raise error
            custom_objects.update({custom_object_name: custom_object})

        # There is an "update" in TF2.2 which does not make metrics available by default
        # from loaded models, see https://github.com/tensorflow/tensorflow/issues/37990
        # Therefore we do a fake "fit" round to make everything available
        return Evaluator._fake_fit(model)

    def _init_data_generators(self, labels_only: bool) -> dict:
        """
        Initialize data generators for all sample folders.

        Args:
            labels_only (bool): Whether to only load labels

        Returns:
            dict: initalized data generators
        """
        sub_folders = IO_utils.get_subdirectories(self.data_folder)
        data_generators = {}
        for i_sub_folder in sub_folders:
            folder_name = IO_utils.get_root_name(i_sub_folder)
            if (
                folder_name == PrognosAIs.Constants.TRAIN_DS_NAME
                and not self.config.get_evaluate_train_set()
            ):
                continue
            data_generators[folder_name] = DataGenerator.HDF5Generator(
                i_sub_folder,
                self.batch_size,
                shuffle=self.config.get_shuffle_evaluation(),
                drop_batch_remainder=False,
                labels_only=labels_only,
            )
        return data_generators

    def init_data_generators(self) -> dict:
        """
        Initialize the data generators.

        Returns:
            dict: DataGenerator for each subfolder of samples
        """

        return self._init_data_generators(False)

    def init_label_generators(self) -> dict:
        """
        Initialize the data generators which only give labels.

        Returns:
            dict: DataGenerator for each subfolder of samples
        """
        return self._init_data_generators(False)

    def init_model_parameters(self) -> None:
        """
        Initialize the parameters from the model.
        """
        self.output_names = self.model.output_names
        self.number_of_outputs = len(self.output_names)
        if self.number_of_outputs == 1:
            self.output_shapes = [self.model.output_shape]
        else:
            self.output_shapes = self.model.output_shape

        self.output_classes = {}
        self.one_hot_outputs = {}
        for i_output_index, i_output_name in enumerate(self.output_names):
            self.output_classes[i_output_name] = self.output_shapes[i_output_index][-1]

            if self.output_shapes[i_output_index][-1] > 1:
                self.one_hot_outputs[i_output_name] = True
            else:
                self.one_hot_outputs[i_output_name] = False

        self.input_names = self.model.input_names
        self.number_of_inputs = len(self.input_names)

        model_input_shape = self.model.input_shape
        if isinstance(model_input_shape, dict):
            self.input_shapes = list(model_input_shape.values())
        elif self.number_of_inputs == 1:
            self.input_shapes = [model_input_shape]
        else:
            self.input_shapes = model_input_shape.values()

    def get_image_output_labels(self) -> dict:
        """
        Whether an output label is a simple class, the label is actually an image.

        Returns:
            dict: Output labels that are image outputs
        """
        image_outputs_labels = {}

        for i_output_name, i_output_shape in zip(self.output_names, self.output_shapes):
            for i_input_name, i_input_shape in zip(self.input_names, self.input_shapes):
                # It is an image of a certain input of the output has as many dimension
                # and the size of each dimension is equal to the input size
                # minus the batch dimension and number of classes

                equal_dimensions = len(i_input_shape) == len(i_output_shape)
                equal_size = i_input_shape[1:-1] == i_output_shape[1:-1]
                if equal_dimensions and equal_size:
                    image_outputs_labels[i_output_name] = i_input_name

        return image_outputs_labels

    def get_real_labels(self) -> dict:
        real_labels = {}
        for i_generator_name in self.data_generator.keys():
            real_labels[i_generator_name] = self.get_real_labels_of_sample_subset(i_generator_name)

        return real_labels

    def get_real_labels_of_sample_subset(self, subset_name: str) -> dict:
        """
        Get the real labels corresponding of all samples from a subset.

        Args:
            subset_name (str): Name of subset to get labels for

        Returns:
            dict: Real labels for each dataset and output
        """
        real_labels = {}
        subset_generator = self.data_generator[subset_name].get_tf_dataset()
        for i_output_name in self.output_names:
            real_labels[i_output_name] = np.concatenate(
                [i_label_batch[1][i_output_name] for i_label_batch in subset_generator], axis=0,
            )
        return real_labels

    def _combine_config_and_model_metrics(self, model_metrics: dict, config_metrics: dict) -> dict:
        """
        Combine the metrics specified in the model and those specified in the config.

        Args:
            model_metrics (dict): Metrics as defined by the model
            config_metrics (dict): Metrics defined in the config

        Returns:
            dict: Combined metrics
        """
        metric_functions = {}
        for i_output_name in self.output_names:
            if i_output_name in model_metrics and i_output_name in config_metrics:
                metric_functions[i_output_name] = (
                    model_metrics[i_output_name] + config_metrics[i_output_name]
                )
            elif i_output_name in model_metrics:
                metric_functions[i_output_name] = model_metrics[i_output_name]
            elif i_output_name in config_metrics:
                metric_functions[i_output_name] = config_metrics[i_output_name]

        return metric_functions

    def get_to_evaluate_metrics(self) -> dict:
        """
        Get the metrics functions which should be evaluated.

        Returns:
            dict: Metric function to be evaluated for the different outputs
        """
        # We get both the metrics that are specified in the config
        # As well as the metrics from the model
        metric_parser = ModelParsers.MetricParser(
            self.config.get_evaluation_metric_settings(),
            self.output_names,
            self.config.get_custom_definitions_file(),
        )
        model_metrics = self.model.metrics

        # First N metrics are always losses, we dont want to evaluate those
        if self.number_of_outputs > 1 and len(model_metrics) > self.number_of_outputs:
            # When we have multiple outputs we need to do + 1 because we also
            # have the total loss
            model_metrics = model_metrics[self.number_of_outputs + 1 :]
        elif len(model_metrics) > self.number_of_outputs:
            model_metrics = model_metrics[self.number_of_outputs :]
        else:
            # In this case there are no metrics, only losses
            model_metrics = []

        model_metrics = metric_parser.convert_metrics_list_to_dict(model_metrics)
        config_metrics = metric_parser.get_metrics()
        config_metrics = metric_parser.convert_metrics_list_to_dict(config_metrics)

        return self._combine_config_and_model_metrics(model_metrics, config_metrics)

    def _format_predictions(self, predictions: Union[list, np.ndarray]) -> dict:
        """
        Format the predictions to match them with the output names

        Args:
            predictions (Union[list, np.ndarray]): The predictions from the model

        Raises:
            ValueError: If the predictions do not match with the expected output names

        Returns:
            dict: Output predictions matched with the output names
        """
        if isinstance(predictions, np.ndarray):
            # There is only one output in this case
            predictions = [predictions]

        if len(predictions) != len(self.output_names):
            raise ValueError("The predictions do not match with the output names!")

        out_predictions = {}
        for i_output_name, i_prediction in zip(self.output_names, predictions):
            out_predictions[i_output_name] = i_prediction

        return out_predictions

    def predict(self) -> dict:
        """
        Get predictions from the model

        Returns:
            dict: Predictions for the different outputs of the model for all samples
        """
        if self.predictions is None:
            # We have not yet determined the predictions, first run
            self.predictions = {}
            predictions = {}
            for i_generator_name, i_generator in self.data_generator.items():
                # We go over all generators
                self.predictions[i_generator_name] = {}
                dataset = i_generator.get_tf_dataset()
                final_predictions = {}
                for i_output_name in self.output_names:
                    final_predictions[i_output_name] = []

                for i_batch in dataset:
                    # We have to go over the different predictions step by step
                    # Otherwise will lead to memory leak
                    # The first index in the batch is the sample (the second is the label)
                    batch_prediction = self.model.predict_on_batch(i_batch[0])

                    # Convert to list if we only have one output
                    if isinstance(batch_prediction, np.ndarray):
                        batch_prediction = [batch_prediction]
                    for i_output_name, i_prediction in zip(self.output_names, batch_prediction):
                        final_predictions[i_output_name].append(i_prediction)

                # We create one single list for all predictions that we got
                for i_output_name in self.output_names:
                    final_predictions[i_output_name] = np.concatenate(
                        final_predictions[i_output_name], axis=0
                    )

                predictions[i_generator_name] = final_predictions
            self.predictions = predictions
        return self.predictions

    def evaluate_metrics_from_predictions(self, predictions: dict, real_labels: dict) -> dict:
        """
        Evaluate the metrics based on the model predictions

        Args:
            predictions (dict): Predictions obtained from the model
            real_labels (dict): The true labels of the samples for the different outputs

        Returns:
            dict: The different evaluated metrics
        """

        out_metrics = {}
        for i_output_name, i_output_prediction in predictions.items():
            out_metrics[i_output_name] = {}
            to_evaluate_functions = self.metric_functions[i_output_name]
            for i_function in to_evaluate_functions:
                for i_real_label, i_prediction in zip(
                    real_labels[i_output_name], i_output_prediction
                ):
                    i_real_label = np.asarray(i_real_label)
                    i_prediction = np.asarray(i_prediction)
                    if len(i_real_label.shape) == 1:
                        i_real_label = np.expand_dims(i_real_label, 0)
                        i_prediction = np.expand_dims(i_prediction, 0)

                    i_function.update_state(i_real_label, i_prediction)
                out_metrics[i_output_name][i_function.name] = i_function.result().numpy()
                i_function.reset_states()

        return out_metrics

    def evaluate_metrics(self) -> dict:
        """
        Evaluate all metrics for all samples

        Returns:
            dict: The evaluated metrics
        """
        if self.metrics == {}:
            predictions = self.predict()

            for i_dataset_name, i_predictions in predictions.items():
                real_labels = self.get_real_labels_of_sample_subset(i_dataset_name)
                self.metrics[i_dataset_name] = self.evaluate_metrics_from_predictions(
                    i_predictions, real_labels,
                )
        return self.metrics

    def evaluate_sample_metrics(self) -> dict:
        """
        Evaluate the metrics based on a full sample instead of based on individual batches

        Returns:
            dict: The evaluated metrics
        """
        if self.sample_metrics is None:
            self.sample_metrics = {}
            _, sample_predictions = self.get_sample_predictions_from_patch_predictions()
            _, sample_labels = self.get_sample_labels_from_patch_labels()

            for i_dataset_name, i_predictions in sample_predictions.items():
                self.sample_metrics[i_dataset_name] = self.evaluate_metrics_from_predictions(
                    i_predictions, sample_labels[i_dataset_name],
                )
        return self.sample_metrics

    @staticmethod
    def combine_predictions(
        predictions: np.ndarray, are_one_hot: bool, label_combination_type: str
    ) -> np.ndarray:
        if label_combination_type == "vote":
            if are_one_hot and predictions.ndim >= 2:
                # In case of one-hot we have to sum over the sample dimension
                # We then can just take the argmax to get the real label
                prediction_occurrence = np.sum(np.round(predictions), axis=0)
                majority_vote_index = np.argmax(prediction_occurrence)

                combined_prediction = np.zeros_like(prediction_occurrence)
                combined_prediction[majority_vote_index] = 1
            elif are_one_hot:
                combined_prediction = predictions
            else:
                combined_prediction = np.argmax(np.bincount(predictions))
        elif label_combination_type == "average":
            if are_one_hot and predictions.ndim >= 2:
                combined_prediction = np.mean(predictions, axis=0)
            elif are_one_hot:
                combined_prediction = predictions
            else:
                err_msg = (
                    "Predictions can only be combined when given as probability score"
                    "thus the labels must be one-hot encoded."
                )
                raise ValueError(err_msg)
        else:
            raise ValueError("Unknown combination type")

        return combined_prediction

    def patches_to_sample_image(
        self,
        datagenerator: PrognosAIs.IO.DataGenerator.HDF5Generator,
        filenames: list,
        output_name: str,
        predictions: np.ndarray,
        labels_are_one_hot: bool,
        label_combination_type: str,
    ) -> np.ndarray:

        if not labels_are_one_hot and label_combination_type == "average":
            err_msg = (
                "Predictions can only be combined when given as probability score"
                "thus the labels must be one-hot encoded."
            )
            raise ValueError(err_msg)

        input_name = self.image_output_labels[output_name]
        image_size = self.sample_metadata[input_name]["original_size"]
        transpose_dims = np.arange(len(image_size) - 1, -1, -1)
        number_of_classes = self.output_classes[output_name]
        image_size = np.append(image_size, number_of_classes)

        original_image = np.zeros(image_size)
        number_of_hits = np.zeros(image_size)

        if isinstance(filenames, str):
            filenames = [filenames]

        # TODO REMOVE ONLY FOR IF NOT REALLY PATCHES
        if (
            len(predictions.shape) == len(original_image.shape)
            and predictions.shape[-1] == original_image.shape[-1]
        ):
            predictions = np.expand_dims(predictions, axis=0)

        for i_filename, i_prediction in zip(filenames, predictions):
            i_sample_metadata = datagenerator.get_feature_metadata_from_sample(i_filename)
            i_sample_metadata = i_sample_metadata[input_name]

            in_sample_index_start = np.copy(i_sample_metadata["index"])
            in_sample_index_end = in_sample_index_start + i_sample_metadata["size"]

            # Parts of the patch can be outside of the original image, because of padding
            # Thus here we take only the parts of the patch that are within the original image
            in_sample_index_start[in_sample_index_start < 0] = 0
            sample_indices = tuple(
                slice(*i) for i in zip(in_sample_index_start, in_sample_index_end)
            )

            patch_index_start = np.copy(i_sample_metadata["index"])
            patch_index_end = i_sample_metadata["size"]
            # We also need to cut out the part of the patch that is normally outside of the image
            # we do this here
            patch_index_start[patch_index_start > 0] = 0
            patch_index_start = -1 * patch_index_start
            patch_slices = tuple(slice(*i) for i in zip(patch_index_start, patch_index_end))

            if not labels_are_one_hot:
                i_prediction = i_prediction.astype(np.int32)
                i_prediction = np.eye(number_of_classes)[i_prediction]
            elif label_combination_type == "vote":
                i_prediction = np.round(i_prediction)

            original_image[sample_indices] += i_prediction[patch_slices]
            number_of_hits[sample_indices] += 1

        number_of_hits[number_of_hits == 0] = 1
        if label_combination_type == "vote":
            original_image = np.argmax(original_image, axis=-1)
        elif label_combination_type == "average":
            original_image = np.argmax(np.round(original_image / number_of_hits), axis=-1)
        else:
            raise ValueError("Unknown combination type")

        # Need to transpose because of different indexing between numpy and simpleitk
        original_image = np.transpose(original_image, transpose_dims)
        return original_image

    def image_array_to_sitk(self, image_array: np.ndarray, input_name: str) -> sitk.Image:
        original_image_direction = self.sample_metadata[input_name]["original_direction"]
        original_image_origin = self.sample_metadata[input_name]["original_origin"]
        original_image_spacing = self.sample_metadata[input_name]["original_spacing"]
        img = sitk.GetImageFromArray(image_array)
        img.SetDirection(original_image_direction)
        img.SetOrigin(original_image_origin)
        img.SetSpacing(original_image_spacing)
        # To ensure proper loading
        img = sitk.Cast(img, sitk.sitkFloat32)

        return img

    def _find_sample_names_from_patch_names(self, data_generator):
        filenames = data_generator.sample_files
        file_locations = np.asarray(data_generator.sample_locations)

        # Get the unique names of the files
        sample_names = np.unique([i_file.split("_patch")[0] for i_file in filenames])

        sample_indices = {}
        for i_sample_name in sample_names:
            sample_indices[i_sample_name] = np.squeeze(
                np.argwhere(
                    [i_sample_name == i_filename.split("_patch")[0] for i_filename in filenames]
                )
            )

        return sample_names, sample_indices

    def get_sample_result_from_patch_results(self, patch_results):
        sample_results = {}
        sample_names = {}
        for i_dataset_name, i_dataset_generator in self.data_generator.items():
            i_patch_results = patch_results[i_dataset_name]
            sample_results[i_dataset_name] = {}
            file_locations = np.asarray(i_dataset_generator.sample_locations)

            sample_names[i_dataset_name], sample_indices = self._find_sample_names_from_patch_names(
                i_dataset_generator
            )

            for i_output_name, i_output_prediction in i_patch_results.items():
                sample_results[i_dataset_name][i_output_name] = []

                if i_output_name in self.image_output_labels:
                    for i_sample_name, i_sample_indices in sample_indices.items():
                        patches_from_sample_results = i_output_prediction[i_sample_indices]
                        sample_results[i_dataset_name][i_output_name].append(
                            self.patches_to_sample_image(
                                i_dataset_generator,
                                file_locations[i_sample_indices],
                                i_output_name,
                                patches_from_sample_results,
                                self.one_hot_outputs[i_output_name],
                                self.config.get_label_combination_type(),
                            )
                        )
                else:
                    for i_sample_name, i_sample_indices in sample_indices.items():
                        patches_from_sample_results = i_output_prediction[i_sample_indices]
                        sample_results[i_dataset_name][i_output_name].append(
                            self.combine_predictions(
                                patches_from_sample_results,
                                self.one_hot_outputs[i_output_name],
                                self.config.get_label_combination_type(),
                            )
                        )
                for i_key, i_value in sample_results[i_dataset_name].items():
                    sample_results[i_dataset_name][i_key] = np.asarray(i_value)

        return sample_names, sample_results

    def get_sample_labels_from_patch_labels(self):
        patch_labels = self.get_real_labels()
        sample_names, sample_labels = self.get_sample_result_from_patch_results(patch_labels)

        return sample_names, sample_labels

    def get_sample_predictions_from_patch_predictions(self):
        patch_predictions = self.predict()
        sample_names, sample_predictions = self.get_sample_result_from_patch_results(
            patch_predictions
        )
        return sample_names, sample_predictions

    @staticmethod
    def one_hot_labels_to_flat_labels(labels: np.ndarray) -> np.ndarray:
        flat_labels = np.argmax(labels, axis=-1)
        flat_labels[labels[..., 0] == -1] = -1
        return flat_labels

    def make_dataframe(self, sample_names, predictions, labels) -> pd.DataFrame:
        df_columns = ["Sample"]
        for i_output_name in self.output_names:
            if i_output_name not in self.image_output_labels:
                df_columns.append("Label_" + i_output_name)
                for i_class in range(self.output_classes[i_output_name]):
                    df_columns.append("Prediction_" + i_output_name + "_class_" + str(i_class))

        results_df = pd.DataFrame(columns=df_columns)
        results_df["Sample"] = sample_names
        for i_output_name, i_output_prediction in predictions.items():
            if i_output_name not in self.image_output_labels:
                # i_output_labels = labels[i_output_name]
                # if labels_one_hot[i_output_name]:
                #     i_output_prediction = self.one_hot_labels_to_flat_labels(i_output_prediction,)
                # i_output_labels = self.one_hot_labels_to_flat_labels(i_output_labels)

                if self.one_hot_outputs[i_output_name]:
                    results_df["Label_" + i_output_name] = self.one_hot_labels_to_flat_labels(
                        labels[i_output_name]
                    )
                else:
                    results_df["Label_" + i_output_name] = labels[i_output_name]
                for i_class in range(self.output_classes[i_output_name]):
                    results_df[
                        "Prediction_" + i_output_name + "_class_" + str(i_class)
                    ] = i_output_prediction[:, i_class]

        return results_df

    def write_image_predictions_to_files(self, sample_names, predictions, labels_one_hot) -> None:
        for i_output_name, i_output_prediction in predictions.items():
            if i_output_name in self.image_output_labels:
                if labels_one_hot is not None and labels_one_hot[i_output_name]:
                    i_output_prediction = self.one_hot_labels_to_flat_labels(i_output_prediction,)

                i_output_prediction_images = [
                    self.image_array_to_sitk(
                        i_sample_output_prediction, self.image_output_labels[i_output_name]
                    )
                    for i_sample_output_prediction in i_output_prediction
                ]

                for i_pred_image, i_sample_name in zip(i_output_prediction_images, sample_names):
                    out_file = os.path.join(
                        self.output_folder, i_sample_name.split(".")[0] + "_prediction.nii.gz"
                    )
                    sitk.WriteImage(i_pred_image, out_file)

    def write_predictions_to_file(self) -> None:
        predictions = self.predict()

        for i_dataset_name, i_dataset_generator in self.data_generator.items():
            if self.config.get_patch_predictions():
                out_file = os.path.join(self.output_folder, i_dataset_name + "_predictions.csv")
                i_prediction = predictions[i_dataset_name]

                results_df = self.make_dataframe(
                    i_dataset_generator.sample_files,
                    i_prediction,
                    self.get_real_labels_of_sample_subset(i_dataset_name),
                )

                results_df.to_csv(out_file, index=False)

                self.write_image_predictions_to_files(
                    i_dataset_generator.sample_files, i_prediction, self.one_hot_outputs,
                )

            if self.config.get_combine_patch_predictions():
                (
                    sample_names,
                    sample_predictions,
                ) = self.get_sample_predictions_from_patch_predictions()
                out_file = os.path.join(
                    self.output_folder, i_dataset_name + "_predictions_combined_patches.csv"
                )

                _, sample_labels = self.get_sample_labels_from_patch_labels()
                results_df = self.make_dataframe(
                    sample_names[i_dataset_name],
                    sample_predictions[i_dataset_name],
                    sample_labels[i_dataset_name],
                )
                results_df.to_csv(out_file, index=False)

                self.write_image_predictions_to_files(
                    sample_names[i_dataset_name], sample_predictions[i_dataset_name], None,
                )

    def make_metric_dataframe(self, metrics: dict) -> pd.DataFrame:
        metrics_df = pd.DataFrame(columns=["Metric"] + self.dataset_names)
        for i_dataset_name in self.dataset_names:
            dataset_metrics = metrics[i_dataset_name]
            metric_names = []
            metric_values = []
            for i_output_name, i_output_metrics in dataset_metrics.items():
                metric_names.extend(
                    [
                        i_output_name + "_" + i_metric_name
                        for i_metric_name in i_output_metrics.keys()
                    ]
                )
                metric_values.extend(list(i_output_metrics.values()))

            metrics_df["Metric"] = metric_names
            metrics_df[i_dataset_name] = metric_values
        return metrics_df

    def write_metrics_to_file(self) -> None:
        if self.config.get_patch_predictions():
            metrics = self.evaluate_metrics()
            out_file = os.path.join(self.output_folder, "metrics.csv")

            metrics_df = self.make_metric_dataframe(metrics)
            metrics_df.to_csv(out_file, index=False)

        if self.config.get_combine_patch_predictions():
            out_file = os.path.join(self.output_folder, "metrics_combined_patches.csv")
            sample_metrics = self.evaluate_sample_metrics()
            sample_metrics_df = self.make_metric_dataframe(sample_metrics)
            sample_metrics_df.to_csv(out_file, index=False)

    def evaluate(self):
        if self.config.get_write_predictions():
            self.write_predictions_to_file()

        if self.config.get_evaluate_metrics():
            self.write_metrics_to_file()

    @classmethod
    def init_from_sys_args(cls, args_in):
        parser = argparse.ArgumentParser(description="Train a CNN")

        parser.add_argument(
            "-c",
            "--config",
            required=True,
            help="The location of the PrognosAIs config file",
            metavar="configuration file",
            dest="config",
            type=str,
        )

        parser.add_argument(
            "-i",
            "--input",
            required=True,
            help="The input directory where the to evaluate samples are located",
            metavar="Input directory",
            dest="input_dir",
            type=str,
        )

        parser.add_argument(
            "-o",
            "--output",
            required=True,
            help="The output directory where to store the results",
            metavar="Output directory",
            dest="output_dir",
            type=str,
        )

        parser.add_argument(
            "-m",
            "--model",
            required=True,
            help="The model file to evaluate",
            metavar="Modefile",
            dest="model_file",
            type=str,
        )

        args = parser.parse_args(args_in)

        return cls(args.model_file, args.input_dir, args.config, args.output_dir)


if __name__ == "__main__":
    evaluator = Evaluator.init_from_sys_args(sys.argv[1:])
    evaluator.evaluate()
