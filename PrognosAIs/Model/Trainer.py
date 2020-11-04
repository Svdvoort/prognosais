from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys

from typing import Tuple
from typing import Union

import tensorflow as tf
from tensorflow.keras.models import load_model


from tensorflow.keras.mixed_precision import experimental as mixed_precision

import PrognosAIs.Constants
import PrognosAIs.IO.utils as IO_utils
import PrognosAIs.Model.Architectures
import PrognosAIs.Model.Parsers as ModelParsers

from PrognosAIs.IO import ConfigLoader
from PrognosAIs.IO import DataGenerator
from PrognosAIs.Model.Architectures import VGG
from PrognosAIs.Model.Architectures import AlexNet
from PrognosAIs.Model.Architectures import DDSNet
from PrognosAIs.Model.Architectures import DenseNet
from PrognosAIs.Model.Architectures import InceptionNet
from PrognosAIs.Model.Architectures import ResNet
from PrognosAIs.Model.Architectures import UNet


class Trainer:
    """Trainer to be used for training a model."""

    def __init__(
        self,
        config: ConfigLoader.ConfigLoader,
        sample_folder: str,
        output_folder: str,
        tmp_data_folder: str = None,
        save_name: str = None,
    ) -> None:
        """
        Trainer to be used for training a model.

        Args:
            config (ConfigLoader.ConfigLoader): Config to be used
            sample_folder (str): Folder containing the train and validation samples
            output_folder (str): Folder to put the resulting model
            tmp_data_folder (str): Folder to copy samples to and load from. Defaults to None.
            save_name (str): Specify a name to save the model as instead of
                using a automatically generated one. Defaults to None.
        """
        self._model = None
        self._train_data_generator_is_setup = False
        self._validation_data_generator_is_setup = False
        self.cluster_resolver = None
        self.total_memory_used = 0
        self.multiworker = False
        self.worker_index = 0
        self.n_workers = 1
        self.steps_per_epoch = None
        self.validation_steps = None

        self.config = copy.deepcopy(config)
        self.output_folder = os.path.join(output_folder, PrognosAIs.Constants.MODEL_SUBFOLDER)

        self.sample_folder = sample_folder
        self.tmp_data_folder = tmp_data_folder

        IO_utils.setup_logger()
        logging.info(
            "Using configuration file: {config_file}".format(config_file=self.config.config_file),
        )
        logging.info("Loading samples from: {input_dir}".format(input_dir=self.sample_folder))
        logging.info("Putting output in: {output}".format(output=self.output_folder))

        if save_name is None:
            self.save_name = self.config.get_save_name()
        else:
            self.save_name = save_name

        self.model_save_file = ".".join(
            [os.path.join(self.output_folder, self.save_name), PrognosAIs.Constants.HDF5_EXTENSION],
        )

        logging.info("Will save model as: {save_name}".format(save_name=self.model_save_file))

        IO_utils.create_directory(self.output_folder)
        self.config.copy_config(self.output_folder, self.save_name)

        # First thing we need to do is get the precision strategy
        # After this we will clear up the GPUs again, so this needs to be done
        # before tensorflow allocates anything (such as in the distribution
        # strategy)
        self.set_precision_strategy(self.config.get_float_policy())

        self.distribution_strategy = self.get_distribution_strategy()

        self.custom_definitions_file = self.config.get_custom_definitions_file()

        self.class_weights = self.load_class_weights()

        self.validation_ds_dir = os.path.join(
            self.sample_folder, PrognosAIs.Constants.VALIDATION_DS_NAME,
        )
        self.do_validation = os.path.exists(self.validation_ds_dir)

    # ===============================================================
    # Distribution and precision strategies
    # ===============================================================

    @staticmethod
    def set_tf_config(
        cluster_resolver: tf.distribute.cluster_resolver.ClusterResolver, environment: str = None,
    ) -> None:
        """
        Set the TF_CONFIG env variable from the given cluster resolver.

        From https://github.com/tensorflow/tensorflow/issues/37693

        Args:
            cluster_resolver (tf.distribute.cluster_resolver.ClusterResolver): cluster
                resolver to use.
            environment (str): Environment to set in TF_CONFIG. Defaults to None.
        """
        cfg = {
            "cluster": cluster_resolver.cluster_spec().as_dict(),
            "task": {
                "type": cluster_resolver.get_task_info()[0],
                "index": cluster_resolver.get_task_info()[1],
            },
            "rpc_layer": cluster_resolver.rpc_layer,
        }
        if environment:
            cfg["environment"] = environment
        os.environ["TF_CONFIG"] = json.dumps(cfg)
        logging.info(
            "Set up TF config environmentas : {TF_CONFIG}".format(
                TF_CONFIG=os.environ["TF_CONFIG"],
            ),
        )

    def set_precision_strategy(self, float_policy_setting: Union[str, bool]) -> None:
        """
        Set the appropiate precision strategy for GPUs.

        If the GPUs support it a mixed float16 precision will be used
        (see tf.keras.mixe_precision for more information), which reduces the memory overhead
        of the training, while doing computation in float32.
        If GPUs dont support mixed precision, we will try a float16 precision setting.
        If that doesn't work either the normal policy is used.
        If you get NaN values for loss or loss doesn't converge it might be because of the policy.
        Try running the model without a policy setting.

        Args:
            float_policy_setting (float_policy_setting: Union[str, bool]): Which policy to select
                if set to PrognosAIs.Constants.AUTO, we will automatically determine what can be done.
                "mixed" will only consider mixed precision, "float16" only considers float16 policy.
                Set to False to not use a policy
        """
        gpus = IO_utils.get_gpu_devices()
        gpu_supports_mixed_precision = all(
            IO_utils.gpu_supports_mixed_precision(i_gpu) for i_gpu in gpus
        )
        gpu_supports_float16_precision = all(IO_utils.gpu_supports_float16(i_gpu) for i_gpu in gpus)

        mixed_precision_allowed = float_policy_setting in [PrognosAIs.Constants.AUTO, "mixed"]
        float16_precision_allowed = float_policy_setting in [PrognosAIs.Constants.AUTO, "float16"]

        if gpu_supports_mixed_precision:
            logging.info("GPU supports a mixed float16 policy")

        if gpu_supports_float16_precision:
            logging.info("GPU support float16 precision policy")

        if gpu_supports_mixed_precision and mixed_precision_allowed and len(gpus) > 0:
            policy = mixed_precision.Policy("mixed_float16", loss_scale="dynamic")
            mixed_precision.set_policy(policy)
            logging.info(
                (
                    "Using a mixed float16 policy, with compute dtype {cdtype} "
                    "and variable dtype {vdtype}.\n"
                    "Loss scaling applied: {loss_scale}"
                ).format(
                    cdtype=policy.compute_dtype,
                    vdtype=policy.variable_dtype,
                    loss_scale=policy.loss_scale,
                ),
            )
            logging.warning(
                "If model is not converging or loss give NaNs, consider turning of this policy",
            )
        elif gpu_supports_float16_precision and float16_precision_allowed and len(gpus) > 0:
            policy = mixed_precision.Policy("float16", loss_scale="dynamic")
            mixed_precision.set_policy(policy)
            logging.info(
                (
                    "Using a float16 policy, with compute dtype {cdtype} "
                    "and variable dtype {vdtype}.\n"
                    "Loss scaling applied: {loss_scale}"
                ).format(
                    cdtype=policy.compute_dtype,
                    vdtype=policy.variable_dtype,
                    loss_scale=policy.loss_scale,
                ),
            )
            logging.warning(
                "If model is not converging or loss give NaNs, consider turning of this policy",
            )
        else:
            logging.info("No float policy is being used")

    def get_distribution_strategy(self) -> tf.distribute.Strategy:
        """
        Get the appropiate distribution strategy.

        A strategy will be returned that can either distribute the training over
        multiple SLURM nodes, over multi GPUs, train on a single GPU or on a
        single CPU (in that order).

        Returns:
            tf.distribute.Strategy: The distribution strategy to be used in training.
        """
        if IO_utils.get_number_of_slurm_nodes() > 1:
            # We are in a slurm environment for multiworker
            self.cluster_resolver = tf.distribute.cluster_resolver.SlurmClusterResolver()
            self.set_tf_config(self.cluster_resolver)
            strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
                cluster_resolver=self.cluster_resolver,
                communication=tf.distribute.experimental.CollectiveCommunication.NCCL,
            )
            self.multiworker = True
            self.worker_index = self.cluster_resolver.get_task_info()[1]
            self.n_workers = (
                len(self.cluster_resolver.get_task_info()[0])
                * self.cluster_resolver.max_tasks_per_node
            )
            logging.info(
                "Using a multi-worker distribution environment with {nodes} nodes and {tasks} tasks per node".format(
                    nodes=len(self.cluster_resolver.get_task_info()[0]),
                    tasks=self.cluster_resolver.max_tasks_per_node,
                ),
            )
        elif IO_utils.get_number_of_gpu_devices() > 1:
            strategy = tf.distribute.MirroredStrategy()
            logging.info(
                "Using a mirrored distribution environment with {gpus} parallel GPUs".format(
                    gpus=strategy.num_replicas_in_sync,
                ),
            )
        elif IO_utils.get_number_of_gpu_devices() == 1:
            gpus = IO_utils.get_gpu_devices()
            gpu_device_name = ":".join(gpus[0].name.split(":")[-2:])
            strategy = tf.distribute.OneDeviceStrategy(gpu_device_name)
            logging.info(
                "Using a single device strategy with {device} as device".format(
                    device=gpu_device_name,
                ),
            )
        else:
            cpus = IO_utils.get_cpu_devices()
            cpu_device_name = ":".join(cpus[0].name.split(":")[-2:])
            strategy = tf.distribute.OneDeviceStrategy(cpu_device_name)
            logging.info(
                "Using a single device strategy with {device} as device".format(
                    device=cpu_device_name,
                ),
            )
        return strategy

    # ===============================================================
    # Set-up of data
    # ===============================================================

    def load_class_weights(self) -> Union[None, dict]:
        """
        Load the class weight from the class weight file.

        Returns:
            Union[None, dict]: Class weights if requested and the class weight file exists,
                otherwise None.
        """
        class_weight_file = os.path.join(self.sample_folder, PrognosAIs.Constants.CLASS_WEIGHT_FILE)
        has_class_weight_file = os.path.exists(class_weight_file)
        class_weights = self.config.get_class_weights()

        if not has_class_weight_file and class_weights is None:
            logging.warning("Class weight file not found, not using class weights")

        if self.config.get_use_class_weights() and (
            class_weights is not None or has_class_weight_file
        ):
            if class_weights is None:
                with open(class_weight_file, "r") as the_class_weight_file:
                    class_weights = json.load(the_class_weight_file)

            out_class_weights = {}

            for i_key, i_value in class_weights.items():
                this_class_weight = {}
                for i_class, i_weight in i_value.items():
                    this_class_weight[int(i_class)] = float(i_weight)

                out_class_weights[i_key] = this_class_weight

            if len(out_class_weights.keys()) == 1:
                out_class_weights = list(out_class_weights.values())[0]
            logging.info(
                "Using the following class weights: {weights}".format(weights=out_class_weights)
            )
        else:
            out_class_weights = None
            logging.info(
                "Class weight file found, but requested to not use class weights, thus not using class weights"
            )

        return out_class_weights

    def move_data_to_temporary_folder(self, data_folder: str) -> str:
        """
        Move the data to a temporary directory before loading.

        Args:
            data_folder (str): The original data folder

        Returns:
            str: Folder to which the data has been moved
        """
        if self.tmp_data_folder is not None and self.config.get_copy_files():
            IO_utils.copy_directory(data_folder, self.tmp_data_folder)
            new_folder = os.path.join(self.tmp_data_folder, IO_utils.get_root_name(data_folder))
            logging.info("Loading data from temporary directory {temp}".format(temp=new_folder))
        else:
            new_folder = data_folder
        return new_folder

    @property
    def train_data_generator(self) -> DataGenerator.HDF5Generator:
        """
        The train data generator to be used in training.

        Returns:
            DataGenerator.HDF5Generator: The train data generator
        """
        if not self._train_data_generator_is_setup:
            logging.info("Setting up train data generator")
            train_ds_folder = os.path.join(self.sample_folder, PrognosAIs.Constants.TRAIN_DS_NAME)
            train_ds_folder = self.move_data_to_temporary_folder(train_ds_folder)
            self._train_data_generator = self.setup_data_generator(train_ds_folder)
            self._train_data_generator_is_setup = True
            if self.multiworker:
                self.steps_per_epoch = self.train_data_generator.steps
        return self._train_data_generator

    @property
    def validation_data_generator(self) -> DataGenerator.HDF5Generator:
        """
        The validation data generator to be used in training.

        Returns:
            DataGenerator.HDF5Generator: The validation data generator
        """
        if not self._validation_data_generator_is_setup and self.do_validation:
            logging.info("Setting up validation data generator")
            validation_ds_folder = self.move_data_to_temporary_folder(self.validation_ds_dir)
            self._validation_data_generator = self.setup_data_generator(validation_ds_folder)
            self._validation_data_generator_is_setup = True
            if self.multiworker:
                self.validation_steps = self.validation_data_generator.steps
        else:
            self._validation_data_generator = None
            self._validation_data_generator_is_setup = True
        return self._validation_data_generator

    def setup_data_generator(self, sample_folder: str) -> DataGenerator.HDF5Generator:
        """
        Set up a data generator for a folder containg train samples.

        Args:
            sample_folder (str): The path to the folder containing the sample files.

        Raises:
            ValueError: If the sample folder does not exist or does not contain any samples.

        Returns:
            DataGenerator.HDF5Generator: Datagenerator of the sample in the sample folder.
        """
        if (
            IO_utils.get_root_name(sample_folder) == PrognosAIs.Constants.VALIDATION_DS_NAME
            and not self.config.get_shuffle_val()
        ):
            # In the validation data we don't shuffle and don't apply data augmentation
            # In that way we can be sure that validation metrics will not depend
            # on random components
            data_augmentation = False
            shuffle = False
            logging.warning("Turned off data augmentation and shuffling for validation data")
        else:
            data_augmentation = self.config.get_do_augmentation()
            shuffle = self.config.get_shuffle()

        batch_size = self.config.get_batch_size() * self.distribution_strategy.num_replicas_in_sync
        if self.distribution_strategy.num_replicas_in_sync > 1:
            logging.info(
                (
                    "Requested batch size {batch}, changed to {global_batch} to work correctly with "
                    "distribution strategy"
                ).format(batch=self.config.get_batch_size(), global_batch=batch_size),
            )

        if os.path.exists(sample_folder) and len(os.listdir(sample_folder)) > 0:
            with self.distribution_strategy.scope():
                # Need to multiply the requested batch size by number of replicas
                # in the distribution strategy to get the total global patch sizes
                # the distribution strategy will then make sure that on each device
                # the batch size is the same as the one requested
                data_generator = DataGenerator.HDF5Generator(
                    sample_folder,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    max_steps=self.config.get_max_steps_per_epoch(),
                )

                # We will try to set-up caching of the dataset in memory
                data_generator.setup_caching(
                    self.config.get_cache_in_memory(), self.total_memory_used,
                )
                if data_generator.cache_in_memory:
                    self.total_memory_used += data_generator.memory_size

                if data_augmentation:
                    data_generator.setup_augmentation(
                        self.config.get_data_augmentation_factor(),
                        self.config.get_data_augmentation_settings(),
                    )

                if self.multiworker:
                    data_generator.repeat = True
                    data_generator.setup_sharding(self.worker_index, self.n_workers)
                    logging.info(
                        "Set up sharding with {workers} workers for worker index {index} for the data generator".format(
                            workers=self.n_workers, index=self.worker_index,
                        ),
                    )

        else:
            raise ValueError(
                "Dataset directory {ds_name} does not exist, cannot create data generator!".format(
                    ds_name=sample_folder,
                ),
            )
        return data_generator

    # ===============================================================
    # Model setup
    # ===============================================================

    @property
    def model(self) -> tf.keras.Model:
        """
        Model to be used in training.

        Returns:
            tf.keras.Model: The model
        """
        if self._model is None:
            self._model = self.setup_model()
            logging.info(
                "Using the following model:\n{model}".format(
                    model=self._model.summary(line_length=120),
                ),
            )
        return self._model

    @staticmethod
    def _get_architecture_name(model_name: str, input_dimensionality: dict) -> Tuple[str, str]:
        """
        Get the full architecture name from the model name and input dimensionality.

        Args:
            model_name (str): Name of the model
            input_dimensionality (dict): Dimensionality of the different inputs

        Returns:
            Tuple[str, str]: Class name of architecture and full achitecture name
        """
        separator = "_"
        architecture_name_parts = model_name.split(separator)
        architecture_class_name = architecture_name_parts[0]

        # We get the model that will fit the maximum dimensionality of our inputs
        max_dimensionality = int(max(input_dimensionality.values()))
        full_architecture_name = "{model_name}_{input_dimensionality}D".format(
            model_name=model_name, input_dimensionality=max_dimensionality,
        )

        return architecture_class_name, full_architecture_name

    def _setup_model(self) -> tf.keras.Model:
        """
        Get the model architecture from the architecture name (not yet compiled).

        Raises:
            ValueError: If architecture is not known

        Returns:
            tf.keras.Model: The loaded architecture
        """
        architecture_class_name, full_architecture_name = self._get_architecture_name(
            self.config.get_model_name(), self.train_data_generator.get_feature_dimensionality(),
        )
        architecture_class = getattr(PrognosAIs.Model.Architectures, architecture_class_name, None)
        if architecture_class is None:
            architecture_class = IO_utils.load_module_from_file(
                self.config.get_custom_definitions_file(),
            )
        architecture = getattr(architecture_class, full_architecture_name, None)
        if architecture is None:
            architecture_class = IO_utils.load_module_from_file(
                self.config.get_custom_definitions_file(),
            )
            architecture = getattr(architecture_class, full_architecture_name, None)

        if architecture is None:
            err_msg = "Could not find requested model {model}!".format(model=full_architecture_name)
            raise ValueError(err_msg)

        return architecture(
            self.train_data_generator.get_feature_shape(),
            self.train_data_generator.get_number_of_classes(),
            model_config=self.config.get_model_settings(),
            # TODO SET FROM CONFIG
            input_data_type=tf.keras.backend.floatx(),
        ).create_model()

    def _load_model(self) -> tf.keras.Model:
        logging.info("Loaded the model")
        return load_model(self.config.get_model_file(), compile=False)

    def setup_model(self) -> tf.keras.Model:
        """
        Set up model to be used during train.

        Returns:
            tf.keras.Model: The compiled model to be trained.
        """
        if self.config.get_use_class_weights_in_losses() and self.class_weights is not None:
            loss_parser = ModelParsers.LossParser(
                self.config.get_loss_settings(), self.class_weights, self.custom_definitions_file,
            )
            self.class_weights = None
            logging.info("Using class weights directly inside losses instead of during fit")
        else:
            loss_parser = ModelParsers.LossParser(
                self.config.get_loss_settings(), module_paths=self.custom_definitions_file,
            )
        model_loss = loss_parser.get_losses()

        loss_weights = self.config.get_loss_weights()

        with self.distribution_strategy.scope():
            # Set up the model and compile the model
            # Using the strategy to make sure everything is distributed properly
            logging.info("Setting up model")
            if self.config.get_resume_training_from_model():
                model = self._load_model()
            else:
                model = self._setup_model()

            logging.info("Setting up optimizer")
            optimizer = ModelParsers.OptimizerParser(
                self.config.get_optimizer_settings(), self.custom_definitions_file,
            ).get_optimizer()

            logging.info("Setting up metrics")
            model_metrics = ModelParsers.MetricParser(
                self.config.get_metric_settings(),
                self.train_data_generator.label_names,
                self.custom_definitions_file,
            ).get_metrics()

            model_message = (
                "The following settings are used to set up the model:"
                "Loss: {loss}"
                "Optimizer: {optimizer}"
                "Metrics: {metrics}"
                "Loss weights: {loss_weights}"
            ).format(
                loss=model_loss,
                optimizer=optimizer,
                metrics=model_metrics,
                loss_weights=loss_weights,
            )
            logging.info(model_message)

            model.compile(
                loss=model_loss,
                optimizer=optimizer,
                metrics=model_metrics,
                loss_weights=loss_weights,
            )
        return model

    def setup_callbacks(self) -> list:
        """
        Set up callbacks to be used during training.

        Returns:
            list: the callbacks
        """
        with self.distribution_strategy.scope():
            logging.info("Setting up callbacks")
            callbacks = ModelParsers.CallbackParser(
                self.config.get_callback_settings(),
                self.output_folder,
                self.custom_definitions_file,
                self.save_name,
            ).get_callbacks()
            logging.info("Using the following callbacks: {callbacks}".format(callbacks=callbacks))

        return callbacks

    # ===============================================================
    # Model training
    # ===============================================================

    def train_model(self) -> str:
        """
        Train the model.

        Returns:
            str: The location where the model has been saved
        """
        with self.distribution_strategy.scope():
            train_data = self.train_data_generator.get_tf_dataset()
            if self.do_validation:
                validation_data = self.validation_data_generator.get_tf_dataset()
            else:
                validation_data = None

        epochs = self.config.get_N_epoch()
        callbacks = self.setup_callbacks()
        logging.info("Starting training")
        logging.debug(
            (
                "Training with following parameters:\n"
                "Train data: {train}\n"
                "Validation data: {val}\n"
                "Epochs: {epoch}\n"
                "Callbacks: {callback}\n"
                "Class weights: {weights}\n"
                "Steps per epoch: {steps}\n"
                "Validation setps: {val_steps}\n"
            ).format(
                train=train_data,
                val=validation_data,
                epoch=epochs,
                callback=callbacks,
                weights=self.class_weights,
                steps=self.steps_per_epoch,
                val_steps=self.validation_steps,
            ),
        )

        self.model.fit(
            train_data,
            validation_data=validation_data,
            epochs=epochs,
            callbacks=callbacks,
            shuffle=False,
            class_weight=self.class_weights,
            verbose=1,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
        )

        logging.info("Finished training")
        if self.worker_index == 0:
            self.model.save(self.model_save_file)
            logging.info("Model saved to {save_file}".format(save_file=self.model_save_file))
        else:
            # We need to save the model for other workers as well, otherwise
            # We run into errors, however we instantly delete because we dont actually
            # Need the other models
            model_save_file = ".".join(
                [
                    os.path.join(self.output_folder, self.save_name + "_" + str(self.worker_index)),
                    PrognosAIs.Constants.HDF5_EXTENSION,
                ],
            )

            self.model.save(model_save_file)
            os.remove(model_save_file)
        return self.model_save_file

    # ===============================================================
    # External use
    # ===============================================================

    @classmethod
    def init_from_sys_args(cls: Trainer, args_in: list) -> Trainer:
        """
        Initialize a Trainer object from the command line.

        Args:
            args_in (list): Arguments to parse to the trainer

        Returns:
            Trainer: The trainer object
        """
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
            help="The input directory where the samples to train on are located",
            metavar="Input directory",
            dest="input_dir",
            type=str,
        )

        parser.add_argument(
            "-o",
            "--output",
            required=True,
            help="The output directory where to store the saved model",
            metavar="Output directory",
            dest="output_dir",
            type=str,
        )
        parser.add_argument(
            "-T",
            "--tmp",
            required=False,
            help="The temporary directory",
            metavar="Temp directory",
            dest="tmp_dir",
            type=str,
            default=None,
        )
        parser.add_argument(
            "-s",
            "--savename",
            required=False,
            help="Save name for model",
            metavar="Save name",
            dest="save_name",
            type=str,
            default=None,
        )

        args = parser.parse_args(args_in)

        config = ConfigLoader.ConfigLoader(args.config)

        return cls(config, args.input_dir, args.output_dir, args.tmp_dir, args.save_name)


if __name__ == "__main__":
    trainer = Trainer.init_from_sys_args(sys.argv[1:])
    trainer.train_model()
