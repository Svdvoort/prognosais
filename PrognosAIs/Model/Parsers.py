import copy
import importlib
import os

import tensorflow.keras.callbacks
import tensorflow.keras.losses
import tensorflow.keras.metrics
import tensorflow.keras.optimizers

import PrognosAIs.Model.Callbacks
import PrognosAIs.Model.Losses
import PrognosAIs.Model.Metrics

from PrognosAIs.IO import utils as IO_utils


class StandardParser:
    def __init__(self, config: dict, module_paths: list):
        self.config = config
        self.module_paths = module_paths

    def parse_settings(self):
        if self.config is not None:
            if "name" in self.config:
                parsed_settings = self._initiate_class(self.config["name"], self.config["settings"])
            else:
                parsed_settings = {}

                for key, value in self.config.items():
                    if "name" in value:
                        parsed_settings[key] = self._initiate_class(
                            value["name"], value["settings"]
                        )
                    else:
                        settings = []
                        for key_deep, value_deep in value.items():
                            settings.append(
                                self._initiate_class(value_deep["name"], value_deep["settings"])
                            )
                        parsed_settings[key] = settings
        else:
            parsed_settings = None

        return parsed_settings

    def _initiate_class(self, class_name, class_settings):
        class_function = self.get_class(class_name)

        if class_settings is None:
            initiated_class = class_function()
        else:
            initiated_class = class_function(**class_settings)

        return initiated_class

    def get_class(self, class_name):
        class_function = None
        module_index = 0
        while class_function is None and module_index < len(self.module_paths):
            module_path = self.module_paths[module_index]
            if isinstance(module_path, str):
                module_path = IO_utils.load_module_from_file(module_path)
            class_function = getattr(module_path, class_name, None)

            module_index += 1

        if class_function is None:
            raise ValueError("Requested class {} not found!".format(class_name))

        return class_function


class LossParser(StandardParser):
    def __init__(self, loss_settings: dict, class_weights: dict = None, module_paths=None):
        """
        Parse loss settings to actual losses

        Args:
            loss_settings: Settings for the losses

        Returns:
            None
        """

        # TODO remove the need for a "settings" index if there is no settings
        # Index just need to default to None.
        # Also get rid of all the "vague" keywords that are then in the config

        self.class_weights = class_weights
        self.module_paths = [tensorflow.keras.losses, PrognosAIs.Model.Losses]
        if module_paths is not None:
            if isinstance(module_paths, list):
                self.module_paths.extend(module_paths)
            else:
                self.module_paths.append(module_paths)

        if "name" in loss_settings and class_weights is not None:
            loss_settings["settings"]["class_weight"] = self.class_weights
        elif "name" not in loss_settings:
            for key, value in loss_settings.items():
                if (
                    value["settings"] is not None
                    and class_weights is not None
                    and key in self.class_weights.keys()
                ):
                    value["settings"]["class_weight"] = self.class_weights[key]

        super().__init__(loss_settings, self.module_paths)

    def get_losses(self):
        return self.parse_settings()


class OptimizerParser(StandardParser):
    def __init__(self, optimizer_settings: dict, module_paths=None) -> None:
        """
        Interfacing class to easily get a tf.keras.optimizers optimizer

        Args:
            optimizer_settings: Arguments to be passed to the optimizer

        Returns:
            None
        """

        self.module_paths = [tensorflow.keras.optimizers]
        if module_paths is not None:
            if isinstance(module_paths, list):
                self.module_paths.extend(module_paths)
            else:
                self.module_paths.append(module_paths)

        super().__init__(optimizer_settings, self.module_paths)

        return

    def get_optimizer(self):
        return self.parse_settings()


class CallbackParser(StandardParser):
    def __init__(
        self, callback_settings: dict, root_path: str = None, module_paths=None, save_name=None
    ):
        """
        Parse callback settings to actual callbacks

        Args:
            callback_settings: Settings for the callbacks

        Returns:
            None
        """
        self.module_paths = [tensorflow.keras.callbacks, PrognosAIs.Model.Callbacks]
        self.save_name = save_name
        if module_paths is not None:
            if isinstance(module_paths, list):
                self.module_paths.extend(module_paths)
            else:
                self.module_paths.append(module_paths)

        super().__init__(callback_settings, self.module_paths)

        if root_path is not None:
            self.config = self.replace_root_path(self.config, root_path)
        return

    def replace_root_path(self, settings, root_path):
        for key, value in settings.items():
            if type(value) == dict:
                settings[key] = self.replace_root_path(value, root_path)
            else:
                # TODO  this gives error, for example with TensorBoard which has profile_batch option
                # (because file is in profile)
                if "file" in key:
                    settings[key] = os.path.join(root_path, value)
                    if self.save_name is not None:
                        settings[key] = settings[key].format(savename=self.save_name)

        return settings

    def get_callbacks(self):
        # Need to make sure that CSVLogger is the last so that everything is properly stored
        parsed_callbacks = list(self.parse_settings().values())
        out_parsed_callbacks = []

        logger_index = -1
        for i_i_parsed_callback, i_parsed_callback in enumerate(parsed_callbacks):
            if type(i_parsed_callback) == tensorflow.keras.callbacks.CSVLogger:
                logger_index = i_i_parsed_callback
            else:
                out_parsed_callbacks.append(i_parsed_callback)

        if logger_index != -1:
            out_parsed_callbacks.append(parsed_callbacks[logger_index])

        return out_parsed_callbacks


class MetricParser(StandardParser):
    def __init__(self, metric_settings: dict, label_names: list = None, module_paths=None) -> None:
        """
        Parse metrics settings to actual metrics

        Args:
            loss_settings: Settings for the losses
        """

        self.module_paths = [tensorflow.keras.metrics, PrognosAIs.Model.Metrics]
        if module_paths is not None:
            if isinstance(module_paths, list):
                self.module_paths.extend(module_paths)
            else:
                self.module_paths.append(module_paths)

        super().__init__(metric_settings, self.module_paths)
        if label_names is not None:
            self.label_names = label_names
        else:
            self.label_names = []

    def get_metrics(self):
        parsed_callbacks = self.parse_settings()

        if not isinstance(parsed_callbacks, dict) and parsed_callbacks is not None:
            parsed_callbacks = [parsed_callbacks]
        elif (
            parsed_callbacks is not None
            and self.label_names != []
            and sorted(parsed_callbacks.keys()) != sorted(self.label_names)
        ):
            parsed_callbacks = list(parsed_callbacks.values())
        elif parsed_callbacks is None:
            parsed_callbacks = []
        return parsed_callbacks

    def convert_metrics_list_to_dict(self, metrics: list) -> dict:
        if isinstance(metrics, list):
            new_metrics = {}

            for i_label_name in self.label_names:
                new_metrics[i_label_name] = []
                other_label_names = copy.deepcopy(self.label_names)
                other_label_names.remove(i_label_name)

                for i_metric in metrics:
                    i_metric_name = i_metric.name
                    i_metric_base_name = i_metric.__class__().name
                    i_metric_label_name = i_metric_name.split(i_metric_base_name)[0]
                    # No output specific metrics, append all metrics to all outputs
                    if "_" not in i_metric_label_name:
                        new_metrics[i_label_name].append(i_metric)
                    else:
                        # Need to cut off the last underscore to get the label name
                        i_metric_label_name = i_metric_label_name[:-1]
                        if i_metric_label_name == i_label_name:
                            new_metrics[i_label_name].append(i_metric)
        elif isinstance(metrics, dict):
            for key, value in metrics.items():
                if not isinstance(value, list):
                    metrics[key] = [value]

            new_metrics = metrics
        else:
            new_metrics = metrics
        return new_metrics
