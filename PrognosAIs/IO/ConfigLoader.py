import datetime
import hashlib
import os
import random
import re
import shutil

import PrognosAIs.IO.utils as utils
import yaml


# TODO make sure in testing that all floats/ number are actually loaded as such
class ConfigLoader:
    def __init__(self, config_file):
        self.config_file = config_file
        loader = yaml.SafeLoader
        # We add the following to safely load scientific notation as floats
        loader.add_implicit_resolver(
            "tag:yaml.org,2002:float",
            re.compile(
                """^(?:
                       [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
                       |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
                       |\\.[0-9_]+(?:[eE][-+][0-9]+)?
                       |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
                       |[-+]?\\.(?:inf|Inf|INF)
                       |\\.(?:nan|NaN|NAN))$""",
                re.X,
            ),
            list("-+0123456789."),
        )

        with open(config_file, "r") as ymlfile:
            self.cfg = yaml.load(ymlfile, Loader=loader)

        if "original_location" not in self.cfg:
            self.cfg["original_location"] = utils.get_parent_directory(self.config_file)

    def copy_config(self, output_folder, save_name=None):
        if save_name is not None:
            new_config_location = os.path.join(
                output_folder, "config_{savename}.yaml".format(savename=save_name)
            )
        else:
            new_config_location = os.path.join(output_folder, "config.yaml")

        with open(new_config_location, "w") as f:
            yaml.dump(self.cfg, f, default_flow_style=False)

        return new_config_location

    def get_config_file(self):
        return self.config_file

    def get_data_folder(self):
        return self.cfg["preprocessing"]["data_folder"]

    def get_test_data_folder(self):
        return self.cfg["testing"]["data_folder"]

    def get_resample_images(self):
        return self.cfg["preprocessing"]["resample_images"]

    def get_make_patches(self):
        return self.cfg["preprocessing"]["make_patches"]

    def get_patch_size(self):
        return self.cfg["preprocessing"]["patch_size"]

    def get_reject_patches(self):
        return self.cfg["preprocessing"]["reject_patches"]

    def get_min_patch_voxels(self):
        return self.cfg["preprocessing"]["min_patch_voxels"]

    def get_extra_input_file(self):
        return self.cfg["preprocessing"]["extra_input_file"]

    def get_mask_file(self):
        return self.cfg["preprocessing"]["mask_file"]

    def get_fsl_reorient_bin(self):
        return self.cfg["program_paths"]["fslreorient2std_bin"]

    def get_fsl_val_bin(self):
        return self.cfg["program_paths"]["fslval_bin"]

    def get_label_file(self):
        return self.cfg["preprocessing"]["label_file"]

    def get_test_label_file(self):
        return self.cfg["testing"]["label_file"]

    def get_dataset_distribution(self):
        return self.cfg["preprocessing"]["dataset_distribution"]

    def get_resample_size(self):
        return self.cfg["preprocessing"]["resample_image_size"]

    def get_random_state(self):
        random_state = self.cfg["preprocessing"]["random_state"]
        if random_state == -1:
            random_state = random.randint(1, 5000000)
        return random_state

    def get_stratify_index(self):
        stratify_index = self.cfg["preprocessing"]["stratify_index"]
        if stratify_index == "None":
            stratify_index = None
        return stratify_index

    def get_multi_channels_patches(self):
        return self.cfg["preprocessing"]["multi_channel_patches"]

    def get_N_max_patches(self):
        return self.cfg["preprocessing"]["N_max_patches"]

    def get_use_mask_as_channel(self):
        return self.cfg["preprocessing"]["use_mask_as_channel"]

    def get_use_mask_as_label(self):
        return self.cfg["preprocessing"]["use_mask_as_label"]

    def get_keep_rejected_patches(self):
        return self.cfg["preprocessing"]["keep_rejected_patches"]

    def get_mask_keyword(self):
        return self.cfg["preprocessing"]["mask_keyword"]

    def get_center_patch_around_mask(self):
        return self.cfg["preprocessing"]["center_patch_around_mask"]

    def get_rescale_mask_intensity(self):
        return self.cfg["preprocessing"]["rescale_mask_intensity"]

    def get_model_name(self):
        return self.cfg["model"]["architecture"]["model_name"]

    def get_model_settings(self):
        if "settings" in self.cfg["model"]["architecture"]:
            model_settings = self.cfg["model"]["architecture"]["settings"]
            if model_settings is None:
                model_settings = {}
        else:
            model_settings = {}
        return model_settings

    def get_cache_in_memory(self):
        if "cache_in_memory" in self.cfg["training"]:
            cache_in_memory = self.cfg["training"]["cache_in_memory"]
        else:
            cache_in_memory = False
        return cache_in_memory

    def get_float_policy(self):
        if "float_policy" in self.cfg["training"]:
            float_policy = self.cfg["training"]["float_policy"]
        else:
            float_policy = False
        return float_policy

    def get_gpu_workers(self):
        return self.cfg["training"]["N_workers"]

    def get_shuffle(self):
        return self.cfg["training"]["shuffle"]

    def get_data_augmentation(self):
        return self.cfg["training"]["data_augmentation"]

    def get_data_augmentation_factor(self):
        return self.cfg["training"]["augmentation_factor"]

    def get_batch_size(self):
        return self.cfg["model"]["architecture"]["batch_size"]

    def get_N_epoch(self):
        return self.cfg["model"]["architecture"]["N_epoch"]

    def get_N_classes(self):
        return self.cfg["model"]["architecture"]["N_output"]

    def get_make_one_hot(self):
        return self.cfg["training"]["make_one_hot"]

    def get_filter_missing(self):
        return self.cfg["training"]["filter_missing"]

    def get_dtype(self):
        return self.cfg["network"]["dtype"]

    def get_copy_files(self):
        return self.cfg["training"]["copy_files"]

    def get_test_model_file(self):
        return self.cfg["testing"]["model_file"]

    def get_max_steps_per_epoch(self):
        return self.cfg["training"]["max_steps_per_epoch"]

    def get_training_multi_processing(self):
        return self.cfg["training"]["multi_processing"]

    def get_float16_epsilon(self):
        if "float16_epsilon" in self.cfg["training"]:
            return float(self.cfg["training"]["float16_epsilon"])
        else:
            return 1e-4

    def get_save_name(self):
        time_string = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
        return time_string

    def get_image_size(self):
        if self.get_resample_images():
            return self.cfg["preprocessing"]["resample_image_size"]
        elif self.get_make_patches():
            return self.cfg["preprocessing"]["patch_size"]
        return None

    def get_size_string(self):
        size = self.get_image_size()
        if size is not None:
            size_string = [str(size[i]) + "x" for i in range(len(size))]
            size_string = "".join(size_string)
            size_string = size_string[0:-1]
        else:
            size_string = None
        return size_string

    def get_processed_samples_folder(self):
        base_dir = utils.get_parent_directory(self.get_data_folder())
        processed_samples_folder = os.path.join(base_dir, "NPZ_samples_" + self.get_size_string())

        return processed_samples_folder

    def get_output_folder(self):
        return self.cfg["general"]["output_folder"]

    def get_input_folder(self):
        return self.cfg["general"]["input_folder"]

    def get_cluster_setting(self):
        if "cluster_type" in self.cfg["general"]:
            return self.cfg["general"]["cluster_type"]
        else:
            return None

    def get_specific_output_folder(self):
        hasher = hashlib.sha512()
        with open(self.config_file, "rb") as afile:
            buf = afile.read()
            hasher.update(buf)
        hash_string = hasher.hexdigest()
        output_specifications = [
            self.get_model_name(),
            self.get_save_name(),
            hash_string,
        ]
        output_string = "_".join(output_specifications)
        # specific_output_folder = os.path.join(self.get_output_folder(), output_string)

        # utils.create_directory(specific_output_folder)

        return output_string

    def get_N_jobs(self):
        return self.cfg["cluster"]["N_jobs"]

    def get_cluster_type(self):
        return self.cfg["cluster"]["type"]

    def get_use_labels_from_rejection(self):
        return self.cfg["preprocessing"]["labels_from_rejection"]

    def get_use_class_weights(self):
        return self.cfg["training"]["use_class_weights"]

    def get_use_class_weights_in_losses(self):
        return self.cfg["training"]["use_class_weights_in_losses"]

    def get_optimizer_settings(self):
        return self.cfg["model"]["optimizer"]

    def get_loss_settings(self):
        return self.cfg["model"]["losses"]

    def get_metric_settings(self):
        return self.cfg["model"]["metrics"]

    def get_callback_settings(self):
        return self.cfg["model"]["callbacks"]

    def get_loss_weights(self):
        if "loss_weights" in self.cfg["model"]:
            loss_weights = self.cfg["model"]["loss_weights"]
        else:
            loss_weights = None
        return loss_weights

    def get_evaluation_metric_settings(self):
        return self.cfg["evaluation"]["metrics"]

    # ===============================================================
    # NEW
    # ===============================================================
    def get_preprocessings_settings(self):
        return self.cfg["preprocessing"]

    def get_evaluation_mask_labels(self):
        if "image_outputs" in self.cfg["evaluation"]:
            mask_labels = self.cfg["evaluation"]["image_outputs"]
            if mask_labels is None:
                mask_labels = []
        else:
            mask_labels = []

        return mask_labels

    def get_combine_patch_predictions(self):
        if "combine_patch_predictions" in self.cfg["evaluation"]:
            combine_patch_predictions = self.cfg["evaluation"]["combine_patch_predictions"]
        else:
            combine_patch_predictions = False

        return combine_patch_predictions

    def get_patch_predictions(self):
        if "patch_predictions" in self.cfg["evaluation"]:
            patch_predictions = self.cfg["evaluation"]["patch_predictions"]
        else:
            patch_predictions = True

        return patch_predictions

    def get_evaluate_train_set(self):
        if "evaluate_train_set" in self.cfg["evaluation"]:
            evaluate_train_set = self.cfg["evaluation"]["evaluate_train_set"]
        else:
            evaluate_train_set = True

        return evaluate_train_set

    def get_label_combination_type(self):
        if "combination_type" in self.cfg["evaluation"]:
            combination_type = self.cfg["evaluation"]["combination_type"]
        else:
            combination_type = None
        return combination_type

    def get_write_predictions(self):
        if "write_predictions" in self.cfg["evaluation"]:
            write_predictions = self.cfg["evaluation"]["write_predictions"]
        else:
            write_predictions = False
        return write_predictions

    def get_evaluate_metrics(self):
        if "evaluate_metrics" in self.cfg["evaluation"]:
            evaluate_metrics = self.cfg["evaluation"]["evaluate_metrics"]
        else:
            evaluate_metrics = True
        return evaluate_metrics

    def get_custom_definitions_file(self):
        file_found = False
        if "custom_definitions_file" in self.cfg["general"]:
            custom_definitions_file = self.cfg["general"]["custom_definitions_file"]
            if not os.path.exists(custom_definitions_file):
                custom_definitions_file = os.path.join(
                    self.cfg["original_location"], custom_definitions_file
                )

            if os.path.exists(custom_definitions_file):
                file_found = True

            if not file_found:
                raise ValueError("Custom definitions file was not found!")
        else:
            custom_definitions_file = None

        return custom_definitions_file

    def get_do_augmentation(self):
        if "data_augmentation" in self.cfg["training"]:
            return self.cfg["training"]["data_augmentation"]
        else:
            return False

    def get_data_augmentation_settings(self):
        if "augmentation_settings" in self.cfg["training"]:
            return self.cfg["training"]["augmentation_settings"]
        else:
            return {}

    def get_shuffle_val(self):
        if "shuffle_validation" in self.cfg["training"]:
            return self.cfg["training"]["shuffle_validation"]
        else:
            return False

    def get_class_weights(self):
        if "class_weights" in self.cfg["training"]:
            return self.cfg["training"]["class_weights"]
        else:
            return None

    def get_shuffle_evaluation(self):
        if "shuffle" in self.cfg["evaluation"]:
            shuffle_evaluation = self.cfg["evaluation"]["shuffle"]
        else:
            shuffle_evaluation = False

        return shuffle_evaluation

    def get_resume_training_from_model(self):
        if "resume_training" in self.cfg["training"]:
            return self.cfg["training"]["resume_training"]
        else:
            return False

    def get_model_file(self):
        if "model_file" in self.cfg["training"]:
            return self.cfg["training"]["model_file"]
        else:
            return None
