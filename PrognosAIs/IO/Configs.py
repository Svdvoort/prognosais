import copy

from typing import Tuple
from typing import Union

import numpy as np
import SimpleITK as sitk
import PrognosAIs.Constants


class config:
    def __init__(self, config_settings: Union[dict, None]):
        if config_settings is not None:
            config_settings = copy.deepcopy(config_settings)
            self.perform_step = True
            for i_config_name, i_config_value in config_settings.items():
                setattr(self, i_config_name, i_config_value)

        else:
            self.perform_step = False

    @staticmethod
    def get_step_type(config: Union[dict, None]) -> Tuple[bool, bool, dict]:
        if config is not None:
            if "type" in config:
                step_type = config["type"]
                if step_type == "image":
                    perform_step_on_image = True
                    perform_step_on_patch = False
                elif step_type == "patch":
                    perform_step_on_image = False
                    perform_step_on_patch = True
                elif step_type == "both":
                    perform_step_on_image = True
                    perform_step_on_patch = True
                else:
                    raise NotImplementedError(
                        "You have specified an unknown step type {}!".format(step_type)
                    )
            else:
                perform_step_on_image = True
                perform_step_on_patch = False
        else:
            perform_step_on_image = perform_step_on_patch = False

        return perform_step_on_image, perform_step_on_patch, config


class general_config(config):
    def __init__(self, config_settings: dict):
        config_settings = copy.deepcopy(config_settings)
        general_config = config_settings.pop("general", None)
        self.pipeline = [
            "multi_dimension_extracting",
            "bias_field_correcting",
            "masking",
            "resampling",
            "normalizing",
            "rejecting",
            "patching",
            "saving",
        ]

        self.mask_keyword = "mask"
        self.max_cpus = 999
        self.output_channel_names=[]
        super().__init__(general_config)


class multi_dimension_extracting_config(config):
    def __init__(self, config_settings: dict):
        self.extraction_type = None
        self.max_dimensions = -1
        self.perform_step_on_image = True
        self.perform_step_on_patch = False
        self.extract_masks = False
        self.apply_to_output = False

        super().__init__(config_settings)


class masking_config(config):
    def __init__(self, config_settings: dict):
        self.mask_background = False
        self.crop_to_mask = False
        self.background_value = 0.0
        self.process_masks = True
        self.apply_to_output = False
        self._mask_file = None
        self._mask = None

        (
            self.perform_step_on_image,
            self.perform_step_on_patch,
            config_settings,
        ) = super().get_step_type(config_settings)
        super().__init__(config_settings)

    @property
    def mask_file(self):
        return self._mask_file

    @mask_file.setter
    def mask_file(self, mask_file: str):
        self._mask_file = mask_file
        self.mask = self._mask_file

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, mask_file: str):
        mask = sitk.ReadImage(mask_file, sitk.sitkUInt8)
        mask = sitk.RescaleIntensity(mask, 0, 1)
        self._mask = mask


class resampling_config(config):
    def __init__(self, config_settings: dict):
        self.resample_size = [0, 0, 0]
        self.apply_to_output = False
        (
            self.perform_step_on_image,
            self.perform_step_on_patch,
            config_settings,
        ) = super().get_step_type(config_settings)

        super().__init__(config_settings)


class normalizing_config(config):
    def __init__(self, config_settings: dict):
        self._mask = None
        self._mask_file = None
        self.normalization_range = [0, 100]
        self.output_range = None
        self.mask_normalization = None
        self.normalization_method = None
        self.mask_smoothing = False
        self.apply_to_output = False

        (
            self.perform_step_on_image,
            self.perform_step_on_patch,
            config_settings,
        ) = super().get_step_type(config_settings)

        super().__init__(config_settings)

    @property
    def mask_file(self):
        return self._mask_file

    @mask_file.setter
    def mask_file(self, mask_file: str):
        self._mask_file = mask_file
        self.mask = self._mask_file

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, mask_file: str):
        mask = sitk.ReadImage(mask_file, sitk.sitkUInt8)
        mask = sitk.RescaleIntensity(mask, 0, 1)
        self._mask = mask


class bias_field_correcting_config(config):
    def __init__(self, config_settings: dict):
        self.apply_to_output = False
        self._mask_file = None
        self._mask = None

        (
            self.perform_step_on_image,
            self.perform_step_on_patch,
            config_settings,
        ) = super().get_step_type(config_settings)

        super().__init__(config_settings)

    @property
    def mask_file(self):
        return self._mask_file

    @mask_file.setter
    def mask_file(self, mask_file: str):
        self._mask_file = mask_file
        self.mask = self._mask_file

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, mask_file: str):
        mask = sitk.ReadImage(mask_file, sitk.sitkUInt8)
        mask = sitk.RescaleIntensity(mask, 0, 1)
        self._mask = mask


class patching_config(config):
    def __init__(self, config_settings: dict):
        self._patch_size = np.asarray([0, 0, 0])
        self.pad_if_needed = False
        self.pad_constant = 0.0
        self.extraction_type = "fitting"
        self.max_number_of_patches = -1
        self.overlap_fraction = 0.5

        self.perform_step_on_image = True
        self.perform_step_on_patch = False

        self.apply_to_output = False

        super().__init__(config_settings)

    @property
    def patch_size(self):
        return self._patch_size

    @patch_size.setter
    def patch_size(self, patch_size: list):
        self._patch_size = np.asarray(patch_size)


class rejecting_config(config):
    def __init__(self, config_settings: dict):
        self.rejection_limit = 0
        self.rejection_as_label = False
        self.apply_to_output = False
        self._mask_file = None
        self._mask = None
        (
            self.perform_step_on_image,
            self.perform_step_on_patch,
            config_settings,
        ) = super().get_step_type(config_settings)

        super().__init__(config_settings)

    @property
    def mask_file(self):
        return self._mask_file

    @mask_file.setter
    def mask_file(self, mask_file: str):
        self._mask_file = mask_file
        self.mask = self._mask_file

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, mask_file: str):
        mask = sitk.ReadImage(mask_file, sitk.sitkUInt8)
        mask = sitk.RescaleIntensity(mask, 0, 1)
        self._mask = mask


class saving_config(config):
    def __init__(self, config_settings: dict):
        self.use_mask_as_channel = False
        self.use_mask_as_label = False
        self.out_dir_name = "Samples"
        self.sample_npz_keyword = PrognosAIs.Constants.FEATURE_INDEX
        self.label_npz_keyword = PrognosAIs.Constants.LABEL_INDEX
        self.default_keyword = "default"
        self.named_channels = False
        self.impute_missing_channels = False
        self.save_as_float16 = False
        self.float16_percentage_diff = 0.1
        self.combine_labels = False
        self.channel_names = None
        self.mask_channels = 0
        (
            self.perform_step_on_image,
            self.perform_step_on_patch,
            config_settings,
        ) = super().get_step_type(config_settings)

        super().__init__(config_settings)
        self.perform_step = True
        if not self.perform_step_on_image and not self.perform_step_on_patch:
            self.perform_step_on_patch = True

class labeling_config(config):
    def __init__(self, config_settings: dict):
        config_settings = copy.deepcopy(config_settings)
        labeling_config = config_settings.pop("labeling", None)

        self.label_file = None
        self.train_fraction = 1.0
        self.validation_fraction = 0.0
        self.test_fraction = 0.0
        self.stratify_label_name = None
        self.filter_missing = False
        self.missing_value = -1
        self.make_one_hot = False

        super().__init__(labeling_config)
