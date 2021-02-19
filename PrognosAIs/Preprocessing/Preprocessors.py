import argparse
import copy
import itertools
import json
import os
import shutil
import sys

from multiprocessing import Pool
from typing import Tuple
from typing import Union

import h5py
import numpy as np
import PrognosAIs.IO.ConfigLoader
import PrognosAIs.IO.Configs
import PrognosAIs.IO.LabelParser
import PrognosAIs.IO.utils as IO_utils
import SimpleITK as sitk
import sklearn.model_selection

from PrognosAIs.Preprocessing import Samples
from PrognosAIs.Preprocessing.Samples import ImageSample


class SingleSamplePreprocessor:
    def __init__(
        self, sample: ImageSample, config: dict, output_directory: str = None,
    ):
        # We make sure we dont change to the original objects
        self.sample = sample.copy()
        self.config = copy.deepcopy(config)
        self.output_directory = output_directory
        self.save_names = None

        self.general_config = PrognosAIs.IO.Configs.general_config(self.config)

        self._HDF5_EXTENSION = ".hdf5"
        self._PATCH_SEPARATOR = "_patch_"

        self._init_configs()

    def _init_configs(self):
        for i_possible_step in self.general_config.pipeline:
            config_name = i_possible_step + "_config"
            config_class = getattr(PrognosAIs.IO.Configs, config_name)

            if i_possible_step in self.config:
                setattr(self, config_name, config_class(self.config[i_possible_step]))
            else:
                setattr(self, config_name, config_class(None))

    def build_pipeline(self) -> list:
        # TODO add potential extra inputs
        # TODO add reorienting to standard space
        # TODO registration of different channels in sample
        # TODO RGB rejecting/to gray

        pipeline_image = []
        pipeline_patches = []

        for i_possible_step in self.general_config.pipeline:
            i_step_config = getattr(self, i_possible_step + "_config")
            if i_step_config.perform_step and i_step_config.perform_step_on_image:
                pipeline_image.append(getattr(self, i_possible_step))
            elif i_step_config.perform_step and i_step_config.perform_step_on_patch:
                pipeline_patches.append(getattr(self, i_possible_step))

        pipeline = pipeline_image + pipeline_patches

        return pipeline

    def apply_pipeline(self, pipeline=None):
        if pipeline is None:
            pipeline = self.build_pipeline()

        for i_step in pipeline:
            success = i_step()
            if success is False:
                break

    # ===============================================================
    # Image dimension extraction
    # ===============================================================
    def multi_dimension_extracting(self):
        """
        Extract invidiual images from a multi-dimensional sequence.

        Raises:
            NotImplementedError: If an extraction type is requested that is not supported.
        """
        if (
            self.multi_dimension_extracting_config.max_number_of_dimensions
            < self.sample.number_of_dimensions
        ):
            max_dims = self.multi_dimension_extracting_config.max_number_of_dimensions
            if self.multi_dimension_extracting_config.extraction_type == "first":
                extraction_fuction = self._get_first_image_from_sequence
            elif self.multi_dimension_extracting_config.extraction_type == "all":
                extraction_fuction = self._get_all_images_from_sequence
            else:
                raise NotImplementedError(
                    "Unknown extraction type {}".format(
                        self.multi_dimension_extracting_config.extraction_type
                    )
                )

            self.sample.channels = (extraction_fuction, [max_dims])
            if self.multi_dimension_extracting_config.extract_masks:
                self.sample.masks = (extraction_fuction, [max_dims])

            if self.multi_dimension_extracting_config.apply_to_output:
                self.sample.output_channels = (extraction_fuction, [max_dims])

    @staticmethod
    def _get_first_image_from_sequence(image: sitk.Image, max_dims: int) -> sitk.Image:
        """
        Extract the first image from a sequence of images

        Args:
            image (sitk.Image): Multi-dimensional image containg the sequence.
            max_dims (int): The maximum number of dimension the output can be.

        Returns:
            sitk.Image: The first image extracted from the sequence
        """

        image_size = list(image.GetSize())
        image_dims = len(image_size)
        to_cut_dim = image_dims - max_dims

        image_size[max_dims:] = [0] * to_cut_dim
        image = sitk.Extract(image, size=image_size, index=[0] * image_dims)

        return image

    @staticmethod
    def _get_all_images_from_sequence(image: sitk.Image, max_dims: int) -> list:
        """
        Get all of the images from a sequence of images.

        Args:
            image (sitk.Image): Multi-dimensional image containg the sequence.
            max_dims (int): The number of dimension of each individual image.
             This should be equal to the dimensionality of the input image - 1.
             Otherwise, we do not know how to extract the appropiate images

        Raises:
            ValueError: If the maximum number of dimensions does not fit with the sequences.

        Returns:
            list: All images extracted from the sequence.
        """

        image_size = list(image.GetSize())
        image_dims = len(image_size)

        if max_dims + 1 != image_dims:
            err_msg = """When extracting all dimensions of an image, the image can only have one
            more dimension than the max dimension.
            Image had {} dimensions, but max dimensions was set to {}."""
            raise ValueError(err_msg.format(image_dims, max_dims))

        image_extractor = sitk.ExtractImageFilter()
        N_patches = image_size[-1]
        patches = []
        image_size[-1] = 0
        image_extractor.SetSize(image_size)
        for i_patch in range(N_patches):
            image_extractor.SetIndex([0] * max_dims + [i_patch])
            patches.append(image_extractor.Execute(image))

        return patches

    # ===============================================================
    # Masking functions
    # ===============================================================

    def masking(self):
        if self.masking_config.mask_background:
            self.mask_background(
                self.masking_config.mask,
                self.masking_config.background_value,
                self.masking_config.process_masks,
                self.masking_config.apply_to_output
            )
        if self.masking_config.crop_to_mask:
            self.crop_to_mask(self.masking_config.mask, self.masking_config.process_masks, self.masking_config.apply_to_output)

    def mask_background(
        self, ROI_mask: sitk.Image, background_value: float = 0.0, process_masks: bool = True,
        apply_to_output: bool = False
    ):
        mask_image_filter = sitk.MaskImageFilter()

        mask_image_filter.SetMaskingValue(0)
        if background_value == "min":
            self.sample.channels = (self.mask_background_to_min, [ROI_mask])
            if apply_to_output:
                self.sample.output_channels = (self.mask_background_to_min, [ROI_mask])

        else:
            mask_image_filter.SetOutsideValue(background_value)
            self.sample.channels = (mask_image_filter.Execute, [ROI_mask])
            if apply_to_output:
                self.sample.output_channels = (mask_image_filter.Execute, [ROI_mask])

        if process_masks:
            # background_dtype = ImageSample.get_appropiate_dtype_from_scalar(background_value)
            # if background_dtype != self.sample.get_example_mask().GetPixelID():
            #     common_type = ImageSample.promote_simpleitk_types(
            #         background_dtype, self.sample.get_example_mask().GetPixelID()
            #     )
            #     self.sample.masks = (sitk.Cast, [common_type])
            # TODO fix here as well, we set 0 automatically, but perhaps need to fix this
            mask_image_filter.SetOutsideValue(0.0)
            self.sample.masks = (mask_image_filter.Execute, [ROI_mask])

    @staticmethod
    def mask_background_to_min(image, mask):
        mask_label_filter = sitk.LabelIntensityStatisticsImageFilter()
        mask_label_filter.Execute(mask, image)

        img_min = mask_label_filter.GetMinimum(1)

        image = sitk.Mask(image, mask, img_min)
        return image

    def crop_to_mask(self, ROI_mask: sitk.Image, process_masks: bool = True, apply_to_output: bool = False):
        statics_image_filter = sitk.LabelShapeStatisticsImageFilter()
        statics_image_filter.Execute(ROI_mask)

        mask_bounding_box = statics_image_filter.GetBoundingBox(1)
        N_dimensions = int(len(mask_bounding_box) / 2)
        bounding_box_index = mask_bounding_box[0:N_dimensions]
        bounding_box_size = mask_bounding_box[N_dimensions:]

        self.sample.channels = (
            sitk.RegionOfInterest,
            {"index": bounding_box_index, "size": bounding_box_size},
        )
        if process_masks:
            self.sample.masks = (
                sitk.RegionOfInterest,
                {"index": bounding_box_index, "size": bounding_box_size},
            )

    # ===============================================================
    # Resampling functions
    # ===============================================================
    def resampling(self):
        channel_resampler = sitk.ResampleImageFilter()
        channel_resampler.SetInterpolator(sitk.sitkBSpline)

        mask_resampler = sitk.ResampleImageFilter()
        mask_resampler.SetInterpolator(sitk.sitkNearestNeighbor)

        self.sample.channels = (
            self._resample,
            [self.resampling_config.resample_size, channel_resampler],
        )
        self.sample.masks = (
            self._resample,
            [self.resampling_config.resample_size, mask_resampler],
        )

        if self.resampling_config.apply_to_output:
            self.sample.output_channels = (
                self._resample,
                [self.resampling_config.resample_size, channel_resampler],
            )

    @staticmethod
    def _resample(image, resample_size, resampler):
        original_size = np.asarray(image.GetSize())
        original_spacing = np.asarray(image.GetSpacing())
        resample_size = np.asarray(resample_size)

        new_spacing = original_size * original_spacing / resample_size

        resampler.SetOutputSpacing(new_spacing.tolist())
        resampler.SetSize(resample_size.tolist())

        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetOutputPixelType(image.GetPixelID())
        resampler.SetTransform(sitk.Transform())

        image = resampler.Execute(image)
        return image

    # ===============================================================
    # Normalizing functions
    # ===============================================================
    def normalizing(self):
        if (
            self.normalizing_config.normalization_method == "range"
            and self.normalizing_config.mask is None
        ):
            self.sample.channels = (
                self._rescale_image_intensity_range,
                [
                    self.normalizing_config.normalization_range,
                    self.normalizing_config.output_range,
                ],
            )
            if self.normalizing_config.apply_to_output:
                self.sample.output_channels = (
                    self._rescale_image_intensity_range,
                    [
                        self.normalizing_config.normalization_range,
                        self.normalizing_config.output_range,
                    ],
                )

        elif (
            self.normalizing_config.normalization_method == "range"
            and self.normalizing_config.mask is not None
        ):
            self.sample.channels = (
                self._rescale_image_intensity_range_with_mask,
                [
                    self.normalizing_config.mask,
                    self.normalizing_config.normalization_range,
                    self.normalizing_config.output_range,
                ],
            )
            if self.normalizing_config.apply_to_output:
                self.sample.output_channels = (
                    self._rescale_image_intensity_range_with_mask,
                    [
                        self.normalizing_config.mask,
                        self.normalizing_config.normalization_range,
                        self.normalizing_config.output_range,
                    ],
            )

        elif (
            self.normalizing_config.normalization_method == "zscore"
            and self.normalizing_config.mask is None
        ):
            self.sample.channels = self._zscore_image_intensity
            if self.normalizing_config.apply_to_output:
                self.sample.output_channels = self._zscore_image_intensity
        elif (
            self.normalizing_config.normalization_method == "zscore"
            and self.normalizing_config.mask is not None
        ):
            self.sample.channels = (
                self._zscore_image_intensity_with_mask,
                [self.normalizing_config.mask],
            )
            if self.normalizing_config.apply_to_output:
                self.sample.output_channels = (
                    self._zscore_image_intensity_with_mask,
                    [self.normalizing_config.mask],
                )

        if self.normalizing_config.mask_normalization == "collapse":
            self.sample.masks = self._collapse_mask
        elif self.normalizing_config.mask_normalization == "consecutively":
            self.sample.masks = self._make_consecutive_mask

        if self.normalizing_config.mask_smoothing:
            self.sample.masks = self._smooth_mask

    @staticmethod
    def _rescale_image_intensity_range(
        image: sitk.Image, percentile_range: list, output_range: list = None
    ) -> sitk.Image:
        image_array = sitk.GetArrayViewFromImage(image)

        low_intensity = np.percentile(image_array, percentile_range[0])
        high_intensity = np.percentile(image_array, percentile_range[1])
        if np.isclose(low_intensity, high_intensity):
            raise ValueError(
                """Percentiles are too close, or image intensity is
         too imbalanced, cannot normalize"""
            )

        image = sitk.IntensityWindowing(
            image, low_intensity, high_intensity, low_intensity, high_intensity
        )
        if output_range is not None:
            image = sitk.RescaleIntensity(image, output_range[0], output_range[1])
        return image

    @staticmethod
    def _rescale_image_intensity_range_with_mask(
        image: sitk.Image, mask: sitk.Image, percentile_range: list, output_range: list = None,
    ) -> sitk.Image:

        image_array = sitk.GetArrayViewFromImage(image)
        mask_array = sitk.GetArrayViewFromImage(mask)

        masked_values = image_array[mask_array > 0].flatten()

        low_intensity = np.percentile(masked_values, percentile_range[0])
        high_intensity = np.percentile(masked_values, percentile_range[1])

        if np.isclose(low_intensity, high_intensity):
            raise ValueError(
                """Percentiles are too close, or image intensity is
         too imbalanced, cannot normalize"""
            )

        image = sitk.IntensityWindowing(
            image, low_intensity, high_intensity, low_intensity, high_intensity
        )

        if output_range is not None:
            image = sitk.RescaleIntensity(image, output_range[0], output_range[1])
        return image

    @staticmethod
    def _zscore_image_intensity(image: sitk.Image) -> sitk.Image:
        return sitk.Normalize(image)

    @staticmethod
    def _zscore_image_intensity_with_mask(image: sitk.Image, mask: sitk.Image) -> sitk.Image:
        mask_label_filter = sitk.LabelIntensityStatisticsImageFilter()
        mask_label_filter.Execute(mask, image)

        img_mean = mask_label_filter.GetMean(1)
        img_std = mask_label_filter.GetStandardDeviation(1)

        image = sitk.ShiftScale(image, -1.0 * img_mean, 1.0 / img_std)
        return image

    @staticmethod
    def _make_mask_positive(mask: sitk.Image) -> sitk.Image:
        original_mask = sitk.Image(mask)
        # Make the mask positive, but keep 0 as 0
        minmax_filter = sitk.MinimumMaximumImageFilter()
        minmax_filter.Execute(mask)
        # We make sure that everything is positive
        mask_min = minmax_filter.GetMinimum()
        if mask_min < 0:
            # First we make sure that we dont lose any data in the new data type
            new_max = minmax_filter.GetMaximum() - mask_min
            new_data_type = ImageSample.get_appropiate_dtype_from_scalar(new_max)
            mask = sitk.Subtract(mask, mask_min - 1)
            # Multiply here so what was originally 0 is still 0
            mask = sitk.Multiply(
                mask, sitk.Cast(sitk.NotEqual(original_mask, 0), mask.GetPixelID())
            )
            mask = sitk.Cast(mask, new_data_type)

        return mask

    @staticmethod
    def _collapse_mask(mask: sitk.Image) -> sitk.Image:
        mask = SingleSamplePreprocessor._make_mask_positive(mask)
        mask = sitk.LabelImageToLabelMap(mask)
        mask = sitk.AggregateLabelMap(mask)
        mask = sitk.RelabelLabelMap(mask)
        mask = sitk.LabelMapToLabel(mask)
        # Since we collapsed, it for sure fits into a uint8
        # As there are only two values, so we cast here to save memory
        mask = sitk.Cast(mask, sitk.sitkUInt8)

        return mask

    @staticmethod
    def _smooth_mask(mask: sitk.Image) -> sitk.Image:
        mask = sitk.BinaryMedian(mask, [3, 3, 3])
        return mask

    @staticmethod
    def _make_consecutive_mask(mask: sitk.Image) -> sitk.Image:
        mask = SingleSamplePreprocessor._make_mask_positive(mask)
        mask = sitk.LabelImageToLabelMap(mask)
        mask = sitk.RelabelLabelMap(mask)
        mask = sitk.LabelMapToLabel(mask)

        return mask

    # ===============================================================
    # Padding
    # ===============================================================
    @staticmethod
    def _pad_image_to_size(
        image: sitk.Image, output_size: list, pad_constant: float = 0.0
    ) -> Tuple[sitk.Image, np.ndarray, np.ndarray]:
        image_size = np.asarray(image.GetSize())
        output_size = np.asarray(output_size)
        if any(image_size < output_size):
            required_padding = np.maximum(output_size - image_size, 0)
            left_padding = np.floor(required_padding / 2.0) + np.mod(required_padding, 2.0)
            right_padding = np.floor(required_padding / 2.0)
            image = SingleSamplePreprocessor._pad_image_from_parameters(
                image, left_padding, right_padding, pad_constant
            )
        else:
            left_padding = np.zeros(len(image_size))
            right_padding = np.zeros_like(left_padding)
        return image, left_padding, right_padding

    @staticmethod
    def _pad_image_from_parameters(
        image: sitk.Image, left_padding: np.ndarray, right_padding: np.ndarray, pad_constant: float,
    ) -> sitk.Image:

        left_padding = left_padding.astype(np.int).tolist()
        right_padding = right_padding.astype(np.int).tolist()
        image = sitk.ConstantPad(image, left_padding, right_padding, pad_constant)
        return image

    # ===============================================================
    # Patching
    # ===============================================================
    def patching(self) -> None:
        patch_parameters = self._get_patch_parameters()

        self.sample.channels = (
            self._make_patches,
            [patch_parameters, self.patching_config.pad_constant, self.patching_config.patch_size],
        )
        # TODO this pad constant is 0, because that makes sense for
        # Masks but perhaps lets let users set it themselves
        self.sample.masks = (
            self._make_patches,
            [patch_parameters, 0, self.patching_config.patch_size],
        )

        if self.patching_config.apply_to_output:
            self.sample.output_channels = (
                self._make_patches,
                [patch_parameters, self.patching_config.pad_constant, self.patching_config.patch_size],
            )

    def _get_patch_parameters(self) -> dict:
        patch_parameters = {}
        patch_parameters["left_padding"] = np.zeros(self.sample.number_of_dimensions)
        patch_parameters["right_padding"] = np.zeros(self.sample.number_of_dimensions)
        patch_parameters["patch_indices"] = None

        # Make sure that all samples are the same size, otherwise the patches
        # wont make sense
        self.sample.assert_all_channels_same_size()
        self.sample.assert_all_masks_same_size()

        example_sample = self.sample.get_example_channel()

        if self.patching_config.pad_if_needed:
            example_sample, left_padding, right_padding = self._pad_image_to_size(
                example_sample, self.patching_config.patch_size, self.patching_config.pad_constant,
            )
            patch_parameters["left_padding"] += left_padding
            patch_parameters["right_padding"] += right_padding

        elif any(example_sample.GetSize() < self.patching_config.patch_size):
            raise ValueError(
                """The sample is smaller than the patch and padding not requested,
                 cannot make patches."""
            )


        if self.patching_config.extraction_type not in ["random", "fitting", "overlap"]:
            raise NotImplementedError(
                "The specified extraction type {} is not specified!".format(
                    self.patching_config.extraction_type
                )
            )

        if self.patching_config.extraction_type == "random":
            patch_parameters["patch_indices"] = self._get_random_patching_parameters(
                self.patching_config.patch_size,
                self.patching_config.max_number_of_patches,
                example_sample,
            )
        elif self.patching_config.extraction_type == "fitting":
            patch_parameters["patch_indices"] = self._get_fitting_patching_parameters(
                self.patching_config.patch_size, example_sample
            )
        elif self.patching_config.extraction_type == "overlap":
            (patch_indices, left_padding, right_padding,) = self._get_overlap_patching_parameters(
                self.patching_config.patch_size,
                self.patching_config.overlap_fraction,
                self.patching_config.pad_constant,
                example_sample,
            )
            patch_parameters["patch_indices"] = patch_indices
            patch_parameters["left_padding"] += left_padding
            patch_parameters["right_padding"] += right_padding

        return patch_parameters

    @staticmethod
    def _make_patches(
        image: sitk.Image, patch_parameters: dict, pad_constant: float, patch_size: np.ndarray,
    ) -> list:
        image = SingleSamplePreprocessor._pad_image_from_parameters(
            image,
            patch_parameters["left_padding"],
            patch_parameters["right_padding"],
            pad_constant,
        )

        patches = []
        patch_size = patch_size.tolist()
        patch_filter = sitk.RegionOfInterestImageFilter()
        patch_filter.SetSize(patch_size)

        for i_patch_index in patch_parameters["patch_indices"]:
            patch_filter.SetIndex(i_patch_index.tolist())
            cur_patch = patch_filter.Execute(image)
            patches.append(cur_patch)
        return patches

    @staticmethod
    def _get_random_patching_parameters(
        patch_size: np.ndarray, max_number_of_patches: int, example_sample: ImageSample
    ) -> np.ndarray:
        if max_number_of_patches > 0:
            N_patches = max_number_of_patches
            max_patch_index = example_sample.GetSize() - patch_size
            patch_indices = np.zeros((N_patches, len(max_patch_index)))

            for i_i_max_patch_index, i_max_patch_index in enumerate(max_patch_index):
                patch_indices[:, i_i_max_patch_index] = np.random.randint(
                    0, i_max_patch_index, size=N_patches
                )
            patch_indices = np.floor(patch_indices).astype(np.int)
        else:
            raise ValueError("If extraction is random, number of patches should be specified!")

        return patch_indices

    @staticmethod
    def _get_fitting_patching_parameters(
        patch_size: np.ndarray, example_sample: sitk.Image
    ) -> np.ndarray:
        # sample_size = sample_array.shape
        sample_size = np.asarray(example_sample.GetSize())

        patches_per_dim = np.floor(sample_size / patch_size)
        per_dim_patch_indice_number = []
        for i_patches_per_dim in patches_per_dim:
            per_dim_patch_indice_number.append(range(int(i_patches_per_dim)))
        patch_indice_numbers = np.asarray(list(itertools.product(*per_dim_patch_indice_number)))
        # Spacing is determined by the patch size, and the possible missed voxels
        # because patches dont fit perfectly
        between_patch_spacing = patch_size + np.mod(sample_size, patch_size)/(patches_per_dim -1)
        patch_indices = patch_indice_numbers * between_patch_spacing
        patch_indices = np.floor(patch_indices).astype(np.int)
        return patch_indices

    @staticmethod
    def _get_overlap_patching_parameters(
        patch_size: np.ndarray,
        overlap_fraction: Union[int, list],
        pad_constant: float,
        example_sample: sitk.Image,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        if not isinstance(overlap_fraction, list):
            overlap_fraction = [overlap_fraction] * example_sample.GetDimension()

        # Need to flip to convert between sitk and numpy coordinates
        image_size = np.asarray(example_sample.GetSize())

        overlap_fraction = np.asarray(overlap_fraction)
        if 0 <= overlap_fraction[0] < 1:
            stride_size = (patch_size - np.ceil(overlap_fraction * patch_size)).astype(np.int)
        else:
            stride_size = (patch_size - overlap_fraction).astype(np.int)

        # Need to calculate the number of stride steps we can take
        # This formula is base so that the full image is covered as muc has possible
        # and as equally as possible, with possible padding as well
        # The total size extracted with the patches = (patch_size + (N-1)*stride_size)
        # With N the number of patches
        # We need to make sure that this is larger than the total image size to ensure
        # that the whole image is cover, hence we take one extra patch (+2 instead of +1)
        # and floor in case the patches dont fit perfectly because we already take an extra one.
        # If we would have used ceil and +1 this would not work for cases where the patches fit
        # perfectly in the image.
        N_stride_steps = np.floor((image_size - patch_size) / stride_size) + 2
        N_stride_steps = N_stride_steps.astype(np.int)

        # We pad so that the steps fit nicely
        required_size = (N_stride_steps - 1) * stride_size + patch_size
        (
            example_sample,
            left_padding,
            right_padding,
        ) = SingleSamplePreprocessor._pad_image_to_size(example_sample, required_size, pad_constant)

        stride_steps_per_dim = []
        for i_i_N_stride_steps, i_N_stride_steps in enumerate(N_stride_steps):
            stride_steps_per_dim.append(range(i_N_stride_steps))

        stride_steps = np.asarray(list(itertools.product(*stride_steps_per_dim)))
        patch_indices = stride_steps * stride_size
        # Need to invert to correspond with the simpleitk indices
        patch_indices = patch_indices.astype(np.int)

        return patch_indices, left_padding, right_padding

    # ===============================================================
    # Rejecting
    # ===============================================================

    def rejecting(self):
        if not self.sample.has_masks:
            raise ValueError("Sample does not have masks, cannot reject patches!")
        rejection_status = self._get_to_reject_patches(
            self.sample.get_example_mask_patches(), self.rejecting_config.rejection_limit,
        )
        if self.rejecting_config.rejection_as_label:
            accepted_status = np.logical_not(rejection_status)
            accepted_status = accepted_status.astype(np.uint8)
            if self.sample.are_labels_one_hot:
                accepted_status = np.eye(2)[accepted_status].astype(np.uint8)
            accepted_labels = [{"accepted": i_status} for i_status in accepted_status]
            self.sample.add_to_labels(accepted_labels, {"accepted": 2})
        else:
            self.sample.channels = (self._get_accepted_patches, [rejection_status])
            self.sample.masks = (self._get_accepted_patches, [rejection_status])
            if self.rejecting_config.apply_to_output:
                self.sample.output_channels = (self._get_accepted_patches, [rejection_status])

            return self.sample.number_of_patches > 0

    @staticmethod
    def _get_to_reject_patches(mask: Union[sitk.Image, list], rejection_limit: float) -> list:
        if isinstance(mask, sitk.Image):
            mask = [mask]

        rejection_limit = rejection_limit * np.prod(mask[0].GetSize())

        rejection_status = [
            np.count_nonzero(sitk.GetArrayViewFromImage(i_mask_patch)) < rejection_limit
            for i_mask_patch in mask
        ]

        return rejection_status

    @staticmethod
    def _get_accepted_patches(patches: Union[sitk.Image, list], rejection_status: list) -> list:
        accepted_patches = []

        if isinstance(patches, sitk.Image):
            patches = [patches]

        for i_patch, is_rejected in zip(patches, rejection_status):
            if not is_rejected:
                accepted_patches.append(i_patch)
        return accepted_patches

    # ===============================================================
    # Bias field correcting
    # ===============================================================
    def bias_field_correcting(self):
        bias_field_corrector = sitk.N4BiasFieldCorrectionImageFilter()
        if self.bias_field_correcting_config.mask is not None:
            bias_field_corrector.SetUseMaskLabel(True)
            args = [self.bias_field_correcting_config.mask]
        else:
            bias_field_corrector.SetUseMaskLabel(False)
            args = []
        self.sample.channels = (bias_field_corrector.Execute, args)
        if self.bias_field_correcting_config.apply_to_output:
            self.sample.output_channels = (bias_field_corrector.Execute, args)

    # ===============================================================
    # Saving
    # ===============================================================

    @staticmethod
    def _convert_sitk_arrays_to_numpy(images: list):
        N_images = len(images)
        if N_images > 0:
            image_size = images[0].GetSize()
            dtypes = []
            for i_image in images:
                dtypes.append(i_image.GetPixelID())
            # Simpleitk dtypes are ints in increasing order
            # Thus we can get the max and it will it be appropiate everything
            sitk_dtype = np.max(dtypes)
            np_dtype = ImageSample.get_numpy_type_from_sitk_type(sitk_dtype)
            np_array = np.empty((*image_size, N_images), dtype=np_dtype)
            for i_i_image, i_image in enumerate(images):
                np_array[..., i_i_image] = np.transpose(sitk.GetArrayFromImage(i_image))
        else:
            np_array = None
        return np_array

    def _patch_to_data_structure(
        self, patch_channels: list, patch_output_channels: list, patch_masks: list, patch_labels: list
    ) -> dict:
        N_channels = len(patch_channels)
        patch_channels = self._convert_sitk_arrays_to_numpy(patch_channels)
        if patch_masks is not None:
            N_masks = len(patch_masks)
            patch_masks = self._convert_sitk_arrays_to_numpy(patch_masks)
        else:
            N_masks = 0

        if patch_output_channels is not None:
            N_output_channels = len(patch_output_channels)
            patch_output_channels = self._convert_sitk_arrays_to_numpy(patch_output_channels)
        else:
            N_output_channels = 0

        if self.saving_config.impute_missing_channels:
            patch_channels = self.channel_imputation(patch_channels)

        if self.saving_config.save_as_float16:
            patch_channels = self.channels_to_float16(patch_channels)
            if N_output_channels > 0:
                patch_output_channels = self.channels_to_float16(patch_output_channels)

        if self.saving_config.use_mask_as_channel and patch_masks is not None:
            patch_names = self.sample.channel_names + self.sample.mask_names
            patches = np.concatenate((patch_channels, patch_masks), axis=-1)
            N_patches = N_channels + N_masks
        else:
            patch_names = self.sample.channel_names
            patches = patch_channels
            N_patches = N_channels

        data_structure = {
            self.saving_config.sample_npz_keyword: {},
            self.saving_config.label_npz_keyword: {},
        }
        if self.saving_config.named_channels:
            data_structure[self.saving_config.sample_npz_keyword] = dict(
                zip(patch_names, np.split(patches, N_patches, axis=-1))
            )
        else:
            data_structure[self.saving_config.sample_npz_keyword] = {
                self.saving_config.sample_npz_keyword: patches
            }

        if self.saving_config.use_mask_as_label:
            if not self.sample.has_masks:
                raise ValueError(
                    "You request to use masks as labels, but no masks were found in the sample!"
                )
            if self.sample.are_labels_one_hot:
                patch_masks = np.squeeze(patch_masks)

                # for i_i_patch_mask, i_patch_mask in enumerate(patch_masks):
                patch_masks = np.eye(self.saving_config.mask_channels)[patch_masks].astype(np.uint8)
            if self.saving_config.named_channels:
                data_structure[self.saving_config.label_npz_keyword] = dict(
                    zip(self.sample.mask_names, np.split(patch_masks, N_masks, axis=-1),)
                )
            elif len(patch_labels) > 0 or N_output_channels > 0:
                # If we have other labels as well we ensure that we
                # Give a name to the mask, as we have multi-outputs
                data_structure[self.saving_config.label_npz_keyword] = {
                    self.sample.mask_keyword: patch_masks
                }

            else:
                data_structure[self.saving_config.label_npz_keyword] = {
                    self.saving_config.label_npz_keyword: patch_masks
                }

        if len(patch_labels) > 0:

            if self.sample.are_labels_one_hot:
                for i_key, i_value in patch_labels.items():
                    patch_labels[i_key] = np.asarray(i_value).astype(np.int8)
            if self.saving_config.combine_labels:
                label_keys = self.saving_config.label_npz_keyword
                labels = [
                    value for key, value in sorted(patch_labels.items(), key=lambda item: item[0])
                ]
                patch_labels = {self.saving_config.label_npz_keyword: np.asarray(labels)}
            else:
                label_keys = list(patch_labels.keys())
            if len(label_keys) > 1 or self.saving_config.use_mask_as_label:
                data_structure[self.saving_config.label_npz_keyword].update(patch_labels)
            else:
                data_structure[self.saving_config.label_npz_keyword] = {
                    self.saving_config.label_npz_keyword: patch_labels[label_keys[0]]
                }

        if N_output_channels > 0:
            output_channel_structure = dict(
                zip(self.sample.output_channel_names, np.split(patch_output_channels, N_output_channels, axis=-1),)
            )
            if len(patch_labels) > 0 or self.saving_config.use_mask_as_label:
                # If we have other labels as well we ensure that we
                # Give a name to the mask, as we have multi-outputs
                data_structure[self.saving_config.label_npz_keyword].update(output_channel_structure)
            else:
                data_structure[self.saving_config.label_npz_keyword] = output_channel_structure


        return data_structure

    def _get_number_of_classes(self, data_structure: dict):
        labels = data_structure[self.saving_config.label_npz_keyword]
        label_classes = self.sample.number_of_label_classes
        N_labels = len(labels)
        if N_labels == 0:
            number_of_classes = None
        elif N_labels == 1:
            label_key = list(labels.keys())[0]
            if label_classes != {}:
                # we get the label from the sample
                N_classes = list(label_classes.values())[0]
            else:
                # It is a mask, so we get the unique number of classes
                # in the mask
                N_classes = int(len(np.unique(labels[label_key])))
            number_of_classes = {label_key: N_classes}
        else:
            number_of_classes = {}
            for i_key, i_value in labels.items():
                if i_key in label_classes:
                    number_of_classes[i_key] = label_classes[i_key]
                else:
                    number_of_classes[i_key] = int(len(np.unique(i_value)))

        return number_of_classes

    def _write_to_h5py(
        self, filename: str, data_structure: dict, number_of_classes: dict, metadata: dict
    ):
        with h5py.File(filename, "w") as h5f:
            h5f.attrs["sample_name"] = self.sample.sample_name
            sample_group = h5f.create_group(self.saving_config.sample_npz_keyword)
            for i_sample_name, i_sample in data_structure[
                self.saving_config.sample_npz_keyword
            ].items():
                sample_ds = sample_group.create_dataset(i_sample_name, data=i_sample)
                sample_ds.attrs["N_channels"] = np.asarray(i_sample.shape[-1]).astype(np.uint16)
                sample_ds.attrs["size"] = np.asarray(metadata["patch_size"]).astype(np.uint16)
                sample_ds.attrs["dimensionality"] = np.asarray(len(i_sample.shape[0:-1])).astype(
                    np.uint8
                )
                sample_ds.attrs["origin"] = np.asarray(metadata["patch_origin"]).astype(np.float32)
                sample_ds.attrs["index"] = np.asarray(metadata["patch_index"]).astype(np.int32)
                sample_ds.attrs["direction"] = np.asarray(metadata["patch_direction"]).astype(
                    np.float32
                )
                sample_ds.attrs["spacing"] = np.asarray(metadata["patch_spacing"]).astype(
                    np.float32
                )
                sample_ds.attrs["original_size"] = np.asarray(
                    self.sample.original_metadata["image_size"]
                ).astype(np.uint32)
                sample_ds.attrs["original_origin"] = self.sample.original_metadata["image_origin"]
                sample_ds.attrs["original_direction"] = self.sample.original_metadata[
                    "image_direction"
                ]
                sample_ds.attrs["original_spacing"] = self.sample.original_metadata["image_spacing"]

            label_group = h5f.create_group(self.saving_config.label_npz_keyword)
            for i_label_name, i_label in data_structure[
                self.saving_config.label_npz_keyword
            ].items():
                label_ds = label_group.create_dataset(i_label_name, data=i_label)
                label_ds.attrs["N_classes"] = np.asarray(number_of_classes[i_label_name]).astype(
                    np.uint16
                )
                label_ds.attrs["one_hot"] = self.sample.are_labels_one_hot

    def channel_imputation(self, sample_channels):
        sample_channel_names = np.asarray(self.sample.channel_names)
        expected_channel_names = np.asarray(sorted(self.saving_config.channel_names))
        channels_dtype = sample_channels[0].dtype

        imputed_sample_channels = np.zeros(
            sample_channels.shape[0:-1] + (len(expected_channel_names),),
            dtype=sample_channels[0].dtype,
        )

        for i_i, i_expected_channel_name in enumerate(expected_channel_names):
            if i_expected_channel_name in sample_channel_names:
                channel_location = np.squeeze(
                    np.argwhere(sample_channel_names == i_expected_channel_name)
                )
                imputed_sample_channels[..., i_i] = sample_channels[..., channel_location]

        return imputed_sample_channels

    def channels_to_float16(self, sample_channels):
        channels_float16 = sample_channels.astype(np.float16)
        machine_epsilon = np.finfo(sample_channels.dtype).eps
        to_compare = np.copy(sample_channels)

        to_compare[np.abs(to_compare) <= np.finfo(np.float16).eps] = 0

        percentage_diff = np.abs(channels_float16 - to_compare) / (to_compare) * 100.0
        percentage_diff = np.nan_to_num(percentage_diff, posinf=0)
        largest_diff = np.amax(percentage_diff)

        print("largest diff is:")
        print(largest_diff)

        if largest_diff <= self.saving_config.float16_percentage_diff:
            out_channel = channels_float16
        else:
            out_channel = sample_channels

        return out_channel

    def saving(self):
        # First convert to a dict for easy saving in npz format
        sample_channels = self.sample.get_grouped_channels()

        if self.sample.has_output_channels:
            sample_output_channels = self.sample.get_grouped_output_channels()
        else:
            sample_output_channels = [None] * len(sample_channels)

        if self.sample.has_masks:
            sample_masks = self.sample.get_grouped_masks()
        else:
            sample_masks = [None] * len(sample_channels)

        sample_metadata = self.sample.metadata
        sample_labels = self.sample.labels

        data_structure_patches = []
        for i_channel_patch, i_output_channel_patch, i_mask_patch, i_label_patch in zip(
            sample_channels, sample_output_channels, sample_masks, sample_labels
        ):
            data_structure_patches.append(
                self._patch_to_data_structure(i_channel_patch, i_output_channel_patch, i_mask_patch, i_label_patch)
            )

        if data_structure_patches:
            number_of_classes = self._get_number_of_classes(data_structure_patches[0])
        else:
            number_of_classes = None

        if self.output_directory is not None:
            output_directory = os.path.join(self.output_directory, self.saving_config.out_dir_name)
        else:
            output_directory = self.saving_config.out_dir_name

        IO_utils.create_directory(output_directory)

        default_patch_name = self.sample.sample_name + self._PATCH_SEPARATOR
        if len(data_structure_patches) > 0:
            N_digits = int(np.log10(len(data_structure_patches))) + 1
        else:
            N_digits = 0

        file_names = []

        for (i_i_patch, i_patch), i_metadata in zip(
            enumerate(data_structure_patches), self.sample.metadata
        ):
            out_patch_name = (
                default_patch_name + str(i_i_patch).zfill(N_digits) + self._HDF5_EXTENSION
            )
            out_patch_file = os.path.join(output_directory, out_patch_name)
            self._write_to_h5py(out_patch_file, i_patch, number_of_classes, i_metadata)
            file_names.append(out_patch_file)

        self.save_names = file_names


class BatchPreprocessor:
    def __init__(self, samples_path: str, output_directory: str, config: dict):
        self.sample_directories = IO_utils.get_subdirectories(samples_path)
        self.config = copy.deepcopy(config)
        self.output_directory = output_directory
        self.general_config = PrognosAIs.IO.Configs.general_config(config)
        self.sample_class = Samples.get_sample_class(self.general_config.sample_type)
        self._CLASS_WEIGHT_FILE = "class_weights.json"

        self.labeling_config = PrognosAIs.IO.Configs.labeling_config(config)
        if self.labeling_config.perform_step and self.labeling_config.label_file is not None:
            self.label_loader = PrognosAIs.IO.LabelParser.LabelLoader(
                self.labeling_config.label_file,
                filter_missing=self.labeling_config.filter_missing,
                missing_value=self.labeling_config.missing_value,
                make_one_hot=self.labeling_config.make_one_hot,
            )
        else:
            self.label_loader = None

    def _run_single_sample(self, sample_directory: str):
        sample = self.sample_class(
            root_path=sample_directory, mask_keyword=self.general_config.mask_keyword, output_channel_names=self.general_config.output_channel_names
        )
        print(sample.sample_name)

        if self.label_loader is not None:
            # print(sample.sample_name in self.label_loader.get_samples())
            if sample.sample_name in self.label_loader.get_samples():
                sample_label = self.label_loader.get_label_from_sample(sample.sample_name)
                number_of_classes = self.label_loader.get_number_of_classes()
                labels_one_hot = self.label_loader.one_hot_encoded
            else:
                return None, None
        else:
            sample_label = {}
            number_of_classes = {}
            labels_one_hot = self.labeling_config.make_one_hot

        sample.add_to_labels(sample_label, number_of_classes)
        sample.are_labels_one_hot = labels_one_hot

        preprocessor = SingleSamplePreprocessor(
            sample, self.config, output_directory=self.output_directory,
        )
        preprocessor.apply_pipeline()
        return preprocessor.save_names, sample_label

    def start(self):
        N_cpus = np.minimum(IO_utils.get_number_of_cpus(), self.general_config.max_cpus)
        print("Number of cpus:")
        print(N_cpus)
        if N_cpus > 1:
            with Pool(N_cpus) as p:
                sample_info = p.map(self._run_single_sample, self.sample_directories)
        else:
            sample_info  = []
            for i_sample_directory in self.sample_directories:
                sample_info.append(self._run_single_sample(i_sample_directory))

        sample_save_names = [
            i_sample_info[0] for i_sample_info in sample_info if i_sample_info[0] is not None
        ]
        single_save_name = sample_save_names[0][0]
        samples_directory = IO_utils.get_parent_directory(single_save_name)

        sample_labels = [
            i_sample_info[1] for i_sample_info in sample_info if i_sample_info[1] is not None
        ]

        self.N_samples = len(sample_save_names)

        subsets = self.split_into_subsets(sample_save_names, sample_labels)

        for subset_name, subset_samples in subsets.items():
            if subset_samples is not None:
                output_folder = os.path.join(samples_directory, subset_name)
                IO_utils.create_directory(output_folder)
                for i_sample in subset_samples:
                    for i_patch in i_sample:
                        shutil.move(i_patch, output_folder)

        if self.label_loader is not None:
            class_weights = self.label_loader.get_class_weights(json_serializable=True)
            with open(
                os.path.join(samples_directory, self._CLASS_WEIGHT_FILE), "w"
            ) as the_json_file:
                json.dump(class_weights, the_json_file, indent=4)

        return samples_directory

    # ===============================================================
    # Labeling
    # ===============================================================

    @staticmethod
    def _get_number_of_samples_in_subsets(
        train_fraction: float,
        validation_fraction: float,
        test_fraction: float,
        number_of_samples: int,
    ):
        set_fractions = np.asarray([train_fraction, validation_fraction, test_fraction])

        # Normalize the fractions
        set_fractions = set_fractions / np.sum(set_fractions)
        N_samples_in_set = set_fractions * number_of_samples

        fractional_samples, N_samples_in_set = np.modf(N_samples_in_set)

        # All the fractional parts we assign to the train set
        N_samples_in_set[0] += np.sum(fractional_samples)
        N_samples_in_set = N_samples_in_set.astype(np.int)

        (N_train_samples, N_val_samples, N_test_samples) = N_samples_in_set

        return N_train_samples, N_val_samples, N_test_samples

    @staticmethod
    def _get_data_split(samples, test_size, stratification_labels=None) -> dict:
        samples = np.asarray(samples)
        if test_size == 0:
            return samples, None, None

        if stratification_labels is not None:
            splitter = sklearn.model_selection.StratifiedShuffleSplit(
                n_splits=1, test_size=test_size
            )
        else:
            splitter = sklearn.model_selection.ShuffleSplit(n_splits=1, test_size=test_size)

        train_indices, test_indices = next(splitter.split(samples, stratification_labels))
        if stratification_labels is not None:
            stratification_labels = np.asarray(stratification_labels)
            train_strat_labels = stratification_labels[train_indices]
        else:
            train_strat_labels = None

        return samples[train_indices], samples[test_indices], train_strat_labels

    def split_into_subsets(self, samples: list, sample_labels: list) -> dict:
        N_train, N_val, N_test = self._get_number_of_samples_in_subsets(
            self.labeling_config.train_fraction,
            self.labeling_config.validation_fraction,
            self.labeling_config.test_fraction,
            self.N_samples,
        )

        if self.labeling_config.stratify_label_name is not None:
            stratification_labels = []
            for i_sample_label in sample_labels:
                stratification_labels.append(
                    i_sample_label[self.labeling_config.stratify_label_name]
                )
        else:
            stratification_labels = None

        samples, test_samples, stratification_labels = self._get_data_split(
            samples, N_test, stratification_labels
        )
        train_samples, val_samples, _ = self._get_data_split(samples, N_val, stratification_labels)

        subsets = {
            "train": train_samples,
            "test": test_samples,
            "validation": val_samples,
        }

        return subsets

    @classmethod
    def init_from_sys_args(cls, args_in):
        parser = argparse.ArgumentParser(
            description="Pre-process (medical) images for use in a neural network"
        )
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
            help="The input directory where the samples to pre-process are located",
            metavar="Input directory",
            dest="input_dir",
            type=str,
        )
        parser.add_argument(
            "-o",
            "--output",
            required=True,
            help="The output directory to store the pre-processed samples in",
            metavar="Output directory",
            dest="output_dir",
            type=str,
        )

        args = parser.parse_args(args_in)

        config = PrognosAIs.IO.ConfigLoader.ConfigLoader(args.config)

        batch_preprocessor = cls(
            args.input_dir, args.output_dir, config.get_preprocessings_settings()
        )

        return batch_preprocessor


if __name__ == "__main__":
    batch_preprocessor = BatchPreprocessor.init_from_sys_args(sys.argv[1:])
    batch_preprocessor.start()
