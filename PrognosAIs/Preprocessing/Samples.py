import copy

from abc import ABC
from abc import abstractmethod
from typing import List
from typing import Union

import numpy as np
import SimpleITK as sitk

from PrognosAIs.IO import utils as IO_utils


class ImageSample(ABC):
    """
    ImageSample base class

    To be implemented by subclasses:

        * `init_image_files`: Contains logic for retrieval of channel filepaths
        * `load_channels`: Contains logic for loading of channels from filepaths
        * `load_output_channels`: Contains logic for loading of output channels from filepaths
        * `load_masks`: Contains logic of loading masks from filepaths

    Args:
        root_path: Path of the sample.
            Should contain folders or directories of channels and masks
        extension_keyword: Extension of the files to load
        mask_keyword (optional): Keyword to identify which filepaths are masks.
            Defaults to None.
    """

    def __init__(
        self,
        root_path: str,
        extension_keyword: str = None,
        mask_keyword: str = None,
        labels: dict = None,
        number_of_label_classes: dict = None,
        are_labels_one_hot: bool = False,
        output_channel_names: list = [],
    ):
        self.root_path = copy.deepcopy(root_path)
        if isinstance(labels, dict):
            self.labels = [copy.deepcopy(labels)]
        elif isinstance(labels, list):
            self.labels = copy.deepcopy(labels)
        else:
            self.labels = [{}]

        if number_of_label_classes is not None:
            self.number_of_label_classes = copy.deepcopy(number_of_label_classes)
        else:
            self.number_of_label_classes = {}
        self.are_labels_one_hot = copy.deepcopy(are_labels_one_hot)
        self.mask_keyword = mask_keyword
        self.sample_name = IO_utils.get_root_name(self.root_path)
        self.output_channel_names = output_channel_names

        for i_output_channel_name in self.output_channel_names:
            self.number_of_label_classes[i_output_channel_name] = -1

        if extension_keyword is not None:
            self.image_extension = extension_keyword
        else:
            self.image_extension = ""
        self.image_files = self.init_image_files()

        self.channel_files = self._init_channel_files(self.image_files)
        self.output_channel_files = self._init_output_channel_files(self.image_files)
        self.mask_files = self._init_mask_files(self.image_files)
        self.number_of_channels = len(self.channel_files)
        self.number_of_output_channels = len(self.output_channel_files)
        self.number_of_masks = len(self.mask_files)

        self._channels = self.load_channels(self.channel_files)
        self._output_channels = self.load_channels(self.output_channel_files)
        self._masks = self.load_masks(self.mask_files)

        self.has_patches = False
        self.number_of_patches = 0
        self.number_of_output_patches = 0
        self.has_masks = self.number_of_masks > 0
        self.has_output_channels = self.number_of_output_channels > 0
        self._channel_patches = False
        self._output_channel_patches = False
        self._mask_patches = False

        self.update_channel_size()
        self.update_output_channel_size()
        self.update_mask_size()

        self._perform_sanity_checks()

        example_image = self.get_example_channel()
        self.original_metadata = {
            "image_size": example_image.GetSize(),
            "image_spacing": example_image.GetSpacing(),
            "image_origin": example_image.GetOrigin(),
            "image_direction": example_image.GetDirection(),
            "image": example_image,
        }
        self.update_metadata()
        self.update_labels()

    def copy(self):
        """
        Returns a (deep) copy of the instance

        Returns:
            ImageSample: Deep copy of the instance
        """

        return self.__class__(
            root_path=self.root_path,
            extension_keyword=self.image_extension,
            mask_keyword=self.mask_keyword,
            labels=self.labels,
            number_of_label_classes=self.number_of_label_classes,
            are_labels_one_hot=self.are_labels_one_hot,
            output_channel_names=self.output_channel_names
        )

    def _perform_sanity_checks(self):
        """
        Automatic sanity check to see if we can process the sample

        Raises:
            NotImplementedError: If the configuration has not been implemented
        """

        if self.number_of_masks not in [0, 1, self.number_of_channels]:
            err_str = """Expect the number of masks to either be 0, 1 or equal to the number
             of channels but got {}!\nNumber of channels: {}"""
            raise NotImplementedError(err_str.format(self.number_of_masks, self.number_of_channels))

    def _identify_mask_file(self, image_file: str) -> bool:
        """
        Identify whether an image file is a mask based on the mask keyword

        Args:
            image_file: Image file to check

        Returns:
            bool: True if image_file is mask, False otherwise
        """

        if self.mask_keyword is None:
            is_mask_file = False
        else:
            is_mask_file = self.mask_keyword in IO_utils.get_root_name(image_file)

        return is_mask_file

    def _identify_output_channel_file(self, image_file: str) -> bool:
        """
        Identify whether an image file is a output channel

        Args:
            image_file: Image file to check

        Returns:
            bool: True if image_file is output channel, False otherwise
        """

        return IO_utils.get_file_name(image_file, self.image_extension) in self.output_channel_names

    @abstractmethod
    def init_image_files(self) -> list:
        """
        Get the filepaths (folders or files) of the channels for a single sample.
        To be implemented by the subclass

        Returns:
            list: The filepaths of the channels
        """

    @abstractmethod
    def load_channels(self, channel_files: list) -> dict:
        """
        Load the channels from the channel files.
        To be implemented by the subclass

        Example subclass implementation::

            def load_channels(self, channel_files):
                channels = {}

                nifti_reader = sitk.ImageFileReader()
                nifti_reader.SetImageIO("NiftiImageIO")
                for i_channel_file in channel_files:
                    nifti_reader.SetFileName(i_channel_file)
                    i_channel = nifti_reader.Execute()
                    i_channel_name = os.path.basename(i_channel_file)
                    channels[i_channel_name] = i_channel
                return channels

        Args:
            channel_files (list): Paths to the channels to be loaded

        Returns:
            dict: mapping the channel file to the loaded image
        """

    @abstractmethod
    def load_masks(self, mask_files: list) -> dict:
        """
        Load the masks from the mask files.
        To be implemented by the subclass

        Example subclass implementation::

            def load_masks(self, mask_files):
                masks = {}
                nifti_reader = sitk.ImageFileReader()
                nifti_reader.SetImageIO("NiftiImageIO")
                for i_mask_file in mask_files:
                    nifti_reader.SetFileName(i_mask_file)
                    i_mask = nifti_reader.Execute()
                    i_mask = sitk.Cast(i_mask, sitk.sitkUInt8)
                    i_mask_name = IO_utils.get_file_name(i_mask_file, self.image_extension)
                    masks[i_mask_name] = i_mask
                return masks

        Args:
            mask_files (list): Paths to the masks to be loaded

        Returns:
            dict: mapping the mask file to the loaded mask
        """

    def _init_channel_files(self, image_files: list) -> list:
        """
        Get only the channel files from the image files, filtering out masks.

        Args:
            image_files (list): Paths to the image files

        Returns:
            list: The paths to the channel files
        """

        channel_files = [
            i_image_file
            for i_image_file in image_files
            if (
                not self._identify_mask_file(i_image_file)
                and not self._identify_output_channel_file(i_image_file)
            )
        ]
        return channel_files

    def _init_output_channel_files(self, image_files: list) -> list:
        """
        Get the output channel files from the image files.

        Args:
            image_files (list): Paths to the image files

        Returns:
            list: The paths to the output channel files
        """

        output_channel_files = [
            i_image_file
            for i_image_file in image_files
            if (
                not self._identify_mask_file(i_image_file)
                and self._identify_output_channel_file(i_image_file)
            )
        ]
        return output_channel_files

    def _init_mask_files(self, image_files: list) -> list:
        """
        Get only the mask files from the image files, filtering out channels.

        Args:
            image_files (list): Paths to the image files

        Returns:
            list: The paths to the mask files
        """
        mask_files = [
            i_image_file for i_image_file in image_files if self._identify_mask_file(i_image_file)
        ]
        return mask_files

    @property
    def channel_size(self) -> np.ndarray:
        """
        The image size of the channels
        """
        return self._channel_size

    @channel_size.setter
    def channel_size(self, channel_size):
        self._channel_size = np.asarray(channel_size)

    def update_channel_size(self):
        """
        Update the channel size according to the current channels
        """
        example_channel = self.get_example_channel()
        if example_channel is not None:
            self.channel_size = example_channel.GetSize()
            self.number_of_dimensions = len(example_channel.GetSize())
        else:
            self.channel_size = None
            self.number_of_dimensions = None

    @property
    def output_channel_size(self) -> np.ndarray:
        """
        The image size of the channels
        """
        return self._output_channel_size

    @output_channel_size.setter
    def output_channel_size(self, output_channel_size):
        self._output_channel_size = np.asarray(output_channel_size)

    def update_output_channel_size(self):
        """
        Update the channel size according to the current channels
        """
        example_channel = self.get_example_output_channel()
        if example_channel is not None:
            self.output_channel_size = example_channel.GetSize()
            self.output_number_of_dimensions = len(example_channel.GetSize())
        else:
            self.output_channel_size = None
            self.output_number_of_dimensions = None

    @property
    def mask_size(self) -> Union[np.ndarray, None]:
        """The image size of the masks."""
        if self.has_masks:
            return self._mask_size

        return None

    @mask_size.setter
    def mask_size(self, mask_size) -> None:
        if self.has_masks:
            self._mask_size = np.asarray(mask_size)

    def update_mask_size(self) -> None:
        """
        Update the mask size according to the current masks
        """
        if self.has_masks:
            example_mask = self.get_example_mask()
            if example_mask is not None:
                self.mask_size = example_mask.GetSize()
            else:
                self.mask_size = None

    def update_metadata(self) -> None:
        example_patches = self.get_example_channel_patches()
        self.metadata = []

        for i_patch in example_patches:
            # If the dimensionality changes then we need to do a little trick
            # we always just get the first dimension
            # TODO allow recontrsuction from multi-dimensional images
            if self.original_metadata["image"].GetDimension() > i_patch.GetDimension():
                patch_index = self.original_metadata["image"].TransformPhysicalPointToIndex(
                    i_patch.GetOrigin() + (0,),
                )
            else:
                patch_index = self.original_metadata["image"].TransformPhysicalPointToIndex(
                    i_patch.GetOrigin(),
                )

            i_metadata = {
                "patch_size": i_patch.GetSize(),
                "patch_direction": i_patch.GetDirection(),
                "patch_origin": i_patch.GetOrigin(),
                "patch_spacing": i_patch.GetSpacing(),
                "patch_index": patch_index,
            }

            self.metadata.append(i_metadata)

    def update_labels(self) -> None:
        if len(self.labels) < self.number_of_patches:
            new_labels = []
            for _ in range(self.number_of_patches):
                new_labels.append(copy.deepcopy(self.labels[0]))
            self.labels = new_labels

    def add_to_labels(
        self, to_add_labels: List[dict], to_add_number_of_label_classes: dict
    ) -> None:
        # If the user parses just a dict, we fix this here
        if isinstance(to_add_labels, dict):
            to_add_labels = [to_add_labels]

        if len(to_add_labels) != 1 and len(to_add_labels) != self.number_of_patches:
            err_msg = (
                "Labels to add should have either length 1 or length equal to"
                " number of patches.\n"
                "Number of patches is {}, but you have supplied {} labels.\n"
                "Please check your code and try again."
            ).format(self.number_of_patches, len(to_add_labels))
            raise ValueError(err_msg)

        if len(to_add_labels) == 1 and self.has_patches:
            to_add_labels = to_add_labels * self.number_of_patches

        for i_label, i_to_add_label in zip(self.labels, to_add_labels):
            i_label.update(i_to_add_label)

        self.number_of_label_classes.update(to_add_number_of_label_classes)

    def _parse_function_parameters(self, function_parameters):
        """
        Parse the function parameters

        Args:
            function_parameters (function or tuple): Function and possible args
                and kw_args.

        Returns:
            tuple: function, args, and kw_args
        """
        if isinstance(function_parameters, tuple):
            if len(function_parameters) == 2:
                function = function_parameters[0]
                if isinstance(function_parameters[1], list):
                    arguments = function_parameters[1]
                    kw_arguments = {}
                elif isinstance(function_parameters[1], dict):
                    arguments = []
                    kw_arguments = function_parameters[1]
            elif len(function_parameters) == 3:
                function = function_parameters[0]
                if isinstance(function_parameters[1], list):
                    arguments = function_parameters[1]
                    kw_arguments = function_parameters[2]
                else:
                    arguments = function_parameters[2]
                    kw_arguments = function_parameters[1]
        else:
            function = function_parameters
            arguments = []
            kw_arguments = {}

        return function, arguments, kw_arguments

    @property
    def channels(self) -> list:
        """
        The channels present in the sample

        Channels of a sample can be set by providing either a function,
         or a tuple consisting of a function, possible function argument
         and function keyword arguments.
        This function will then be applied to all channels in the sample.
        The function has to output either a SimpleITK Image or a list.
        In the last case it is assumed that these are patches and the class is updated accordingly


        Returns:
            list: Channels present in the sample
        """
        return [value for key, value in sorted(self._channels.items(), key=lambda item: item[0])]

    @channels.setter
    def channels(self, function_parameters):
        (
            function,
            function_arguments,
            function_kw_arguments,
        ) = self._parse_function_parameters(function_parameters)

        example_function_output = function(
            self.get_example_channel(), *function_arguments, **function_kw_arguments
        )
        returns_patches = isinstance(example_function_output, list)

        if self._channel_patches and not returns_patches:
            for i_channel_name in self.channel_names:
                for i_i_patch, i_patch in enumerate(self._channels[i_channel_name]):
                    self._channels[i_channel_name][i_i_patch] = function(
                        i_patch, *function_arguments, **function_kw_arguments
                    )
        else:
            channel_items = sorted(self._channels.items(), key=lambda item: item[0])
            if not returns_patches:
                # We already have the first example, so we can use it for efficiency
                # If the function does return patches we should pass all patches at once
                # So we still need to calculate the output
                self._channels[channel_items[0][0]] = example_function_output
                channel_items = channel_items[1:]
            for i_i_channel, (i_channel_name, i_channel) in enumerate(channel_items):
                self._channels[i_channel_name] = function(
                    i_channel, *function_arguments, **function_kw_arguments
                )

        if returns_patches:
            self.has_patches = True
            self._channel_patches = True
            self.number_of_patches = len(self.get_example_channel_patches())
            self._number_of_channel_patches = len(self.get_example_channel_patches())

        self.update_channel_size()
        self.update_metadata()
        self.update_labels()

    @property
    def output_channels(self) -> list:
        """
        The output channels present in the sample

        Output channels of a sample can be set by providing either a function,
         or a tuple consisting of a function, possible function argument
         and function keyword arguments.
        This function will then be applied to all output channels in the sample.
        The function has to output either a SimpleITK Image or a list.
        In the last case it is assumed that these are patches and the class is updated accordingly


        Returns:
            list: Channels present in the sample
        """
        return [value for key, value in sorted(self._output_channels.items(), key=lambda item: item[0])]

    @output_channels.setter
    def output_channels(self, function_parameters):
        if self.has_output_channels:
            (
                function,
                function_arguments,
                function_kw_arguments,
            ) = self._parse_function_parameters(function_parameters)

            example_function_output = function(
                self.get_example_output_channel(), *function_arguments, **function_kw_arguments
            )
            returns_patches = isinstance(example_function_output, list)

            if self._output_channel_patches and not returns_patches:
                for i_output_channel_name in self.output_channel_names:
                    for i_i_patch, i_patch in enumerate(self._output_channels[i_output_channel_name]):
                        self._output_channels[i_output_channel_name][i_i_patch] = function(
                            i_patch, *function_arguments, **function_kw_arguments
                        )
            else:
                output_channel_items = sorted(self._output_channels.items(), key=lambda item: item[0])
                if not returns_patches:
                    # We already have the first example, so we can use it for efficiency
                    # If the function does return patches we should pass all patches at once
                    # So we still need to calculate the output
                    self._output_channels[output_channel_items[0][0]] = example_function_output
                    output_channel_items = output_channel_items[1:]
                for i_output_channel_name, i_output_channel in output_channel_items:
                    self._output_channels[i_output_channel_name] = function(
                        i_output_channel, *function_arguments, **function_kw_arguments
                    )

            if returns_patches:
                self.has_patches = True
                self._output_channel_patches = True
                self.number_of_output_patches = len(self.get_example_output_channel_patches())
                self._number_of_output_channel_patches = len(self.get_example_output_channel_patches())

            self.update_output_channel_size()
            self.update_metadata()
            self.update_labels()

    @property
    def masks(self):
        """
        The masks present in the sample

        Masks of a sample can be set by providing either a function,
         or a tuple consisting of a function, possible function argument
         and function keyword arguments.
        This function will then be applied to all masks in the sample.
        The function has to output either a SimpleITK Image or a list.
        In the last case it is assumed that these are patches and the class is updated accordingly


        Returns:
            list: masks present in the sample
        """
        return [value for key, value in sorted(self._masks.items(), key=lambda item: item[0])]

    @masks.setter
    def masks(self, function_parameters):
        if self.has_masks:
            (
                function,
                function_arguments,
                function_kw_arguments,
            ) = self._parse_function_parameters(function_parameters)

            example_function_output = function(
                self.get_example_mask(), *function_arguments, **function_kw_arguments
            )
            returns_patches = isinstance(example_function_output, list)

            if self._mask_patches and not returns_patches:
                for i_mask_name in self.mask_names:
                    for i_i_patch, i_patch in enumerate(self._masks[i_mask_name]):
                        self._masks[i_mask_name][i_i_patch] = function(
                            i_patch, *function_arguments, **function_kw_arguments
                        )
            else:
                for i_mask_name, i_mask in self._masks.items():
                    self._masks[i_mask_name] = function(
                        i_mask, *function_arguments, **function_kw_arguments
                    )

            if returns_patches:
                self.has_patches = True
                self._mask_patches = True
                self._number_of_mask_patches = len(self.get_example_channel_patches())
                self.number_of_patches = len(self.get_example_mask_patches())
            self.update_mask_size()
            self.update_metadata()
            self.update_labels()

    @property
    def channel_names(self) -> list:
        """
        Names of the channels

        Returns:
            list: Channel names
        """
        return [key for key in sorted(self._channels.keys())]

    @property
    def mask_names(self) -> list:
        """
        Names of the masks

        Returns:
            list: Mask names
        """
        return [key for key in sorted(self._masks.keys())]

    def get_example_channel(self) -> sitk.Image:
        """
        Provides an example channel of the samples

        Returns:
            sitk.Image: Single channel of the sample
        """

        example = self.channels[0]
        if self._channel_patches and self._number_of_channel_patches > 0:
            example = example[0]
        elif self._channel_patches:
            # If we removed all patches, we will return None
            return None
        return sitk.Image(example)

    def get_example_output_channel(self) -> sitk.Image:
        """
        Provides an example output channel of the samples

        Returns:
            sitk.Image: Single channel of the sample
        """

        if self.has_output_channels:
            example = self.output_channels[0]
            if self._output_channel_patches and self._number_of_output_channel_patches > 0:
                example = example[0]
            elif self._output_channel_patches:
                # If we removed all patches, we will return None
                return None
            example = sitk.Image(example)
        else:
            example = None

        return example

    def get_example_mask(self) -> sitk.Image:
        """
        Provides an example mask of the samples

        Returns:
            sitk.Image: Single mask of the sample
        """

        if self.has_masks:
            example = self.masks[0]
            if self._mask_patches and self._number_of_mask_patches > 0:
                example = example[0]
            elif self._mask_patches:
                return None
            example = sitk.Image(example)
        else:
            example = None
        return example

    def get_example_channel_patches(self) -> list:
        """
        Provides an example of all patches of a channel, even if there is only one patch

        Returns:
            list: Patch(es) of a single channel of the sample
        """

        temp_example = self.channels[0]
        if not self._channel_patches:
            temp_example = [temp_example]

        # We ensure we make a copy so no operations are done on the
        # real sample
        example = []
        for i_patch in temp_example:
            example.append(sitk.Image(i_patch))
        return example

    def get_example_output_channel_patches(self) -> list:
        """
        Provides an example of all patches of a output channel, even if there is only one patch

        Returns:
            list: Patch(es) of a single output channel of the sample
        """

        temp_example = self.output_channels[0]
        if not self._output_channel_patches:
            temp_example = [temp_example]

        # We ensure we make a copy so no operations are done on the
        # real sample
        example = []
        for i_patch in temp_example:
            example.append(sitk.Image(i_patch))
        return example

    def get_example_mask_patches(self) -> list:
        """
        Provides an example of all patches of a mask, even if there is only one patch

        Returns:
            list: Patch(es) of a single mask of the sample
        """

        if self.has_masks:
            temp_example = self.masks[0]
            if not self._mask_patches:
                temp_example = [temp_example]

            # We ensure we make a copy so no operations are done on the
            # real sample
            example = []
            for i_patch in temp_example:
                example.append(sitk.Image(i_patch))
        else:
            example = [None]
        return example

    def assert_all_channels_same_size(self):
        """
        Check wheter all channels have the same size

        Raises:
            ValueError: Raised when not all channels have same size
        """

        default_size = self.channel_size

        same_size = []
        for i_channel in self.channels:
            if self.has_patches:
                i_channel = i_channel[0]
            same_size.append(all(i_channel.GetSize() == default_size))

        if not all(same_size):
            raise ValueError("Not all channels have the same size, cannot continue!")

    def assert_all_masks_same_size(self):
        """
        Check wheter all masks have the same size

        Raises:
            ValueError: Raised when not all masks have same size
        """

        if self.has_masks:
            default_size = self.mask_size

            same_size = []
            for i_mask in self.masks:
                if self.has_patches:
                    i_mask = i_mask[0]
                same_size.append(all(i_mask.GetSize() == default_size))

            if not all(same_size):
                raise ValueError("Not all masks have the same size, cannot continue!")

    def get_grouped_channels(self) -> list:
        """
        Groups the channels on a per-patch basis instead of a per-channel basis

        The channels property indexes first by channel and then by (possibly) patches.
        This function instead first indexes by patches (or the whole sample of no patches).
        This can be handy when all channels are needed at the same time

        Returns:
            list: Grouped channels for each patch
        """

        grouped_channels = []
        if self.has_patches:
            channels = self.channels
            for i_patch in range(self.number_of_patches):
                grouped_channels.append(
                    [channels[i_channel][i_patch] for i_channel in range(self.number_of_channels)]
                )
        elif self.channels:
            grouped_channels.append(self.channels)
        else:
            # We dont have any channels
            grouped_channels = []

        return grouped_channels

    def get_grouped_output_channels(self) -> list:
        """
        Groups the output channels on a per-patch basis instead of a per-channel basis

        The channels property indexes first by channel and then by (possibly) patches.
        This function instead first indexes by patches (or the whole sample of no patches).
        This can be handy when all channels are needed at the same time

        Returns:
            list: Grouped channels for each patch
        """

        grouped_output_channels = []
        if self.has_patches:
            output_channels = self.output_channels
            for i_patch in range(self.number_of_output_channels):
                grouped_output_channels.append(
                    [output_channels[i_output_channel][i_patch] for i_output_channel in range(self.number_of_output_channels)]
                )
        elif self.output_channels:
            grouped_output_channels.append(self.output_channels)
        else:
            # We dont have any channels
            grouped_output_channels = []

        return grouped_output_channels

    def get_grouped_masks(self) -> list:
        """
        Groups the masks on a per-patch basis instead of a per-channel basis

        The masks property indexes first by channel and then by (possibly) patches.
        This function instead first indexes by patches (or the whole sample of no patches).
        This can be handy when all masks are needed at the same time

        Returns:
            list: Grouped channels for each patch. Empty lists if sample doesn't have mask
        """
        grouped_masks = []
        if self.has_patches:
            masks = self.masks
            for i_patch in range(self.number_of_patches):
                grouped_masks.append(
                    [masks[i_mask][i_patch] for i_mask in range(self.number_of_masks)]
                )
        else:
            grouped_masks.append(self.masks)

        return grouped_masks

    @staticmethod
    def get_sitk_type_from_numpy_type(numpy_type: np.dtype) -> int:
        conversion_table = {
            "uint8": sitk.sitkUInt8,
            "uint16": sitk.sitkUInt16,
            "uint32": sitk.sitkUInt32,
            "uint64": sitk.sitkUInt64,
            "int8": sitk.sitkInt8,
            "int16": sitk.sitkInt16,
            "int32": sitk.sitkInt32,
            "int64": sitk.sitkInt64,
            # Simple ITK doesnt support float16
            "float16": sitk.sitkFloat32,
            "float32": sitk.sitkFloat32,
            "float64": sitk.sitkFloat64,
        }

        return conversion_table[str(numpy_type)]

    @staticmethod
    def get_numpy_type_from_sitk_type(sitk_type: int) -> np.dtype:
        conversion_table = {
            sitk.sitkUInt8: "uint8",
            sitk.sitkUInt16: "uint16",
            sitk.sitkUInt32: "uint32",
            sitk.sitkUInt64: "uint64",
            sitk.sitkInt8: "int8",
            sitk.sitkInt16: "int16",
            sitk.sitkInt32: "int32",
            sitk.sitkInt64: "int64",
            # Simple ITK doesnt support float16
            sitk.sitkFloat32: "float32",
            sitk.sitkFloat64: "float64",
        }

        return conversion_table[sitk_type]

    @staticmethod
    def get_appropiate_dtype_from_scalar(
        value: Union[int, float], return_np_type: bool = False
    ) -> Union[int, np.dtype]:
        """
        Find the minimum SimpleITK type need to represent the value

        Args:
            value (float): The value to check
            return_np_type (bool): If True returns the numpy type instead of the SimpleITK type.
             Defaults to False.

        Returns:
            int: The appropiate SimpleITK to which the value can be casted
        """

        # Add 1e10 to make sure that very large values are still converted to floats
        # Otherwise we have a problem when the value is larger than the maximum value
        # of a uint64
        if isinstance(value, float) and value.is_integer() and value < 1e10:
            value = int(value)

        appropiate_dtype = np.min_scalar_type(value)
        if not return_np_type:
            appropiate_dtype = ImageSample.get_sitk_type_from_numpy_type(appropiate_dtype)

        return appropiate_dtype

    @staticmethod
    def get_appropiate_dtype_from_image(image: sitk.Image) -> int:
        """
        Find the minimum SimpleITK type need to represent the value

        Args:
            value (float): The value to check

        Returns:
            int: The appropiate SimpleITK to which the value can be casted
        """

        datatypes = [
            "uint8",
            "uint16",
            "uint32",
            "uint64",
            "int8",
            "int16",
            "int32",
            "int64",
            "float16",
            "float32",
            "float64",
        ]

        image_array = sitk.GetArrayViewFromImage(image)
        min_image_value = np.amin(image_array)
        max_image_value = np.amax(image_array)

        has_only_integers = np.array_equal(image_array, image_array.astype(np.int))

        if has_only_integers:
            min_image_value = int(min_image_value)
            max_image_value = int(max_image_value)
            min_type = ImageSample.get_appropiate_dtype_from_scalar(
                min_image_value, return_np_type=True
            )
            # If the max value also fits within this type we are done.
            # Otherwise we might get too big of a type (e.g. when min = -10 and max = 10, it will give int16 instead of int8)
            # when promoting types
            if np.can_cast(max_image_value, min_type):
                data_type = min_type
            else:
                test_array = np.asarray([min_image_value, max_image_value])
                for i_data_type in datatypes:
                    if np.array_equal(test_array, test_array.astype(i_data_type)):
                        data_type = i_data_type
                        break
        else:
            # In the case of floats we need to make sure that the  whole range fits in a float
            if min_image_value < 0:
                value_range = max_image_value - min_image_value
            else:
                value_range = max_image_value

            range_data_type = ImageSample.get_appropiate_dtype_from_scalar(
                value_range, return_np_type=True
            )
            # If the min/max values are int, here we make sure that we cast it to at least a float
            data_type = np.promote_types(range_data_type, np.float32)

        return ImageSample.get_sitk_type_from_numpy_type(data_type)

    @staticmethod
    def promote_simpleitk_types(type_1: int, type_2: int) -> int:
        """
        Get the datatype that can represent both datatypes

        Args:
            type_1 (int): SimpleITK datype of variable 1
            type_2 (int): SimpleITK datatype of variable 2

        Returns:
            int: SimpleITK datatype that can represent both datatypes
        """

        np_type_1 = ImageSample.get_numpy_type_from_sitk_type(type_1)
        np_type_2 = ImageSample.get_numpy_type_from_sitk_type(type_2)

        np_common_type = np.promote_types(np_type_1, np_type_2)

        sitk_common_type = ImageSample.get_sitk_type_from_numpy_type(np_common_type)

        return sitk_common_type


class NIFTISample(ImageSample):
    def __init__(self, **kwds):
        if "extension_keyword" not in kwds:
            kwds["extension_keyword"] = ".nii"
        super().__init__(**kwds)

    def load_channels(self, channel_files):
        channels = {}

        nifti_reader = sitk.ImageFileReader()
        nifti_reader.SetImageIO("NiftiImageIO")
        for i_channel_file in channel_files:
            nifti_reader.SetFileName(i_channel_file)
            i_channel = nifti_reader.Execute()
            i_channel_name = IO_utils.get_file_name(i_channel_file, self.image_extension)
            channel_dtype = self.get_appropiate_dtype_from_image(i_channel)
            # Channel has to be at minimum float32 because of pre-processing operations
            channel_dtype = self.promote_simpleitk_types(channel_dtype, sitk.sitkFloat32)
            i_channel = sitk.Cast(i_channel, channel_dtype)
            channels[i_channel_name] = i_channel
        return channels

    def load_masks(self, mask_files):
        masks = {}
        nifti_reader = sitk.ImageFileReader()
        nifti_reader.SetImageIO("NiftiImageIO")
        for i_mask_file in mask_files:
            nifti_reader.SetFileName(i_mask_file)
            i_mask = nifti_reader.Execute()
            mask_dtype = self.get_appropiate_dtype_from_image(i_mask)
            i_mask = sitk.Cast(i_mask, mask_dtype)
            i_mask_name = IO_utils.get_file_name(i_mask_file, self.image_extension)
            masks[i_mask_name] = i_mask
        return masks

    def init_image_files(self):
        image_files = IO_utils.find_files_with_extension(self.root_path, self.image_extension)
        return image_files


def get_sample_class(sample_type_name: str):
    if sample_type_name == "nifti":
        return NIFTISample
