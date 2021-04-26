from typing import Any, Tuple, Union
import logging

import h5py
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
import scipy

import PrognosAIs.Constants
import PrognosAIs.IO.utils as IO_utils


class HDF5Generator:
    def __init__(
        self,
        root_folder: str,
        batch_size: int = 16,
        shuffle: bool = False,
        max_steps: int = -1,
        drop_batch_remainder: bool = True,
        labels_only: bool = False,
    ) -> None:
        """
        Generate data from HDF5 files to be used in a TensorFlow pipeline.

        This generator loads sample data from HDF5 files, and does this efficiently
        making us of TensorFlow dataset functions.
        The inputs and outputs are dict, which allows for easy us in a
        multi-input and/or multi-output model

        Args:
            root_folder (str): Folder in which the HDF5 files are stored
            batch_size (int, optional): Batch size of the generator. Defaults to 16.
            shuffle (bool, optional): Whether datset should be shuffled. Defaults to False.
            data_augmentation (bool, optional): Whether data augmentation should be applied.
                Defaults to False.
            augmentation_factor (int, optional): Number of times dataset should be repeated for augmentation.
                Defaults to 5.
            augmentation_settings (dict, optional): Setting for the data augmenation. Defaults to None.
            max_steps (int, optional): Maximum number of (iteration) steps to provide.
                Defaults to -1, in which case all samples are provied.
            drop_batch_remainder (bool, optional): Whether to drop the remainder of the batch if it does not fit perfectly.
                Defaults to True.
            labels_only (bool, optional): Whether to only provide labels. Defaults to False.
            feature_index (str, optional): Name of the feature group in the HDF5 file. Defaults to "sample".
            label_index (str, optional): Name of the label group in the HDF5 file. Defaults to "label".
        """
        self.cache_in_memory = False
        self.TF_dataset = None
        self.augmentation_factor = 1
        self.augmentation_settings = {}
        self.augmentors = {}
        self.data_augmentation = False
        self.feature_index = PrognosAIs.Constants.FEATURE_INDEX
        self.label_index = PrognosAIs.Constants.LABEL_INDEX
        self.repeat = False
        self.shard = False
        self.n_workers = 1
        self.worker_index = 0

        self.sample_locations = IO_utils.find_files_with_extension(
            root_folder, PrognosAIs.Constants.HDF5_EXTENSION,
        )

        self.sample_files = [
            IO_utils.get_file_name_from_full_path(i_sample_location)
            for i_sample_location in self.sample_locations
        ]

        self.example_sample_file = self.sample_locations[0]
        self.max_steps = max_steps
        self.memory_size = IO_utils.get_dir_size(root_folder)

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.N_samples = len(self.sample_locations)
        self.drop_batch_remainder = drop_batch_remainder

        self.labels_only = labels_only
        self.dataset_attributes = self.get_all_dataset_attributes()

        self.hdf_dataset_names = self.get_dataset_names()
        self.feature_dataset_names = [
            i_ds_name for i_ds_name in self.hdf_dataset_names if self.feature_index in i_ds_name
        ]
        self.label_dataset_names = [
            i_ds_name for i_ds_name in self.hdf_dataset_names if self.label_index in i_ds_name
        ]
        self.feature_names = [
            i_ds_name.split(PrognosAIs.Constants.HDF_SEPARATOR)[-1]
            for i_ds_name in self.feature_dataset_names
        ]
        self.label_names = [
            i_ds_name.split(PrognosAIs.Constants.HDF_SEPARATOR)[-1]
            for i_ds_name in self.label_dataset_names
        ]

        self.spec = self.get_spec()
        self.possible_steps = int(np.floor(self.N_samples / self.batch_size))
        self.labels_are_one_hot = self.get_labels_are_one_hot()

        if self.max_steps != -1:
            self.steps = np.minimum(self.possible_steps, self.max_steps)
        else:
            self.steps = self.possible_steps

        if self.labels_only:
            self.loader = self.label_loader
        else:
            self.loader = self.features_and_labels_loader

    def get_spec(self) -> dict:
        """
        Get the TensorSpec for all input features.

        Returns:
            dict: Maps the name of each input feature to the TensorSpec of the input.
        """
        spec = {}
        for i_ds_name in self.hdf_dataset_names:
            dtype = self.get_dataset_attribute(i_ds_name, "dtype")
            shape = self.get_dataset_attribute(i_ds_name, "shape")
            spec[i_ds_name] = tf.TensorSpec(shape, dtype=dtype, name=i_ds_name)

        return spec

    def get_dataset_names(self) -> list:
        """
        Get the names of all datasets in the sample.

        Returns:
            list: Dataset names in the sample
        """
        with h5py.File(self.example_sample_file, "r") as example_5py:
            ds_names = self._get_dataset_names(example_5py)
        return ds_names

    def _get_dataset_names(self, h5py_object: Union[h5py.File, h5py.Dataset, h5py.Group]) -> list:
        """
        Run through all groups and dataset to get the names.

        Args:
            h5py_object (Union[h5py.File, h5py.Dataset, h5py.Group]): Object for which to return
                the dataset names

        Returns:
            list: Dataset names in object
        """
        if isinstance(h5py_object, h5py.Dataset):
            dataset_names = [h5py_object.name]
        else:
            dataset_names = []
            for i_key in h5py_object.keys():
                dataset_names.extend(self._get_dataset_names(h5py_object.get(i_key)))
        return dataset_names

    def get_all_dataset_attributes(self, sample_file: str = None) -> dict:
        """
        Get the attributes of the features and labels stored in the file.

        Returns:
            dict: Mapping of the feature/label name to its attributes
        """
        if sample_file is None:
            sample_file = self.example_sample_file
        with h5py.File(sample_file, "r") as hdf5_example:
            ds_attributes = self._get_all_dataset_attributes(hdf5_example)

        return ds_attributes

    def _get_all_dataset_attributes(
        self, h5py_object: Union[h5py.File, h5py.Dataset, h5py.Group],
    ) -> dict:
        """
        Run through al groups and dataset to get the attributes.

        Args:
            h5py_object (Union[h5py.File, h5py.Dataset, h5py.Group]): Object for which to return
                the attributes

        Returns:
            dict: Mapping between feature/label name and its attributes
        """
        if isinstance(h5py_object, h5py.Dataset):
            ds_attributes = dict(h5py_object.attrs)
            ds_attributes["shape"] = h5py_object.shape
            ds_attributes["dtype"] = h5py_object.dtype
            return {h5py_object.name: ds_attributes}
        else:
            ds_attributes = {h5py_object.name: dict(h5py_object.attrs)}
            for i_key in h5py_object.keys():
                ds_attributes.update(self._get_all_dataset_attributes(h5py_object.get(i_key)))
            return ds_attributes

    def get_dataset_attribute(self, dataset_name: str, attribute_name: str) -> Any:
        """
        Get the attribute of a specific dataset

        Args:
            dataset_name (str): Name of dataset for which to get the attribute
            attribute_name (str): Name of attribute to get

        Returns:
            Any: The value of the attribute
        """
        return self.dataset_attributes[dataset_name][attribute_name]

    def get_feature_attribute(self, attribute_name: str) -> dict:
        """
        Get a specific attribute for all features.

        Args:
            attribute_name (str): Name of attribute to get

        Returns:
            dict: Mapping between feature names and the attribute value
        """
        attribute = {}
        for i_feature_name, i_feature_dataset_name in zip(
            self.feature_names, self.feature_dataset_names,
        ):
            attribute[i_feature_name] = self.get_dataset_attribute(
                i_feature_dataset_name, attribute_name,
            )

        return attribute

    def get_label_attribute(self, attribute_name: str) -> dict:
        """
        Get a specific attribute for all labels.

        Args:
            attribute_name (str): Name of attribute to get

        Returns:
            dict: Mapping between label names and the attribute value
        """
        attribute = {}
        for i_label_name, i_label_dataset_name in zip(self.label_names, self.label_dataset_names):
            attribute[i_label_name] = self.get_dataset_attribute(
                i_label_dataset_name, attribute_name,
            )

        return attribute

    def get_feature_metadata(self) -> dict:
        """
        Get all metadata of all features.

        Returns:
            dict: The metadata of all features
        """
        feature_metadata = {}

        for i_feature_name, i_feature_dataset_name in zip(
            self.feature_names, self.feature_dataset_names,
        ):
            feature_metadata[i_feature_name] = self.dataset_attributes[i_feature_dataset_name]

        return feature_metadata

    def get_feature_metadata_from_sample(self, sample_location: str) -> dict:
        """
        Get the feature metadata of a specific sample.

        Args:
            sample_location (str): The file location of the sample

        Returns:
            dict: The feature metadata of the sample
        """
        ds_attributes = self.get_all_dataset_attributes(sample_location)
        feature_metadata = {}
        for i_feature_name, i_feature_dataset_name in zip(
            self.feature_names, self.feature_dataset_names,
        ):
            feature_metadata[i_feature_name] = ds_attributes[i_feature_dataset_name]
        return feature_metadata

    def get_number_of_classes(self) -> dict:
        """
        Get the number of output classes.

        Returns:
            dict: Number of output classes for each label
        """
        return self.get_label_attribute("N_classes")

    def get_feature_dimensionality(self) -> dict:
        """
        Get the dimensionality of each feature.

        Returns:
            dict: Dimensionality of each feature
        """
        return self.get_feature_attribute("dimensionality")

    def get_feature_size(self) -> dict:
        """
        Get the size of each feature.

        The size only of the feature does not take into account the number of channels
        and only represents the size of an individual channel of the feature.

        Returns:
            dict: Size of each feature
        """
        return self.get_feature_attribute("size")

    def get_feature_shape(self) -> dict:
        """
        Get the shape of each feature.

        Returns:
            dict: Shape of each feature
        """
        return self.get_feature_attribute("shape")

    def get_number_of_channels(self) -> dict:
        """
        Get the number of feature channels.

        Returns:
            dict: Number of channels for each feature
        """
        return self.get_feature_attribute("N_channels")

    def get_labels_are_one_hot(self) -> dict:
        """
        Get whether labels are one-hot encoded.

        Returns:
            dict: One-hot encoding status of each label
        """
        return self.get_label_attribute("one_hot")

    def setup_augmentation(
        self, augmentation_factor: int = 1, augmentation_settings: dict = {},
    ) -> None:
        """
        Set up data augmentation in the generator.

        Args:
            augmentation_factor (int): Repeat dataset this many times in augmentation. Defaults to 1.
            augmentation_settings (dict): Setting to parse to augmentation instance. Defaults to {}.
        """

        logging.info(
            (
                "Setting up data augmentation with the following settings:\n"
                "Augmentation factor: {aug_fac}\n"
                "Augmentation settings: {aug_set}"
            ).format(aug_fac=augmentation_factor, aug_set=augmentation_settings)
        )
        self.augmentation_factor = augmentation_factor
        self.augmentation_settings = augmentation_settings
        self.data_augmentation = True
        for i_feature_ds_name, i_feature_name in zip(
            self.feature_dataset_names, self.feature_names,
        ):
            self.augmentors[i_feature_name] = Augmentor(
                self.spec[i_feature_ds_name], **self.augmentation_settings,
            )

        for i_label_ds_name, i_label_name in zip(self.label_dataset_names, self.label_names,):
            self.augmentors[i_label_name] = Augmentor(
                self.spec[i_label_ds_name], **self.augmentation_settings,
            )

    def features_and_labels_loader(
        self, sample_location: tf.Tensor
    ) -> Tuple[dict, dict, tf.Tensor]:
        """
        Load the features and labels from a hdf5 file to be used in a TensorFlow dataset pipeline.

        This loader loads the features and labels from a hdf5 file using TensorFlowIO.
        The outputs are therefor directly cast to tensor and can be used in a TensorFlow graph.
        All features and labels from the file are loaded, and a dict is returned mapping
        the name of each feature and label to its respective value

        Args:
            sample_location (tf.Tensor): Location of the sample file

        Returns:
            Tuple[dict, dict]: The features (first output) and labels (second output) loaded
                from the sample.
        """
        # tf.strings.split doesnt work because we can then not use the
        # value of that tensor as an index for the dict.
        # Works in eager execution but not in graph execution.
        # if that works in the future, this can be replace as it is probably faster.
        # Maybe we can do something by loading each dataset, concatenating and then returning
        # the name as well.
        # This could be done using tfio.IODataset instead of IOTensor, but might
        # be complicated then with concatenation and such
        loaded_hdf5 = tfio.IOTensor.from_hdf5(sample_location, spec=self.spec)
        features = self.load_features(loaded_hdf5)
        labels = self.load_labels(loaded_hdf5)

        return features, labels

    def label_loader(self, sample_location: tf.Tensor) -> dict:
        """
        Load the labels from a hdf5 sample file.

        This loader only loads the labels, instead of the features and labels
        as done by features_and_labels_loader

        Args:
            sample_location (tf.Tensor): Location of the sample file

        Returns:
            dict: Labels loaded from the sample file
        """
        loaded_hdf5 = tfio.IOTensor.from_hdf5(sample_location, spec=self.spec)
        return self.load_labels(loaded_hdf5)

    def feature_loader(self, sample_location: tf.Tensor) -> dict:
        """
        Load the features from a hdf5 sample file.

        This loader only loads the labels, instead of the features and labels
        as done by features_and_labels_loader

        Args:
            sample_location (tf.Tensor): Location of the sample file

        Returns:
            dict: Features loaded from the sample file
        """
        loaded_hdf5 = tfio.IOTensor.from_hdf5(sample_location, spec=self.spec)
        return self.load_features(loaded_hdf5)

    def load_features(self, loaded_hdf5: tfio.IOTensor) -> dict:
        """
        Load the features from a HDF5 tensor.

        Args:
            loaded_hdf5 (tfio.IOTensor): Tensor from which to load features

        Returns:
            dict: Mapping between feature names and features
        """
        features = {}
        for i_feature_ds_name, i_feature_name in zip(
            self.feature_dataset_names, self.feature_names,
        ):
            features[i_feature_name] = loaded_hdf5(i_feature_ds_name).to_tensor()

        return features

    def load_labels(self, loaded_hdf5: tfio.IOTensor) -> dict:
        """
        Load the labels from a HDF5 tensor.

        Args:
            loaded_hdf5 (tfio.IOTensor): Tensor from which to load labels

        Returns:
            dict: Mapping between label names and labels
        """
        labels = {}
        for i_label_ds_name, i_label_name in zip(self.label_dataset_names, self.label_names):
            labels[i_label_name] = loaded_hdf5(i_label_ds_name).to_tensor()
        return labels

    def apply_augmentation(self, features: dict, labels: dict) -> Tuple[dict, dict]:
        seed = tf.random.uniform([2], 0, 10000000, dtype=tf.dtypes.int32)
        for i_key, i_value in features.items():
            features[i_key] = self.augmentors[i_key].augment_sample(i_value, seed)

        for i_key, i_value in labels.items():
            if i_key == "MASK":
                labels[i_key] = self.augmentors[i_key].augment_sample(i_value, seed, True)

        return features, labels

    def fits_in_memory(self, used_memory: int = 0):
        ds_size = self.memory_size * PrognosAIs.Constants.MEM_SAFETY_FACTOR
        return ds_size <= IO_utils.get_available_ram(used_memory)

    def setup_caching(
        self, cache_in_memory: Union[bool, str] = PrognosAIs.Constants.AUTO, used_memory: int = 0,
    ) -> None:
        """
        Set up caching of the dataset in RAM.

        Args:
            cache_in_memory (Union[bool, str]): Whether dataset should be cached in memory.
                Defaults to PrognosAIs.Constants.AUTO, in which case the dataset will be
                cached in memory if it fits, otherwise it will not be cached
            used_memory (int): Amount of RAM (in bytes) that is already being used. Defaults to 0.

        Raises:
            ValueError: If an unknown cache setting is requested
        """
        if cache_in_memory == PrognosAIs.Constants.AUTO and self.fits_in_memory(used_memory):
            self.cache_in_memory = True
        elif isinstance(cache_in_memory, bool):
            self.cache_in_memory = cache_in_memory
        elif cache_in_memory not in [PrognosAIs.Constants.AUTO, True, False]:
            err_msg = (
                "Unknown cache in memory setting {cach_set}"
                ", should be either True, False or {auto}"
            ).format(cach_set=cache_in_memory, auto=PrognosAIs.Constants.AUTO,)
            raise ValueError(err_msg)

    def setup_sharding(self, n_workers: int, worker_index: int) -> None:
        """
        Shard the dataset according to the number of workers and worker index

        Args:
            n_workers (int): number of workers
            worker_index (int): worker index
        """
        self.n_workers = n_workers
        self.worker_index = worker_index
        self.shard = True

    def setup_caching_shuffling_steps(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """
        Set-up caching, shuffling and the iteration step in the dataset pipeline.

        This function helps to ensure that caching, shuffling and step limiting is done properly
        and efficiently, no matter where in the dataset pipeline it is included.

        Args:
            dataset (tf.data.Dataset): Datset for which to include the steps

        Returns:
            tf.data.Dataset: Datset with caching, shuffling and iteration steps included
        """
        dataset = dataset.cache()
        # Lets go over the dataset to make sure that we cached everything
        # Otherwise it might be that in the first iteration we do not full iterate
        # Over the dataset, making our cache useless
        logging.info("Caching dataset")
        for _ in dataset:
            pass
        logging.info("Done caching dataset")

        if self.data_augmentation:
            dataset = dataset.repeat(self.augmentation_factor)

        if self.shuffle:
            dataset = dataset.shuffle(
                self.N_samples * self.augmentation_factor, reshuffle_each_iteration=True
            )

        if self.steps < self.possible_steps:
            dataset = dataset.take(self.steps * self.batch_size)

        return dataset

    def get_tf_dataset(
        self, num_parallel_calls: int = tf.data.experimental.AUTOTUNE,
    ) -> tf.data.Dataset:
        """
        Construct a TensorFlow dataset.

        The dataset is constructed based on the settings supplied to the DataGenerator.
        The dataset can then directly be used to train or evaluate a TensorFlow model

        Args:
            num_parallel_calls (int): Number of parallel process to use.
                Defaults to tf.data.experimental.AUTOTUNE.

        Returns:
            tf.data.Dataset: The constructed dataset
        """
        if self.TF_dataset is None:
            dataset = tf.data.Dataset.from_tensor_slices(self.sample_locations)

            if self.shard:
                dataset = dataset.shard(self.n_workers, self.worker_index)

            # If we dont cache the whole dataset in memory, we just cache the
            # file names, and perform as many steps before the data loading
            # as possible to reduce the loading times
            if not self.cache_in_memory:
                dataset = self.setup_caching_shuffling_steps(dataset)

            dataset = dataset.map(self.loader, num_parallel_calls=num_parallel_calls)

            if self.cache_in_memory:
                dataset = self.setup_caching_shuffling_steps(dataset)

            if self.data_augmentation:
                dataset = dataset.map(
                    self.apply_augmentation, num_parallel_calls=num_parallel_calls,
                )

            dataset = dataset.batch(self.batch_size, drop_remainder=self.drop_batch_remainder)

            if self.repeat:
                dataset = dataset.repeat()

            dataset = dataset.prefetch(num_parallel_calls)
            self.TF_dataset = dataset
        return self.TF_dataset

    def get_numpy_iterator(self) -> np.nditer:
        """
        Construct a numpy iterator instead of TensorFlow dataset.

        The numpy iterator will provide exactly the same data as the TensorFlow dataset.
        However, it might be easier to inspect the data when using a numpy iterator
        instead of a TensorFlow dataset

        Returns:
            np.nditer: The dataset
        """
        dataset = self.get_tf_dataset()

        return dataset.as_numpy_iterator()


class Augmentor(object):
    """
    Augmentor
    """

    def __init__(
        self,
        example_sample: tf.Tensor,
        brightness_probability: float = 0,
        brightness_delta: float = 0,
        contrast_probability: float = 0,
        contrast_min_factor: float = 1,
        contrast_max_factor: float = 1,
        flip_probability: float = 0,
        to_flip_axis: Union[int, list] = 0,
        crop_probability: float = 0,
        crop_size: list = None,
        rotate_probability: float = 0,
        max_rotate_angle: float = 0,
        to_rotate_axis: Union[int, list] = 0,
    ) -> None:
        """
        Augmentor to randomly augment the features of a sample.

        Args:
            example_sample (tf.Tensor): Example sample from which settings for augmentation
                will be derived
            brightness_probability (float, optional): Probability of augmenting brightness. Defaults to 0.
            brightness_delta (float, optional): Brightness will be adjusted with value from -delta to delta.
                Defaults to 0.
            contrast_probability (float, optional): Probability of augmenting contrast.
                Defaults to 0.
            contrast_min_factor (float, optional): Minimum contrast adjustment factor.
                Defaults to 1.
            contrast_max_factor (float, optional): Maximum contrast adjustment factor.
                Defaults to 1.
            flip_probability (float, optional): Probability of a random flip. Defaults to 0.
            to_flip_axis (Union[int, list], optional): Axis to flip the feature over. Defaults to 0.
            crop_probability (float, optional): Probability of cropping the feature. Defaults to 0.
            crop_size (list, optional): Size to crop the feature to. Defaults to None.
        """

        self.sample_size = example_sample.shape
        # Minus 1 because last dimension is channel and we dont want to augment
        # those (they will be augmented all the same, not individually)
        self.n_dim = len(example_sample.shape) - 1
        self.to_reduce_axis = tf.range(0, self.n_dim, 1)

        self.brightness_probability = brightness_probability
        self.brightness_delta = brightness_delta

        self.contrast_probability = contrast_probability
        self.contrast_min = contrast_min_factor
        self.contrast_max = contrast_max_factor

        self.flip_probability = flip_probability
        if isinstance(to_flip_axis, list):
            self.to_flip_axis = to_flip_axis
        elif isinstance(to_flip_axis, int):
            self.to_flip_axis = [to_flip_axis]

        self.crop_probability = crop_probability
        if crop_size is None:
            self.crop_size = [0, 0, 0]
        else:
            self.crop_size = crop_size

        self.rotate_probability = rotate_probability
        self.max_rotate_angle = max_rotate_angle
        if isinstance(to_rotate_axis, list):
            self.to_rotate_axis = to_rotate_axis
        else:
            self.to_rotate_axis = [to_rotate_axis]

    def get_seed(self) -> tf.Tensor:
        """
        Get a random seed that can be used to make other operation repeatable.

        Returns:
            tf.Tensor: The seed
        """
        return tf.random.uniform([], 0, tf.dtypes.int32.max - 1, dtype=tf.dtypes.int32)

    def apply_augmentation(self, augmentation_probability: float, seed: tf.Tensor = None) -> bool:
        """
        Whether the the augmentation step should be applied based on the probability.

        Args:
            augmentation_probability (float): The probability with which the step should be applied
            seed (tf.Tensor): Seed to make operation repeatable. Defaults to None.

        Returns:
            bool: Whether the step should be applied
        """
        if seed is not None and isinstance(seed, tf.Tensor):
            return (
                tf.random.stateless_uniform([], seed, 0, 1, dtype=tf.float32)
                < augmentation_probability
            )
        else:
            return tf.random.uniform([], 0, 1, dtype=tf.float32) < augmentation_probability

    def random_brightness(self, sample: tf.Tensor, seed: tf.Tensor = None) -> tf.Tensor:
        """
        Randomly adjusts the brightness of a sample.

        Brightness is adjusted by a constact factor over the whole image, drawn from
        a distribution between -delta and delta as set during the initialization of
        the augmentator.

        Args:
            sample (tf.Tensor): Sample for which to adjust brightness.
            seed (tf.Tensor): Seed to make operation repeatable. Defaults to None.

        Returns:
            tf.Tensor: The augmented sample.
        """
        if self.apply_augmentation(self.brightness_probability, seed):
            sample += tf.random.stateless_uniform(
                [], seed, -self.brightness_delta, self.brightness_delta, dtype=sample.dtype,
            )
        return sample

    def random_contrast(self, sample: tf.Tensor, seed: tf.Tensor = None) -> tf.Tensor:
        """
        Randomly adjust the contrast of a sample.

        The contrast is adjusted by keeping the mean of the sample the same as
        for the original sample, and squeezing or expending the distribution of
        the intensities around the mean. The amount of squeezing or expanding is
        randomly drawn from the minimum and maximum contrast set during initialization.

        Args:
            sample (tf.Tensor): Sample for which to adjust contrast
            seed (tf.Tensor): Seed to make operation repeatable. Defaults to None.

        Returns:
            tf.Tensor: The augmented sample
        """
        if self.apply_augmentation(self.contrast_probability, seed):
            contrast_factor = tf.random.stateless_uniform(
                [], seed, self.contrast_min, self.contrast_max, dtype=sample.dtype
            )
            sample_mean = tf.math.reduce_mean(sample, axis=self.to_reduce_axis)
            sample = (sample - sample_mean) * contrast_factor + sample_mean

        return sample

    def random_flipping(self, sample: tf.Tensor, seed: tf.Tensor = None) -> tf.Tensor:
        """
        Randomly flip the sample along one or multiple axis.

        Args:
            sample (tf.Tensor): Sample for which to apply flipping
            seed (tf.Tensor): Seed to make operation repeatable. Defaults to None.

        Returns:
            tf.Tensor: The augmented sample
        """
        for i_flip_axis in self.to_flip_axis:
            if self.apply_augmentation(self.flip_probability, seed + i_flip_axis):
                sample = tf.reverse(sample, [i_flip_axis])
        return sample

    def random_cropping(self, sample: tf.Tensor, seed: tf.Tensor = None) -> tf.Tensor:
        """
        Randomly crop a part of the sample.

        The crop will have the size of the crop size defined upon initialization of the augmentator.
        The crop will happen for all channels in the same way, but will not crop out channels.
        The location of the crop will be randomly drawn from throughout the whole image.

        Args:
            sample (tf.Tensor): The sample to be cropped
            seed (tf.Tensor): Seed to make operation repeatable. Defaults to None.

        Returns:
            tf.Tensor: The augmented sample
        """
        if self.apply_augmentation(self.crop_probability, seed):
            crop_start = []

            for i_dim in range(self.n_dim):
                dim_seed = seed + i_dim
                crop_start.append(
                    tf.random.stateless_uniform(
                        [],
                        dim_seed,
                        0,
                        tf.shape(sample)[i_dim] - self.crop_size[i_dim],
                        dtype=tf.int32,
                    ),
                )

            # Need to make sure we do not crop channels
            crop_start.append(tf.constant(0))
            crop_size = self.crop_size + [tf.shape(sample)[-1]]

            sample = tf.slice(sample, crop_start, crop_size)

        return sample

    def _random_rotate(self, feature, seed, to_rotate_axis, interpolation_order):
        seed = seed + tf.cast(to_rotate_axis, seed.dtype)
        all_axis = tf.range(0, tf.rank(feature) - 1)
        np.random.seed(seed)
        angle = np.random.uniform(-self.max_rotate_angle, self.max_rotate_angle)

        rotation_axis = np.setdiff1d(all_axis, to_rotate_axis)
        feature = scipy.ndimage.rotate(
            feature, angle, reshape=False, order=interpolation_order, axes=rotation_axis
        )
        return feature

    def _rotate(self, feature, angle, interpolation_order, axis):
        return scipy.ndimage.rotate(
            feature, angle, reshape=False, order=interpolation_order, axes=axis,
        )

    def random_rotate(
        self, feature: tf.Tensor, seed: tf.Tensor = None, interpolation_order: int = 3
    ) -> tf.Tensor:
        all_axis = tf.range(0, tf.rank(feature) - 1)
        for i_to_rotate_axis in self.to_rotate_axis:
            axis_seed = seed + i_to_rotate_axis
            if self.apply_augmentation(self.rotate_probability, axis_seed):
                angle = tf.random.stateless_uniform(
                    [], axis_seed, -self.max_rotate_angle, self.max_rotate_angle, dtype=tf.float32,
                )
                rotation_axis, _ = tf.compat.v1.setdiff1d(all_axis, tf.constant([i_to_rotate_axis]))
                feature = tf.numpy_function(
                    self._rotate,
                    [feature, angle, interpolation_order, rotation_axis],
                    feature.dtype,
                )
        return feature

    def pad_to_original_size(self, sample: tf.Tensor) -> tf.Tensor:
        """
        Pad back a (potentially) augmented sample to its original size.

        Args:
            sample (tf.Tensor): The sample to pad

        Returns:
            tf.Tensor: The padded sample with the same size as before any augmentation steps
        """
        sample_size = tf.shape(sample)

        required_paddings = [
            [tf.math.ceil((m - sample_size[i]) / 2), tf.math.floor((m - sample_size[i]) / 2)]
            for (i, m) in enumerate(self.sample_size)
        ]

        return tf.pad(sample, required_paddings)

    def augment_sample(self, sample: tf.Tensor, seed=None, is_mask=False) -> tf.Tensor:
        """
        Apply random augmentations to the sample based on the config.

        Args:
            sample (tf.Tensor): sample to be augmented

        Returns:
            tf.Tensor: augmented sample
        """
        if not is_mask:
            sample = self.random_brightness(sample, seed)
            sample = self.random_contrast(sample, seed)
        sample = self.random_flipping(sample, seed)
        sample = self.random_cropping(sample, seed)
        if not is_mask:
            sample = self.random_rotate(sample, seed)
            # sample = tf.py_function(func=self.random_rotate, inp=[sample, seed], Tout=tf.float32)
        else:
            sample = self.random_rotate(sample, seed, 0)
            # sample = tf.py_function(func=self.random_rotate, inp=[sample, seed, 0], Tout=tf.uint8)

        return self.pad_to_original_size(sample)
