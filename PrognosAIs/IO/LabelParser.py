import numpy as np
import pandas as pd
import tensorflow as tf

from PrognosAIs.IO import utils


class LabelLoader:
    def __init__(
        self,
        label_file: str,
        filter_missing: bool = False,
        missing_value: int = -1,
        make_one_hot: bool = False,
        new_root_path: str = None,
    ) -> None:
        """
        Create a label loader, that can load the image paths and labels
        from a text file to be used for a data generator

        Args:
            label_file: The label file from which to read the labels
            filter_missing: Whether missing values should be masked when generating one hot labels and class weights
            missing_value: If filter_missing is True, this value is used to mask
            make_one_hot: Whether labels should be transformed to one hot labels
            new_root_path: If you want to move the files, this will be the new root path
        """

        self.label_file = label_file
        self.label_data = pd.read_csv(label_file, sep="\t", header=0, index_col=0)
        self._original_label_data = self.label_data.copy(deep=True)

        self.filter_missing = filter_missing
        self.missing_value = missing_value
        self.make_one_hot = make_one_hot
        self.one_hot_encoded = False
        self.new_root_path = new_root_path

        self.total_weight_sum = 1.0

        if self.new_root_path is not None:
            self.replace_root_path()

        if self.make_one_hot:
            self.encode_labels_one_hot()

        return

    def get_labels(self) -> list:
        """Get all labels of all samples

        Args:
            None
        Returns:
            labels: List of labels
        """
        labels = np.squeeze(self.label_data.values)

        if isinstance(labels, np.ndarray) and labels.size > 1:
            labels = labels.tolist()
        elif isinstance(labels, np.ndarray):
            # Otherwise if it is 1 element it will remove the list,
            # and return only a string
            labels = [labels.tolist()]
        else:
            labels = [labels]

        return labels

    def get_samples(self) -> list:
        """Get all labels of all samples

        Args:
            None
        Returns:
            samples: List of samples
        """

        return self.label_data.index.to_list()

    def get_data(self) -> dict:
        """Get all data from the label file

        Args:
            None
        Returns:
            data: Dictionary mapping each sample to each label
        """
        return self.label_data.to_dict(orient="index")

    def get_label_from_sample(self, sample: str) -> dict:
        """Get label from a sample

        Args:
            sample: The sample from which to get the label
        Returns:
            label: Label of the sample
        """

        return self.label_data.loc[sample].to_dict()

    def get_label_categories(self) -> list:
        """Get categories of labels

        Args:
            None
        Returns:
            label_categories: Category names
        """

        return self.label_data.columns.to_numpy(copy=True).tolist()

    def get_labels_from_category(self, category_name: str) -> list:
        """Get labels of a specific category/class

        Args:
            category_name: Name of the category/class to get

        Returns:
            list: Labels of the category
        """

        return self.label_data[category_name].to_numpy(copy=True).tolist()

    def get_original_labels_from_category(self, category_name: str) -> list:
        """Get original labels of a specific category/class

        Args:
            category_name: Name of the category/class to get

        Returns:
            list: Original labels of the category
        """

        return self._original_label_data[category_name].to_numpy(copy=True).tolist()

    def get_label_category_type(self, category_name: str) -> type:
        """Get the type of a label of a specific category/class

        Args:
            category_name: Name of the category/class to get type of

        Returns:
            type: Type of the labels of the category
        """

        category_label_type = type(self.label_data[category_name][0])

        return category_label_type

    def get_original_label_category_type(self, category_name: str) -> type:
        """Get the original type of a label of a specific category/class

        Args:
            category_name: Name of the category/class to get type of

        Returns:
            type: Type of the labels of the category
        """

        category_label_type = type(self._original_label_data[category_name][0])

        return category_label_type

    def encode_labels_one_hot(self) -> None:
        """Encode sample labels as one hot

        Args:
            None
        Returns:
            None
        """

        if self.one_hot_encoded:
            return

        label_categories = self.get_label_categories()

        for i_label_category in label_categories:
            category_type = self.get_label_category_type(i_label_category)
            if np.issubdtype(category_type, np.integer):
                category_labels = self.get_labels_from_category(i_label_category)
                category_labels = np.asarray(category_labels)
                labels_min = np.amin(category_labels)
                if self.filter_missing and labels_min == self.missing_value:
                    labels_min = np.amin(category_labels[category_labels != self.missing_value])
                N_labels = np.amax(category_labels) - labels_min + 1
                category_labels -= labels_min
                one_hot_labels = tf.one_hot(category_labels, N_labels, dtype=tf.int8).numpy()
                # We need to replace the one hot labels with the value again
                missing_label_index = np.sum(one_hot_labels, axis=-1)
                one_hot_labels[missing_label_index == 0] = self.missing_value
                self.label_data[i_label_category] = one_hot_labels.tolist()

        self.one_hot_encoded = True

        return

    def replace_root_path(self) -> None:
        """Replace the root path of the sample files in case they have been
        moved to a different a different directory.

        Args:
            new_root_path: Path in which the files are now located
        Returns:
            None
        """

        if self.new_root_path is not None:
            samples = self.get_samples()

            new_root_path = utils.normalize_path(self.new_root_path)

            for i_i_sample, i_sample in enumerate(samples):
                old_root_path = utils.get_file_path(i_sample)
                samples[i_i_sample] = i_sample.replace(old_root_path, new_root_path)

            self.label_data.index = samples

            label_categories = self.get_label_categories()
            for i_label_category in label_categories:
                if self.get_label_category_type(i_label_category) is str:
                    category_labels = self.get_labels_from_category(i_label_category)

                    for i_i_label, i_label in enumerate(category_labels):
                        old_root_path = utils.get_file_path(i_label)
                        category_labels[i_i_label] = i_label.replace(old_root_path, new_root_path)

                    self.label_data[i_label_category] = category_labels
            return

    def get_class_weights(self, json_serializable=False) -> dict:
        """ Get class weights for unbalanced labels

        Args:
            None
        Returns:
            Scaled_weights: the weights for each class of each label category, scaled
                such that the total weights*number of samples of each class
                approximates the total number of samples
        """

        out_scaled_weights = {}

        for i_label_category in self.get_label_categories():
            category_type = self.get_original_label_category_type(i_label_category)
            if np.issubdtype(category_type, np.integer):
                category_labels = self.get_original_labels_from_category(i_label_category)
                category_labels = np.asarray(category_labels)
                if self.filter_missing:
                    category_labels = category_labels[category_labels != self.missing_value]
                if self.make_one_hot:
                    category_labels = category_labels - np.amin(category_labels)
                classes, counts = np.unique(category_labels, return_counts=True)
                N_samples = len(category_labels)

                weights = N_samples / (counts * len(classes))
                if json_serializable:
                    classes = [str(i_class) for i_class in classes]
                # Dictionary for each category, with dictionary of weight for each class in that category
                # This allows for easy input with tensorflow
                out_scaled_weights[i_label_category] = dict(zip(classes, weights))
            else:
                out_scaled_weights[i_label_category] = None

        return out_scaled_weights

    def get_number_of_classes_from_category(self, category_name: str) -> int:
        """ Get number of classes for a label category

        Args:
            category_name: Category to get number of classes for
        Returns:
            number_of_classes: The number of classes for the category
        """

        category_type = self.get_original_label_category_type(category_name)
        if np.issubdtype(category_type, np.integer):
            category_labels = self.get_original_labels_from_category(category_name)
            category_labels = np.asarray(category_labels)
            if self.filter_missing:
                category_labels = category_labels[category_labels != self.missing_value]
            classes = np.unique(category_labels)
            number_of_classes = len(classes)
        else:
            number_of_classes = -1

        return number_of_classes

    def get_number_of_classes(self) -> dict:
        """ Get number of classes for all categories

        Args:
            None
        Returns:
            number_of_classes: The number of classes for each category
        """

        number_of_classes = {}

        for i_label_category in self.get_label_categories():
            number_of_classes[i_label_category] = self.get_number_of_classes_from_category(
                i_label_category
            )

        return number_of_classes

    def get_number_of_samples(self) -> int:
        """ Get number of samples

        Args:
            None
        Returns:
            number_of_samples: The number of samples
        """

        return len(self.get_samples())
