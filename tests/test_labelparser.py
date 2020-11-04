import itertools
import os

import numpy as np
import pytest

from PrognosAIs.IO import LabelParser

from . import TestDataGenerator
from . import data_tests


LabelLoaderTestSettings = {
    "make_one_hot": [None, True, False],
    "new_root_path": [None, os.path.join(data_tests._tmpdir, "test_location/")],
    "filter_missing": [None, True, False],
    "missing_value": [None, -1, 15],
}


# Now we apply all of them
def pytest_generate_tests(metafunc):
    idlist = []
    argvalues = []
    all_loader_settings = list(itertools.product(*metafunc.cls.test_settings.values()))
    # print(list(all_loader_settings))
    loader_setting_names = metafunc.cls.test_settings.keys()
    for scenario in metafunc.cls.scenarios.values():
        label_file = scenario["label_file"]

        argnames = ["label_file", "loader_settings", "true_output"]

        for i_loader_setting in all_loader_settings:
            loader_settings = {}
            for i_setting_name, i_setting in zip(loader_setting_names, i_loader_setting):
                if i_setting is not None:
                    loader_settings[i_setting_name] = i_setting
            # We make the true outputs
            true_output = []

            true_label_index = "labels"
            true_label_category_index = "category_labels"
            if "make_one_hot" in loader_settings and loader_settings["make_one_hot"]:
                true_label_index = "one_hot_" + true_label_index
                true_label_category_index = "one_hot_" + true_label_category_index
            if (
                "filter_missing" in loader_settings
                and loader_settings["filter_missing"]
                and (
                    "missing_value" not in loader_settings
                    or loader_settings["missing_value"]
                    == scenario["expected_output"]["true_missing_value"]
                )
            ):
                if true_label_index + "_filtered" in scenario["expected_output"]:
                    true_label_index = true_label_index + "_filtered"
                    true_label_category_index = true_label_category_index + "_filtered"

            if "new_root_path" in loader_settings:
                if "moved_" + true_label_index in scenario["expected_output"]:
                    true_label_index = "moved_" + true_label_index
                    true_label_category_index = "moved_" + true_label_category_index

            true_output.append(scenario["expected_output"][true_label_index])
            true_output.append(scenario["expected_output"][true_label_category_index])

            if "new_root_path" in loader_settings:
                true_output.append(scenario["expected_output"]["moved_sample_locations"])
            else:
                true_output.append(scenario["expected_output"]["sample_locations"])

            class_weight_index = "class_weights"
            if "make_one_hot" in loader_settings and loader_settings["make_one_hot"]:
                class_weight_index = "one_hot_" + class_weight_index
            if (
                "filter_missing" in loader_settings
                and loader_settings["filter_missing"]
                and (
                    "missing_value" not in loader_settings
                    or loader_settings["missing_value"]
                    == scenario["expected_output"]["true_missing_value"]
                )
            ):
                if class_weight_index + "_filtered" in scenario["expected_output"]:
                    class_weight_index = class_weight_index + "_filtered"

            true_output.append(scenario["expected_output"][class_weight_index])

            number_of_classes_index = "number_of_classes"
            if (
                "filter_missing" in loader_settings
                and loader_settings["filter_missing"]
                and (
                    "missing_value" not in loader_settings
                    or loader_settings["missing_value"]
                    == scenario["expected_output"]["true_missing_value"]
                )
            ):
                if number_of_classes_index + "_filtered" in scenario["expected_output"]:
                    number_of_classes_index = number_of_classes_index + "_filtered"

            true_output.append(scenario["expected_output"][number_of_classes_index])

            argvalues.append(([label_file, loader_settings, true_output]))
            idlist.append(scenario["name"])

    metafunc.parametrize(argnames, argvalues, ids=idlist, scope="class")


class TestLabelLoaderWithScenarios:
    test_settings = LabelLoaderTestSettings
    test_data_generator = TestDataGenerator.TestDataGenerator()
    scenarios = test_data_generator.get_data_configs()

    def approx_assert(self, expected_value, true_value, rel=None, abs=None, nan_ok=False):
        if isinstance(expected_value, list):
            for idx, sub_expected_value in enumerate(expected_value):
                self.approx_assert(sub_expected_value, true_value[idx])
        elif isinstance(expected_value, dict):
            for key, sub_expected_value in expected_value.items():
                assert key in true_value
                self.approx_assert(sub_expected_value, true_value[key])
        elif isinstance(expected_value, str) or expected_value is None:
            assert expected_value == true_value
        else:
            assert expected_value == pytest.approx(true_value)

    def standard_tests_labels(
        self,
        label_loader,
        true_labels,
        true_labels_categories,
        true_samples,
        true_scaled_weights,
        true_number_of_classes,
    ):

        assert label_loader.get_number_of_samples() == len(true_samples)

        # Test labels
        assert type(label_loader.get_labels()) == list
        labels = label_loader.get_labels()
        assert all(isinstance(elem, type(labels[0])) for elem in labels)
        self.approx_assert(label_loader.get_labels(), true_labels)

        # Test samples
        assert type(label_loader.get_samples()) == list
        assert label_loader.get_samples() == true_samples

        # Test labels for each category
        label_categories = label_loader.get_label_categories()

        for i_label_category in label_categories:
            category_labels = label_loader.get_labels_from_category(i_label_category)

            assert type(category_labels) == list
            assert all(isinstance(elem, type(category_labels[0])) for elem in category_labels)

            self.approx_assert(category_labels, true_labels_categories[i_label_category])

        # Test labels per sample
        samples = label_loader.get_samples()
        for i_sample in samples:
            true_sample_index = np.squeeze(np.argwhere(np.asarray(true_samples) == i_sample))
            true_sample_label = {}
            for key, values in true_labels_categories.items():
                true_sample_label[key] = values[true_sample_index]

            assert type(label_loader.get_label_from_sample(i_sample)) == dict

            self.approx_assert(label_loader.get_label_from_sample(i_sample), true_sample_label)

        # Test data getting
        assert type(label_loader.get_data()) == dict

        label_data = label_loader.get_data()
        for i_i_sample, i_sample in enumerate(label_loader.get_samples()):
            for i_label_category in label_loader.get_label_categories():
                self.approx_assert(
                    label_data[i_sample][i_label_category],
                    true_labels_categories[i_label_category][i_i_sample],
                )

        # Test class weights
        class_weights = label_loader.get_class_weights()

        assert type(class_weights) == dict
        self.approx_assert(class_weights, true_scaled_weights)

        # Test number of classes
        assert type(label_loader.get_number_of_classes()) == dict
        self.approx_assert(label_loader.get_number_of_classes(), true_number_of_classes)

        for i_label_category in label_categories:
            self.approx_assert(
                label_loader.get_number_of_classes_from_category(i_label_category),
                true_number_of_classes[i_label_category],
            )

        return

    def test_label_loading(self, rootdir, label_file, loader_settings, true_output):
        label_file = os.path.join(rootdir, label_file)
        LabelParser.LabelLoader(label_file)

    def test_standard_label_loader(self, rootdir, label_file, loader_settings, true_output):
        label_file = os.path.join(rootdir, label_file)
        label_loader = LabelParser.LabelLoader(label_file, **loader_settings)
        self.standard_tests_labels(label_loader, *true_output)

    def test_function_order(self, rootdir, label_file, loader_settings, true_output):
        label_file = os.path.join(rootdir, label_file)
        label_loader = LabelParser.LabelLoader(label_file, **loader_settings)
        label_loader.encode_labels_one_hot()
        label_loader.replace_root_path()
        label_loader.get_class_weights()
        label_loader.get_class_weights()
        label_loader.replace_root_path()
        label_loader.encode_labels_one_hot()
