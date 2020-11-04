import PrognosAIs.Model.Metrics
import tensorflow.keras.metrics

from PrognosAIs.Model.Parsers import MetricParser


CONFIG_SINGLE_TF = {
    "name": "Single_TF_metric",
    "Settings": {"name": "Accuracy", "is_custom": False, "settings": None},
    "expected_output": {"type": None, "class": tensorflow.keras.metrics.Accuracy,},
}

CONFIG_MULTI_TF = {
    "name": "Multi_TF_metric",
    "Settings": {
        "label_1": {"name": "Accuracy", "is_custom": False, "settings": None},
        "label_2": {"name": "Precision", "is_custom": False, "settings": None},
    },
    "expected_output": {
        "type": dict,
        "class": {
            "label_1": tensorflow.keras.metrics.Accuracy,
            "label_2": tensorflow.keras.metrics.Precision,
        },
    },
}

CONFIG_SINGLE_CUSTOM = {
    "name": "Single_custom_metric",
    "Settings": {"name": "MaskedAUC", "is_custom": True, "settings": None},
    "expected_output": {"type": None, "class": PrognosAIs.Model.Metrics.MaskedAUC},
}

CONFIG_MULTI_CUSTOM = {
    "name": "Multi_custom_metric",
    "Settings": {
        "label_1": {"name": "MaskedAUC", "is_custom": True, "settings": None},
        "label_2": {"name": "MaskedCategoricalAccuracy", "is_custom": True, "settings": None},
    },
    "expected_output": {
        "type": dict,
        "class": {
            "label_1": PrognosAIs.Model.Metrics.MaskedAUC,
            "label_2": PrognosAIs.Model.Metrics.MaskedCategoricalAccuracy,
        },
    },
}

CONFIG_MIXED_TF_CUSTOM = {
    "name": "Mixed_TF_custom_metric",
    "Settings": {
        "label_1": {"name": "Accuracy", "is_custom": False, "settings": None},
        "label_2": {"name": "MaskedCategoricalAccuracy", "is_custom": True, "settings": None},
    },
    "expected_output": {
        "type": dict,
        "class": {
            "label_1": tensorflow.keras.metrics.Accuracy,
            "label_2": PrognosAIs.Model.Metrics.MaskedCategoricalAccuracy,
        },
    },
}

CONFIG_SINGLE_TF_WITH_SETTINGS = {
    "name": "Single_TF_metric_config",
    "Settings": {
        "name": "SensitivityAtSpecificity",
        "is_custom": False,
        "settings": {"specificity": 0.5},
    },
    "expected_output": {"type": None, "class": tensorflow.keras.metrics.SensitivityAtSpecificity,},
}

CONFIG_SINGLE_CUSTOM_WITH_SETTINGS = {
    "name": "Single_custom_metric_config",
    "Settings": {
        "name": "MaskedCategoricalAccuracy",
        "is_custom": True,
        "settings": {"mask_value": 5},
    },
    "expected_output": {"type": None, "class": PrognosAIs.Model.Metrics.MaskedCategoricalAccuracy},
}


def pytest_generate_tests(metafunc):
    idlist = []
    argvalues = []
    argnames = ["metric_config", "expected_output"]

    for scenario in metafunc.cls.scenarios:
        argvalues.append(([scenario["Settings"], scenario["expected_output"]]))
        idlist.append(scenario["name"])

    metafunc.parametrize(argnames, argvalues, ids=idlist, scope="class")


class TestMetricParser:
    scenarios = [
        CONFIG_SINGLE_TF,
        CONFIG_MULTI_TF,
        CONFIG_SINGLE_CUSTOM,
        CONFIG_MULTI_CUSTOM,
        CONFIG_MIXED_TF_CUSTOM,
        CONFIG_SINGLE_TF_WITH_SETTINGS,
        CONFIG_SINGLE_CUSTOM_WITH_SETTINGS,
    ]

    def test_init_parser(self, rootdir, metric_config, expected_output):
        loss_parser = MetricParser(metric_config)
        losses = loss_parser.get_metrics()

        if expected_output["type"] is None:
            assert isinstance(losses, list)
            expected_class = expected_output["class"]
            self.single_metric_test(losses[0], metric_config, expected_class)
        else:
            assert isinstance(losses, expected_output["type"])
            for i_loss in losses.keys():
                expected_class = expected_output["class"][i_loss]
                self.single_metric_test(losses[i_loss], metric_config[i_loss], expected_class)

    def single_metric_test(self, metric, metric_settings, expected_type):
        assert type(metric) == expected_type
        loss_config = metric.get_config()

        if metric_settings["settings"] is not None:
            setting_names = metric_settings["settings"].keys()
            for i_loss_setting in setting_names:
                assert loss_config[i_loss_setting] == metric_settings["settings"][i_loss_setting]
