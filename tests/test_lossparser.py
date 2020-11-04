import PrognosAIs.Model.Losses
import tensorflow.keras.losses

from PrognosAIs.Model.Parsers import LossParser


CONFIG_SINGLE_TF = {
    "name": "Single_TF_loss",
    "Settings": {"name": "BinaryCrossentropy", "is_custom": False, "settings": None},
    "expected_output": {"type": None, "class": tensorflow.keras.losses.BinaryCrossentropy,},
}

CONFIG_MULTI_TF = {
    "name": "Multi_TF_loss",
    "Settings": {
        "label_1": {"name": "BinaryCrossentropy", "is_custom": False, "settings": None},
        "label_2": {"name": "CategoricalCrossentropy", "is_custom": False, "settings": None},
    },
    "expected_output": {
        "type": dict,
        "class": {
            "label_1": tensorflow.keras.losses.BinaryCrossentropy,
            "label_2": tensorflow.keras.losses.CategoricalCrossentropy,
        },
    },
}

CONFIG_SINGLE_CUSTOM = {
    "name": "Single_custom_loss",
    "Settings": {"name": "MaskedCategoricalCrossentropy", "is_custom": True, "settings": None},
    "expected_output": {
        "type": None,
        "class": PrognosAIs.Model.Losses.MaskedCategoricalCrossentropy,
    },
}

CONFIG_MULTI_CUSTOM = {
    "name": "Multicustom_loss",
    "Settings": {
        "label_1": {"name": "MaskedCategoricalCrossentropy", "is_custom": True, "settings": None},
        "label_2": {"name": "MaskedCategoricalCrossentropy", "is_custom": True, "settings": None},
    },
    "expected_output": {
        "type": dict,
        "class": {
            "label_1": PrognosAIs.Model.Losses.MaskedCategoricalCrossentropy,
            "label_2": PrognosAIs.Model.Losses.MaskedCategoricalCrossentropy,
        },
    },
}

CONFIG_MIXED_TF_CUSTOM = {
    "name": "Mixed_TF_custom_loss",
    "Settings": {
        "label_1": {"name": "BinaryCrossentropy", "is_custom": False, "settings": None},
        "label_2": {"name": "MaskedCategoricalCrossentropy", "is_custom": True, "settings": None},
    },
    "expected_output": {
        "type": dict,
        "class": {
            "label_1": tensorflow.keras.losses.BinaryCrossentropy,
            "label_2": PrognosAIs.Model.Losses.MaskedCategoricalCrossentropy,
        },
    },
}

CONFIG_SINGLE_TF_WITH_SETTINGS = {
    "name": "Single_TF_loss_config",
    "Settings": {
        "name": "BinaryCrossentropy",
        "is_custom": False,
        "settings": {"from_logits": True},
    },
    "expected_output": {"type": None, "class": tensorflow.keras.losses.BinaryCrossentropy,},
}

CONFIG_SINGLE_CUSTOM_WITH_SETTINGS = {
    "name": "Single_custom_loss_config",
    "Settings": {
        "name": "MaskedCategoricalCrossentropy",
        "is_custom": True,
        "settings": {"mask_value": 5},
    },
    "expected_output": {
        "type": None,
        "class": PrognosAIs.Model.Losses.MaskedCategoricalCrossentropy,
    },
}


def pytest_generate_tests(metafunc):
    idlist = []
    argvalues = []
    argnames = ["loss_config", "expected_output"]

    for scenario in metafunc.cls.scenarios:
        argvalues.append(([scenario["Settings"], scenario["expected_output"]]))
        idlist.append(scenario["name"])

    metafunc.parametrize(argnames, argvalues, ids=idlist, scope="class")


class TestLossParser:
    scenarios = [
        CONFIG_SINGLE_TF,
        CONFIG_MULTI_TF,
        CONFIG_SINGLE_CUSTOM,
        CONFIG_MULTI_CUSTOM,
        CONFIG_MIXED_TF_CUSTOM,
        CONFIG_SINGLE_TF_WITH_SETTINGS,
        CONFIG_SINGLE_CUSTOM_WITH_SETTINGS,
    ]

    def test_init_parser(self, rootdir, loss_config, expected_output):
        loss_parser = LossParser(loss_config)
        losses = loss_parser.get_losses()

        if expected_output["type"] is None:
            expected_class = expected_output["class"]
            self.single_loss_test(losses, loss_config, expected_class)
        else:
            assert type(losses) == expected_output["type"]
            for i_loss in losses.keys():
                expected_class = expected_output["class"][i_loss]
                self.single_loss_test(losses[i_loss], loss_config[i_loss], expected_class)

    def single_loss_test(self, loss, loss_settings, expected_type):
        assert type(loss) == expected_type
        loss_config = loss.get_config()

        if loss_settings["settings"] is not None:
            setting_names = loss_settings["settings"].keys()
            for i_loss_setting in setting_names:
                assert loss_config[i_loss_setting] == loss_settings["settings"][i_loss_setting]
