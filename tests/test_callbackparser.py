import copy
import os
import tempfile

import PrognosAIs.Model.Callbacks
import tensorflow.keras.callbacks

from PrognosAIs.Model.Parsers import CallbackParser


tmp = tempfile.mkdtemp()

CONFIG_SINGLE_TF = {
    "name": "Single_TF_callback",
    "Settings": {
        "early_stopping": {
            "name": "EarlyStopping",
            "is_custom": False,
            "settings": {"monitor": "val_loss", "patience": 10, "verbose": 1},
        }
    },
    "expected_output": {"class": [tensorflow.keras.callbacks.EarlyStopping],},
}

CONFIG_MULTI_TF = {
    "name": "Multi_TF_callback",
    "Settings": {
        "early_stopping": {
            "name": "EarlyStopping",
            "is_custom": False,
            "settings": {"monitor": "val_loss", "patience": 10, "verbose": 1},
        },
        "lr_scheduler": {"name": "ReduceLROnPlateau", "is_custom": False, "settings": None},
    },
    "expected_output": {
        "class": [
            tensorflow.keras.callbacks.EarlyStopping,
            tensorflow.keras.callbacks.ReduceLROnPlateau,
        ],
    },
}

CONFIG_LOGGER = {
    "name": "TF_logger",
    "Settings": {
        "parser_settings": {"root_path": tmp},
        "logger": {"name": "CSVLogger", "is_custom": False, "settings": {"filename": "log.csv"}},
    },
    "expected_output": {
        "class": [tensorflow.keras.callbacks.CSVLogger],
        "filename": os.path.join(tmp, "log.csv"),
    },
}

CONFIG_MULTI_LOGGER = {
    "name": "TF_multi_logger",
    "Settings": {
        "parser_settings": {"root_path": tmp},
        "logger": {"name": "CSVLogger", "is_custom": False, "settings": {"filename": "log.csv"}},
        "early_stopping": {
            "name": "EarlyStopping",
            "is_custom": False,
            "settings": {"monitor": "val_loss", "patience": 10, "verbose": 1},
        },
    },
    "expected_output": {
        # Here we switch the classes as CSVLogger always has to go last
        "class": [tensorflow.keras.callbacks.EarlyStopping, tensorflow.keras.callbacks.CSVLogger],
        "order": [1, 0],
        "filename": os.path.join(tmp, "log.csv"),
    },
}


CONFIG_SINGLE_CUSTOM = {
    "name": "Single_custom_callback",
    "Settings": {"timer": {"name": "Timer", "is_custom": True, "settings": None}},
    "expected_output": {"class": [PrognosAIs.Model.Callbacks.Timer],},
}


CONFIG_MIXED_TF_CUSTOM = {
    "name": "Mixed_TF_custom_callback",
    "Settings": {
        "parser_settings": {"root_path": tmp},
        "timer": {"name": "Timer", "is_custom": True, "settings": None},
        "logger": {"name": "CSVLogger", "is_custom": False, "settings": {"filename": "log.csv"}},
        "lr_scheduler": {"name": "ReduceLROnPlateau", "is_custom": False, "settings": None},
    },
    "expected_output": {
        "class": [
            PrognosAIs.Model.Callbacks.Timer,
            tensorflow.keras.callbacks.ReduceLROnPlateau,
            tensorflow.keras.callbacks.CSVLogger,
        ],
        "filename": os.path.join(tmp, "log.csv"),
        "order": [0, 2, 1],
    },
}


def pytest_generate_tests(metafunc):
    idlist = []
    argvalues = []
    argnames = ["callback_config", "expected_output"]

    for scenario in metafunc.cls.scenarios:
        argvalues.append(([scenario["Settings"], scenario["expected_output"]]))
        idlist.append(scenario["name"])

    metafunc.parametrize(argnames, argvalues, ids=idlist, scope="class")


class TestCallbackParser:
    scenarios = [
        CONFIG_SINGLE_TF,
        CONFIG_MULTI_TF,
        CONFIG_LOGGER,
        CONFIG_SINGLE_CUSTOM,
        CONFIG_MULTI_LOGGER,
        CONFIG_MIXED_TF_CUSTOM,
    ]

    def test_parser(self, rootdir, callback_config, expected_output):
        if "parser_settings" in callback_config:
            parser_settings = callback_config.pop("parser_settings")
        else:
            parser_settings = {}

        callback_parser = CallbackParser(callback_config, **parser_settings)
        callbacks = callback_parser.get_callbacks()

        assert isinstance(callbacks, list)
        assert len(callbacks) == len(expected_output["class"])

        true_callback_configs = list(callback_config.values())
        if "order" in expected_output:
            new_true_callback_configs = []
            for i_i in expected_output["order"]:
                new_true_callback_configs.append(true_callback_configs[i_i])
            true_callback_configs = new_true_callback_configs

        for i_callback, i_true_callback_config, i_true_callback in zip(
            callbacks, true_callback_configs, expected_output["class"]
        ):
            assert isinstance(i_callback, i_true_callback)

            if i_true_callback_config["settings"] is not None:
                for i_callback_config_setting in i_true_callback_config["settings"].keys():
                    if i_callback_config_setting in expected_output.keys():
                        true_callback_comparison = expected_output[i_callback_config_setting]
                    else:
                        true_callback_comparison = i_true_callback_config["settings"][
                            i_callback_config_setting
                        ]

                    assert (
                        getattr(i_callback, i_callback_config_setting) == true_callback_comparison
                    )
