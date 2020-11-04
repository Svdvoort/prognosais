import tensorflow.keras.optimizers

from PrognosAIs.Model.Parsers import OptimizerParser


CONFIG_SINGLE_TF = {
    "name": "single_optimizer",
    "Settings": {"name": "Adam", "is_custom": False, "settings": None},
    "expected_output": {"class": tensorflow.keras.optimizers.Adam},
}

CONFIG_SINGLE_TF_WITH_SETTINGS = {
    "name": "single_optimizer_settings",
    "Settings": {
        "name": "Adam",
        "is_custom": False,
        "settings": {"learning_rate": 0.001, "beta_1": 0.8, "beta_2": 0.3},
    },
    "expected_output": {"class": tensorflow.keras.optimizers.Adam},
}

CONFIG_SINGLE_TF_WITH_SETTINGS_2 = {
    "name": "single_optimizer_settings",
    "Settings": {
        "name": "Nadam",
        "is_custom": False,
        "settings": {"learning_rate": 0.001, "beta_1": 0.8, "epsilon": 1e-3},
    },
    "expected_output": {"class": tensorflow.keras.optimizers.Nadam},
}


def pytest_generate_tests(metafunc):
    idlist = []
    argvalues = []
    argnames = ["optimizer_config", "expected_output"]

    for scenario in metafunc.cls.scenarios:
        argvalues.append(([scenario["Settings"], scenario["expected_output"]]))
        idlist.append(scenario["name"])

    metafunc.parametrize(argnames, argvalues, ids=idlist, scope="class")


class TestLossParser:
    scenarios = [CONFIG_SINGLE_TF, CONFIG_SINGLE_TF_WITH_SETTINGS, CONFIG_SINGLE_TF_WITH_SETTINGS_2]

    def test_init_parser(self, rootdir, optimizer_config, expected_output):
        optimizer_parser = OptimizerParser(optimizer_config)
        optimizer = optimizer_parser.get_optimizer()

        assert isinstance(optimizer, expected_output["class"])
        this_optimizer_config = optimizer.get_config()

        if optimizer_config["settings"] is not None:
            for i_optimizer_setting in optimizer_config["settings"].keys():
                assert (
                    this_optimizer_config[i_optimizer_setting]
                    == optimizer_config["settings"][i_optimizer_setting]
                )
