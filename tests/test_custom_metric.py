import numpy as np
import PrognosAIs.Model.Metrics
import pytest
import tensorflow
import tensorflow.keras.metrics


CONFIG_MASKEDCATEGORICALACCURACY = {
    "name": "MaskedCategoricalAccuracy_default",
    "loss_config": {},
    "test_input": {
        "y_true": [
            [[1, 0], [0, 1], [1, 0]],
            [[-1, -1], [0, 1], [1, 0]],
            [[-1, -1], [-1, -1], [-1, -1]],
        ],
        "y_pred": [
            [[0.8, 0.2], [0.3, 0.7], [0.1, 0.9]],
            [[0.8, 0.2], [0.3, 0.7], [0.1, 0.9]],
            [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]],
        ],
        "masked": [[False, False, False], [True, False, False], [True, True, True]],
    },
    "expected_output": {
        "metric_name": "MaskedCategoricalAccuracy",
        "class": PrognosAIs.Model.Metrics.MaskedCategoricalAccuracy,
        "unmasked_loss": tensorflow.keras.metrics.CategoricalAccuracy,
    },
}


def pytest_generate_tests(metafunc):
    idlist = []
    argvalues = []
    argnames = ["loss_settings", "test_input", "expected_output"]

    for scenario in metafunc.cls.scenarios:
        argvalues.append(
            ([scenario["loss_config"], scenario["test_input"], scenario["expected_output"]])
        )
        idlist.append(scenario["name"])

    metafunc.parametrize(argnames, argvalues, ids=idlist, scope="class")


class TestCustomLosses:
    scenarios = [CONFIG_MASKEDCATEGORICALACCURACY]

    def test_custom_metric(self, rootdir, loss_settings, test_input, expected_output):
        loss_function = expected_output["class"](**loss_settings)
        true_loss_function = expected_output["unmasked_loss"]()

        y_true = test_input["y_true"]
        y_pred = test_input["y_pred"]
        is_masked = test_input["masked"]

        assert isinstance(loss_function, expected_output["class"])
        # Make sure we can serialize it
        serialized_metric = tensorflow.keras.metrics.serialize(loss_function)
        # And deserialize
        tensorflow.keras.metrics.deserialize(
            serialized_metric,
            custom_objects={expected_output["metric_name"]: expected_output["class"]},
        )

        metric_config = loss_function.get_config()
        expected_output["class"].from_config(metric_config)

        for i_y_true, i_y_pred, i_is_masked in zip(y_true, y_pred, is_masked):
            self.single_test_metric_values(
                i_y_true, i_y_pred, i_is_masked, loss_function, true_loss_function, loss_settings
            )

    def single_test_metric_values(
        self, y_true, y_pred, is_masked, loss_function, true_loss_function, loss_settings
    ):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        is_masked = np.asarray(is_masked)

        y_true_masked = y_true[np.logical_not(is_masked)]
        y_pred_masked = y_pred[np.logical_not(is_masked)]

        true_loss_function.update_state(y_true_masked, y_pred_masked)
        true_nonmasked_loss = true_loss_function.result().numpy()
        loss_function.update_state(y_true, y_pred)
        loss = loss_function.result()
        # loss = loss_function.call(y_true, y_pred)
        assert tensorflow.is_tensor(loss)
        assert tensorflow.rank(loss) == 0
        loss = loss.numpy()

        assert true_nonmasked_loss == pytest.approx(loss)
        assert 0 <= loss <= 1
