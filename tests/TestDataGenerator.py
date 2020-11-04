# define the test cases here
from . import data_tests


class TestDataGenerator:
    def __init__(self):
        self.test_cases = self.get_data_configs()
        print(self.test_cases)

    def get_data_configs(self):
        module = globals().get("data_tests", None)
        test_cases = {}
        if module:
            test_cases = {
                key: value
                for key, value in module.__dict__.items()
                if not (key.startswith("__") or key.startswith("_"))
            }
        return test_cases
