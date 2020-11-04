import os

import pytest


@pytest.fixture
def rootdir():
    return os.path.dirname(os.path.abspath(__file__))


def pytest_addoption(parser):
    parser.addoption("--gpu", action="store_true", default=False, help="run GPU tests")
    parser.addoption("--cpu", action="store_true", default=False, help="run CPU tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: mark test as needs gpu to run")
    config.addinivalue_line("markers", "cpu: mark test as only runs on cpu")


def pytest_collection_modifyitems(config, items):
    skip_gpu = pytest.mark.skip(reason="need --gpu option to run")
    skip_cpu = pytest.mark.skip(reason="need --cpu option to run")

    for item in items:
        if "gpu" in item.keywords and not config.getoption("--gpu"):
            item.add_marker(skip_gpu)
        if "cpu" in item.keywords and not config.getoption("--cpu"):
            item.add_marker(skip_cpu)
