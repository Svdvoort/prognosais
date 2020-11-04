import os
import shutil

import pytest

from PrognosAIs import Pipeline


CONFIG_FILE = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "test_data", "test_config.yaml"
)

CONFIG_FILE_CUSTOM = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "test_data", "test_config_custom_functions.yaml"
)


OUTPUT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output")


def test_init():
    pipeline = Pipeline.Pipeline(CONFIG_FILE)
    output_folder = pipeline.output_folder
    shutil.rmtree(output_folder)


def test_run_pipeline():
    pipeline = Pipeline.Pipeline(CONFIG_FILE)
    pipeline.start_local_pipeline()
    output_folder = pipeline.output_folder
    shutil.rmtree(output_folder)


def test_run_pipeline_preprocess_only():
    pipeline = Pipeline.Pipeline(CONFIG_FILE, train=False)
    pipeline.start_local_pipeline()
    output_folder = pipeline.output_folder

    assert os.path.exists(os.path.join(output_folder, "Samples"))
    assert not os.path.exists(os.path.join(output_folder, "MODEL"))
    shutil.rmtree(output_folder)


def test_run_pipeline_custom_functions():
    pipeline = Pipeline.Pipeline(CONFIG_FILE_CUSTOM)
    pipeline.start_local_pipeline()
    output_folder = pipeline.output_folder
    shutil.rmtree(output_folder)
