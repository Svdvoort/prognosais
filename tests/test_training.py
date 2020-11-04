import os
import tempfile
import types

import PrognosAIs.IO.ConfigLoader
import tensorflow

from PrognosAIs.Model import Trainer


FIXTURE_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "test_data", "HDF5_Data", "Samples"
)

CONFIG_FILE = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "test_data", "test_config.yaml"
)

SAMPLES_DIR_MASK = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "test_data", "HDF5_Data_mask_patches", "Samples"
)

CONFIG_FILE_MASK = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "test_data", "test_config_mask_patches.yaml"
)


def test_init():
    tmp = tempfile.mkdtemp()
    config = PrognosAIs.IO.ConfigLoader.ConfigLoader(CONFIG_FILE)

    trainer = Trainer.Trainer(config, FIXTURE_DIR, tmp)
    assert trainer.train_data_generator is not None


def test_init_from_cli():
    tmp = tempfile.mkdtemp()
    trainer = Trainer.Trainer.init_from_sys_args(
        ["--config", CONFIG_FILE, "--input", FIXTURE_DIR, "--output", tmp]
    )
    assert isinstance(trainer, Trainer.Trainer)
    assert trainer.sample_folder == FIXTURE_DIR


def test_training():
    tmp = tempfile.mkdtemp()
    config = PrognosAIs.IO.ConfigLoader.ConfigLoader(CONFIG_FILE)

    trainer = Trainer.Trainer(config, FIXTURE_DIR, tmp)
    model_save_file = trainer.train_model()

    assert os.path.exists(os.path.join(tmp, "MODEL"))
    assert os.path.exists(model_save_file)

    # Check if we can actually load the model as well
    tensorflow.keras.models.load_model(model_save_file)


def test_training_mask_patches():
    tmp = tempfile.mkdtemp()
    config = PrognosAIs.IO.ConfigLoader.ConfigLoader(CONFIG_FILE_MASK)

    trainer = Trainer.Trainer(config, SAMPLES_DIR_MASK, tmp)
    model_save_file = trainer.train_model()

    assert os.path.exists(os.path.join(tmp, "MODEL"))
    assert os.path.exists(model_save_file)

    # Check if we can actually load the model as well
    tensorflow.keras.models.load_model(model_save_file)
