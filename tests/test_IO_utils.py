import PrognosAIs.IO.utils as IO_utils
import pytest
import tensorflow as tf


@pytest.mark.gpu
def test_gpu_compute_capability():
    gpu_devices = tf.config.get_visible_devices("GPU")

    result = IO_utils.get_gpu_compute_capability(gpu_devices[0])
    # In the CI pipeline we have a P4000, with compute capability 6.1
    assert result == (6, 1)


@pytest.mark.cpu
def test_gpu_compute_capability_no_gpu():
    # Create a fake device
    gpu_device = tf.config.PhysicalDevice("gpu:1", "GPU")

    result = IO_utils.get_gpu_compute_capability(gpu_device)

    assert result == (0, 0)
