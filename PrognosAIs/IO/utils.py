import importlib
import os
import shutil
import logging
import sys

from pathlib import Path

import numba.cuda
import psutil
import tensorflow as tf

from slurmpie import slurmpie


def create_directory(file_path, exist_ok=True):
    if not os.path.exists(file_path):
        os.makedirs(file_path, exist_ok=exist_ok)


def delete_directory(file_path):
    if os.path.exists(file_path):
        shutil.rmtree(file_path)


def copy_directory(original_directory, out_directory):
    shutil.copytree(original_directory, out_directory, dirs_exist_ok=True)


def get_root_name(file_path):
    return os.path.basename(os.path.normpath(file_path))


def get_file_name_from_full_path(file_path):
    return os.path.basename(os.path.normpath(file_path))


def get_file_name(file_path, file_extension):
    root_name = get_root_name(file_path)
    if file_extension[0] != ".":
        file_extension = ".".join(file_extension)
    return root_name.split(file_extension)[0]


def find_files_with_extension(file_path, file_extension):
    if file_extension[0] == ".":
        file_extension = file_extension[1:]
    return sorted(
        f_path.path
        for f_path in os.scandir(file_path)
        if (f_path.is_file() and file_extension in "".join(Path(f_path.name).suffixes))
    )


def get_parent_directory(file_path):
    return os.path.dirname(os.path.normpath(os.path.abspath(file_path)))


def get_file_path(file_path):
    file_name = get_root_name(file_path)
    file_path = file_path.split(file_name)[0]
    return normalize_path(file_path)


def normalize_path(path):
    if path[-1] == os.sep:
        path = path[:-1]
    return path


def get_number_of_cpus():
    return len(os.sched_getaffinity(0))


def get_subdirectories(root_dir: str) -> list:
    return [f_path.path for f_path in os.scandir(root_dir) if f_path.is_dir()]


def get_available_ram(used_memory: int = 0) -> int:
    """
    Get the available RAM in bytes.

    Returns:
        int: available in RAM in bytes
    """
    slurm_mem = slurmpie.System().get_job_memory()
    if slurm_mem is None:
        available_ram = psutil.virtual_memory().available
    else:
        # Convert from megabytes to bytes (*1024*1024)
        slurm_mem *= 1048576
        available_ram = slurm_mem - used_memory
    return available_ram


def get_dir_size(root_dir):
    """Returns total size of all files in dir (and subdirs)"""
    root_directory = Path(os.path.normpath(root_dir))
    return sum(f.stat().st_size for f in root_directory.glob("**/*") if f.is_file())


def get_gpu_compute_capability(gpu: tf.config.PhysicalDevice) -> tuple:
    try:
        gpu_number = int(gpu.name.split(":")[-1])
        cuda_device = numba.cuda.select_device(gpu_number)
        cuda_capability = cuda_device.compute_capability
        cuda_device.reset()
    except numba.cuda.cudadrv.error.CudaSupportError:
        # We do not actually have a cuda device
        cuda_capability = (0, 0)
    return cuda_capability


def gpu_supports_float16(gpu: tf.config.PhysicalDevice) -> bool:
    gpu_compute_capability = get_gpu_compute_capability(gpu)
    # Float16 support is supported with at least compute capability 5.3
    supports_float16 = (gpu_compute_capability[0] == 5 and gpu_compute_capability[1] >= 3) or (
        gpu_compute_capability[0] > 5
    )
    return supports_float16


def gpu_supports_mixed_precision(gpu: tf.config.PhysicalDevice) -> bool:
    gpu_compute_capability = get_gpu_compute_capability(gpu)
    # Mixed precision has benefits on compute 7.5 and higher
    return gpu_compute_capability[0] >= 7 and gpu_compute_capability[1] >= 5


def get_gpu_devices() -> list:
    return tf.config.list_physical_devices("GPU")


def get_number_of_gpu_devices() -> int:
    return len(tf.config.list_physical_devices("GPU"))


def get_cpu_devices() -> list:
    return tf.config.list_physical_devices("CPU")


def get_number_of_slurm_nodes() -> int:
    if "SLURM_JOB_NUM_NODES" in os.environ:
        number_of_slurm_nodes = int(os.environ["SLURM_JOB_NUM_NODES"])
    else:
        number_of_slurm_nodes = 0

    return number_of_slurm_nodes


def load_module_from_file(module_path):
    if module_path is None:
        return None
    class_name = get_root_name(module_path).split(".")[0]
    module_file_spec = importlib.util.spec_from_file_location(class_name, module_path,)
    module = importlib.util.module_from_spec(module_file_spec)
    module_file_spec.loader.exec_module(module)

    return module


def setup_logger():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s prognosais %(levelname)-1s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
