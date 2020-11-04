import collections
import os

import numpy as np
import pytest
import tensorflow as tf
import scipy

import PrognosAIs.IO.DataGenerator
import PrognosAIs.IO.utils as IO_utils
import PrognosAIs.Constants


SAMPLES_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "test_data", "HDF5_Data", "Samples", "train",
)


def test_init():
    generator = PrognosAIs.IO.DataGenerator.HDF5Generator(SAMPLES_DIR)
    assert generator.N_samples == 6
    assert generator.label_names == [PrognosAIs.Constants.LABEL_INDEX]


def test_get_spec():
    generator = PrognosAIs.IO.DataGenerator.HDF5Generator(SAMPLES_DIR)
    feature_loc = (
        "/" + PrognosAIs.Constants.FEATURE_INDEX + "/" + PrognosAIs.Constants.FEATURE_INDEX
    )
    label_loc = "/" + PrognosAIs.Constants.LABEL_INDEX + "/" + PrognosAIs.Constants.LABEL_INDEX

    result = generator.get_spec()

    assert isinstance(result, dict)
    assert result == {
        feature_loc: tf.TensorSpec([30, 30, 30, 4], dtype=tf.float32, name=feature_loc),
        # Maybe this should be int8? Or otherwsie float for use in training?
        label_loc: tf.TensorSpec([3], dtype=tf.int64, name=label_loc),
    }


def test_get_number_of_channels():
    generator = PrognosAIs.IO.DataGenerator.HDF5Generator(SAMPLES_DIR)

    result = generator.get_number_of_channels()

    assert isinstance(result, dict)
    assert result == {PrognosAIs.Constants.FEATURE_INDEX: 4}


def test_get_input_size():
    generator = PrognosAIs.IO.DataGenerator.HDF5Generator(SAMPLES_DIR)

    result = generator.get_feature_size()

    assert isinstance(result, dict)
    assert PrognosAIs.Constants.FEATURE_INDEX in result
    assert result[PrognosAIs.Constants.FEATURE_INDEX] == pytest.approx(np.asarray([30, 30, 30]))


def test_get_input_dimensionality():
    generator = PrognosAIs.IO.DataGenerator.HDF5Generator(SAMPLES_DIR)
    input_dimensionality = generator.get_feature_dimensionality()
    assert isinstance(input_dimensionality, dict)
    assert input_dimensionality == {PrognosAIs.Constants.FEATURE_INDEX: 3}


def test_get_number_of_classes():
    generator = PrognosAIs.IO.DataGenerator.HDF5Generator(SAMPLES_DIR)
    N_classes = generator.get_number_of_classes()
    assert isinstance(N_classes, dict)
    assert N_classes == {PrognosAIs.Constants.LABEL_INDEX: 3}


def test_get__feature_shape():
    generator = PrognosAIs.IO.DataGenerator.HDF5Generator(SAMPLES_DIR)

    result = generator.get_feature_shape()

    assert isinstance(result, dict)
    assert result == {PrognosAIs.Constants.FEATURE_INDEX: (30, 30, 30, 4)}


def test_get_tf_dataset():
    generator = PrognosAIs.IO.DataGenerator.HDF5Generator(SAMPLES_DIR)

    result = generator.get_tf_dataset()

    assert isinstance(result, tf.data.Dataset)


def test_get_numpy_iterator():
    generator = PrognosAIs.IO.DataGenerator.HDF5Generator(SAMPLES_DIR)
    np_generator = generator.get_numpy_iterator()
    assert isinstance(np_generator, collections.abc.Iterable)


def test_tf_dataset_data_generation():
    generator = PrognosAIs.IO.DataGenerator.HDF5Generator(
        SAMPLES_DIR, batch_size=1, shuffle=False, max_steps=-1,
    )
    tf_dataset = generator.get_tf_dataset()
    N_iterations = 4

    for i_iter in range(N_iterations):
        seen_samples = 0
        for i_batch in tf_dataset:
            # This should always be 2, one sample, one label
            assert len(i_batch) == 2
            assert isinstance(i_batch, tuple)

            assert isinstance(i_batch[0], dict)
            assert isinstance(i_batch[1], dict)
            assert PrognosAIs.Constants.FEATURE_INDEX in i_batch[0]
            assert PrognosAIs.Constants.LABEL_INDEX in i_batch[1]

            sample = i_batch[0][PrognosAIs.Constants.FEATURE_INDEX]
            label = i_batch[1][PrognosAIs.Constants.LABEL_INDEX]

            assert isinstance(sample, tf.Tensor)
            assert isinstance(label, tf.Tensor)

            sample = sample.numpy()
            sample = sample[0]

            label = label.numpy()
            label = label[0]

            if seen_samples == 0:
                assert label == pytest.approx(np.asarray([1, 0, 0]))
            elif seen_samples == 1:
                assert label == pytest.approx(np.asarray([0, 1, 0]))
            else:
                assert label == pytest.approx(np.asarray([0, 0, 1]))
            seen_samples += 1

        assert seen_samples == 6


def test_tf_dataset_label_only():
    generator = PrognosAIs.IO.DataGenerator.HDF5Generator(
        SAMPLES_DIR, batch_size=1, shuffle=False, max_steps=-1, labels_only=True,
    )

    tf_dataset = generator.get_tf_dataset()
    N_iterations = 4

    for i_iter in range(N_iterations):
        seen_samples = 0
        for i_batch in tf_dataset:
            # This should always be 2, one sample, one label
            assert len(i_batch) == 1
            assert isinstance(i_batch, dict)

            assert PrognosAIs.Constants.LABEL_INDEX in i_batch

            label = i_batch[PrognosAIs.Constants.LABEL_INDEX]

            assert isinstance(label, tf.Tensor)

            label = label.numpy()
            label = label[0]

            if seen_samples == 0:
                assert label == pytest.approx(np.asarray([1, 0, 0]))
            elif seen_samples == 1:
                assert label == pytest.approx(np.asarray([0, 1, 0]))
            else:
                assert label == pytest.approx(np.asarray([0, 0, 1]))
            seen_samples += 1

        assert seen_samples == 6


def test_load_hdf5():
    generator = PrognosAIs.IO.DataGenerator.HDF5Generator(SAMPLES_DIR, batch_size=1)
    hdf5_files = IO_utils.find_files_with_extension(SAMPLES_DIR, "hdf5")
    assert len(hdf5_files) == 6
    for i_hdf5_file in hdf5_files:
        i_hdf5_file = tf.convert_to_tensor(i_hdf5_file)
        hdf5_loaded = generator.features_and_labels_loader(i_hdf5_file)
        assert isinstance(hdf5_loaded, tuple)
        assert len(hdf5_loaded) == 2
        sample = hdf5_loaded[0]
        label = hdf5_loaded[1]

        assert isinstance(sample, dict)
        assert isinstance(label, dict)

        assert PrognosAIs.Constants.FEATURE_INDEX in sample
        assert PrognosAIs.Constants.LABEL_INDEX in label

        assert isinstance(sample[PrognosAIs.Constants.FEATURE_INDEX], tf.Tensor)
        assert isinstance(label[PrognosAIs.Constants.LABEL_INDEX], tf.Tensor)


def test_tf_dataset_batch_size():
    to_test_batch_size = 3
    generator = PrognosAIs.IO.DataGenerator.HDF5Generator(
        SAMPLES_DIR, batch_size=to_test_batch_size
    )
    tf_dataset = generator.get_tf_dataset()

    for i_batch in tf_dataset:
        assert i_batch[0][PrognosAIs.Constants.FEATURE_INDEX].shape[0] == to_test_batch_size
        assert i_batch[1][PrognosAIs.Constants.LABEL_INDEX].shape[0] == to_test_batch_size


def test_tf_dataset_shuffle():
    generator = PrognosAIs.IO.DataGenerator.HDF5Generator(SAMPLES_DIR, batch_size=1, shuffle=True)
    tf_dataset = generator.get_tf_dataset()

    labels_first_iter = []
    for i_batch in tf_dataset:
        labels_first_iter.append(i_batch[1][PrognosAIs.Constants.LABEL_INDEX].numpy()[0])

    max_retries = 5

    passed_test = False
    # sometimes we can have the situation where the dataset is the same by accident
    # This should resolves it in most of the cases
    # Best solution would be to set a seed to the datset, but this would require
    # settings just for the testing

    for i_retry in range(max_retries):
        labels_second_iter = []
        for i_batch in tf_dataset:
            labels_second_iter.append(i_batch[1][PrognosAIs.Constants.LABEL_INDEX].numpy()[0])

        if np.asarray(labels_first_iter) != pytest.approx(np.asarray(labels_second_iter)):
            passed_test = True
            break

    assert passed_test


def test_tf_dataset_max_steps():
    generator = PrognosAIs.IO.DataGenerator.HDF5Generator(SAMPLES_DIR, max_steps=3, batch_size=1)
    tf_dataset = generator.get_tf_dataset()

    seen_samples = 0
    for i_batch in tf_dataset:
        seen_samples += 1

    assert seen_samples == 3


def test_memory_caching():
    generator = PrognosAIs.IO.DataGenerator.HDF5Generator(SAMPLES_DIR, max_steps=3, batch_size=1,)
    generator.setup_caching()

    result = generator.get_tf_dataset()

    assert isinstance(result, tf.data.Dataset)


def test_get_feature_metadata():
    generator = PrognosAIs.IO.DataGenerator.HDF5Generator(SAMPLES_DIR, batch_size=1)

    result = generator.get_feature_metadata()

    assert isinstance(result, dict)
    assert PrognosAIs.Constants.FEATURE_INDEX in result
    assert isinstance(result[PrognosAIs.Constants.FEATURE_INDEX], dict)
    assert sorted(result[PrognosAIs.Constants.FEATURE_INDEX].keys()) == [
        "N_channels",
        "dimensionality",
        "direction",
        "dtype",
        "index",
        "origin",
        "original_direction",
        "original_origin",
        "original_size",
        "original_spacing",
        "shape",
        "size",
        "spacing",
    ]


def test_get_feature_metadata_from_sample():
    generator = PrognosAIs.IO.DataGenerator.HDF5Generator(SAMPLES_DIR, batch_size=1)
    sample_file = generator.sample_locations[0]

    result = generator.get_feature_metadata_from_sample(sample_file)

    assert isinstance(result, dict)
    assert PrognosAIs.Constants.FEATURE_INDEX in result
    assert isinstance(result[PrognosAIs.Constants.FEATURE_INDEX], dict)
    assert sorted(result[PrognosAIs.Constants.FEATURE_INDEX].keys()) == [
        "N_channels",
        "dimensionality",
        "direction",
        "dtype",
        "index",
        "origin",
        "original_direction",
        "original_origin",
        "original_size",
        "original_spacing",
        "shape",
        "size",
        "spacing",
    ]


# ===============================================================
# Augmentation test
# ===============================================================


def test_augmentor_init():
    sample = tf.random.uniform(shape=[5, 5, 5, 2])

    augmentor = PrognosAIs.IO.DataGenerator.Augmentor(sample)

    assert isinstance(augmentor, PrognosAIs.IO.DataGenerator.Augmentor)
    assert augmentor.n_dim == 3
    assert isinstance(augmentor.sample_size, tf.TensorShape)
    assert augmentor.sample_size == sample.shape


def test_seeding():
    sample = tf.random.uniform(shape=[5, 5, 5, 2])
    augmentor = PrognosAIs.IO.DataGenerator.Augmentor(sample)

    seed = augmentor.get_seed()

    assert isinstance(seed, tf.Tensor)
    assert tf.rank(seed) == 0
    assert seed.dtype == tf.dtypes.int32


def test_augmentation_step_probability_always():
    sample = tf.random.uniform(shape=[5, 5, 5, 2])
    augmentor = PrognosAIs.IO.DataGenerator.Augmentor(sample)
    n_trials = 100
    do_augmentation_step = []

    for _ in range(n_trials):
        do_augmentation_step.append(augmentor.apply_augmentation(1))

    assert all(do_augmentation_step)


def test_augmentation_step_probability_never():
    sample = tf.random.uniform(shape=[5, 5, 5, 2])
    augmentor = PrognosAIs.IO.DataGenerator.Augmentor(sample)
    n_trials = 100
    do_augmentation_step = []

    for _ in range(n_trials):
        do_augmentation_step.append(augmentor.apply_augmentation(0))

    assert not any(do_augmentation_step)


def test_augmentation_step_probability_half():
    sample = tf.random.uniform(shape=[5, 5, 5, 2])
    augmentor = PrognosAIs.IO.DataGenerator.Augmentor(sample)
    n_trials = 5000
    do_augmentation_step = []

    for _ in range(n_trials):
        do_augmentation_step.append(augmentor.apply_augmentation(0.5))

    # Need to give a bit of margin because will never come out perfectly to 0.5
    assert np.count_nonzero(do_augmentation_step) / n_trials == pytest.approx(0.5, abs=0.05)


def test_random_brightness():
    sample = tf.random.uniform(shape=[5, 5, 5, 2])
    augmentor = PrognosAIs.IO.DataGenerator.Augmentor(
        sample, brightness_probability=1, brightness_delta=0.3
    )

    # Seed of 1317 will give brightness delta of 0.08449766
    out_sample = augmentor.random_brightness(sample, seed=[1317, 1317])

    assert isinstance(out_sample, tf.Tensor)
    assert tf.shape(out_sample).numpy() == pytest.approx(tf.shape(sample).numpy())
    assert out_sample.dtype == sample.dtype
    diff = out_sample.numpy() - sample.numpy()
    assert np.allclose(diff, diff[0])
    assert diff.flatten()[0] != 0
    assert out_sample.numpy() == pytest.approx(sample.numpy() + 0.08449766)


def test_random_contrast():
    sample = tf.random.uniform(shape=[5, 5, 5, 2])
    augmentor = PrognosAIs.IO.DataGenerator.Augmentor(
        sample, contrast_probability=1, contrast_min_factor=0.5, contrast_max_factor=1.5
    )

    out_sample = augmentor.random_contrast(sample, seed=[1317, 1317])

    assert isinstance(out_sample, tf.Tensor)
    assert tf.shape(out_sample).numpy() == pytest.approx(tf.shape(sample).numpy())
    assert out_sample.dtype == sample.dtype
    assert np.mean(out_sample.numpy()) == pytest.approx(np.mean(sample.numpy()))
    assert np.mean(out_sample.numpy()[..., 0]) == pytest.approx(np.mean(sample.numpy()[..., 0]))
    assert np.mean(out_sample.numpy()[..., 1]) == pytest.approx(np.mean(sample.numpy()[..., 1]))
    assert out_sample.numpy() != pytest.approx(sample.numpy())


def test_random_flipping_single_axis():
    sample = tf.random.uniform(shape=[5, 5, 5, 2])
    augmentor = PrognosAIs.IO.DataGenerator.Augmentor(sample, flip_probability=1, to_flip_axis=0)

    out_sample = augmentor.random_flipping(sample, seed=np.asarray([1317, 1317]))

    assert isinstance(out_sample, tf.Tensor)
    assert tf.shape(out_sample).numpy() == pytest.approx(tf.shape(sample).numpy())
    assert out_sample.dtype == sample.dtype
    assert out_sample.numpy() == pytest.approx(np.flip(sample.numpy(), 0))


def test_random_flipping_multi_axis():
    sample = tf.random.uniform(shape=[5, 5, 5, 2])
    augmentor = PrognosAIs.IO.DataGenerator.Augmentor(
        sample, flip_probability=1, to_flip_axis=[0, 1]
    )

    out_sample = augmentor.random_flipping(sample, seed=np.asarray([1317, 1317]))

    assert isinstance(out_sample, tf.Tensor)
    assert tf.shape(out_sample).numpy() == pytest.approx(tf.shape(sample).numpy())
    assert out_sample.dtype == sample.dtype
    # make sure that the order of flipping doesnt matter
    assert out_sample.numpy() == pytest.approx(np.flip(np.flip(sample.numpy(), 1), 0))
    assert out_sample.numpy() == pytest.approx(np.flip(np.flip(sample.numpy(), 0), 1))


def test_random_cropping():
    sample = tf.random.uniform(shape=[50, 50, 50, 2])
    crop_size = [30, 30, 30]
    augmentor = PrognosAIs.IO.DataGenerator.Augmentor(
        sample, crop_probability=1, crop_size=crop_size
    )

    # Seed of 1317 will give us a crop start of [3, 3, 7, 0]
    out_sample = augmentor.random_cropping(sample, seed=np.asarray([1317, 1317]))

    assert isinstance(out_sample, tf.Tensor)
    assert tf.shape(out_sample).numpy() == pytest.approx(np.asarray([30, 30, 30, 2]))
    assert out_sample.dtype == sample.dtype
    assert out_sample.numpy() == pytest.approx(sample.numpy()[3:33, 3:33, 7:37, :])


def test_pad_to_original_size():
    sample = tf.random.uniform(shape=[50, 50, 50, 2])
    crop_size = [30, 30, 30, 2]
    crop_start = [5, 5, 5, 0]
    padding_size = [10, 10, 10, 0]
    augmentor = PrognosAIs.IO.DataGenerator.Augmentor(sample)
    cropped_sample = tf.slice(sample, crop_start, crop_size)
    pad_indices = tuple(
        [slice(*i) for i in zip(padding_size, np.asarray(padding_size) + np.asarray(crop_size))]
    )
    zero_indices_left = tuple([slice(*i) for i in zip([0] * len(pad_indices), padding_size)])
    zero_indices_right = tuple(
        [
            slice(*i)
            for i in zip(np.asarray(padding_size) + np.asarray(crop_size), tf.shape(sample).numpy())
        ]
    )
    crop_indices = tuple(
        [slice(*i) for i in zip(crop_start, np.asarray(crop_start) + np.asarray(crop_size))]
    )

    out_sample = augmentor.pad_to_original_size(cropped_sample)

    assert isinstance(out_sample, tf.Tensor)
    assert tf.shape(out_sample).numpy() == pytest.approx(tf.shape(sample).numpy())
    assert out_sample.numpy()[pad_indices] == pytest.approx(sample.numpy()[crop_indices])
    assert out_sample.numpy()[zero_indices_left] == pytest.approx(0)
    assert out_sample.numpy()[zero_indices_right] == pytest.approx(0)


def test_random_rotate():
    sample = tf.random.uniform(shape=[50, 50, 50, 2])
    augmentor = PrognosAIs.IO.DataGenerator.Augmentor(
        sample, rotate_probability=1, max_rotate_angle=90, to_rotate_axis=2
    )
    np_sample = sample.numpy()
    rotated_np_sample = scipy.ndimage.rotate(np_sample, 54.804337, reshape=False)

    # Seed of 1317 will give us an angle of 54.804337
    out_sample = augmentor.augment_sample(sample, seed=np.asarray([1317, 1317]))

    assert isinstance(out_sample, tf.Tensor)
    assert tf.shape(out_sample).numpy() == pytest.approx(np.asarray([50, 50, 50, 2]))
    assert out_sample.dtype == sample.dtype
    assert out_sample.numpy() == pytest.approx(rotated_np_sample, abs=1e-5)
    assert out_sample.numpy() != pytest.approx(sample.numpy())


def test_random_rotate_mask():
    sample = tf.random.uniform(shape=[50, 50, 50, 2])
    augmentor = PrognosAIs.IO.DataGenerator.Augmentor(
        sample, rotate_probability=1, max_rotate_angle=90, to_rotate_axis=2
    )
    np_sample = sample.numpy()
    rotated_np_sample = scipy.ndimage.rotate(np_sample, 54.804337, reshape=False, order=0)

    # Seed of 1317 will give us an angle of 54.804337
    out_sample = augmentor.augment_sample(sample, seed=np.asarray([1317, 1317]), is_mask=True)

    assert isinstance(out_sample, tf.Tensor)
    assert tf.shape(out_sample).numpy() == pytest.approx(np.asarray([50, 50, 50, 2]))
    assert out_sample.dtype == sample.dtype
    assert out_sample.numpy() == pytest.approx(rotated_np_sample, abs=1e-5)
    assert out_sample.numpy() != pytest.approx(sample.numpy())


# ===============================================================
# Datagenerator with augmentation
# ===============================================================


def test_tf_dataset_data_augmentation_init():
    generator = PrognosAIs.IO.DataGenerator.HDF5Generator(
        SAMPLES_DIR, batch_size=1, shuffle=False, max_steps=-1,
    )
    generator.setup_augmentation(
        augmentation_factor=1, augmentation_settings={"flip_probability": 0.5, "to_flip_axis": 1}
    )

    result = generator.get_tf_dataset()

    assert isinstance(result, tf.data.Dataset)
