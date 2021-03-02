import copy
import itertools
import os
import tempfile

import h5py
import numpy as np
import PrognosAIs.IO.Configs
import pytest
import SimpleITK as sitk
import tensorflow as tf
import yaml

from PrognosAIs.IO import ConfigLoader
from PrognosAIs.Preprocessing.Preprocessors import BatchPreprocessor
from PrognosAIs.Preprocessing.Preprocessors import SingleSamplePreprocessor
from PrognosAIs.Preprocessing.Samples import NIFTISample
import PrognosAIs.Constants


FIXTURE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_data", "Nifti_Data",)

NIFTI_FILES = pytest.mark.datafiles(
    os.path.join(FIXTURE_DIR, "Subject-000"),
    os.path.join(FIXTURE_DIR, "Subject-001"),
    os.path.join(FIXTURE_DIR, "Subject-002"),
    keep_top_dir=True,
)

CONFIG_FILE = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "test_data", "test_config.yaml"
)


def get_samples(datafiles):
    samples = []
    for i_sample_location in datafiles.listdir():
        samples.append(
            NIFTISample(
                root_path=i_sample_location, extension_keyword=".nii.gz", mask_keyword="MASK",
            )
        )
    return samples


@NIFTI_FILES
def test_init(datafiles):
    with open(CONFIG_FILE, "r") as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    samples = get_samples(datafiles)
    original_config = copy.deepcopy(config)
    preprocessor = SingleSamplePreprocessor(samples[0], config["preprocessing"])
    assert preprocessor.masking_config.perform_step
    assert preprocessor.bias_field_correcting_config.perform_step
    assert not preprocessor.bias_field_correcting_config.perform_step_on_patch
    assert preprocessor.bias_field_correcting_config.perform_step_on_image
    assert preprocessor.normalizing_config.perform_step_on_image
    assert not preprocessor.normalizing_config.perform_step_on_patch
    assert preprocessor.rejecting_config.perform_step_on_patch
    assert not preprocessor.rejecting_config.perform_step_on_image
    # Make sure we didnt change the original config, as this would give problems later on
    # when we do bath preprocssing
    assert config["preprocessing"] == original_config["preprocessing"]


@NIFTI_FILES
def test_pipeline_building(datafiles):
    samples = get_samples(datafiles)

    with open(CONFIG_FILE, "r") as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.SafeLoader)

    preprocessor = SingleSamplePreprocessor(samples[0], config["preprocessing"])
    pipeline = preprocessor.build_pipeline()

    assert isinstance(pipeline, list)
    assert pipeline[0] == preprocessor.multi_dimension_extracting


@NIFTI_FILES
def test_pipeline_configuration(datafiles):
    samples = get_samples(datafiles)

    with open(CONFIG_FILE, "r") as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    config["preprocessing"]["general"] = {"pipeline": ["resampling", "saving"]}

    preprocessor = SingleSamplePreprocessor(samples[0], config["preprocessing"])
    pipeline = preprocessor.build_pipeline()

    assert isinstance(pipeline, list)
    assert pipeline == [preprocessor.resampling, preprocessor.saving]


@NIFTI_FILES
def test_pipeline_applying(datafiles):
    with open(CONFIG_FILE, "r") as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.SafeLoader)

    tmp = tempfile.mkdtemp()

    samples = get_samples(datafiles)
    for i_sample in samples:
        preprocessor = SingleSamplePreprocessor(
            i_sample, config["preprocessing"], output_directory=tmp
        )
        preprocessor.apply_pipeline()


# ===============================================================
# Dimensionality extraction
# ===============================================================


def test_single_image_from_4D_sequence_extraction():
    sample = np.ones([4, 30, 30, 30])
    sample[0, ...] = 500
    sample = sitk.GetImageFromArray(sample, isVector=False)

    extracted_image = SingleSamplePreprocessor._get_first_image_from_sequence(sample, 3)

    assert isinstance(extracted_image, sitk.Image)
    image_size = extracted_image.GetSize()
    assert len(image_size) == 3
    assert image_size == (30, 30, 30)

    image_arr = sitk.GetArrayFromImage(extracted_image)
    assert image_arr == pytest.approx(500)


def test_single_image_from_3D_sequence_extraction():
    sample = np.ones([30, 30, 30])
    sample[0, ...] = 500
    sample = sitk.GetImageFromArray(sample,)

    extracted_image = SingleSamplePreprocessor._get_first_image_from_sequence(sample, 2)

    assert isinstance(extracted_image, sitk.Image)
    image_size = extracted_image.GetSize()
    assert len(image_size) == 2
    assert image_size == (30, 30)

    image_arr = sitk.GetArrayFromImage(extracted_image)
    assert image_arr == pytest.approx(500)


def test_all_images_from_4D_sequence_extraction():
    sample = np.zeros([4, 30, 30, 30])
    sample[1, :] = 1
    sample[2, :] = 2
    sample[3, :] = 3
    sample = sitk.GetImageFromArray(sample, isVector=False)

    extracted_images = SingleSamplePreprocessor._get_all_images_from_sequence(sample, 3)
    assert isinstance(extracted_images, list)
    assert len(extracted_images) == 4
    for i_i_image, i_image in enumerate(extracted_images):
        assert i_image.GetSize() == (30, 30, 30)
        assert isinstance(i_image, sitk.Image)
        assert sitk.GetArrayFromImage(i_image) == pytest.approx(i_i_image)


def test_all_images_from_3D_sequence_extraction():
    sample = np.zeros([4, 30, 30])
    sample[1, :] = 1
    sample[2, :] = 2
    sample[3, :] = 3
    sample = sitk.GetImageFromArray(sample)

    extracted_images = SingleSamplePreprocessor._get_all_images_from_sequence(sample, 2)
    assert isinstance(extracted_images, list)
    assert len(extracted_images) == 4
    for i_i_image, i_image in enumerate(extracted_images):
        assert isinstance(i_image, sitk.Image)
        assert i_image.GetSize() == (30, 30)
        assert sitk.GetArrayFromImage(i_image) == pytest.approx(i_i_image)


@NIFTI_FILES
def test_multi_dimension_extracting(datafiles):
    samples = get_samples(datafiles)
    for i_sample in samples:
        preprocessor = SingleSamplePreprocessor(
            i_sample,
            {
                "multi_dimension_extracting": {
                    "extraction_type": "first",
                    "max_number_of_dimensions": 2,
                }
            },
        )
        preprocessor.multi_dimension_extracting()

        example_channel = preprocessor.sample.get_example_channel()
        example_mask = preprocessor.sample.get_example_mask()

        assert example_channel.GetSize() == (30, 30)
        assert not preprocessor.sample.has_patches
        assert example_mask.GetSize() == (30, 30, 30)

        preprocessor = SingleSamplePreprocessor(
            i_sample,
            {
                "multi_dimension_extracting": {
                    "extraction_type": "first",
                    "max_number_of_dimensions": 2,
                    "extract_masks": True,
                }
            },
        )
        preprocessor.multi_dimension_extracting()

        example_channel = preprocessor.sample.get_example_channel()
        example_mask = preprocessor.sample.get_example_mask()

        assert example_channel.GetSize() == (30, 30)
        assert example_mask.GetSize() == (30, 30)
        assert not preprocessor.sample.has_patches

        preprocessor = SingleSamplePreprocessor(
            i_sample,
            {
                "multi_dimension_extracting": {
                    "extraction_type": "all",
                    "max_number_of_dimensions": 2,
                    "extract_masks": True,
                }
            },
        )
        preprocessor.multi_dimension_extracting()

        example_channel = preprocessor.sample.get_example_channel()
        example_mask = preprocessor.sample.get_example_mask()

        assert example_channel.GetSize() == (30, 30)
        assert example_mask.GetSize() == (30, 30)
        assert preprocessor.sample.has_patches
        assert preprocessor.sample.number_of_patches == 30


# ===============================================================
# Masking
# ===============================================================


@NIFTI_FILES
def test_background_masking(datafiles):
    samples = get_samples(datafiles)

    mask = np.zeros([30, 30, 30])
    mask[0, :, :] = 1
    mask = sitk.GetImageFromArray(mask)
    mask = sitk.Cast(mask, sitk.sitkUInt8)

    for i_sample in samples:
        example_channel = sitk.GetArrayFromImage(i_sample.get_example_channel())
        preprocessor = SingleSamplePreprocessor(i_sample, {})

        preprocessor.mask_background(mask)
        example_channel = sitk.GetArrayFromImage(preprocessor.sample.get_example_channel())
        assert example_channel[0, 0, 0] == pytest.approx(100)
        assert example_channel[0, 1:, 1:] == pytest.approx(50)

        # Before this values was 250
        assert example_channel[1, 1, 1] == pytest.approx(0)


@NIFTI_FILES
def test_background_masking_value(datafiles):
    samples = get_samples(datafiles)

    mask = np.zeros([30, 30, 30])
    mask[0, :, :] = 1
    mask = sitk.GetImageFromArray(mask)
    mask = sitk.Cast(mask, sitk.sitkUInt8)

    for i_sample in samples:
        preprocessor = SingleSamplePreprocessor(i_sample, {})

        preprocessor.mask_background(mask, background_value=-15)
        example_channel = sitk.GetArrayFromImage(preprocessor.sample.get_example_channel())
        assert example_channel[0, 0, 0] == pytest.approx(100)
        assert example_channel[0, 1:, 1:] == pytest.approx(50)

        # Before this values was 250
        assert example_channel[1, 1, 1] == pytest.approx(-15)


@NIFTI_FILES
def test_crop_to_mask(datafiles):
    samples = get_samples(datafiles)

    mask = np.ones([6, 10, 13])
    mask = np.pad(mask, ((0, 24), (1, 20), (8, 9)))
    mask = sitk.GetImageFromArray(mask)
    mask = sitk.Cast(mask, sitk.sitkUInt8)

    for i_sample in samples:
        true_example = sitk.GetArrayFromImage(i_sample.get_example_channel())
        preprocessor = SingleSamplePreprocessor(i_sample, {})

        preprocessor.crop_to_mask(mask)
        example_channel = sitk.GetArrayFromImage(preprocessor.sample.get_example_channel())
        assert example_channel.shape == pytest.approx([6, 10, 13])

        example_mask = sitk.GetArrayFromImage(preprocessor.sample.get_example_mask())
        assert example_mask.shape == pytest.approx([6, 10, 13])

        assert example_channel == pytest.approx(true_example[0:6, 10:20, 8:21])
        assert example_channel[0, 1:, 1:] == pytest.approx(50)

        assert example_mask[:, 0, :] == pytest.approx(105)


@NIFTI_FILES
def test_masking(datafiles):
    samples = get_samples(datafiles)
    for i_sample in samples:
        preprocessor = SingleSamplePreprocessor(
            i_sample,
            {
                "masking": {
                    "mask_file": os.path.join(FIXTURE_DIR, "ATLAS", "atlas_mask.nii.gz"),
                    "mask_background": True,
                    "background_value": -25,
                }
            },
        )

        preprocessor.masking()

        example_channel = sitk.GetArrayFromImage(preprocessor.sample.get_example_channel())
        example_mask = sitk.GetArrayFromImage(preprocessor.sample.get_example_mask())
        assert preprocessor.sample.get_example_channel().GetSize() == (30, 30, 30)
        assert preprocessor.sample.get_example_mask().GetSize() == (30, 30, 30)

        assert example_channel[20:, 20:, 20:] == pytest.approx(-25)
        assert example_mask[20:, 20:, 20:] == pytest.approx(0)

        preprocessor = SingleSamplePreprocessor(
            i_sample,
            {
                "masking": {
                    "mask_file": os.path.join(FIXTURE_DIR, "ATLAS", "atlas_mask.nii.gz"),
                    "crop_to_mask": True,
                }
            },
        )

        preprocessor.masking()

        example_channel = sitk.GetArrayFromImage(preprocessor.sample.get_example_channel())
        example_mask = sitk.GetArrayFromImage(preprocessor.sample.get_example_mask())
        assert preprocessor.sample.get_example_channel().GetSize() == (20, 20, 20)
        assert preprocessor.sample.get_example_mask().GetSize() == (20, 20, 20)

        preprocessor = SingleSamplePreprocessor(
            i_sample,
            {
                "masking": {
                    "mask_file": os.path.join(FIXTURE_DIR, "ATLAS", "atlas_mask.nii.gz"),
                    "mask_background": True,
                    "process_masks": False,
                    "background_value": -25,
                }
            },
        )

        preprocessor.masking()

        example_channel = sitk.GetArrayFromImage(preprocessor.sample.get_example_channel())
        example_mask = sitk.GetArrayFromImage(preprocessor.sample.get_example_mask())
        assert preprocessor.sample.get_example_channel().GetSize() == (30, 30, 30)
        assert preprocessor.sample.get_example_mask().GetSize() == (30, 30, 30)
        assert example_channel[20:, 20:, 20:] == pytest.approx(-25)
        assert example_mask[20:, 20:, 20:] == pytest.approx(0)


# ===============================================================
# Resampling
# ===============================================================


@NIFTI_FILES
def test_resampling(datafiles):
    samples = get_samples(datafiles)
    for i_sample in samples:
        preprocessor = SingleSamplePreprocessor(
            i_sample, {"resampling": {"resample_size": [123, 30, 12]}}
        )
        preprocessor.resampling()

        i_sample = preprocessor.sample
        example_channel = i_sample.get_example_channel()
        assert example_channel.GetSize() == pytest.approx([123, 30, 12])

        example_mask = i_sample.get_example_mask()
        assert example_mask.GetSize() == pytest.approx([123, 30, 12])

        example_mask = sitk.GetArrayFromImage(example_mask)
        mask_values = np.unique(example_mask)

        assert len(mask_values) == 3
        assert all(
            [np.issubdtype(i_mask_value, np.unsignedinteger) for i_mask_value in mask_values]
        )
        assert mask_values == pytest.approx([0, 1, 105])


# ===============================================================
# Normalizing
# ===============================================================


def test_mask_normalization_consecutively():
    mask = np.zeros([30, 30, 30])
    mask[5, 5, 5] = 1
    mask = sitk.Cast(sitk.GetImageFromArray(mask), sitk.sitkUInt64)

    out_mask = SingleSamplePreprocessor._make_consecutive_mask(mask)
    out_mask = sitk.GetArrayFromImage(out_mask)

    assert np.unique(out_mask) == pytest.approx(np.asarray([0, 1]))

    mask = np.zeros([30, 30, 30])
    mask[5, 5, 5] = 1
    mask[6, 6, 6] = 15
    mask = sitk.Cast(sitk.GetImageFromArray(mask), sitk.sitkUInt64)

    out_mask = SingleSamplePreprocessor._make_consecutive_mask(mask)
    out_mask = sitk.GetArrayFromImage(out_mask)

    assert np.unique(out_mask) == pytest.approx(np.asarray([0, 1, 2]))

    mask = np.zeros([30, 30, 30])
    mask[5, 5, 5] = 2
    mask[6, 6, 6] = 15
    mask = sitk.Cast(sitk.GetImageFromArray(mask), sitk.sitkUInt64)

    out_mask = SingleSamplePreprocessor._make_consecutive_mask(mask)
    out_mask = sitk.GetArrayFromImage(out_mask)

    assert np.unique(out_mask) == pytest.approx(np.asarray([0, 1, 2]))

    mask = np.zeros([30, 30, 30])
    mask[5, 5, 5] = 2
    mask[6, 6, 6] = 15
    mask[7, 7, 7] = 25
    mask[8, 8, 8] = 3
    mask = sitk.Cast(sitk.GetImageFromArray(mask), sitk.sitkUInt64)

    out_mask = SingleSamplePreprocessor._make_consecutive_mask(mask)
    out_mask = sitk.GetArrayFromImage(out_mask)

    assert np.unique(out_mask) == pytest.approx(np.asarray([0, 1, 2, 3, 4]))
    assert out_mask[5, 5, 5] == 1
    assert out_mask[6, 6, 6] == 3
    assert out_mask[7, 7, 7] == 4
    assert out_mask[8, 8, 8] == 2

    # Make sure it even works for negative masks
    mask = np.zeros([30, 30, 30])
    mask[5, 5, 5] = -5
    mask[6, 6, 6] = 15
    mask = sitk.Cast(sitk.GetImageFromArray(mask), sitk.sitkInt64)

    out_mask = SingleSamplePreprocessor._make_consecutive_mask(mask)
    out_mask = sitk.GetArrayFromImage(out_mask)

    assert np.unique(out_mask) == pytest.approx(np.asarray([0, 1, 2]))
    assert out_mask[5, 5, 5] == 1
    assert out_mask[6, 6, 6] == 2


def test_mask_normalization_collapse():
    mask = np.zeros([30, 30, 30])
    mask[5, 5, 5] = 1
    mask = sitk.Cast(sitk.GetImageFromArray(mask), sitk.sitkUInt64)

    out_mask = SingleSamplePreprocessor._collapse_mask(mask)
    out_mask = sitk.GetArrayFromImage(out_mask)

    assert np.unique(out_mask) == pytest.approx(np.asarray([0, 1]))
    assert out_mask[5, 5, 5] == 1

    mask = np.zeros([30, 30, 30])
    mask[5, 5, 5] = 1
    mask[6, 6, 6] = 15
    mask = sitk.Cast(sitk.GetImageFromArray(mask), sitk.sitkUInt64)

    out_mask = SingleSamplePreprocessor._collapse_mask(mask)
    out_mask = sitk.GetArrayFromImage(out_mask)

    assert np.unique(out_mask) == pytest.approx(np.asarray([0, 1]))
    assert out_mask[5, 5, 5] == 1
    assert out_mask[6, 6, 6] == 1

    mask = np.zeros([30, 30, 30])
    mask[5, 5, 5] = 2
    mask[6, 6, 6] = 15
    mask = sitk.Cast(sitk.GetImageFromArray(mask), sitk.sitkUInt64)

    out_mask = SingleSamplePreprocessor._collapse_mask(mask)
    out_mask = sitk.GetArrayFromImage(out_mask)

    assert np.unique(out_mask) == pytest.approx(np.asarray([0, 1]))
    assert out_mask[5, 5, 5] == 1
    assert out_mask[6, 6, 6] == 1

    # Make sure it even works for negative masks
    mask = np.zeros([30, 30, 30])
    mask[5, 5, 5] = -5
    mask[6, 6, 6] = 15
    mask = sitk.Cast(sitk.GetImageFromArray(mask), sitk.sitkInt64)

    out_mask = SingleSamplePreprocessor._collapse_mask(mask)
    out_mask = sitk.GetArrayFromImage(out_mask)

    assert np.unique(out_mask) == pytest.approx(np.asarray([0, 1]))
    assert out_mask[5, 5, 5] == 1
    assert out_mask[6, 6, 6] == 1


def test_rescale_intensity_range():
    sample = np.ones([30, 30, 30]) * 50
    sample = np.pad(sample, 5)

    image_sample = sitk.GetImageFromArray(sample)
    image_sample = SingleSamplePreprocessor._rescale_image_intensity_range(
        image_sample, [0, 100], [0, 1]
    )

    assert np.amax(sitk.GetArrayFromImage(image_sample)) == pytest.approx(1)
    assert np.amin(sitk.GetArrayFromImage(image_sample)) == pytest.approx(0)

    image_sample = sitk.GetImageFromArray(sample)
    image_sample = SingleSamplePreprocessor._rescale_image_intensity_range(
        image_sample, [0, 100], [5, 17.5]
    )

    assert np.amax(sitk.GetArrayFromImage(image_sample)) == pytest.approx(17.5)
    assert np.amin(sitk.GetArrayFromImage(image_sample)) == pytest.approx(5)

    image_sample = sitk.GetImageFromArray(sample)
    image_sample = SingleSamplePreprocessor._rescale_image_intensity_range(
        image_sample, [5, 100], None
    )

    assert np.amax(sitk.GetArrayFromImage(image_sample)) == pytest.approx(np.amax(sample))
    assert np.amin(sitk.GetArrayFromImage(image_sample)) == pytest.approx(np.percentile(sample, 5))

    image_sample = sitk.GetImageFromArray(sample)
    image_sample = SingleSamplePreprocessor._rescale_image_intensity_range(
        image_sample, [0, 95], None
    )

    assert np.amax(sitk.GetArrayFromImage(image_sample)) == pytest.approx(np.percentile(sample, 95))
    assert np.amin(sitk.GetArrayFromImage(image_sample)) == pytest.approx(np.amin(sample))

    image_sample = sitk.GetImageFromArray(sample)
    image_sample = SingleSamplePreprocessor._rescale_image_intensity_range(
        image_sample, [5, 95], None
    )

    assert np.amax(sitk.GetArrayFromImage(image_sample)) == pytest.approx(np.percentile(sample, 95))
    assert np.amin(sitk.GetArrayFromImage(image_sample)) == pytest.approx(np.percentile(sample, 5))

    image_sample = sitk.GetImageFromArray(sample)
    image_sample = SingleSamplePreprocessor._rescale_image_intensity_range(
        image_sample, [5, 95], [0, 1]
    )

    assert np.amax(sitk.GetArrayFromImage(image_sample)) == pytest.approx(1)
    assert np.amin(sitk.GetArrayFromImage(image_sample)) == pytest.approx(0)
    percentile_5_index = sample <= np.percentile(sample, 5)
    percentile_95_index = sample >= np.percentile(sample, 95)
    normalized_sample = sitk.GetArrayFromImage(image_sample)
    assert normalized_sample[percentile_5_index] == pytest.approx(0)
    assert normalized_sample[percentile_95_index] == pytest.approx(1)

    image_sample = sitk.GetImageFromArray(sample)
    image_sample = SingleSamplePreprocessor._rescale_image_intensity_range(
        image_sample, [2, 89], [3, 7.5]
    )

    assert np.amax(sitk.GetArrayFromImage(image_sample)) == pytest.approx(7.5)
    assert np.amin(sitk.GetArrayFromImage(image_sample)) == pytest.approx(3)
    percentile_5_index = sample <= np.percentile(sample, 2)
    percentile_95_index = sample >= np.percentile(sample, 89)
    normalized_sample = sitk.GetArrayFromImage(image_sample)
    assert normalized_sample[percentile_5_index] == pytest.approx(3)
    assert normalized_sample[percentile_95_index] == pytest.approx(7.5)


def test_rescale_intensity_range_with_mask():
    sample = np.zeros([30, 30, 30])
    sample[5:25, 5:25, 5:25] = 1
    sample[15:20, 15:20, 15:20] = 2
    sample[10:15, 10:15, 10:15] = 5
    mask = np.zeros([30, 30, 30])
    mask[5:25, 5:25, 5:25] = 1

    image_sample = sitk.GetImageFromArray(sample)
    mask_sample = sitk.GetImageFromArray(mask)
    mask_sample = sitk.Cast(mask_sample, sitk.sitkUInt8)
    image_sample = SingleSamplePreprocessor._rescale_image_intensity_range_with_mask(
        image_sample, mask_sample, [0, 100], [0, 1]
    )

    assert np.amax(sitk.GetArrayFromImage(image_sample)) == pytest.approx(1)
    assert np.amin(sitk.GetArrayFromImage(image_sample)) == pytest.approx(0)

    image_sample = sitk.GetImageFromArray(sample)
    mask_sample = sitk.GetImageFromArray(mask)
    mask_sample = sitk.Cast(mask_sample, sitk.sitkUInt8)
    image_sample = SingleSamplePreprocessor._rescale_image_intensity_range_with_mask(
        image_sample, mask_sample, [0, 100], [5, 17.5]
    )

    assert np.amax(sitk.GetArrayFromImage(image_sample)) == pytest.approx(17.5)
    assert np.amin(sitk.GetArrayFromImage(image_sample)) == pytest.approx(5)

    image_sample = sitk.GetImageFromArray(sample)
    mask_sample = sitk.GetImageFromArray(mask)
    mask_sample = sitk.Cast(mask_sample, sitk.sitkUInt8)
    image_sample = SingleSamplePreprocessor._rescale_image_intensity_range_with_mask(
        image_sample, mask_sample, [5, 100], None
    )

    assert np.amax(sitk.GetArrayFromImage(image_sample)) == pytest.approx(
        np.amax(sample[mask.astype(np.bool)])
    )
    assert np.amin(sitk.GetArrayFromImage(image_sample)) == pytest.approx(
        np.percentile(sample[mask.astype(np.bool)], 5)
    )

    image_sample = sitk.GetImageFromArray(sample)
    mask_sample = sitk.GetImageFromArray(mask)
    mask_sample = sitk.Cast(mask_sample, sitk.sitkUInt8)
    image_sample = SingleSamplePreprocessor._rescale_image_intensity_range_with_mask(
        image_sample, mask_sample, [0, 98], None
    )

    assert np.amax(sitk.GetArrayFromImage(image_sample)) == pytest.approx(
        np.percentile(sample[mask.astype(np.bool)], 98)
    )
    assert np.amin(sitk.GetArrayFromImage(image_sample)) == pytest.approx(
        np.min(sample[mask.astype(np.bool)])
    )
    # Test whether the background is really not taken into account
    assert np.amin(sitk.GetArrayFromImage(image_sample)) != pytest.approx(np.min(sample))

    image_sample = sitk.GetImageFromArray(sample)
    mask_sample = sitk.GetImageFromArray(mask)
    mask_sample = sitk.Cast(mask_sample, sitk.sitkUInt8)
    image_sample = SingleSamplePreprocessor._rescale_image_intensity_range_with_mask(
        image_sample, mask_sample, [2, 98], None
    )

    assert np.amax(sitk.GetArrayFromImage(image_sample)) == pytest.approx(
        np.percentile(sample[mask.astype(np.bool)], 98)
    )
    assert np.amin(sitk.GetArrayFromImage(image_sample)) == pytest.approx(
        np.percentile(sample[mask.astype(np.bool)], 2)
    )

    image_sample = sitk.GetImageFromArray(sample)
    mask_sample = sitk.GetImageFromArray(mask)
    mask_sample = sitk.Cast(mask_sample, sitk.sitkUInt8)
    image_sample = SingleSamplePreprocessor._rescale_image_intensity_range_with_mask(
        image_sample, mask_sample, [2, 98], [0, 1]
    )

    assert np.amax(sitk.GetArrayFromImage(image_sample)) == pytest.approx(1)
    assert np.amin(sitk.GetArrayFromImage(image_sample)) == pytest.approx(0)
    percentile_2_index = sample <= np.percentile(sample[mask.astype(np.bool)], 2)
    percentile_98_index = sample >= np.percentile(sample[mask.astype(np.bool)], 98)
    normalized_sample = sitk.GetArrayFromImage(image_sample)
    assert normalized_sample[percentile_2_index] == pytest.approx(0)
    assert normalized_sample[percentile_98_index] == pytest.approx(1)


def test_image_zscoring():
    sample = np.random.rand(10, 10, 10)
    image = sitk.GetImageFromArray(sample)

    image = SingleSamplePreprocessor._zscore_image_intensity(image)
    image_arr = sitk.GetArrayFromImage(image)

    assert np.mean(image_arr) == pytest.approx(0)
    assert np.std(image_arr) == pytest.approx(1, abs=1e-3)


def test_image_zscoring_with_mask():
    sample = np.random.rand(10, 10, 10)
    image = sitk.GetImageFromArray(sample)
    mask = np.zeros([10, 10, 10])
    mask[3:8, 3:8, 3:8] = 1
    mask_sample = sitk.GetImageFromArray(mask)
    mask_sample = sitk.Cast(mask_sample, sitk.sitkUInt8)

    image = SingleSamplePreprocessor._zscore_image_intensity_with_mask(image, mask_sample)
    image_arr = sitk.GetArrayFromImage(image)

    assert np.mean(image_arr[mask.astype(np.bool)]) == pytest.approx(0)
    assert np.mean(image_arr) != pytest.approx(0)
    assert np.std(image_arr[mask.astype(np.bool)]) == pytest.approx(1, abs=1e-2)


@NIFTI_FILES
def test_channel_normalization(datafiles):
    samples = get_samples(datafiles)

    for i_sample in samples:
        preprocessor = SingleSamplePreprocessor(
            i_sample,
            {
                "normalizing": {
                    "type": "image",
                    "normalization_method": "range",
                    "normalization_range": [2, 98],
                    "output_range": [0, 1],
                }
            },
        )
        preprocessor.normalizing()
        example_channel = sitk.GetArrayFromImage(preprocessor.sample.get_example_channel())
        assert np.amax(example_channel) == pytest.approx(1)
        assert np.amin(example_channel) == pytest.approx(0)

        example_mask = sitk.GetArrayFromImage(preprocessor.sample.get_example_mask())
        assert np.amax(example_mask) == pytest.approx(105)
        assert np.amin(example_mask) == pytest.approx(0)
        preprocessor = SingleSamplePreprocessor(
            i_sample,
            {
                "normalizing": {
                    "type": "image",
                    "normalization_method": "range",
                    "normalization_range": [2, 98],
                    "output_range": [0, 1],
                    "mask_normalization": "collapse",
                }
            },
        )
        preprocessor.normalizing()
        example_channel = sitk.GetArrayFromImage(preprocessor.sample.get_example_channel())
        assert np.amax(example_channel) == pytest.approx(1)
        assert np.amin(example_channel) == pytest.approx(0)

        example_mask = sitk.GetArrayFromImage(preprocessor.sample.get_example_mask())
        assert np.amax(example_mask) == pytest.approx(1)
        assert np.amin(example_mask) == pytest.approx(0)

        preprocessor = SingleSamplePreprocessor(
            i_sample,
            {
                "normalizing": {
                    "type": "image",
                    "normalization_method": "range",
                    "normalization_range": [2, 98],
                    "output_range": [0, 1],
                    "mask_file": os.path.join(FIXTURE_DIR, "ATLAS", "atlas_mask.nii.gz"),
                    "mask_normalization": "consecutively",
                }
            },
        )
        preprocessor.normalizing()

        preprocessor = SingleSamplePreprocessor(
            i_sample,
            {
                "normalizing": {
                    "type": "image",
                    "normalization_method": "zscore",
                    "normalization_range": [2, 98],
                    "output_range": [0, 1],
                    "mask_file": os.path.join(FIXTURE_DIR, "ATLAS", "atlas_mask.nii.gz"),
                    "mask_normalization": "collapse",
                }
            },
        )
        preprocessor.normalizing()

        preprocessor = SingleSamplePreprocessor(
            i_sample,
            {
                "normalizing": {
                    "type": "image",
                    "normalization_method": "zscore",
                    "normalization_range": [2, 98],
                    "output_range": [0, 1],
                    "mask_normalization": "collapse",
                }
            },
        )
        preprocessor.normalizing()


# ===============================================================
# Padding
# ===============================================================


def test_padding_from_parameters():
    test_image = sitk.GetImageFromArray(np.ones((15, 15, 15)))

    padded_image = SingleSamplePreprocessor._pad_image_from_parameters(
        test_image, np.zeros(3), np.zeros(3), 0.0
    )

    assert isinstance(padded_image, sitk.Image)
    assert padded_image.GetSize() == (15, 15, 15)

    padded_image = SingleSamplePreprocessor._pad_image_from_parameters(
        test_image, 5 * np.ones(3), 5 * np.ones(3), 0.0
    )

    assert isinstance(padded_image, sitk.Image)
    assert padded_image.GetSize() == (25, 25, 25)

    padded_image = sitk.GetArrayFromImage(padded_image)
    assert padded_image[0:5, 0:5, 0:5] == pytest.approx(np.zeros([5, 5, 5]))
    assert padded_image[5:20, 5:20, 5:20] == pytest.approx(np.ones([15, 15, 15]))
    assert padded_image[20:25, 20:25, 20:25] == pytest.approx(np.zeros([5, 5, 5]))

    padded_image = SingleSamplePreprocessor._pad_image_from_parameters(
        test_image, 5 * np.ones(3), 5 * np.ones(3), 15.0
    )

    assert isinstance(padded_image, sitk.Image)
    assert padded_image.GetSize() == (25, 25, 25)

    padded_image = sitk.GetArrayFromImage(padded_image)
    assert padded_image[0:5, 0:5, 0:5] == pytest.approx(15 * np.ones([5, 5, 5]))
    assert padded_image[5:20, 5:20, 5:20] == pytest.approx(np.ones([15, 15, 15]))
    assert padded_image[20:25, 20:25, 20:25] == pytest.approx(15 * np.ones([5, 5, 5]))


def test_size_padding():
    test_image = sitk.GetImageFromArray(np.ones((15, 15, 15)))
    (padded_image, left_padding, right_padding,) = SingleSamplePreprocessor._pad_image_to_size(
        test_image, [10, 10, 10]
    )

    assert isinstance(padded_image, sitk.Image)
    assert isinstance(left_padding, np.ndarray)
    assert isinstance(right_padding, np.ndarray)

    assert padded_image.GetSize() == (15, 15, 15)
    assert left_padding == pytest.approx(np.asarray([0, 0, 0]))
    assert right_padding == pytest.approx(np.asarray([0, 0, 0]))

    (padded_image, left_padding, right_padding,) = SingleSamplePreprocessor._pad_image_to_size(
        test_image, [17, 15, 15]
    )

    assert isinstance(padded_image, sitk.Image)
    assert isinstance(left_padding, np.ndarray)
    assert isinstance(right_padding, np.ndarray)
    assert padded_image.GetSize() == (17, 15, 15)
    assert left_padding == pytest.approx(np.asarray([1, 0, 0]))
    assert right_padding == pytest.approx(np.asarray([1, 0, 0]))

    (padded_image, left_padding, right_padding,) = SingleSamplePreprocessor._pad_image_to_size(
        test_image, [20, 15, 15]
    )

    assert isinstance(padded_image, sitk.Image)
    assert isinstance(left_padding, np.ndarray)
    assert isinstance(right_padding, np.ndarray)
    assert padded_image.GetSize() == (20, 15, 15)
    assert left_padding == pytest.approx(np.asarray([3, 0, 0]))
    assert right_padding == pytest.approx(np.asarray([2, 0, 0]))

    (padded_image, left_padding, right_padding,) = SingleSamplePreprocessor._pad_image_to_size(
        test_image, [20, 55, 18]
    )

    assert isinstance(padded_image, sitk.Image)
    assert isinstance(left_padding, np.ndarray)
    assert isinstance(right_padding, np.ndarray)
    assert padded_image.GetSize() == (20, 55, 18)
    assert left_padding == pytest.approx(np.asarray([3, 20, 2]))
    assert right_padding == pytest.approx(np.asarray([2, 20, 1]))

    (padded_image, left_padding, right_padding,) = SingleSamplePreprocessor._pad_image_to_size(
        test_image, [20, 55, 18], 23
    )

    assert isinstance(padded_image, sitk.Image)
    assert isinstance(left_padding, np.ndarray)
    assert isinstance(right_padding, np.ndarray)
    assert padded_image.GetSize() == (20, 55, 18)
    assert left_padding == pytest.approx(np.asarray([3, 20, 2]))
    assert right_padding == pytest.approx(np.asarray([2, 20, 1]))

    padded_image = sitk.GetArrayFromImage(padded_image)
    # Numpy array so the first and last dimensions are switched
    assert padded_image[0:2, 0:20, 0:3] == pytest.approx(23 * np.ones([2, 20, 3]))
    assert padded_image[2:17, 20:35, 3:18] == pytest.approx(np.ones([15, 15, 15]))
    assert padded_image[17:18, 35:55, 18:20] == pytest.approx(23 * np.ones([1, 20, 2]))


# ===============================================================
# Patching
# ===============================================================


def test_random_patch_parameters():
    image_size = [20, 30, 40]
    sample = np.ones(np.flip(image_size))
    sample = sitk.GetImageFromArray(sample)
    patch_size = np.asarray([5, 7, 3])
    patch_indices = SingleSamplePreprocessor._get_random_patching_parameters(
        patch_size, 250, sample
    )

    assert isinstance(patch_indices, np.ndarray)
    assert len(patch_indices) == 250
    for i_patch_index in patch_indices:
        assert np.all(
            [
                np.issubdtype(type(i_dimension_index), np.integer)
                for i_dimension_index in i_patch_index
            ]
        )
        assert np.all(np.asarray(i_patch_index) <= np.asarray([15, 23, 37]))


def test_fitting_patch_parameters_2D():
    sample = np.ones([4, 4])
    sample = sitk.GetImageFromArray(sample)
    patch_size = np.asarray([2, 2])
    patch_indices = SingleSamplePreprocessor._get_fitting_patching_parameters(patch_size, sample)

    assert isinstance(patch_indices, np.ndarray)
    assert len(patch_indices) == 4
    for i_patch_index in patch_indices:
        assert np.all(
            [
                np.issubdtype(type(i_dimension_index), np.integer)
                for i_dimension_index in i_patch_index
            ]
        )
        assert np.all(np.asarray(i_patch_index) <= np.asarray([2, 2]))


def test_fitting_patch_parameters_3D():
    image_size = [10, 6, 7]
    # Need to flip for correct orientation
    sample = np.ones(np.flip(image_size))
    sample = sitk.GetImageFromArray(sample)
    patch_size = np.asarray([2, 1, 3])
    patch_indices = SingleSamplePreprocessor._get_fitting_patching_parameters(patch_size, sample)

    assert isinstance(patch_indices, np.ndarray)
    assert len(patch_indices) == 60
    for i_patch_index in patch_indices:
        assert np.all(
            [
                np.issubdtype(type(i_dimension_index), np.integer)
                for i_dimension_index in i_patch_index
            ]
        )
        # Here the maximum is inverted because we the simpleitk indices are inverted
        assert np.all(np.asarray(i_patch_index) <= np.asarray([8, 5, 4]))

    # Make sure that the minimum and maximum patches are in there
    assert len(np.where(np.all(patch_indices == np.asarray([0, 0, 0]), axis=1))) == 1
    assert len(np.where(np.all(patch_indices == np.asarray([8, 5, 4]), axis=1))) == 1
    assert len(np.where(np.all(patch_indices == np.asarray([8, 0, 0]), axis=1))) == 1
    assert len(np.where(np.all(patch_indices == np.asarray([0, 0, 3]), axis=1))) == 1
    # Make sure we skipped over this one, we want maximum spaced patches
    # So this one should not be in there
    assert 3 not in patch_indices[:, 2]
    assert len(np.where(np.all(patch_indices == np.asarray([0, 5, 0]), axis=1))) == 1


def test_overlap_parameters_2D():
    image_size = [10, 10]
    # Need to flip for correct orientation
    sample = np.ones(np.flip(image_size))
    sample = sitk.GetImageFromArray(sample)
    patch_size = np.asarray([4, 4])

    (
        patch_indices,
        left_padding,
        right_padding,
    ) = SingleSamplePreprocessor._get_overlap_patching_parameters(patch_size, 0.5, 0, sample)

    assert isinstance(patch_indices, np.ndarray)
    assert len(patch_indices) == 25
    assert np.unique(patch_indices[:, 0]) == pytest.approx(np.asarray([0, 2, 4, 6, 8]))
    assert np.unique(patch_indices[:, 1]) == pytest.approx(np.asarray([0, 2, 4, 6, 8]))
    assert left_padding == pytest.approx(np.asarray([1, 1]))
    assert right_padding == pytest.approx(np.asarray([1, 1]))

    image_size = [10, 10]
    # Need to flip for correct orientation
    sample = np.ones(np.flip(image_size))
    sample = sitk.GetImageFromArray(sample)
    patch_size = np.asarray([3, 5])

    (
        patch_indices,
        left_padding,
        right_padding,
    ) = SingleSamplePreprocessor._get_overlap_patching_parameters(patch_size, 0.5, 0, sample)

    assert isinstance(patch_indices, np.ndarray)
    assert len(patch_indices) == 36
    assert np.unique(patch_indices[:, 0]) == pytest.approx(np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8]))
    assert np.unique(patch_indices[:, 1]) == pytest.approx(np.asarray([0, 2, 4, 6]))
    assert left_padding == pytest.approx(np.asarray([1, 1]))
    assert right_padding == pytest.approx(np.asarray([0, 0]))

    image_size = [80, 35]
    # Need to flip for correct orientation
    sample = np.ones(np.flip(image_size))
    sample = sitk.GetImageFromArray(sample)
    patch_size = np.asarray([10, 4])

    (
        patch_indices,
        left_padding,
        right_padding,
    ) = SingleSamplePreprocessor._get_overlap_patching_parameters(patch_size, [0.3, 0.5], 0, sample)

    assert isinstance(patch_indices, np.ndarray)
    assert len(patch_indices) == 204
    assert left_padding == pytest.approx(np.asarray([4, 1]))
    assert right_padding == pytest.approx(np.asarray([3, 0]))
    assert np.unique(patch_indices[:, 0]) == pytest.approx(
        np.asarray([0, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77])
    )
    assert np.unique(patch_indices[:, 1]) == pytest.approx(
        np.asarray([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32])
    )

    # Test with set number of overlap voxels
    image_size = [80, 35]
    # Need to flip for correct orientation
    sample = np.ones(np.flip(image_size))
    sample = sitk.GetImageFromArray(sample)
    patch_size = np.asarray([10, 4])

    (
        patch_indices,
        left_padding,
        right_padding,
    ) = SingleSamplePreprocessor._get_overlap_patching_parameters(patch_size, [3, 2], 0, sample)

    assert isinstance(patch_indices, np.ndarray)
    assert len(patch_indices) == 204
    assert left_padding == pytest.approx(np.asarray([4, 1]))
    assert right_padding == pytest.approx(np.asarray([3, 0]))
    assert np.unique(patch_indices[:, 0]) == pytest.approx(
        np.asarray([0, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77])
    )
    assert np.unique(patch_indices[:, 1]) == pytest.approx(
        np.asarray([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32])
    )


def test_overlap_parameters_3D():
    image_size = [80, 35, 21]
    # Need to flip for correct orientation
    sample = np.ones(np.flip(image_size))
    sample = sitk.GetImageFromArray(sample)
    patch_size = np.asarray([10, 4, 3])

    (
        patch_indices,
        left_padding,
        right_padding,
    ) = SingleSamplePreprocessor._get_overlap_patching_parameters(
        patch_size, [0.3, 0.5, 0.33], 0, sample
    )

    assert isinstance(patch_indices, np.ndarray)
    assert len(patch_indices) == 2244
    assert left_padding == pytest.approx(np.asarray([4, 1, 1]))
    assert right_padding == pytest.approx(np.asarray([3, 0, 1]))
    assert np.unique(patch_indices[:, 0]) == pytest.approx(
        np.asarray([0, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77])
    )
    assert np.unique(patch_indices[:, 1]) == pytest.approx(
        np.asarray([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32])
    )
    assert np.unique(patch_indices[:, 2]) == pytest.approx(
        np.asarray([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    )

    # Test with set number of overlap voxels
    image_size = [80, 35, 21]
    # Need to flip for correct orientation
    sample = np.ones(np.flip(image_size))
    sample = sitk.GetImageFromArray(sample)
    patch_size = np.asarray([10, 4, 3])

    (
        patch_indices,
        left_padding,
        right_padding,
    ) = SingleSamplePreprocessor._get_overlap_patching_parameters(patch_size, [3, 2, 1], 0, sample)

    assert isinstance(patch_indices, np.ndarray)
    assert len(patch_indices) == 2244
    assert left_padding == pytest.approx(np.asarray([4, 1, 1]))
    assert right_padding == pytest.approx(np.asarray([3, 0, 1]))
    assert np.unique(patch_indices[:, 0]) == pytest.approx(
        np.asarray([0, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77])
    )
    assert np.unique(patch_indices[:, 1]) == pytest.approx(
        np.asarray([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32])
    )
    assert np.unique(patch_indices[:, 2]) == pytest.approx(
        np.asarray([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    )


@NIFTI_FILES
def test_get_patch_parameters(datafiles):
    samples = get_samples(datafiles)

    for i_sample in samples:
        # Overlap
        preprocessor = SingleSamplePreprocessor(
            i_sample,
            {
                "patching": {
                    "patch_size": [5, 5, 5],
                    "pad_if_needed": True,
                    "pad_constant": 0,
                    "extraction_type": "overlap",
                    "overlap_fraction": 0.5,
                }
            },
        )

        patch_parameters = preprocessor._get_patch_parameters()

        assert isinstance(patch_parameters, dict)
        assert isinstance(patch_parameters["patch_indices"], np.ndarray)
        assert patch_parameters["left_padding"] == pytest.approx(np.asarray([1, 1, 1]))
        assert patch_parameters["right_padding"] == pytest.approx(np.asarray([0, 0, 0]))
        assert len(patch_parameters["patch_indices"]) == 2744
        assert np.unique(patch_parameters["patch_indices"][:, 0]) == pytest.approx(
            np.asarray([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26])
        )
        assert np.unique(patch_parameters["patch_indices"][:, 1]) == pytest.approx(
            np.asarray([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26])
        )
        assert np.unique(patch_parameters["patch_indices"][:, 2]) == pytest.approx(
            np.asarray([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26])
        )

        # Random
        preprocessor = SingleSamplePreprocessor(
            i_sample,
            {
                "patching": {
                    "patch_size": [5, 5, 5],
                    "pad_if_needed": True,
                    "pad_constant": 0,
                    "extraction_type": "random",
                    "max_number_of_patches": 15,
                }
            },
        )

        patch_parameters = preprocessor._get_patch_parameters()

        assert isinstance(patch_parameters, dict)
        assert isinstance(patch_parameters["patch_indices"], np.ndarray)
        assert patch_parameters["left_padding"] == pytest.approx(np.asarray([0, 0, 0]))
        assert patch_parameters["right_padding"] == pytest.approx(np.asarray([0, 0, 0]))
        assert len(patch_parameters["patch_indices"]) == 15

        # fitting
        preprocessor = SingleSamplePreprocessor(
            i_sample,
            {
                "patching": {
                    "patch_size": [5, 5, 5],
                    "pad_if_needed": True,
                    "pad_constant": 0,
                    "extraction_type": "fitting",
                }
            },
        )

        patch_parameters = preprocessor._get_patch_parameters()

        assert isinstance(patch_parameters, dict)
        assert isinstance(patch_parameters["patch_indices"], np.ndarray)
        assert patch_parameters["left_padding"] == pytest.approx(np.asarray([0, 0, 0]))
        assert patch_parameters["right_padding"] == pytest.approx(np.asarray([0, 0, 0]))


def test_patch_making():
    image_size = [80, 35, 21]
    # Need to flip for correct orientation
    sample = np.ones(np.flip(image_size))
    sample[0:3, 0:4, 0:10] = 50
    sample[18:, 31:, 70:] = 25
    sample[15:18, 12:18, 40:60] = 105.5
    sample = sitk.GetImageFromArray(sample)
    patch_size = np.asarray([10, 4, 3])

    patch_indices = np.asarray(
        list(
            itertools.product(
                np.asarray([0, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77]),
                np.asarray([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]),
                np.asarray([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]),
            )
        )
    )

    patch_parameters = {
        "left_padding": np.asarray([4, 1, 1]),
        "right_padding": np.asarray([3, 0, 1]),
        "patch_indices": patch_indices,
    }

    patches = SingleSamplePreprocessor._make_patches(sample, patch_parameters, -15.0, patch_size)
    assert len(patches) == 2244

    # Find indices and check whether correct patches were extraced
    origin_patch_index = np.squeeze(np.argwhere(np.all(patch_indices == np.asarray([0, 0, 0]), 1)))
    assert patches[origin_patch_index].GetSize() == (10, 4, 3)
    obtained_origin_patch = sitk.GetArrayFromImage(patches[origin_patch_index])
    assert obtained_origin_patch[0:1, 0:1, 0:4] == pytest.approx(-15)
    assert obtained_origin_patch[1:3, 1:4, 4:10] == pytest.approx(50)

    last_patch_index = np.squeeze(np.argwhere(np.all(patch_indices == np.asarray([77, 32, 20]), 1)))
    last_patch = sitk.GetArrayFromImage(patches[last_patch_index])

    assert last_patch[0:2, :, 0:7] == pytest.approx(25)
    assert last_patch[2:, :, 7:] == pytest.approx(-15)


@NIFTI_FILES
def test_patching(datafiles):
    samples = get_samples(datafiles)

    for i_sample in samples:
        # Overlap
        preprocessor = SingleSamplePreprocessor(
            i_sample,
            {
                "patching": {
                    "patch_size": [5, 5, 5],
                    "pad_if_needed": True,
                    "pad_constant": 0,
                    "extraction_type": "overlap",
                    "overlap_fraction": 0.5,
                }
            },
        )

        preprocessor.patching()

        assert preprocessor.sample.has_patches
        assert preprocessor.sample.number_of_patches == 2744

        # Make sure that the patches for all the channels are the same
        patches = preprocessor.sample.get_grouped_channels()
        for i_patch in patches:
            first_channel = sitk.GetArrayFromImage(i_patch[0])
            for i_i_patch_channel, i_patch_channel in enumerate(i_patch):
                i_patch_channel = sitk.GetArrayFromImage(i_patch_channel)
                assert (
                    i_patch_channel[i_i_patch_channel, i_i_patch_channel, i_i_patch_channel]
                    == first_channel[0, 0, 0]
                )
                assert (
                    i_patch_channel[
                        i_i_patch_channel + 1, i_i_patch_channel + 1, i_i_patch_channel + 1,
                    ]
                    == first_channel[1, 1, 1]
                )

        preprocessor = SingleSamplePreprocessor(
            i_sample,
            {
                "patching": {
                    "patch_size": [5, 5, 5],
                    "extraction_type": "random",
                    "max_number_of_patches": 10,
                }
            },
        )

        preprocessor.patching()

        assert preprocessor.sample.has_patches
        assert preprocessor.sample.number_of_patches == 10


@NIFTI_FILES
def test_patching_2D(datafiles):
    samples = get_samples(datafiles)

    for i_sample in samples:
        # Overlap
        original_sample_channel = sitk.GetArrayFromImage(i_sample.get_example_channel())
        preprocessor = SingleSamplePreprocessor(
            i_sample,
            {
                "patching": {
                    "patch_size": [30, 30, 1],
                    "pad_if_needed": True,
                    "pad_constant": 0,
                    "extraction_type": "fitting",
                }
            },
        )

        preprocessor.patching()

        assert preprocessor.sample.has_patches
        assert preprocessor.sample.number_of_patches == 30

        # Make sure that the patches are indeed the slices

        sample_patches = preprocessor.sample.get_example_channel_patches()
        for i_i_patch, i_patch in enumerate(sample_patches):
            i_patch = sitk.GetArrayFromImage(i_patch)

            assert i_patch == pytest.approx(original_sample_channel[i_i_patch, :, :])



# ===============================================================
# Rejecting
# ===============================================================


def test_get_rejecting_patches():
    image_size = [10, 10, 10]
    # Need to flip for correct orientation
    mask = np.zeros(np.flip(image_size))
    mask = sitk.Cast(sitk.GetImageFromArray(mask), sitk.sitkUInt8)

    to_reject = SingleSamplePreprocessor._get_to_reject_patches(mask, 0.5)

    assert isinstance(to_reject, list)
    assert len(to_reject) == 1
    assert to_reject[0]

    to_reject = SingleSamplePreprocessor._get_to_reject_patches(mask, 0)

    assert isinstance(to_reject, list)
    assert len(to_reject) == 1
    assert not to_reject[0]

    mask = np.ones(np.flip(image_size))
    mask = sitk.Cast(sitk.GetImageFromArray(mask), sitk.sitkUInt8)

    to_reject = SingleSamplePreprocessor._get_to_reject_patches(mask, 0.5)

    assert isinstance(to_reject, list)
    assert len(to_reject) == 1
    assert not to_reject[0]

    N_patches = 50
    total_voxels = 10 * 10 * 10
    rejection_limit = 0.3
    rejection_limit_voxels = total_voxels * rejection_limit
    patches = []
    true_output = []

    for i_patch in range(N_patches):
        patch_mask = np.random.randint(2, size=image_size)
        true_output.append(np.count_nonzero(patch_mask) < rejection_limit_voxels)
        patches.append(sitk.GetImageFromArray(patch_mask))

    to_reject = SingleSamplePreprocessor._get_to_reject_patches(patches, rejection_limit)

    assert isinstance(to_reject, list)
    assert to_reject == true_output


def test_reject_patches():
    image_size = [10, 10, 10]
    # Need to flip for correct orientation
    sample = np.zeros(np.flip(image_size))
    sample = sitk.GetImageFromArray(sample)

    accepted_patches = SingleSamplePreprocessor._get_accepted_patches(sample, [True])
    assert isinstance(accepted_patches, list)
    assert len(accepted_patches) == 0

    accepted_patches = SingleSamplePreprocessor._get_accepted_patches(sample, [False])
    assert isinstance(accepted_patches, list)
    assert len(accepted_patches) == 1
    assert accepted_patches[0] == sample

    N_patches = 50
    patches = []
    true_output = []
    rejection_status = []

    for i_patch in range(N_patches):
        sample_patch = sitk.GetImageFromArray(np.random.rand(*image_size))
        patches.append(sample_patch)
        to_accept = np.random.choice([True, False])
        rejection_status.append(~to_accept)
        if to_accept:
            true_output.append(sample_patch)

    accepted_patches = SingleSamplePreprocessor._get_accepted_patches(patches, rejection_status)

    assert isinstance(accepted_patches, list)
    assert len(accepted_patches) == len(true_output)
    assert accepted_patches == true_output


@NIFTI_FILES
def test_rejecting(datafiles):
    samples = get_samples(datafiles)

    for i_sample in samples:
        preprocessor = SingleSamplePreprocessor(
            i_sample,
            {
                "patching": {
                    "patch_size": [5, 5, 5],
                    "pad_if_needed": True,
                    "pad_constant": 0,
                    "extraction_type": "overlap",
                    "overlap_fraction": 0.5,
                },
                "rejecting": {"type": "patch", "rejection_limit": 0.9},
            },
        )

        preprocessor.patching()
        preprocessor.rejecting()

        assert preprocessor.sample.number_of_patches == 0

        preprocessor = SingleSamplePreprocessor(
            i_sample,
            {
                "patching": {
                    "patch_size": [5, 5, 5],
                    "pad_if_needed": True,
                    "pad_constant": 0,
                    "extraction_type": "overlap",
                    "overlap_fraction": 0.5,
                },
                "rejecting": {"type": "patch", "rejection_limit": 0.256},
            },
        )

        preprocessor.patching()
        preprocessor.rejecting()

        assert preprocessor.sample.number_of_patches == 196

        preprocessor = SingleSamplePreprocessor(
            i_sample,
            {
                "patching": {
                    "patch_size": [5, 5, 5],
                    "pad_if_needed": True,
                    "pad_constant": 0,
                    "extraction_type": "overlap",
                    "overlap_fraction": 0.5,
                },
                "rejecting": {"type": "patch", "rejection_limit": 0.128},
            },
        )

        preprocessor.patching()
        preprocessor.rejecting()

        assert preprocessor.sample.number_of_patches == 392


@NIFTI_FILES
def test_rejecting_as_labels(datafiles):
    samples = get_samples(datafiles)
    i_sample = samples[0]
    rejection_limit = 0.256
    preprocessor = SingleSamplePreprocessor(
        i_sample,
        {
            "patching": {
                "patch_size": [5, 5, 5],
                "pad_if_needed": True,
                "pad_constant": 0,
                "extraction_type": "overlap",
                "overlap_fraction": 0.5,
            },
            "rejecting": {
                "type": "patch",
                "rejection_limit": rejection_limit,
                "rejection_as_label": True,
            },
        },
    )
    preprocessor.patching()
    preprocessor.rejecting()

    to_reject = SingleSamplePreprocessor._get_to_reject_patches(
        preprocessor.sample.get_example_mask_patches(), rejection_limit
    )

    assert preprocessor.sample.number_of_patches == 2744
    assert len(preprocessor.sample.labels) == 2744
    assert np.asarray(
        [i_label["accepted"] for i_label in preprocessor.sample.labels]
    ) == pytest.approx(np.logical_not(to_reject).astype(np.uint8))


@NIFTI_FILES
def test_rejecting_as_labels_one_hot(datafiles):
    samples = get_samples(datafiles)
    i_sample = samples[0]
    i_sample.are_labels_one_hot = True
    rejection_limit = 0.256
    preprocessor = SingleSamplePreprocessor(
        i_sample,
        {
            "patching": {
                "patch_size": [5, 5, 5],
                "pad_if_needed": True,
                "pad_constant": 0,
                "extraction_type": "overlap",
                "overlap_fraction": 0.5,
            },
            "rejecting": {
                "type": "patch",
                "rejection_limit": rejection_limit,
                "rejection_as_label": True,
            },
        },
    )
    preprocessor.patching()
    preprocessor.rejecting()

    to_reject = SingleSamplePreprocessor._get_to_reject_patches(
        preprocessor.sample.get_example_mask_patches(), rejection_limit
    )

    assert preprocessor.sample.number_of_patches == 2744
    assert len(preprocessor.sample.labels) == 2744
    assert np.asarray(
        [i_label["accepted"] for i_label in preprocessor.sample.labels]
    ) == pytest.approx(
        tf.one_hot(np.logical_not(to_reject).astype(np.uint8), 2, dtype=tf.uint8).numpy()
    )


# ===============================================================
# Bias field correcting
# ===============================================================
@NIFTI_FILES
def test_bias_field_correcting(datafiles):
    samples = get_samples(datafiles)

    for i_sample in samples:
        preprocessor = SingleSamplePreprocessor(
            i_sample, {"bias_field_correcting": {"type": "image"}},
        )

        preprocessor.bias_field_correcting()
        assert isinstance(preprocessor.sample.get_example_channel(), sitk.Image)
        assert not preprocessor.sample.has_patches

        preprocessor = SingleSamplePreprocessor(
            i_sample,
            {
                "patching": {
                    "patch_size": [5, 5, 5],
                    "pad_if_needed": True,
                    "pad_constant": 0,
                    "extraction_type": "random",
                    "max_number_of_patches": 3,
                },
                "bias_field_correcting": {"type": "patch"},
            },
        )

        preprocessor.patching()
        preprocessor.bias_field_correcting()
        assert isinstance(preprocessor.sample.get_example_channel(), sitk.Image)
        assert preprocessor.sample.has_patches
        assert preprocessor.sample.number_of_patches == 3

        preprocessor = SingleSamplePreprocessor(
            i_sample,
            {
                "bias_field_correcting": {
                    "type": "image",
                    "mask_file": os.path.join(FIXTURE_DIR, "ATLAS", "atlas_mask.nii.gz"),
                }
            },
        )

        preprocessor.bias_field_correcting()
        assert isinstance(preprocessor.sample.get_example_channel(), sitk.Image)
        assert not preprocessor.sample.has_patches


# ===============================================================
# Saving
# ===============================================================


@NIFTI_FILES
def test_patches_to_data_structure(datafiles):
    samples = get_samples(datafiles)
    i_sample = samples[0]
    channel_1 = sitk.GetImageFromArray(np.random.random([5, 5, 5]))
    channel_2 = sitk.GetImageFromArray(np.random.random([5, 5, 5]))

    channel_patches = [channel_1, channel_2]

    preprocessor = SingleSamplePreprocessor(i_sample, {"saving": {"use_mask_as_channel": False}})

    npz_patch = preprocessor._patch_to_data_structure(channel_patches, None, None, i_sample.labels[0])

    assert isinstance(npz_patch, dict)

    assert PrognosAIs.Constants.FEATURE_INDEX in npz_patch
    assert isinstance(npz_patch[PrognosAIs.Constants.FEATURE_INDEX], dict)
    assert npz_patch[PrognosAIs.Constants.FEATURE_INDEX][
        PrognosAIs.Constants.FEATURE_INDEX
    ].shape == pytest.approx([5, 5, 5, 2])

    channel_1 = sitk.GetImageFromArray(np.random.random([5, 5, 5]))
    channel_2 = sitk.GetImageFromArray(np.random.random([5, 5, 5]))
    mask_channel = sitk.GetImageFromArray(np.random.random([5, 5, 5]))

    channel_patches = [channel_1, channel_2]
    mask_patches = [mask_channel]

    preprocessor = SingleSamplePreprocessor(i_sample, {"saving": {"use_mask_as_channel": True}})

    npz_patch = preprocessor._patch_to_data_structure(
        channel_patches, None, mask_patches, i_sample.labels[0]
    )

    assert isinstance(npz_patch, dict)

    assert PrognosAIs.Constants.FEATURE_INDEX in npz_patch
    assert isinstance(npz_patch[PrognosAIs.Constants.FEATURE_INDEX], dict)
    assert npz_patch[PrognosAIs.Constants.FEATURE_INDEX][
        PrognosAIs.Constants.FEATURE_INDEX
    ].shape == pytest.approx([5, 5, 5, 3])
    assert np.array_equal(
        npz_patch[PrognosAIs.Constants.FEATURE_INDEX][PrognosAIs.Constants.FEATURE_INDEX][
            :, :, :, 0
        ],
        np.transpose(sitk.GetArrayFromImage(channel_1)),
    )
    assert np.array_equal(
        npz_patch[PrognosAIs.Constants.FEATURE_INDEX][PrognosAIs.Constants.FEATURE_INDEX][
            :, :, :, 1
        ],
        np.transpose(sitk.GetArrayFromImage(channel_2)),
    )
    assert np.array_equal(
        npz_patch[PrognosAIs.Constants.FEATURE_INDEX][PrognosAIs.Constants.FEATURE_INDEX][
            :, :, :, 2
        ],
        np.transpose(sitk.GetArrayFromImage(mask_channel)),
    )

    channel_1 = sitk.GetImageFromArray(np.random.random([5, 5, 5]))
    channel_2 = sitk.GetImageFromArray(np.random.random([5, 5, 5]))
    channel_3 = sitk.GetImageFromArray(np.random.random([5, 5, 5]))

    channel_patches = [channel_1, channel_2, channel_3]

    preprocessor = SingleSamplePreprocessor(
        i_sample, {"saving": {"use_mask_as_channel": False, "named_channels": True}}
    )

    npz_patch = preprocessor._patch_to_data_structure(channel_patches, None, None, i_sample.labels[0])

    assert isinstance(npz_patch, dict)

    assert "Scan-0" in npz_patch[PrognosAIs.Constants.FEATURE_INDEX]
    assert "Scan-1" in npz_patch[PrognosAIs.Constants.FEATURE_INDEX]
    assert "Scan-2" in npz_patch[PrognosAIs.Constants.FEATURE_INDEX]
    assert isinstance(npz_patch[PrognosAIs.Constants.FEATURE_INDEX]["Scan-0"], np.ndarray)
    assert isinstance(npz_patch[PrognosAIs.Constants.FEATURE_INDEX]["Scan-1"], np.ndarray)
    assert isinstance(npz_patch[PrognosAIs.Constants.FEATURE_INDEX]["Scan-2"], np.ndarray)
    assert npz_patch[PrognosAIs.Constants.FEATURE_INDEX]["Scan-0"].shape == pytest.approx(
        [5, 5, 5, 1]
    )
    assert npz_patch[PrognosAIs.Constants.FEATURE_INDEX]["Scan-1"].shape == pytest.approx(
        [5, 5, 5, 1]
    )
    assert npz_patch[PrognosAIs.Constants.FEATURE_INDEX]["Scan-2"].shape == pytest.approx(
        [5, 5, 5, 1]
    )
    assert np.array_equal(
        np.squeeze(npz_patch[PrognosAIs.Constants.FEATURE_INDEX]["Scan-0"]),
        np.transpose(sitk.GetArrayFromImage(channel_1)),
    )
    assert np.array_equal(
        np.squeeze(npz_patch[PrognosAIs.Constants.FEATURE_INDEX]["Scan-1"]),
        np.transpose(sitk.GetArrayFromImage(channel_2)),
    )
    assert np.array_equal(
        np.squeeze(npz_patch[PrognosAIs.Constants.FEATURE_INDEX]["Scan-2"]),
        np.transpose(sitk.GetArrayFromImage(channel_3)),
    )

    channel_1 = sitk.GetImageFromArray(np.random.random([5, 5, 5]))
    channel_2 = sitk.GetImageFromArray(np.random.random([5, 5, 5]))
    channel_3 = sitk.GetImageFromArray(np.random.random([5, 5, 5]))
    mask = sitk.GetImageFromArray(np.random.random([5, 5, 5]))

    channel_patches = [channel_1, channel_2, channel_3]
    mask_patches = [mask]

    preprocessor = SingleSamplePreprocessor(
        i_sample, {"saving": {"use_mask_as_channel": True, "named_channels": True}}
    )

    npz_patch = preprocessor._patch_to_data_structure(
        channel_patches, None, mask_patches, i_sample.labels[0]
    )

    assert isinstance(npz_patch, dict)

    assert "Scan-0" in npz_patch[PrognosAIs.Constants.FEATURE_INDEX]
    assert "Scan-1" in npz_patch[PrognosAIs.Constants.FEATURE_INDEX]
    assert "Scan-2" in npz_patch[PrognosAIs.Constants.FEATURE_INDEX]
    assert "MASK-0" in npz_patch[PrognosAIs.Constants.FEATURE_INDEX]
    assert isinstance(npz_patch[PrognosAIs.Constants.FEATURE_INDEX]["Scan-0"], np.ndarray)
    assert isinstance(npz_patch[PrognosAIs.Constants.FEATURE_INDEX]["Scan-1"], np.ndarray)
    assert isinstance(npz_patch[PrognosAIs.Constants.FEATURE_INDEX]["Scan-2"], np.ndarray)
    assert isinstance(npz_patch[PrognosAIs.Constants.FEATURE_INDEX]["MASK-0"], np.ndarray)
    assert npz_patch[PrognosAIs.Constants.FEATURE_INDEX]["Scan-0"].shape == pytest.approx(
        [5, 5, 5, 1]
    )
    assert npz_patch[PrognosAIs.Constants.FEATURE_INDEX]["Scan-1"].shape == pytest.approx(
        [5, 5, 5, 1]
    )
    assert npz_patch[PrognosAIs.Constants.FEATURE_INDEX]["Scan-2"].shape == pytest.approx(
        [5, 5, 5, 1]
    )
    assert npz_patch[PrognosAIs.Constants.FEATURE_INDEX]["MASK-0"].shape == pytest.approx(
        [5, 5, 5, 1]
    )
    assert np.array_equal(
        np.squeeze(npz_patch[PrognosAIs.Constants.FEATURE_INDEX]["Scan-0"]),
        np.transpose(sitk.GetArrayFromImage(channel_1)),
    )
    assert np.array_equal(
        np.squeeze(npz_patch[PrognosAIs.Constants.FEATURE_INDEX]["Scan-1"]),
        np.transpose(sitk.GetArrayFromImage(channel_2)),
    )
    assert np.array_equal(
        np.squeeze(npz_patch[PrognosAIs.Constants.FEATURE_INDEX]["Scan-2"]),
        np.transpose(sitk.GetArrayFromImage(channel_3)),
    )
    assert np.array_equal(
        np.squeeze(npz_patch[PrognosAIs.Constants.FEATURE_INDEX]["MASK-0"]),
        np.transpose(sitk.GetArrayFromImage(mask)),
    )

    channel_1 = sitk.GetImageFromArray(np.random.random([5, 5, 5]))
    channel_2 = sitk.GetImageFromArray(np.random.random([5, 5, 5]))
    mask_channel = sitk.GetImageFromArray(np.random.random([5, 5, 5]))

    channel_patches = [channel_1, channel_2]
    mask_patches = [mask_channel]

    preprocessor = SingleSamplePreprocessor(i_sample, {"saving": {"use_mask_as_label": True}})

    npz_patch = preprocessor._patch_to_data_structure(
        channel_patches, None, mask_patches, i_sample.labels[0]
    )

    assert isinstance(npz_patch, dict)

    assert PrognosAIs.Constants.FEATURE_INDEX in npz_patch
    assert isinstance(npz_patch[PrognosAIs.Constants.FEATURE_INDEX], dict)
    assert npz_patch[PrognosAIs.Constants.FEATURE_INDEX][
        PrognosAIs.Constants.FEATURE_INDEX
    ].shape == pytest.approx([5, 5, 5, 2])
    assert np.array_equal(
        npz_patch[PrognosAIs.Constants.FEATURE_INDEX][PrognosAIs.Constants.FEATURE_INDEX][
            :, :, :, 0
        ],
        np.transpose(sitk.GetArrayFromImage(channel_1)),
    )
    assert np.array_equal(
        npz_patch[PrognosAIs.Constants.FEATURE_INDEX][PrognosAIs.Constants.FEATURE_INDEX][
            :, :, :, 1
        ],
        np.transpose(sitk.GetArrayFromImage(channel_2)),
    )
    assert PrognosAIs.Constants.LABEL_INDEX in npz_patch
    assert isinstance(npz_patch[PrognosAIs.Constants.LABEL_INDEX], dict)
    assert npz_patch[PrognosAIs.Constants.LABEL_INDEX][
        PrognosAIs.Constants.LABEL_INDEX
    ].shape == pytest.approx([5, 5, 5, 1])
    assert np.array_equal(
        npz_patch[PrognosAIs.Constants.LABEL_INDEX][PrognosAIs.Constants.LABEL_INDEX][:, :, :, 0],
        np.transpose(sitk.GetArrayFromImage(mask_channel)),
    )

    channel_1 = sitk.GetImageFromArray(np.random.random([5, 5, 5]))
    channel_2 = sitk.GetImageFromArray(np.random.random([5, 5, 5]))

    channel_patches = [channel_1, channel_2]

    labels = {"label_1": np.asarray([1, 0])}
    label_classes = {"label_1": 2}
    temp_sample = i_sample.copy()
    temp_sample.add_to_labels(labels, label_classes)

    preprocessor = SingleSamplePreprocessor(temp_sample, {"saving": {"use_mask_as_label": False}})

    npz_patch = preprocessor._patch_to_data_structure(
        channel_patches, None, mask_patches, temp_sample.labels[0]
    )

    assert isinstance(npz_patch, dict)

    assert PrognosAIs.Constants.FEATURE_INDEX in npz_patch
    assert isinstance(npz_patch[PrognosAIs.Constants.FEATURE_INDEX], dict)
    assert npz_patch[PrognosAIs.Constants.FEATURE_INDEX][
        PrognosAIs.Constants.FEATURE_INDEX
    ].shape == pytest.approx([5, 5, 5, 2])
    assert np.array_equal(
        npz_patch[PrognosAIs.Constants.FEATURE_INDEX][PrognosAIs.Constants.FEATURE_INDEX][
            :, :, :, 0
        ],
        np.transpose(sitk.GetArrayFromImage(channel_1)),
    )
    assert np.array_equal(
        npz_patch[PrognosAIs.Constants.FEATURE_INDEX][PrognosAIs.Constants.FEATURE_INDEX][
            :, :, :, 1
        ],
        np.transpose(sitk.GetArrayFromImage(channel_2)),
    )
    assert PrognosAIs.Constants.LABEL_INDEX in npz_patch
    assert isinstance(npz_patch[PrognosAIs.Constants.LABEL_INDEX], dict)
    assert np.array_equal(
        npz_patch[PrognosAIs.Constants.LABEL_INDEX][PrognosAIs.Constants.LABEL_INDEX],
        np.asarray([1, 0]),
    )

    channel_1 = sitk.GetImageFromArray(np.random.random([5, 5, 5]))
    channel_2 = sitk.GetImageFromArray(np.random.random([5, 5, 5]))

    channel_patches = [channel_1, channel_2]

    labels = {"label_1": np.asarray([1, 0]), "label_2": np.asarray([0, 1])}
    label_classes = {"label_1": 2, "label_2": 2}
    temp_sample = i_sample.copy()
    temp_sample.add_to_labels(labels, label_classes)

    preprocessor = SingleSamplePreprocessor(temp_sample, {"saving": {"use_mask_as_label": False}})

    npz_patch = preprocessor._patch_to_data_structure(
        channel_patches, None, mask_patches, temp_sample.labels[0]
    )

    assert isinstance(npz_patch, dict)

    assert PrognosAIs.Constants.FEATURE_INDEX in npz_patch
    assert isinstance(npz_patch[PrognosAIs.Constants.FEATURE_INDEX], dict)
    assert npz_patch[PrognosAIs.Constants.FEATURE_INDEX][
        PrognosAIs.Constants.FEATURE_INDEX
    ].shape == pytest.approx([5, 5, 5, 2])
    assert np.array_equal(
        npz_patch[PrognosAIs.Constants.FEATURE_INDEX][PrognosAIs.Constants.FEATURE_INDEX][
            :, :, :, 0
        ],
        np.transpose(sitk.GetArrayFromImage(channel_1)),
    )
    assert np.array_equal(
        npz_patch[PrognosAIs.Constants.FEATURE_INDEX][PrognosAIs.Constants.FEATURE_INDEX][
            :, :, :, 1
        ],
        np.transpose(sitk.GetArrayFromImage(channel_2)),
    )
    assert PrognosAIs.Constants.LABEL_INDEX in npz_patch
    assert isinstance(npz_patch[PrognosAIs.Constants.LABEL_INDEX], dict)
    assert np.array_equal(
        npz_patch[PrognosAIs.Constants.LABEL_INDEX]["label_1"], np.asarray([1, 0])
    )
    assert np.array_equal(
        npz_patch[PrognosAIs.Constants.LABEL_INDEX]["label_2"], np.asarray([0, 1])
    )

    channel_1 = sitk.GetImageFromArray(np.random.random([5, 5, 5]))
    channel_2 = sitk.GetImageFromArray(np.random.random([5, 5, 5]))
    mask_channel = sitk.GetImageFromArray(np.random.random([5, 5, 5]))

    channel_patches = [channel_1, channel_2]
    mask_patches = [mask_channel]

    labels = {"label_1": np.asarray([1, 0])}
    label_classes = {"label_1": 2}

    temp_sample = i_sample.copy()
    temp_sample.add_to_labels(labels, label_classes)

    preprocessor = SingleSamplePreprocessor(temp_sample, {"saving": {"use_mask_as_label": True}})

    npz_patch = preprocessor._patch_to_data_structure(
        channel_patches, None, mask_patches, temp_sample.labels[0]
    )

    assert isinstance(npz_patch, dict)

    assert PrognosAIs.Constants.FEATURE_INDEX in npz_patch
    assert isinstance(npz_patch[PrognosAIs.Constants.FEATURE_INDEX], dict)
    assert npz_patch[PrognosAIs.Constants.FEATURE_INDEX][
        PrognosAIs.Constants.FEATURE_INDEX
    ].shape == pytest.approx([5, 5, 5, 2])
    assert np.array_equal(
        npz_patch[PrognosAIs.Constants.FEATURE_INDEX][PrognosAIs.Constants.FEATURE_INDEX][
            :, :, :, 0
        ],
        np.transpose(sitk.GetArrayFromImage(channel_1)),
    )
    assert np.array_equal(
        npz_patch[PrognosAIs.Constants.FEATURE_INDEX][PrognosAIs.Constants.FEATURE_INDEX][
            :, :, :, 1
        ],
        np.transpose(sitk.GetArrayFromImage(channel_2)),
    )
    assert PrognosAIs.Constants.LABEL_INDEX in npz_patch
    assert isinstance(npz_patch[PrognosAIs.Constants.LABEL_INDEX], dict)
    assert npz_patch[PrognosAIs.Constants.LABEL_INDEX]["MASK"].shape == pytest.approx([5, 5, 5, 1])
    assert np.array_equal(
        npz_patch[PrognosAIs.Constants.LABEL_INDEX]["MASK"][:, :, :, 0],
        np.transpose(sitk.GetArrayFromImage(mask_channel)),
    )
    assert np.array_equal(
        npz_patch[PrognosAIs.Constants.LABEL_INDEX]["label_1"], np.asarray([1, 0])
    )

    channel_patches = [channel_1, channel_2]
    mask_patches = [mask_channel]

    labels = {"label_1": np.asarray([1, 0])}
    label_classes = {"label_1": 2}

    temp_sample = i_sample.copy()
    temp_sample.add_to_labels(labels, label_classes)

    preprocessor = SingleSamplePreprocessor(
        temp_sample, {"saving": {"use_mask_as_label": True, "named_channels": True}}
    )

    npz_patch = preprocessor._patch_to_data_structure(
        channel_patches, None, mask_patches, temp_sample.labels[0]
    )

    assert isinstance(npz_patch, dict)

    assert PrognosAIs.Constants.FEATURE_INDEX in npz_patch
    assert isinstance(npz_patch[PrognosAIs.Constants.FEATURE_INDEX], dict)
    assert npz_patch[PrognosAIs.Constants.FEATURE_INDEX]["Scan-0"].shape == pytest.approx(
        [5, 5, 5, 1]
    )
    assert npz_patch[PrognosAIs.Constants.FEATURE_INDEX]["Scan-1"].shape == pytest.approx(
        [5, 5, 5, 1]
    )
    assert np.array_equal(
        npz_patch[PrognosAIs.Constants.FEATURE_INDEX]["Scan-0"][:, :, :, 0],
        np.transpose(sitk.GetArrayFromImage(channel_1)),
    )
    assert np.array_equal(
        npz_patch[PrognosAIs.Constants.FEATURE_INDEX]["Scan-1"][:, :, :, 0],
        np.transpose(sitk.GetArrayFromImage(channel_2)),
    )
    assert PrognosAIs.Constants.LABEL_INDEX in npz_patch
    assert isinstance(npz_patch[PrognosAIs.Constants.LABEL_INDEX], dict)
    assert npz_patch[PrognosAIs.Constants.LABEL_INDEX]["MASK-0"].shape == pytest.approx(
        [5, 5, 5, 1]
    )
    assert np.array_equal(
        npz_patch[PrognosAIs.Constants.LABEL_INDEX]["MASK-0"][:, :, :, 0],
        np.transpose(sitk.GetArrayFromImage(mask_channel)),
    )
    assert np.array_equal(
        npz_patch[PrognosAIs.Constants.LABEL_INDEX]["label_1"], np.asarray([1, 0])
    )

    channel_1 = sitk.GetImageFromArray(np.random.random([5, 5, 5]))
    channel_2 = sitk.GetImageFromArray(np.random.random([5, 5, 5]))
    mask_channel = sitk.GetImageFromArray(np.random.random([5, 5, 5]))
    mask_channel_2 = sitk.GetImageFromArray(np.random.random([5, 5, 5]))

    channel_patches = [channel_1, channel_2]
    mask_patches = [mask_channel, mask_channel_2]

    labels = {"label_1": np.asarray([1, 0]), "label_2": np.asarray([0, 1])}
    label_classes = {"label_1": 2, "label_2": 2}
    temp_sample = i_sample.copy()
    temp_sample.add_to_labels(labels, label_classes)

    preprocessor = SingleSamplePreprocessor(
        temp_sample, {"saving": {"use_mask_as_label": True, "named_channels": False}}
    )

    npz_patch = preprocessor._patch_to_data_structure(
        channel_patches, None, mask_patches, temp_sample.labels[0]
    )

    assert isinstance(npz_patch, dict)

    assert PrognosAIs.Constants.FEATURE_INDEX in npz_patch
    assert isinstance(npz_patch[PrognosAIs.Constants.FEATURE_INDEX], dict)
    assert npz_patch[PrognosAIs.Constants.FEATURE_INDEX][
        PrognosAIs.Constants.FEATURE_INDEX
    ].shape == pytest.approx([5, 5, 5, 2])
    assert np.array_equal(
        npz_patch[PrognosAIs.Constants.FEATURE_INDEX][PrognosAIs.Constants.FEATURE_INDEX][
            :, :, :, 0
        ],
        np.transpose(sitk.GetArrayFromImage(channel_1)),
    )
    assert np.array_equal(
        npz_patch[PrognosAIs.Constants.FEATURE_INDEX][PrognosAIs.Constants.FEATURE_INDEX][
            :, :, :, 1
        ],
        np.transpose(sitk.GetArrayFromImage(channel_2)),
    )
    assert PrognosAIs.Constants.LABEL_INDEX in npz_patch
    assert isinstance(npz_patch[PrognosAIs.Constants.LABEL_INDEX], dict)
    assert npz_patch[PrognosAIs.Constants.LABEL_INDEX]["MASK"].shape == pytest.approx([5, 5, 5, 2])
    assert np.array_equal(
        npz_patch[PrognosAIs.Constants.LABEL_INDEX]["MASK"][:, :, :, 0],
        np.transpose(sitk.GetArrayFromImage(mask_channel)),
    )
    assert np.array_equal(
        npz_patch[PrognosAIs.Constants.LABEL_INDEX]["MASK"][:, :, :, 1],
        np.transpose(sitk.GetArrayFromImage(mask_channel_2)),
    )
    assert np.array_equal(
        npz_patch[PrognosAIs.Constants.LABEL_INDEX]["label_1"], np.asarray([1, 0])
    )
    assert np.array_equal(
        npz_patch[PrognosAIs.Constants.LABEL_INDEX]["label_2"], np.asarray([0, 1])
    )


@NIFTI_FILES
def test_get_number_of_classes(datafiles):
    samples = get_samples(datafiles)
    i_sample = samples[0]
    channel_1 = sitk.GetImageFromArray(np.random.random([5, 5, 5]))
    channel_2 = sitk.GetImageFromArray(np.random.random([5, 5, 5]))
    mask_channel = np.ones([5, 5, 5])
    mask_channel[0:2, 0:2, 0:2] = 0
    mask_channel[2:3, 2:3, 2:3] = 12
    mask_channel = sitk.GetImageFromArray(mask_channel)

    channel_patches = [channel_1, channel_2]
    mask_patches = [mask_channel]

    preprocessor = SingleSamplePreprocessor(i_sample, {"saving": {"use_mask_as_channel": False}})

    ds_structure = preprocessor._patch_to_data_structure(channel_patches, None, None, i_sample.labels[0])
    number_of_classes = preprocessor._get_number_of_classes(ds_structure)
    assert number_of_classes is None

    preprocessor = SingleSamplePreprocessor(i_sample, {"saving": {"use_mask_as_label": True}})

    ds_structure = preprocessor._patch_to_data_structure(
        channel_patches, None, mask_patches, i_sample.labels[0]
    )
    number_of_classes = preprocessor._get_number_of_classes(ds_structure)
    assert number_of_classes == {PrognosAIs.Constants.LABEL_INDEX: 3}

    labels = {"label_1": np.asarray([1, 0])}
    label_classes = {"label_1": 2}

    temp_sample = i_sample.copy()
    temp_sample.add_to_labels(labels, label_classes)

    preprocessor = SingleSamplePreprocessor(temp_sample, {"saving": {"use_mask_as_channel": False}})

    ds_structure = preprocessor._patch_to_data_structure(
        channel_patches, None, None, temp_sample.labels[0]
    )
    number_of_classes = preprocessor._get_number_of_classes(ds_structure)
    assert number_of_classes == {PrognosAIs.Constants.LABEL_INDEX: 2}

    labels = {"label_1": np.asarray([1, 0]), "label_2": np.asarray([0, 1, 0])}
    label_classes = {"label_1": 2, "label_2": 3}

    temp_sample = i_sample.copy()
    temp_sample.add_to_labels(labels, label_classes)

    preprocessor = SingleSamplePreprocessor(temp_sample, {"saving": {"use_mask_as_channel": False}})

    ds_structure = preprocessor._patch_to_data_structure(
        channel_patches, None, None, temp_sample.labels[0]
    )
    number_of_classes = preprocessor._get_number_of_classes(ds_structure)
    assert number_of_classes == label_classes

    labels = {"label_1": np.asarray([1, 0]), "label_2": np.asarray([0, 1, 0])}
    label_classes = {"label_1": 2, "label_2": 3}

    temp_sample = i_sample.copy()
    temp_sample.add_to_labels(labels, label_classes)

    preprocessor = SingleSamplePreprocessor(temp_sample, {"saving": {"use_mask_as_label": True}})

    ds_structure = preprocessor._patch_to_data_structure(
        channel_patches, None, mask_patches, temp_sample.labels[0]
    )
    number_of_classes = preprocessor._get_number_of_classes(ds_structure)
    assert number_of_classes == {"label_1": 2, "label_2": 3, "MASK": 3}

    temp_sample = i_sample.copy()
    preprocessor = SingleSamplePreprocessor(
        temp_sample, {"saving": {"use_mask_as_label": True, "named_channels": True}}
    )

    # Hack way to get extra masks in the sample
    preprocessor.sample._masks = {"MASK-0": 0, "MASK-1": 0}

    ds_structure = preprocessor._patch_to_data_structure(
        channel_patches, None, [mask_channel, mask_channel], temp_sample.labels[0]
    )
    number_of_classes = preprocessor._get_number_of_classes(ds_structure)
    assert number_of_classes == {"MASK-0": 3, "MASK-1": 3}


@NIFTI_FILES
def test_channel_imputation_last_channel(datafiles):
    sample = get_samples(datafiles)[0]
    preprocessor = SingleSamplePreprocessor(
        sample,
        {
            "saving": {
                "use_mask_as_channel": False,
                "impute_missing_channels": True,
                "channel_names": ["Scan-0", "Scan-1", "Scan-2", "Scan-3"],
            }
        },
    )
    patch_channels = sample.get_grouped_channels()[0]
    patch_channels = preprocessor._convert_sitk_arrays_to_numpy(patch_channels)

    result = preprocessor.channel_imputation(patch_channels)

    assert patch_channels.shape[-1] == 3
    assert result.shape[-1] == 4
    assert result[..., 0:3] == pytest.approx(patch_channels)
    assert result[..., 3] == pytest.approx(np.zeros(patch_channels.shape[0:-1]))


@NIFTI_FILES
def test_channel_imputation_first_channel(datafiles):
    sample = get_samples(datafiles)[0]
    preprocessor = SingleSamplePreprocessor(
        sample,
        {
            "saving": {
                "use_mask_as_channel": False,
                "impute_missing_channels": True,
                "channel_names": ["Scan-0", "Scan-1", "Scan-2", "ABC"],
            }
        },
    )
    patch_channels = sample.get_grouped_channels()[0]
    patch_channels = preprocessor._convert_sitk_arrays_to_numpy(patch_channels)

    result = preprocessor.channel_imputation(patch_channels)

    assert patch_channels.shape[-1] == 3
    assert result.shape[-1] == 4
    assert result[..., 1:4] == pytest.approx(patch_channels)
    assert result[..., 0] == pytest.approx(np.zeros(patch_channels.shape[0:-1]))


@NIFTI_FILES
def test_saving(datafiles):
    samples = get_samples(datafiles)

    for i_sample in samples:
        tmpdir = tempfile.mkdtemp()
        labels = {"label_1": [1, 0]}
        number_of_label_classes = {"label_1": 2}
        i_sample.labels = labels
        i_sample.number_of_label_classes = number_of_label_classes

        preprocessor = SingleSamplePreprocessor(
            i_sample, {"saving": {"type": "image"}}, output_directory=tmpdir,
        )

        preprocessor.saving()
        assert os.path.exists(os.path.join(tmpdir, "Samples"))
        assert len(os.listdir(os.path.join(tmpdir, "Samples"))) == 1
        assert os.path.exists(
            os.path.join(tmpdir, "Samples", i_sample.sample_name + "_patch_0.hdf5")
        )

        loaded_sample = h5py.File(
            os.path.join(tmpdir, "Samples", i_sample.sample_name + "_patch_0.hdf5"), "r"
        )
        assert list(loaded_sample.keys()) == [
            PrognosAIs.Constants.FEATURE_INDEX,
            PrognosAIs.Constants.LABEL_INDEX,
        ]
        sample = loaded_sample.get(PrognosAIs.Constants.FEATURE_INDEX)

        assert list(sample.keys()) == [PrognosAIs.Constants.FEATURE_INDEX]
        sample = sample.get(PrognosAIs.Constants.FEATURE_INDEX)
        assert sample.dtype.name == "float32"
        assert sample.attrs["size"] == pytest.approx(np.asarray([30, 30, 30]))
        assert sample.attrs["dimensionality"] == 3
        assert sample.attrs["N_channels"] == 3

        label = loaded_sample.get(PrognosAIs.Constants.LABEL_INDEX)

        assert list(label.keys()) == [PrognosAIs.Constants.LABEL_INDEX]

        assert np.asarray(label.get(PrognosAIs.Constants.LABEL_INDEX)) == pytest.approx(
            np.asarray([1, 0])
        )
        assert label.get(PrognosAIs.Constants.LABEL_INDEX).attrs["N_classes"] == 2

    # Check in case of empty channels
    i_sample = samples[0]
    tmpdir = tempfile.mkdtemp()
    labels = {"label_1": [1, 0]}
    number_of_label_classes = {"label_1": 2}
    i_sample.labels = labels
    i_sample.number_of_label_classes = number_of_label_classes

    preprocessor = SingleSamplePreprocessor(
        i_sample, {"saving": {"type": "image"}}, output_directory=tmpdir,
    )

    preprocessor.sample._channels = {}
    preprocessor.sample._masks = {}

    preprocessor.saving()
    assert os.path.exists(os.path.join(tmpdir, "Samples"))
    assert len(os.listdir(os.path.join(tmpdir, "Samples"))) == 0

@NIFTI_FILES
def test_saving_patches(datafiles):
    samples = get_samples(datafiles)
    i_sample = samples[0]
    tmpdir = tempfile.mkdtemp()
    labels = {"label_1": [1, 0]}
    number_of_label_classes = {"label_1": 2}
    i_sample.labels = labels
    i_sample.number_of_label_classes = number_of_label_classes
    preprocessor = SingleSamplePreprocessor(
        i_sample,
        {
            "saving": {"type": "image"},
            "patching": {
                "patch_size": [5, 5, 5],
                "pad_if_needed": True,
                "pad_constant": 0,
                "extraction_type": "random",
                "max_number_of_patches": 15,
            },
        },
        output_directory=tmpdir,
    )

    preprocessor.patching()
    preprocessor.saving()
    assert os.path.exists(os.path.join(tmpdir, "Samples"))
    assert len(os.listdir(os.path.join(tmpdir, "Samples"))) == 15


@NIFTI_FILES
def test_saving_imputed_channels(datafiles):
    samples = get_samples(datafiles)
    i_sample = samples[0]
    tmpdir = tempfile.mkdtemp()
    labels = {"label_1": [1, 0]}
    number_of_label_classes = {"label_1": 2}
    i_sample.labels = labels
    i_sample.number_of_label_classes = number_of_label_classes
    preprocessor = SingleSamplePreprocessor(
        i_sample,
        {
            "saving": {
                "use_mask_as_channel": False,
                "impute_missing_channels": True,
                "channel_names": ["Scan-0", "Scan-1", "Scan-2", "ABC"],
            }
        },
        output_directory=tmpdir,
    )

    preprocessor.saving()
    assert os.path.exists(os.path.join(tmpdir, "Samples"))

    loaded_sample = h5py.File(
        os.path.join(tmpdir, "Samples", i_sample.sample_name + "_patch_0.hdf5"), "r"
    )

    sample_channnels = loaded_sample.get(
        "/" + PrognosAIs.Constants.FEATURE_INDEX + "/" + PrognosAIs.Constants.FEATURE_INDEX
    )

    assert sample_channnels.shape == (30, 30, 30, 4)
    assert sample_channnels[..., 0] == pytest.approx(np.zeros([30, 30, 30]))

    assert sample_channnels.attrs["N_channels"] == 4
    assert sample_channnels.dtype.name == "float32"

@NIFTI_FILES
def test_saving_specific_channels(datafiles):
    tmpdir = tempfile.mkdtemp()

    samples=[]
    for i_sample_location in datafiles.listdir():
        samples.append(
            NIFTISample(
                root_path=i_sample_location, extension_keyword=".nii.gz", mask_keyword="MASK", input_channel_names=["Scan-0", "Scan-1"]
            )
        )

    sample = samples[0]
    preprocessor = SingleSamplePreprocessor(
        sample,
        {
            "saving": {
                "use_mask_as_channel": False,
                "impute_missing_channels": False,
                "channel_names": ["Scan-0", "Scan-1"],
            }
        },
        output_directory=tmpdir
    )

    preprocessor.saving()
    assert os.path.exists(os.path.join(tmpdir, "Samples"))
    assert len(os.listdir(os.path.join(tmpdir, "Samples"))) == 1
    assert os.path.exists(
        os.path.join(tmpdir, "Samples", sample.sample_name + "_patch_0.hdf5")
    )

    loaded_sample = h5py.File(
        os.path.join(tmpdir, "Samples", sample.sample_name + "_patch_0.hdf5"), "r"
    )
    assert list(loaded_sample.keys()) == [
        PrognosAIs.Constants.FEATURE_INDEX,
        PrognosAIs.Constants.LABEL_INDEX,
    ]
    sample = loaded_sample.get(PrognosAIs.Constants.FEATURE_INDEX)

    assert list(sample.keys()) == [PrognosAIs.Constants.FEATURE_INDEX]
    sample = sample.get(PrognosAIs.Constants.FEATURE_INDEX)
    assert sample.dtype.name == "float32"
    assert sample.attrs["size"] == pytest.approx(np.asarray([30, 30, 30]))
    assert sample.attrs["dimensionality"] == 3
    assert sample.attrs["N_channels"] == 2



@NIFTI_FILES
def test_save_float16(datafiles):
    samples = get_samples(datafiles)
    i_sample = samples[0]
    tmpdir = tempfile.mkdtemp()
    labels = {"label_1": [1, 0]}
    number_of_label_classes = {"label_1": 2}
    i_sample.labels = labels
    i_sample.number_of_label_classes = number_of_label_classes
    preprocessor = SingleSamplePreprocessor(
        i_sample,
        {"saving": {"use_mask_as_channel": False, "save_as_float16": True,}},
        output_directory=tmpdir,
    )
    preprocessor.saving()
    loaded_sample = h5py.File(
        os.path.join(tmpdir, "Samples", i_sample.sample_name + "_patch_0.hdf5"), "r"
    )

    result = loaded_sample.get(
        "/" + PrognosAIs.Constants.FEATURE_INDEX + "/" + PrognosAIs.Constants.FEATURE_INDEX
    )

    assert os.path.exists(os.path.join(tmpdir, "Samples"))
    assert result.attrs["N_channels"] == 3
    assert result.dtype.name == "float16"


# ===============================================================
# Labeling
# ===============================================================


def test_number_of_samples_in_subsets():
    tmpdir = tempfile.mkdtemp()
    train_fraction = 1.0
    val_fraction = 0.0
    test_fraction = 0.0
    N_samples = 50

    preprocessor = BatchPreprocessor(
        tmpdir,
        "bla",
        {
            "general": {"sample_type": "nifti"},
            "labeling": {"label_file": os.path.join(FIXTURE_DIR, "label_data.csv")},
        },
    )

    N_train, N_test, N_val = preprocessor._get_number_of_samples_in_subsets(
        train_fraction, val_fraction, test_fraction, N_samples
    )
    assert (N_train + N_test + N_val) == 50
    assert N_train == 50
    assert N_test == 0
    assert N_val == 0

    train_fraction = 0.8
    val_fraction = 0.2
    test_fraction = 0.0
    N_samples = 50

    preprocessor = BatchPreprocessor(
        tmpdir,
        "bla",
        {
            "general": {"sample_type": "nifti"},
            "labeling": {"label_file": os.path.join(FIXTURE_DIR, "label_data.csv")},
        },
    )

    N_train, N_test, N_val = preprocessor._get_number_of_samples_in_subsets(
        train_fraction, val_fraction, test_fraction, N_samples
    )
    assert (N_train + N_test + N_val) == 50
    assert N_train == 40
    assert N_test == 10
    assert N_val == 0

    train_fraction = 0.8
    val_fraction = 0.2
    test_fraction = 0.0
    N_samples = 101

    preprocessor = BatchPreprocessor(
        tmpdir,
        "bla",
        {
            "general": {"sample_type": "nifti"},
            "labeling": {"label_file": os.path.join(FIXTURE_DIR, "label_data.csv")},
        },
    )

    N_train, N_test, N_val = preprocessor._get_number_of_samples_in_subsets(
        train_fraction, val_fraction, test_fraction, N_samples
    )
    assert (N_train + N_test + N_val) == 101
    assert N_train == 81
    assert N_test == 20
    assert N_val == 0

    train_fraction = 0.6
    val_fraction = 0.3
    test_fraction = 0.1
    N_samples = 101

    preprocessor = BatchPreprocessor(
        tmpdir,
        "bla",
        {
            "general": {"sample_type": "nifti"},
            "labeling": {"label_file": os.path.join(FIXTURE_DIR, "label_data.csv")},
        },
    )

    N_train, N_test, N_val = preprocessor._get_number_of_samples_in_subsets(
        train_fraction, val_fraction, test_fraction, N_samples
    )
    assert (N_train + N_test + N_val) == 101
    assert N_train == 61
    assert N_test == 30
    assert N_val == 10

    train_fraction = 0.656
    val_fraction = 0.256
    test_fraction = 0.088
    N_samples = 100

    preprocessor = BatchPreprocessor(
        tmpdir,
        "bla",
        {
            "general": {"sample_type": "nifti"},
            "labeling": {"label_file": os.path.join(FIXTURE_DIR, "label_data.csv")},
        },
    )

    N_train, N_test, N_val = preprocessor._get_number_of_samples_in_subsets(
        train_fraction, val_fraction, test_fraction, N_samples
    )
    assert (N_train + N_test + N_val) == 100
    assert N_train == 67
    assert N_test == 25
    assert N_val == 8


@NIFTI_FILES
def test_batch_preprocessor_with_specific_channels(datafiles):
    tmpdir = tempfile.mkdtemp()

    preprocessor = BatchPreprocessor(
        datafiles,
        tmpdir,
        {
            "general": {"sample_type": "nifti"},
            "saving": {"channel_names": ["Scan-0"]},
        },
    )

    assert preprocessor.saving_config.channel_names == ["Scan-0"]




def test_get_subset():
    tmpdir = tempfile.mkdtemp()
    preprocessor = BatchPreprocessor(
        tmpdir,
        "bla",
        {
            "general": {"sample_type": "nifti"},
            "labeling": {"label_file": os.path.join(FIXTURE_DIR, "label_data.csv")},
        },
    )

    samples = np.zeros(50)

    train, test, strat_labels = preprocessor._get_data_split(samples, 0)

    assert train == pytest.approx(np.zeros(50))
    assert test is None
    assert strat_labels is None

    samples = np.zeros(50)
    train, test, strat_labels = preprocessor._get_data_split(samples, 12)
    assert train == pytest.approx(np.zeros(38))
    assert test == pytest.approx(np.zeros(12))
    assert strat_labels is None

    samples = np.zeros(50)
    train, test, strat_labels = preprocessor._get_data_split(samples, 12)
    assert train == pytest.approx(np.zeros(38))
    assert test == pytest.approx(np.zeros(12))
    assert strat_labels is None

    samples = np.zeros(50)
    samples[25:] = 1
    stratification_labels = np.zeros(50)
    stratification_labels[25:] = 1
    train, test, strat_labels = preprocessor._get_data_split(
        samples, 12, stratification_labels=stratification_labels
    )
    assert len(train) == 38
    assert len(test) == 12
    assert len(strat_labels) == 38
    assert train == pytest.approx(strat_labels)

    assert np.count_nonzero(train) / 38 == pytest.approx(np.count_nonzero(test) / 12)


@NIFTI_FILES
def test_mask_as_label(datafiles):
    single_datafile = list(datafiles.listdir())[0]
    sample = NIFTISample(root_path=single_datafile, mask_keyword="MASK", are_labels_one_hot=False)
    tmpdir = tempfile.mkdtemp()
    preprocessor = SingleSamplePreprocessor(
        sample, {"saving": {"type": "image", "use_mask_as_label": True}}, output_directory=tmpdir,
    )

    result = preprocessor.saving()

    assert result is None
    assert os.path.exists(os.path.join(tmpdir, "Samples"))
    assert len(os.listdir(os.path.join(tmpdir, "Samples"))) == 1
    file_name = os.path.join(tmpdir, "Samples", sample.sample_name + "_patch_0.hdf5")
    assert os.path.exists(file_name)
    with h5py.File(file_name, "r") as the_sample:
        assert the_sample.get(
            "/" + PrognosAIs.Constants.LABEL_INDEX + "/" + PrognosAIs.Constants.LABEL_INDEX
        ).shape == (30, 30, 30, 1)


@NIFTI_FILES
def test_mask_as_label_one_hot(datafiles):
    single_datafile = list(datafiles.listdir())[0]
    sample = NIFTISample(root_path=single_datafile, mask_keyword="MASK", are_labels_one_hot=True)
    tmpdir = tempfile.mkdtemp()
    preprocessor = SingleSamplePreprocessor(
        sample,
        {
            "saving": {"type": "image", "use_mask_as_label": True, "mask_channels": 2},
            "normalizing": {"mask_normalization": "collapse"},
        },
        output_directory=tmpdir,
    )
    preprocessor.normalizing()

    result = preprocessor.saving()

    assert result is None
    assert os.path.exists(os.path.join(tmpdir, "Samples"))
    assert len(os.listdir(os.path.join(tmpdir, "Samples"))) == 1
    file_name = os.path.join(tmpdir, "Samples", sample.sample_name + "_patch_0.hdf5")
    assert os.path.exists(file_name)
    with h5py.File(file_name, "r") as the_sample:
        assert the_sample.get(
            "/" + PrognosAIs.Constants.LABEL_INDEX + "/" + PrognosAIs.Constants.LABEL_INDEX
        ).shape == (30, 30, 30, 2)


# ===============================================================
# Batchprocessor
# ===============================================================


def test_batch_processor_init():
    tmp = tempfile.mkdtemp()
    batch_preprocessor = BatchPreprocessor(
        os.path.join(FIXTURE_DIR, "SAMPLES"),
        tmp,
        {
            "general": {"sample_type": "nifti"},
            "labeling": {"label_file": os.path.join(FIXTURE_DIR, "label_data.csv")},
        },
    )

    assert batch_preprocessor.label_loader is not None

    tmp = tempfile.mkdtemp()
    batch_preprocessor = BatchPreprocessor(
        os.path.join(FIXTURE_DIR, "SAMPLES"), tmp, {"general": {"sample_type": "nifti"}}
    )

    assert batch_preprocessor.label_loader is None


@NIFTI_FILES
def test_batch_single_sample(datafiles):
    tmp = tempfile.mkdtemp()

    batch_preprocessor = BatchPreprocessor(
        os.path.join(FIXTURE_DIR, "SAMPLES"),
        tmp,
        {"general": {"sample_type": "nifti"}, "saving": {"type": "image"}},
    )
    single_sample_folder = sorted(list(datafiles.listdir()))[0]
    save_names, sample_labels = batch_preprocessor._run_single_sample(single_sample_folder)
    assert isinstance(save_names, list)
    assert isinstance(save_names[0], str)
    assert sample_labels == {}

    tmp = tempfile.mkdtemp()
    batch_preprocessor = BatchPreprocessor(
        os.path.join(FIXTURE_DIR, "SAMPLES"),
        tmp,
        {
            "general": {"sample_type": "nifti"},
            "labeling": {"label_file": os.path.join(FIXTURE_DIR, "label_data.csv")},
            "saving": {"type": "image"},
        },
    )
    single_sample_folder = sorted(list(datafiles.listdir()))[0]
    save_names, sample_labels = batch_preprocessor._run_single_sample(single_sample_folder)
    assert isinstance(save_names, list)
    assert isinstance(save_names[0], str)
    assert isinstance(sample_labels, dict)
    assert sample_labels["Label"] == 1


@NIFTI_FILES
def test_batch_single_sample_with_output_channels(datafiles):
    tmp = tempfile.mkdtemp()

    batch_preprocessor = BatchPreprocessor(
        os.path.join(FIXTURE_DIR, "SAMPLES"),
        tmp,
        {"general": {"sample_type": "nifti", "output_channel_names": ["Scan-2"]}, "saving": {"type": "image"}},
    )
    single_sample_folder = sorted(list(datafiles.listdir()))[0]
    save_names, sample_labels = batch_preprocessor._run_single_sample(single_sample_folder)
    assert isinstance(save_names, list)
    assert isinstance(save_names[0], str)
    assert sample_labels == {}


    loaded_sample = h5py.File(save_names[0], "r")
    label_keys = [i_key for i_key in loaded_sample[PrognosAIs.Constants.LABEL_INDEX].keys()]
    assert label_keys == ["Scan-2"]
    assert np.asarray(loaded_sample[PrognosAIs.Constants.FEATURE_INDEX][PrognosAIs.Constants.FEATURE_INDEX]).shape == (30, 30, 30, 3)


def test_batch_running():
    tmp = tempfile.mkdtemp()
    config = {
        "general": {"sample_type": "nifti", "pipeline": ["saving"], "max_cpus": 1},
        "labeling": {
            "label_file": os.path.join(FIXTURE_DIR, "label_data.csv"),
            "make_one_hot": True,
        },
    }
    batch_preprocessor = BatchPreprocessor(os.path.join(FIXTURE_DIR, "SAMPLES"), tmp, config)
    batch_preprocessor.start()
    assert batch_preprocessor.config == config

    sample_file = sorted([i.path for i in os.scandir(os.path.join(tmp, "Samples", "train"))])[0]
    loaded_sample = h5py.File(sample_file, "r")
    assert np.asarray(
        loaded_sample[PrognosAIs.Constants.LABEL_INDEX][PrognosAIs.Constants.LABEL_INDEX]
    ) == pytest.approx(np.asarray([1, 0, 0]))


def test_batch_running_stratify():
    tmp = tempfile.mkdtemp()
    config = {
        "general": {"sample_type": "nifti", "pipeline": ["saving"], "max_cpus": 1},
        "labeling": {
            "label_file": os.path.join(FIXTURE_DIR, "MULTI_SAMPLES", "label_data.csv"),
            "make_one_hot": True,
            "stratify_label_name": "Label",
            "train_fraction": 0.5,
            "test_fraction": 0.5,
        },
    }
    batch_preprocessor = BatchPreprocessor(os.path.join(FIXTURE_DIR, "MULTI_SAMPLES"), tmp, config)
    batch_preprocessor.start()
    assert batch_preprocessor.config == config

    train_samples = [i.path for i in os.scandir(os.path.join(tmp, "Samples", "train"))]
    test_samples = [i.path for i in os.scandir(os.path.join(tmp, "Samples", "test"))]

    train_labels = []
    for i_train_sample in train_samples:
        loaded_sample = h5py.File(i_train_sample, "r")
        train_labels.append(
            np.asarray(
                loaded_sample[PrognosAIs.Constants.LABEL_INDEX][PrognosAIs.Constants.LABEL_INDEX]
            ).tolist()
        )
        loaded_sample.close()

    test_labels = []
    for i_test_sample in test_samples:
        loaded_sample = h5py.File(i_test_sample, "r")
        test_labels.append(
            np.asarray(
                loaded_sample[PrognosAIs.Constants.LABEL_INDEX][PrognosAIs.Constants.LABEL_INDEX]
            ).tolist()
        )
        loaded_sample.close()

    # We make sure that the label in each are the same, then it is properly splitted, stratified
    assert sorted(train_samples) != sorted(test_samples)
    assert sorted(train_labels) == sorted(test_labels)

    # sample_file = sorted([i.path for i in os.scandir(os.path.join(tmp, "Samples", "train"))])[0]
    # assert np.asarray(loaded_sample[PrognosAIs.Constants.LABEL_INDEX][PrognosAIs.Constants.LABEL_INDEX]) == pytest.approx(np.asarray([1, 0, 0]))


def test_batch_from_cli():
    tmp = tempfile.mkdtemp()
    batch_preprocessor = BatchPreprocessor.init_from_sys_args(
        ["--config", CONFIG_FILE, "--input", os.path.join(FIXTURE_DIR, "SAMPLES"), "--output", tmp]
    )
    # batch_preprocessor.start()
    assert isinstance(batch_preprocessor, BatchPreprocessor)
    assert batch_preprocessor.output_directory == tmp
