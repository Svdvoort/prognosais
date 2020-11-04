import os

import numpy as np
import pytest
import SimpleITK as sitk

from PrognosAIs.Preprocessing.Samples import ImageSample
from PrognosAIs.Preprocessing.Samples import NIFTISample


FIXTURE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_data", "Nifti_Data",)

NIFTI_FILES = pytest.mark.datafiles(
    os.path.join(FIXTURE_DIR, "Subject-000"),
    os.path.join(FIXTURE_DIR, "Subject-001"),
    os.path.join(FIXTURE_DIR, "Subject-002"),
    keep_top_dir=True,
)


def _copy_square_image(image):
    second_image = sitk.Square(image)
    return [image, second_image]


def make_sample_with_patches(sample):
    sample.channels = _copy_square_image
    sample.masks = _copy_square_image
    return sample


# ===============================================================
# Base class
# ===============================================================


def test_dtype_from_scalar():
    assert ImageSample.get_appropiate_dtype_from_scalar(254) == sitk.sitkUInt8
    assert ImageSample.get_appropiate_dtype_from_scalar(-1.0) == sitk.sitkInt8
    assert ImageSample.get_appropiate_dtype_from_scalar(0.5) == sitk.sitkFloat32
    assert ImageSample.get_appropiate_dtype_from_scalar(280) == sitk.sitkUInt16
    assert ImageSample.get_appropiate_dtype_from_scalar(-280) == sitk.sitkInt16
    assert ImageSample.get_appropiate_dtype_from_scalar(4e9) == sitk.sitkUInt32
    assert ImageSample.get_appropiate_dtype_from_scalar(-2e9) == sitk.sitkInt32
    assert ImageSample.get_appropiate_dtype_from_scalar(1e80) == sitk.sitkFloat64


def test_dtype_from_image():
    value = 0.0
    image = sitk.GetImageFromArray(np.ones([10, 10, 10]) * value)

    assert ImageSample.get_appropiate_dtype_from_image(image) == sitk.sitkUInt8

    value = -1
    image = sitk.GetImageFromArray(np.ones([10, 10, 10]) * value)

    assert ImageSample.get_appropiate_dtype_from_image(image) == sitk.sitkInt8

    value = 12280.12
    image = sitk.GetImageFromArray(np.ones([10, 10, 10]) * value)

    assert ImageSample.get_appropiate_dtype_from_image(image) == sitk.sitkFloat32

    image_array = np.ones([10, 10, 10])
    image_array[0, ...] = -8103
    image_array[..., 0] = 8402
    image = sitk.GetImageFromArray(image_array)

    assert ImageSample.get_appropiate_dtype_from_image(image) == sitk.sitkInt16

    image_array = np.ones([10, 10, 10])
    image_array[0, ...] = -8103
    image_array[..., 0] = 8402
    image_array[5, 5, 5] = 1.5
    image = sitk.GetImageFromArray(image_array)

    assert ImageSample.get_appropiate_dtype_from_image(image) == sitk.sitkFloat32

    image_array = np.ones([10, 10, 10])
    image_array[0, ...] = -8103
    image_array[..., 0] = 8402
    image_array[5, 5, 5] = 1e80
    image = sitk.GetImageFromArray(image_array)

    assert ImageSample.get_appropiate_dtype_from_image(image) == sitk.sitkFloat64

    image_array = np.ones([10, 10, 10])
    image_array[0, ...] = -8103
    image_array[..., 0] = 8402
    image_array[5, 5, 5] = 1.5
    image = sitk.GetImageFromArray(image_array)
    image = sitk.Cast(image, sitk.sitkFloat32)

    assert ImageSample.get_appropiate_dtype_from_image(image) == sitk.sitkFloat32

    image_array = np.ones([10, 10, 10])
    image_array[0, ...] = -8103.0
    image_array[..., 0] = 402767
    image = sitk.GetImageFromArray(image_array)

    assert ImageSample.get_appropiate_dtype_from_image(image) == sitk.sitkInt32


def test_sitk_common_type():
    assert ImageSample.promote_simpleitk_types(sitk.sitkUInt8, sitk.sitkUInt8) == sitk.sitkUInt8
    assert ImageSample.promote_simpleitk_types(sitk.sitkUInt8, sitk.sitkUInt32) == sitk.sitkUInt32
    assert ImageSample.promote_simpleitk_types(sitk.sitkUInt32, sitk.sitkUInt8) == sitk.sitkUInt32
    assert ImageSample.promote_simpleitk_types(sitk.sitkInt8, sitk.sitkUInt16) == sitk.sitkInt32
    assert (
        ImageSample.promote_simpleitk_types(sitk.sitkFloat32, sitk.sitkUInt16) == sitk.sitkFloat32
    )


# ===============================================================
# NIFTISample
# ===============================================================
@NIFTI_FILES
def test_sample_loading(datafiles):
    for i_sample_location in datafiles.listdir():
        NIFTISample(
            root_path=i_sample_location, extension_keyword=".nii.gz",
        )


@NIFTI_FILES
def test_copying(datafiles):
    for i_sample_location in datafiles.listdir():
        sample = NIFTISample(root_path=i_sample_location, extension_keyword=".nii.gz",)

        copy_sample = sample.copy()

        assert copy_sample.number_of_channels == sample.number_of_channels
        sample.number_of_channels += 5
        assert copy_sample.number_of_channels != sample.number_of_channels


@NIFTI_FILES
def test_channel_loading(datafiles):
    for i_sample_location in datafiles.listdir():
        sample = NIFTISample(root_path=i_sample_location, extension_keyword=".nii.gz",)

        # Check whether number of channels is correctly parsed
        assert isinstance(sample.number_of_channels, int)
        assert sample.number_of_channels == 4
        assert sample.number_of_masks == 0

        assert isinstance(sample.channels, list)
        assert len(sample.channels) == sample.number_of_channels

        assert isinstance(sample.channel_names, list)
        assert sample.channel_names == ["MASK-0", "Scan-0", "Scan-1", "Scan-2"]

        for channel_name, channel in zip(sample.channel_names, sample.channels):
            channel = sitk.GetArrayFromImage(channel)
            sample_number = int(channel_name.split("-")[1])
            if "MASK" in channel_name:
                assert channel[:, sample_number, :] == pytest.approx(1)
            else:
                assert channel[sample_number, sample_number, sample_number] == pytest.approx(100)
                assert channel[
                    sample_number + 1, sample_number + 1, sample_number + 1
                ] == pytest.approx(250)


@NIFTI_FILES
def test_channel_and_mask_loading(datafiles):
    for i_sample_location in datafiles.listdir():
        sample = NIFTISample(
            root_path=i_sample_location, extension_keyword=".nii.gz", mask_keyword="MASK",
        )

        assert sample.number_of_channels == 3

        assert isinstance(sample.number_of_masks, int)
        assert sample.number_of_masks == 1

        assert isinstance(sample.masks, list)
        assert len(sample.masks) == sample.number_of_masks

        assert isinstance(sample.masks, list)
        assert sample.mask_names == ["MASK-0"]
        assert sample.channel_names == ["Scan-0", "Scan-1", "Scan-2"]

        for mask_name, mask in zip(sample.mask_names, sample.masks):
            mask = sitk.GetArrayFromImage(mask)
            sample_number = int(mask_name.split("-")[1])
            assert mask[:, sample_number, :] == pytest.approx(1)
            assert mask[:, sample_number + 1, :] == pytest.approx(105)


@NIFTI_FILES
def test_channel_example(datafiles):
    for i_sample_location in datafiles.listdir():
        sample = NIFTISample(root_path=i_sample_location, extension_keyword=".nii.gz",)

        assert isinstance(sample.get_example_channel(), sitk.Image)
        assert sample.get_example_mask() is None

        sample = make_sample_with_patches(sample)

        assert isinstance(sample.get_example_channel(), sitk.Image)
        assert sample.get_example_mask() is None


@NIFTI_FILES
def test_channel_and_mask_example(datafiles):
    for i_sample_location in datafiles.listdir():
        sample = NIFTISample(
            root_path=i_sample_location, extension_keyword=".nii.gz", mask_keyword="MASK",
        )

        assert isinstance(sample.get_example_channel(), sitk.Image)
        assert isinstance(sample.get_example_mask(), sitk.Image)

        sample = make_sample_with_patches(sample)

        assert isinstance(sample.get_example_channel(), sitk.Image)
        assert isinstance(sample.get_example_mask(), sitk.Image)


@NIFTI_FILES
def test_patch_making(datafiles):
    for i_sample_location in datafiles.listdir():
        sample = NIFTISample(
            root_path=i_sample_location, extension_keyword=".nii.gz", mask_keyword="MASK",
        )

        patch_sample = make_sample_with_patches(sample)

        assert patch_sample.has_patches == True
        assert patch_sample.number_of_patches == 2

        for i_i_channel, i_channel in enumerate(patch_sample.channels):
            assert i_channel[0][i_i_channel, i_i_channel, i_i_channel] == pytest.approx(100)
            assert i_channel[0][i_i_channel + 1, i_i_channel + 1, i_i_channel + 1] == pytest.approx(
                250
            )
            assert i_channel[1][i_i_channel, i_i_channel, i_i_channel] == pytest.approx(10000)
            assert i_channel[1][i_i_channel + 1, i_i_channel + 1, i_i_channel + 1] == pytest.approx(
                62500
            )

        patch_sample.channels = sitk.Square
        assert patch_sample.has_patches == True
        assert patch_sample.number_of_patches == 2

        for i_i_channel, i_channel in enumerate(patch_sample.channels):
            assert i_channel[0][i_i_channel, i_i_channel, i_i_channel] == pytest.approx(10000)
            assert i_channel[0][i_i_channel + 1, i_i_channel + 1, i_i_channel + 1] == pytest.approx(
                62500
            )
            assert i_channel[1][i_i_channel, i_i_channel, i_i_channel] == pytest.approx(10000 ** 2)
            assert i_channel[1][i_i_channel + 1, i_i_channel + 1, i_i_channel + 1] == pytest.approx(
                62500 ** 2
            )

        patch_sample.masks = (sitk.Add, [1])

        for i_i_mask, i_mask in enumerate(patch_sample.masks):
            assert i_mask[0][:, i_i_mask, :] == 2
            assert i_mask[1][:, i_i_mask, :] == 2


@pytest.mark.datafiles(os.path.join(FIXTURE_DIR, "Subject-003"), keep_top_dir=True)
def test_masks_unequal_to_channels(datafiles):
    with pytest.raises(NotImplementedError):
        for i_sample_location in datafiles.listdir():
            NIFTISample(
                root_path=i_sample_location, extension_keyword=".nii.gz", mask_keyword="MASK",
            )


@NIFTI_FILES
def test_size(datafiles):
    for i_sample_location in datafiles.listdir():
        sample = NIFTISample(root_path=i_sample_location, extension_keyword=".nii.gz")

        assert isinstance(sample.channel_size, np.ndarray)
        assert sample.channel_size == pytest.approx(np.asarray([30, 30, 30]))
        assert sample.mask_size is None


@NIFTI_FILES
def test_size_with_mask(datafiles):
    for i_sample_location in datafiles.listdir():
        sample = NIFTISample(
            root_path=i_sample_location, extension_keyword=".nii.gz", mask_keyword="MASK",
        )

        assert isinstance(sample.channel_size, np.ndarray)
        assert sample.channel_size == pytest.approx(np.asarray([30, 30, 30]))
        assert isinstance(sample.mask_size, np.ndarray)
        assert sample.mask_size == pytest.approx(np.asarray([30, 30, 30]))


@NIFTI_FILES
def test_function_parser(datafiles):
    sample_location = datafiles.listdir()[0]
    sample = NIFTISample(
        root_path=sample_location, extension_keyword=".nii.gz", mask_keyword="MASK"
    )

    parsed = sample._parse_function_parameters(os.path.join)
    assert isinstance(parsed, tuple)
    assert len(parsed) == 3
    (fnc, args, kw_args) = parsed

    assert fnc == os.path.join
    assert hasattr(fnc, "__call__")

    assert isinstance(args, list)
    assert isinstance(kw_args, dict)

    assert args == []
    assert kw_args == {}

    parsed = sample._parse_function_parameters((os.path.join))
    assert isinstance(parsed, tuple)
    assert len(parsed) == 3
    (fnc, args, kw_args) = parsed

    assert fnc == os.path.join
    assert hasattr(fnc, "__call__")

    assert isinstance(args, list)
    assert isinstance(kw_args, dict)

    assert args == []
    assert kw_args == {}

    parsed = sample._parse_function_parameters((os.path.join, [5, "abc"]))
    assert isinstance(parsed, tuple)
    assert len(parsed) == 3
    (fnc, args, kw_args) = parsed

    assert fnc == os.path.join
    assert hasattr(fnc, "__call__")

    assert isinstance(args, list)
    assert isinstance(kw_args, dict)

    assert args == [5, "abc"]
    assert kw_args == {}

    parsed = sample._parse_function_parameters((os.path.join, {"input_4": 1, "input_1": 17}))
    assert isinstance(parsed, tuple)
    assert len(parsed) == 3
    (fnc, args, kw_args) = parsed

    assert fnc == os.path.join
    assert hasattr(fnc, "__call__")

    assert isinstance(args, list)
    assert isinstance(kw_args, dict)

    assert args == []
    assert kw_args == {"input_4": 1, "input_1": 17}

    parsed = sample._parse_function_parameters(
        (os.path.join, [5, "abc"], {"input_4": 1, "input_1": 17})
    )
    assert isinstance(parsed, tuple)
    assert len(parsed) == 3
    (fnc, args, kw_args) = parsed

    assert fnc == os.path.join
    assert hasattr(fnc, "__call__")

    assert isinstance(args, list)
    assert isinstance(kw_args, dict)

    assert args == [5, "abc"]
    assert kw_args == {"input_4": 1, "input_1": 17}

    parsed = sample._parse_function_parameters(
        (os.path.join, {"input_4": 1, "input_1": 17}, [5, "abc"])
    )
    assert isinstance(parsed, tuple)
    assert len(parsed) == 3
    (fnc, args, kw_args) = parsed

    assert fnc == os.path.join
    assert hasattr(fnc, "__call__")

    assert isinstance(args, list)
    assert isinstance(kw_args, dict)

    assert args == [5, "abc"]
    assert kw_args == {"input_4": 1, "input_1": 17}


@NIFTI_FILES
def test_channel_modifying(datafiles):
    for i_sample_location in datafiles.listdir():
        sample = NIFTISample(root_path=i_sample_location, extension_keyword=".nii.gz",)

        channels = sample.channels

        sample.channels = sitk.Square

        assert isinstance(sample.channels, list)

        for i_i_channel, i_channel in enumerate(sample.channels):
            assert sitk.GetArrayFromImage(i_channel) == pytest.approx(
                np.power(sitk.GetArrayFromImage(channels[i_i_channel]), 2)
            )

        sample.masks = sitk.Square


@NIFTI_FILES
def test_channel_and_mask_modifying(datafiles):
    for i_sample_location in datafiles.listdir():
        sample = NIFTISample(
            root_path=i_sample_location, extension_keyword=".nii.gz", mask_keyword="MASK",
        )

        masks = sample.masks
        channels = sample.channels

        sample.masks = sitk.Square

        assert isinstance(sample.masks, list)

        for i_i_mask, i_mask in enumerate(sample.masks):
            assert sitk.GetArrayFromImage(i_mask) == pytest.approx(
                np.power(sitk.GetArrayFromImage(masks[i_i_mask]), 2)
            )

        # Make sure the channels didnt get updated
        for i_i_channel, i_channel in enumerate(sample.channels):
            assert sitk.GetArrayFromImage(i_channel) == pytest.approx(
                sitk.GetArrayFromImage(channels[i_i_channel])
            )


@NIFTI_FILES
def test_channel_size_assertion(datafiles):
    for i_sample_location in datafiles.listdir():
        sample = NIFTISample(root_path=i_sample_location, extension_keyword=".nii.gz",)
        sample.assert_all_channels_same_size()
        sample.assert_all_masks_same_size()

        sample.channel_size = [10, 10, 10]

        with pytest.raises(ValueError):
            sample.assert_all_channels_same_size()

        sample = make_sample_with_patches(sample)

        sample.assert_all_channels_same_size()
        sample.assert_all_masks_same_size()


@NIFTI_FILES
def test_channel_and_mask_size_assertion(datafiles):
    for i_sample_location in datafiles.listdir():
        sample = NIFTISample(
            root_path=i_sample_location, extension_keyword=".nii.gz", mask_keyword="MASK",
        )
        sample.assert_all_channels_same_size()

        sample.channel_size = [10, 10, 10]

        with pytest.raises(ValueError):
            sample.assert_all_channels_same_size()

        sample.assert_all_masks_same_size()

        sample.mask_size = [10, 10, 10]
        sample.channel_size = [30, 30, 30]

        with pytest.raises(ValueError):
            sample.assert_all_masks_same_size()

        sample.assert_all_channels_same_size()

        sample = make_sample_with_patches(sample)
        sample.assert_all_channels_same_size()
        sample.assert_all_masks_same_size()


@NIFTI_FILES
def test_grouped_channels_and_masks(datafiles):
    for i_sample_location in datafiles.listdir():
        sample = NIFTISample(
            root_path=i_sample_location, extension_keyword=".nii.gz", mask_keyword="MASK",
        )

        grouped_channels = sample.get_grouped_channels()
        assert isinstance(grouped_channels, list)
        assert len(grouped_channels) == 1
        assert isinstance(grouped_channels[0], list)
        assert len(grouped_channels[0]) == 3

        for i_i_channel, i_channel in enumerate(grouped_channels[0]):
            assert i_channel[i_i_channel, i_i_channel, i_i_channel] == pytest.approx(100)
            assert i_channel[i_i_channel + 1, i_i_channel + 1, i_i_channel + 1] == pytest.approx(
                250
            )

        grouped_masks = sample.get_grouped_masks()
        assert isinstance(grouped_masks, list)
        assert len(grouped_masks) == 1
        assert isinstance(grouped_masks[0], list)
        assert len(grouped_masks[0]) == 1

        for i_i_mask, i_mask in enumerate(grouped_masks[0]):
            assert i_mask[:, i_i_mask, :] == 1

        patches_sample = make_sample_with_patches(sample)
        grouped_channels = patches_sample.get_grouped_channels()
        assert isinstance(grouped_channels, list)
        assert len(grouped_channels) == 2
        assert isinstance(grouped_channels[0], list)
        assert len(grouped_channels[0]) == 3

        for i_i_channel, i_channel in enumerate(grouped_channels[0]):
            assert i_channel[i_i_channel, i_i_channel, i_i_channel] == pytest.approx(100)
            assert i_channel[i_i_channel + 1, i_i_channel + 1, i_i_channel + 1] == pytest.approx(
                250
            )

        for i_i_channel, i_channel in enumerate(grouped_channels[1]):
            assert i_channel[i_i_channel, i_i_channel, i_i_channel] == pytest.approx(10000)
            assert i_channel[i_i_channel + 1, i_i_channel + 1, i_i_channel + 1] == pytest.approx(
                62500
            )

        grouped_masks = patches_sample.get_grouped_masks()
        assert isinstance(grouped_masks, list)
        assert len(grouped_masks) == 2
        assert isinstance(grouped_masks[0], list)
        assert len(grouped_masks[0]) == 1

        for i_i_mask, i_mask in enumerate(grouped_masks[0]):
            assert i_mask[:, i_i_mask, :] == 1


# ===============================================================
# Label adding
# ===============================================================


@NIFTI_FILES
def test_add_label_no_initial_label(datafiles):
    sample_location = datafiles.listdir()[0]
    sample = NIFTISample(
        root_path=sample_location, extension_keyword=".nii.gz", mask_keyword="MASK"
    )
    new_label = [{"class_1": [0, 1]}]
    new_number_of_classes = {"class_1": 2}

    sample.add_to_labels(new_label, new_number_of_classes)

    assert sample.labels == [{"class_1": [0, 1]}]
    assert sample.number_of_label_classes == {"class_1": 2}


@NIFTI_FILES
def test_add_label_initial_label(datafiles):
    sample_location = datafiles.listdir()[0]
    sample = NIFTISample(
        root_path=sample_location,
        extension_keyword=".nii.gz",
        mask_keyword="MASK",
        labels={"class_1": [0, 1]},
        number_of_label_classes={"class_1": 2},
    )
    new_label = [{"class_2": [1, 0]}]
    new_number_of_classes = {"class_2": 2}

    sample.add_to_labels(new_label, new_number_of_classes)

    assert sample.labels == [{"class_1": [0, 1], "class_2": [1, 0]}]
    assert sample.number_of_label_classes == {"class_1": 2, "class_2": 2}


@NIFTI_FILES
def test_add_label_as_dict_initial_label(datafiles):
    sample_location = datafiles.listdir()[0]
    sample = NIFTISample(
        root_path=sample_location,
        extension_keyword=".nii.gz",
        mask_keyword="MASK",
        labels={"class_1": [0, 1]},
        number_of_label_classes={"class_1": 2},
    )
    new_label = {"class_2": [1, 0]}
    new_number_of_classes = {"class_2": 2}

    sample.add_to_labels(new_label, new_number_of_classes)

    assert sample.labels == [{"class_1": [0, 1], "class_2": [1, 0]}]
    assert sample.number_of_label_classes == {"class_1": 2, "class_2": 2}
