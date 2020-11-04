import os

import numpy as np
import SimpleITK as sitk


img_size = [30, 30, 30]
N_channels = 3
N_masks = 1
mask_keyword = "MASK"
N_subjects = 3
test_data_dir = os.path.dirname(os.path.realpath(__file__))

for i_subject in range(N_subjects):
    subject_name = "Subject-" + str(i_subject).zfill(3)
    subject_dir = os.path.join(test_data_dir, subject_name)

    if not os.path.exists(subject_dir):
        os.makedirs(subject_dir)

    for i_channel in range(N_channels):
        scan_data = np.zeros(img_size)
        scan_data[i_channel, ...] = 50
        scan_data[i_channel, i_channel, i_channel] = 100
        scan_data[i_channel + 1, i_channel + 1, i_channel + 1] = 250
        scan_data[-1, -1, -1] = 2.5

        scan = sitk.GetImageFromArray(scan_data)
        sitk.WriteImage(scan, os.path.join(subject_dir, "Scan-" + str(i_channel) + ".nii.gz"))

    for i_mask in range(N_masks):
        mask_data = np.zeros(img_size)
        mask_data[:, i_mask, :] = 1
        mask_data[:, i_mask + 1, :] = 105

        mask = sitk.GetImageFromArray(mask_data)
        sitk.WriteImage(
            mask, os.path.join(subject_dir, mask_keyword + "-" + str(i_mask) + ".nii.gz"),
        )


atlas_dir = os.path.join(test_data_dir, "ATLAS")

if not os.path.exists(atlas_dir):
    os.makedirs(atlas_dir)

atlas_mask = np.zeros(img_size)
atlas_mask[0:20, 0:20, 0:20] = 1

atlas_mask = sitk.GetImageFromArray(atlas_mask)
sitk.WriteImage(atlas_mask, os.path.join(atlas_dir, "atlas_mask.nii.gz"))
