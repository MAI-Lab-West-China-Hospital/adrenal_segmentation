# target-to-target ratio (TTR)
# 计算通过segmentation切patch,是否包含了全部的target
import nibabel as nib
import numpy as np
import glob
import os
import pandas as pd


def TTR(mask, left, right):
    # numpy array
    ttr = (np.sum(left) + np.sum(right)) /np.sum(mask)
    return ttr

def AvgR(mask, left, right):
    # 计算保存下来的总像素是原始图像的几分之一
    # numpy array
    avgr = (left.size + right.size) / mask.size
    return avgr

def read_nii(nii_file):
    data = nib.load(nii_file).get_fdata()
    return data


if __name__ == '__main__':
    mask_path = '/Volumes/Backup Plus/data/adrenal/seg/adrenal112/Mask01'
    mask_path_raw = '/Volumes/Backup Plus/data/adrenal/Mask'
    seg_mask = '/Volumes/Backup Plus/adreanl paper info/wholevspatch/unet09241labelpost_patch/'

    seg_left = sorted(glob.glob(os.path.join(seg_mask, 'l*.nii.gz')))

    case_name = []

    # get TTR
    # case_ttr = []
    # for left_idx in seg_left:
    #     name = left_idx.split('/')[-1][1:]
    #     case_name.append(name)
    #     right_idx = os.path.join(seg_mask, 'r' + name)
    #     mask_idx = os.path.join(mask_path, name)
    #     mask_data = read_nii(mask_idx)
    #     left_data = read_nii(left_idx)
    #     right_data = read_nii(right_idx)
    #     ttr = TTR(mask_data, left_data, right_data)
    #     case_ttr.append(ttr)
    #
    # df = pd.DataFrame(data=case_ttr, columns=['TTR'], index=case_name)
    # df.to_csv(os.path.join(seg_mask, 'TTR.csv'))

    # get avgr
    case_avgr = []
    for left_idx in seg_left:
        name = left_idx.split('/')[-1][1:]
        case_name.append(name)
        right_idx = os.path.join(seg_mask, 'r' + name)
        mask_idx = os.path.join(mask_path_raw, name)
        mask_data = read_nii(mask_idx)
        left_data = read_nii(left_idx)
        right_data = read_nii(right_idx)
        avgr = AvgR(mask_data, left_data, right_data)
        case_avgr.append(round(avgr, 4))

    df = pd.DataFrame(data=case_avgr, columns=['AvgR'], index=case_name)
    df.to_csv(os.path.join(seg_mask, 'AvgR.csv'))
