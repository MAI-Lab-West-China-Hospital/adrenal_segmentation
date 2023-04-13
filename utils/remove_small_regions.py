'''
使用三维的连通成分分析，删除了较远的偏远区域。默认使用26连通域
https://github.com/seung-lab/connected-components-3d/
'''
import cc3d
import nibabel as nib
from pathlib2 import Path
from tqdm import tqdm
import numpy as np
import os


def main(data, output):
    data = Path(data).resolve()
    output = Path(output).resolve()

    assert data != output, f'postprocess data will replace original data, use another output path'

    if not output.exists():
        output.mkdir(parents=True)

    predictions = sorted(data.glob('*_seg.nii.gz'))
    for pred in tqdm(predictions):
        if not pred.name.startswith('.'):
            vol_nii = nib.load(str(pred))
            affine = vol_nii.affine
            vol = vol_nii.get_fdata()
            vol = post_processing(vol)
            vol_nii = nib.Nifti1Image(vol, affine)

            vol_nii_filename = output / pred.name
            vol_nii.to_filename(str(vol_nii_filename))


def post_processing(vol):
    vol_ = vol.copy()
    vol_[vol_ > 0] = 1
    vol_ = vol_.astype(np.int64)
    vol_cc = cc3d.connected_components(vol_)
    cc_sum = [(i, vol_cc[vol_cc == i].shape[0]) for i in range(vol_cc.max() + 1)]
    cc_sum.sort(key=lambda x: x[1], reverse=True)
    cc_sum.pop(0)  # remove background
    reduce_cc = [cc_sum[i][0] for i in range(1, len(cc_sum)) if cc_sum[i][1] < cc_sum[0][1] * 0.1]
    for i in reduce_cc:
        vol[vol_cc == i] = 0

    return vol


if __name__ == '__main__':
    # data = '/Volumes/Backup Plus/data/seg/modelout/wholevspatch/unetretrain09241'
    # output = '/Volumes/Backup Plus/data/seg/modelout/wholevspatch/unet09241post'
    data = '/home/sun/15TB/home/luogt/project/adrenal/output/unetretrain09241/'
    output = '/home/sun/15TB/home/luogt/project/adrenal/output/unetretrain09241remove/'
    if not os.path.exists(output):
        os.makedirs(output)
    main(data, output)
