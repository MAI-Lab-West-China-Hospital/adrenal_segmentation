import os
import nibabel as nib
import glob
from tqdm import tqdm
import torch
from monai.transforms import \
    Compose, LoadNiftid, AddChanneld, ScaleIntensityRanged, Orientationd, ToTensord
from monai.inferers import sliding_window_inference
from monai.networks.layers import Norm
import monai.data
from monai.metrics import compute_meandice

from data_process.segtopatch import get_min_max_coordinate, get_sub_volume, load_case
from network.small_organ import organNet
from utils.remove_small_regions import remove_small
from utils.dice_rve_hd95 import *


def first_seg(test_files, save_dir):

    val_transforms = Compose([
            LoadNiftid(keys=['image', 'label']),
            AddChanneld(keys=['image', 'label']),
            Orientationd(keys=['image', 'label'], axcodes='RAS'),
            ScaleIntensityRanged(keys=['image'], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
            ToTensord(keys=['image', 'label'])
        ])

    val_ds = monai.data.Dataset(data=test_files, transform=val_transforms)
    val_loader = monai.data.DataLoader(val_ds, batch_size=1, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    model = monai.networks.nets.UNet(dimensions=3, in_channels=1, out_channels=2, channels=(16, 32, 64, 128, 256),
                                         strides=(2, 2, 2, 2), num_res_units=2, norm=Norm.BATCH).to(device)

    model.load_state_dict(torch.load('./checkpoints/first_stage_model.pth'))
    model.eval()

    with torch.no_grad():
        metric_sum = 0.
        metric_count = 0
        case_name = []
        dice = []
        for val_data in tqdm(val_loader):
            affine = val_data['label_meta_dict']['affine'][0]
            name = val_data['label_meta_dict']['filename_or_obj'][0]
            name = name.split('/')[-1].replace('mask', 'seg')
            case_name.append(name)
            val_inputs, val_labels = val_data['image'].to(device), val_data['label'].to(device)

            roi_size = (96, 96, 96)
            sw_batch_size = 4
            val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)

            value = compute_meandice(y_pred=val_outputs, y=val_labels, include_background=False,
                                     to_onehot_y=True, mutually_exclusive=True)

            dice.append(value.item())

            test_outputs = torch.argmax(val_outputs, dim=1, keepdim=False)
            test_outputs = test_outputs.cpu().detach().numpy().squeeze()
            pred_img = nib.Nifti1Image(test_outputs, affine=affine)
            pred_img.set_data_dtype(np.float)

            path = os.sep.join([save_dir, name])
            nib.save(pred_img, path)
            metric_count += len(value)
            metric_sum += value.sum().item()
        metric = metric_sum / metric_count

        data = {'dice': dice}
        df = pd.DataFrame(data=data, columns=['dice'], index=case_name)
        df.to_csv(save_dir + 'scores.csv')
        print("evaluation metric:", metric)


def second_seg_patch(seg_path, dataroot):

    transforms = Compose([
        AddChanneld(keys=['image', 'label']),
        ScaleIntensityRanged(keys=['image'], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
        ToTensord(keys=['image', 'label'])
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    model = organNet().to(device)
    model.load_state_dict({k.replace('module.', ''): v for k, v in
                           torch.load('./checkpoints/small_organNet_models/fold1_model.pth').items()})
    model.eval()

    for file in os.listdir(seg_path):
        seg_img = nib.load(os.path.join(seg_path, file)).get_fdata()
        roi_left, roi_right = get_min_max_coordinate(seg_img)
        image = os.path.join(dataroot, 'Image', file.replace('seg', 'v'))
        label = os.path.join(dataroot, 'Mask', file.replace('seg', 'mask'))
        image, label, image_affine, label_affine = load_case(image, label)
        image_left, label_left, cdl = get_sub_volume(image, label, roi_left, range=96)
        image_right, label_right, cdr = get_sub_volume(image, label, roi_right, range=96)

        predict = np.zeros_like(label)

        data_dict = [{'image': image_left, 'label': label_left},
                     {'image': image_right, 'label': label_right}]

        ds = monai.data.Dataset(data=data_dict, transform=transforms)
        loader = monai.data.DataLoader(ds, batch_size=1)
        loader = iter(loader)

        with torch.no_grad():
            name = file.split('/')[-1]

            for idx, data in enumerate(loader):
                inputs, lab = data['image'].type(torch.FloatTensor).to(device), data['label'].to(device)
                outputs = model(inputs)

                test_outputs = torch.argmax(outputs, dim=1, keepdim=False)
                test_outputs = test_outputs.cpu().detach().numpy().squeeze()

                if idx==0:
                    start_x, end_x, start_y, end_y, start_z, end_z = cdl
                else:
                    start_x, end_x, start_y, end_y, start_z, end_z = cdr

                predict[start_x: end_x, start_y: end_y, start_z: end_z] = test_outputs

            pred_img = nib.Nifti1Image(predict, affine=label_affine)
            pred_img.set_data_dtype(np.float)

            path = os.sep.join([seg_path, name])
            nib.save(pred_img, path)


def get_metrics(seg_path, gd_path):

    save_dir = seg_path

    seg = sorted(os.listdir(seg_path))

    dices = []
    hds = []
    rves = []
    case_name = []
    senss = []
    specs = []
    for name in seg:
        if not name.startswith('.') and name.endswith('nii.gz'):
            gd_name = name.replace('seg', 'mask')
            seg_ = nib.load(os.path.join(seg_path, name))
            seg_arr = seg_.get_fdata().astype('float32')
            gd_ = nib.load(os.path.join(gd_path, gd_name))
            gd_arr = gd_.get_fdata().astype('float32')
            case_name.append(name)

            # hausdorff95
            hd_score = get_evaluation_score(seg_arr, gd_arr, spacing=None, metric='hausdorff95')
            hds.append(hd_score)

            # RVE
            rve = get_evaluation_score(seg_arr, gd_arr, spacing=None, metric='rve')
            rves.append(rve)

            # dice
            dice = get_evaluation_score(seg_.get_fdata(), gd_.get_fdata(), spacing=None, metric='dice')
            dices.append(dice)

            # sens, spec
            sens, spec = compute_class_sens_spec(seg_.get_fdata(), gd_.get_fdata())
            senss.append(sens)
            specs.append(spec)
    # save
    data = {'dice': dices, 'RVE': rves, 'Sens': senss, 'Spec': specs, 'HD95': hds}
    df = pd.DataFrame(data=data, columns=['dice', 'RVE', 'Sens', 'Spec', 'HD95'], index=case_name)
    df.to_csv(os.path.join(save_dir, 'metrics.csv'))


if __name__ == '__main__':
    data_root = './data/adrenal_demo'
    images = sorted(glob.glob(os.path.join(data_root, 'Image', '*.nii.gz')))
    labels = sorted(glob.glob(os.path.join(data_root, 'Mask', '*.nii.gz')))
    test_files = [{'image': image, 'label': label}
                  for image, label in zip(images, labels)]
    save_dir = "./output"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # first stage
    print('current stage:  organ localization .....')
    first_seg(test_files, save_dir)

    # post processing
    print('current stage: post processing ......')
    remove_small(save_dir, save_dir)

    # second stage
    print('current stage: second stage ......')
    second_seg_patch(save_dir, data_root)

    # get metrics
    print('current stage: get metrics')
    gd_path = data_root + '/Mask'
    get_metrics(save_dir, gd_path)