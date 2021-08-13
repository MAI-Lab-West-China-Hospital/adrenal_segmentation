'''
adrenal 3D segmentation with MONAI
'''

import os
import glob
import numpy as np
import nibabel as nib
from tqdm import tqdm
import pandas as pd
import torch
import time
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import monai
from monai.transforms import \
    Compose, LoadNiftid, AddChanneld, ScaleIntensityRanged, CropForegroundd, \
    RandCropByPosNegLabeld, RandAffined, RandGaussianNoised, Orientationd, ToTensord
from monai.inferers import sliding_window_inference
from monai.networks.layers import Norm
from monai.metrics import compute_meandice
from monai.utils import set_determinism
import monai.data
from monai.losses import FocalLoss, DiceLoss

from monai.networks.nets import UNet

monai.config.print_config()

# Set deterministic training for reproducibility
set_determinism(seed=0)


def train(train_files, val_files):
    # Setup transforms for training and validation
    train_transforms = Compose([
        LoadNiftid(keys=['image', 'label']),
        AddChanneld(keys=['image', 'label']),
        Orientationd(keys=['image', 'label'], axcodes='RAS'),
        # axcodes=’RAS’ represents 3D orientation: (Left, Right), (Posterior, Anterior), (Inferior, Superior).
        ScaleIntensityRanged(keys=['image'], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
        RandGaussianNoised(keys=['image'], prob=0.2),
        CropForegroundd(keys=['image', 'label'], source_key='image'),
        RandCropByPosNegLabeld(keys=['image', 'label'], label_key='label', spatial_size=(96, 96, 96), pos=1,
                               neg=1, num_samples=4, image_key='image', image_threshold=0),
        ToTensord(keys=['image', 'label'])
    ])
    val_transforms = Compose([
        LoadNiftid(keys=['image', 'label']),
        AddChanneld(keys=['image', 'label']),
        Orientationd(keys=['image', 'label'], axcodes='RAS'),
        ScaleIntensityRanged(keys=['image'], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
        ToTensord(keys=['image', 'label'])
    ])

    train_ds = monai.data.CacheDataset(
        data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=8
    )
    train_loader = monai.data.DataLoader(train_ds, batch_size=6, shuffle=True, num_workers=8, pin_memory=True)

    val_ds = monai.data.CacheDataset(
        data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=8
    )
    val_loader = monai.data.DataLoader(val_ds, batch_size=1, num_workers=8, pin_memory=True)

    # Create Model, Loss, Optimizer
    # standard PyTorch program style: create UNet, DiceLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    model = UNet(dimensions=3, in_channels=1, out_channels=2, channels=(16, 32, 64, 128, 256),
                                     strides=(2, 2, 2, 2), num_res_units=2, norm=Norm.BATCH).to(device)
    # model.load_state_dict(torch.load('./checkpoints/best_metric_model.pth'))
    loss_function = monai.losses.DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=0.0001)

    # Execute a typical PyTorch training process
    val_interval = 1
    best_metric = -1
    best_metric_epoch = -1

    metric_values = list()
    log_path = './runs/1120'
    writer = SummaryWriter(log_path)
    total_start = time.time()
    for epoch in range(500):
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch + 1, 500))
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            epoch_start = time.time()
            step_start = time.time()
            step += 1
            inputs, labels = batch_data['image'].to(device), batch_data['label'].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print('{}/{}, train_loss: {:.4f}, step time: {:.4f}'.format(
                step, len(train_ds) // train_loader.batch_size, loss.item(), time.time() - step_start))

        epoch_loss /= step
        writer.add_scalar("epoch average loss", epoch_loss, epoch + 1)
        print('epoch {} average loss: {:.4f}'.format(epoch + 1, epoch_loss))

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                metric_sum = 0.
                metric_count = 0
                for val_data in val_loader:
                    val_inputs, val_labels = val_data['image'].to(device), val_data['label'].to(device)
                    roi_size = (96, 96, 96)
                    sw_batch_size = 4
                    val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
                    value = compute_meandice(y_pred=val_outputs, y=val_labels, include_background=False,
                                             to_onehot_y=True, mutually_exclusive=True)
                    metric_count += len(value)
                    metric_sum += value.sum().item()
                metric = metric_sum / metric_count
                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    save_dir = 'checkpoints/1120/'
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    save_path = save_dir + str(epoch + 1) + "best_metric_model.pth"
                    torch.save(model.state_dict(), save_path)
                    print('saved new best metric model')
                print('current epoch {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}'.format(
                    epoch + 1, metric, best_metric, best_metric_epoch))
                writer.add_scalar("val_mean_dice", metric, epoch + 1)

        print('time consuming of epoch {} is: {:.4f}'.format(epoch + 1, time.time() - epoch_start))
    print('train completed, best_metric: {:.4f} at epoch: {}, total time: {:.4f}'.format(
        best_metric, best_metric_epoch, time.time() - total_start))
    writer.close()


def test(test_files):
    val_transforms = Compose([
        LoadNiftid(keys=['image', 'label']),
        AddChanneld(keys=['image', 'label']),
        Orientationd(keys=['image', 'label'], axcodes='RAS'),
        ScaleIntensityRanged(keys=['image'], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
        ToTensord(keys=['image', 'label'])
    ])

    test_ds = monai.data.Dataset(data=test_files, transform=val_transforms)
    test_loader = monai.data.DataLoader(test_ds, batch_size=1, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    model = UNet(dimensions=3, in_channels=1, out_channels=2, channels=(16, 32, 64, 128, 256),
                                     strides=(2, 2, 2, 2), num_res_units=2, norm=Norm.BATCH).to(device)
    model.load_state_dict(torch.load('./checkpoints/first_stage_model.pth'))
    model.eval()
    with torch.no_grad():
        metric_sum = 0.
        metric_count = 0
        case_name = []
        dice = []
        for test_data in tqdm(test_loader):
            affine = test_data['label_meta_dict']['affine'][0]
            name = test_data['label_meta_dict']['filename_or_obj'][0]
            name = name.split('/')[-1].replace('mask', 'seg')
            case_name.append(name)
            val_inputs, val_labels = test_data['image'].to(device), test_data['label'].to(device)

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
            save_dir = "./output/first_stage"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            path = os.sep.join([save_dir, name])
            nib.save(pred_img, path)
            metric_count += len(value)
            metric_sum += value.sum().item()
        metric = metric_sum / metric_count
        data = {'dice': dice}
        df = pd.DataFrame(data=data, columns=['dice'], index=case_name)
        df.to_csv(save_dir + 'scores.csv')
        print("evaluation metric:", metric)


if __name__ == '__main__':
    import csv
    import pandas as pd
    import json
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    data_root = '../samples/seg/adrenal'
    train_images = sorted(glob.glob(os.path.join(data_root, 'Image', '*.nii.gz')))
    train_labels = sorted(glob.glob(os.path.join(data_root, 'Mask', '*.nii.gz')))
    data_dicts = [{'image': image_name, 'label': label_name}
                  for image_name, label_name in zip(train_images, train_labels)]
    train_files, val_files = data_dicts[:209], data_dicts[209:279]
    test_files = data_dicts[-70:]

    # train(train_files, val_files)
    test(test_files)