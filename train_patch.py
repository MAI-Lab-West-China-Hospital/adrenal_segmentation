import os
import glob
import numpy as np
import nibabel as nib
from tqdm import tqdm
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.tensorboard import SummaryWriter
import monai
import pandas as pd
from monai.transforms import Compose, LoadNiftid, AddChanneld, ScaleIntensityRanged, \
    Orientationd, RandRotate90d, RandGaussianNoised, RandFlipd, ToTensord
from monai.networks.layers import Norm
from monai.metrics import compute_meandice, DiceMetric
from monai.utils import set_determinism
from monai.networks.utils import one_hot
from network.small_organ import organNet
from loss.BAFocal import weight_FocalLoss

monai.config.print_config()
set_determinism(seed=0)


def train(train_files, val_files):
    # Setup transforms for training and validation
    train_transforms = Compose([
        LoadNiftid(keys=['image', 'label']),
        AddChanneld(keys=['image', 'label']),
        Orientationd(keys=['image', 'label'], axcodes='RAS'),
        ScaleIntensityRanged(keys=['image'], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
        RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=[0, 2]),
        RandGaussianNoised(keys=['image'], prob=0.1),
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
    val_loader = monai.data.DataLoader(val_ds, batch_size=6, num_workers=8, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    loss_function = weight_FocalLoss()

    dice_metric = DiceMetric(include_background=False, reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0001)

    val_interval = 1
    best_metric = -1
    best_metric_epoch = -1

    log_path = './runs/organnet'
    writer = SummaryWriter(log_path)
    for epoch in range(500):
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch + 1, 500))
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data['image'].to(device), batch_data['label'].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if step % 5 == 0:
                print('{}/{}, train_loss: {:.4f}'.format(step, len(train_ds) // train_loader.batch_size, loss.item()))

        epoch_loss /= step
        writer.add_scalar("epoch average loss", epoch_loss, epoch + 1)
        print('epoch {} average loss: {:.4f}'.format(epoch + 1, epoch_loss))

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                metric_sum = 0.
                metric_count = 0
                val_epoch_loss = 0
                val_step = 0
                for val_data in val_loader:
                    val_step += 1
                    val_inputs, val_labels = val_data['image'].to(device), val_data['label'].to(device)

                    val_outputs = model(val_inputs)
                    val_loss = loss_function(val_outputs, val_labels)
                    val_epoch_loss += val_loss.item()

                    val_output = torch.argmax(val_outputs, dim=1, keepdim=True)
                    dice, _ = dice_metric(val_output, val_labels)

                    metric_sum += dice.item()

                val_epoch_loss /= val_step
                metric = metric_sum / val_step

                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    save_dir = 'checkpoints/xxxxx/'
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    save_path = save_dir + str(epoch + 1) + "best_metric_model.pth"
                    torch.save(model.state_dict(), save_path)
                    print('saved new best metric model')

                print('current epoch {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}'.format(
                    epoch + 1, metric, best_metric, best_metric_epoch))
                writer.add_scalar("val_mean_dice", metric, epoch + 1)
                writer.add_scalar("val_epoch_loss", val_epoch_loss, epoch + 1)

    print('train completed, best_metric: {:.4f}  at epoch: {}'.format(best_metric, best_metric_epoch))
    writer.close()


def test(test_files):
    val_transforms = Compose([
        LoadNiftid(keys=['image', 'label']),
        AddChanneld(keys=['image', 'label']),
        Orientationd(keys=['image', 'label'], axcodes='RAS'),
        ScaleIntensityRanged(keys=['image'], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
        ToTensord(keys=['image', 'label'])
    ])

    val_ds = monai.data.Dataset(data=test_files, transform=val_transforms)
    val_loader = monai.data.DataLoader(val_ds, batch_size=1, num_workers=4)

    dice_metric = DiceMetric(include_background=False, reduction='mean')
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    model = organNet().to(device)
    model.load_state_dict(torch.load('./checkpoints/small_organNet_models/fold1_model.pth'))
    model.eval()
    with torch.no_grad():
        metric_sum = 0.
        metric_count = 0
        case_name = []
        dices = []
        for val_data in tqdm(val_loader):
            affine = val_data['image_meta_dict']['original_affine'][0]
            name = val_data['label_meta_dict']['filename_or_obj'][0]
            name = name.split('/')[-1].replace('mask', 'seg')
            case_name.append(name)
            val_inputs, val_labels = val_data['image'].to(device), val_data['label'].to(device)

            val_outputs = model(val_inputs)

            val_output = torch.argmax(val_outputs, dim=1, keepdim=True)
            dice, _ = dice_metric(val_output, val_labels)

            dices.append(dice.item())

            test_outputs = torch.argmax(val_outputs, dim=1, keepdim=False)
            test_outputs = test_outputs.cpu().detach().numpy().squeeze()
            pred_img = nib.Nifti1Image(test_outputs, affine=affine)
            pred_img.set_data_dtype(np.float)
            save_dir = "./output/small-organ/"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            path = os.sep.join([save_dir, name])
            nib.save(pred_img, path)

            metric_sum += dice.item()
        metric = metric_sum / len(dices)
        print("evaluation metric:", metric)


if __name__ == '__main__':
    from sklearn.model_selection import KFold, StratifiedKFold
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    import numpy as np
    import pandas as pd

    set_determinism(seed=0)

    data_root = '../samples/seg/adrenal112'

    images = sorted(glob.glob(os.path.join(data_root, 'ImagePatch', '*.nii.gz')))
    labels = sorted(glob.glob(os.path.join(data_root, 'MaskPatch', '*.nii.gz')))
    data_dicts = [{'image': image_name, 'label': label_name}
                   for image_name, label_name in zip(images1, labels1)]
    all_files = data_dicts

    floder = KFold(n_splits=5, random_state=42, shuffle=True)
    train_files = []
    test_files = []
    for k, (Trindex, Tsindex) in enumerate(floder.split(all_files)):
        train_files.append(np.array(all_files)[Trindex].tolist())
        test_files.append(np.array(all_files)[Tsindex].tolist())

    train(train_files[0], test_files[0])
    test(test_files[0])
