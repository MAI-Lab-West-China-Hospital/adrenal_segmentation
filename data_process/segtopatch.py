import numpy as np
import nibabel as nib


def get_min_max_coordinate(label):
    #  :param label: segmentation image, shape=[x, y, z]
    x, y, z = label.shape
    value_leftz = [0] * z
    value_rightz = [0] * z
    for idx in range(z):
        if 1 in label[0:x // 2, :, idx]:
            value_leftz[idx] = 1.
        if 1 in label[x // 2:, :, idx]:
            value_rightz[idx] = 1.
    value_leftz = np.array(value_leftz)
    slice_1 = np.where(value_leftz == 1)[0]
    min_z1 = int(slice_1.min())
    max_z1 = int(slice_1.max()) + 1
    value_rightz = np.array(value_rightz)
    slice_2 = np.where(value_rightz == 1)[0]
    min_z2 = int(slice_2.min())
    max_z2 = int(slice_2.max()) + 1

    # minx, maxx
    value_leftx = [0] * x
    value_rightx = [0] * x
    for idx in range(x):
        if 0 <= idx <= x//2 and 1 in label[idx, :, :]:
            value_leftx[idx] = 1.
        if idx >= x//2 and 1 in label[idx, :, :]:
            value_rightx[idx] = 1.
    value_leftx = np.array(value_leftx)
    slice_1 = np.where(value_leftx == 1)[0]
    min_x1 = int(slice_1.min())
    max_x1 = int(slice_1.max()) + 1
    value_rightx = np.array(value_rightx)
    slice_2 = np.where(value_rightx == 1)[0]
    min_x2 = int(slice_2.min())
    max_x2 = int(slice_2.max()) + 1

    # miny, maxy
    value_lefty = [0] * y
    value_righty = [0] * y
    for idx in range(y):
        if 1 in label[0:x // 2, idx, :]:
            value_lefty[idx] = 1.
        if 1 in label[x // 2:, idx, :]:
            value_righty[idx] = 1.
    value_lefty = np.array(value_lefty)
    slice_1 = np.where(value_lefty == 1)[0]
    min_y1 = int(slice_1.min())
    max_y1 = int(slice_1.max()) + 1
    value_righty = np.array(value_righty)
    slice_2 = np.where(value_righty == 1)[0]
    min_y2 = int(slice_2.min())
    max_y2 = int(slice_2.max()) + 1

    roi_left = [min_x1, min_y1, min_z1, max_x1, max_y1, max_z1]
    roi_right = [min_x2, min_y2, min_z2, max_x2, max_y2, max_z2]

    return roi_left, roi_right


def get_sub_volume(image, label, roi_list, range=160):
    """
    Extract random sub-volume from original images.

    Args:
        image (np.array): original image,
            of shape (orig_x, orig_y, orig_z)
        label (np.array): original label.
            labels coded using discrete values rather than
            a separate dimension,
            so this is of shape (orig_x, orig_y, orig_z)

    returns:
        X (np.array): sample of original image of dimension
            (output_x, output_y, output_z)
        y (np.array): labels which correspond to X, of dimension
            (output_x, output_y, output_z)
    """
    minx, miny, minz, maxx, maxy, maxz = roi_list
    x, y, z = image.shape  # (512, 512, 112)
    start_x = start_y = start_z = 0
    end_x = x
    end_y = y
    end_z = z

    if maxx - minx <= range:
        x_width = (range - (maxx - minx)) // 2
        start_x = max(start_x, minx - x_width)
        if not start_x + range > end_x:
            end_x = start_x + range

        else:
            end_x = x
            start_x = end_x - range
    else:
        x_width = ((maxx - minx) - range) // 2
        start_x = max(start_x, minx + x_width)
        end_x = start_x + range

    if maxy - miny < range:
        y_width = (range - (maxy - miny)) // 2
        start_y = max(start_y, miny - y_width)
        if not start_y + range > end_y:
            end_y = start_y + range

        else:
            end_y = y
            start_y = end_y - range
    else:
        y_width = ((maxy - miny) - range) // 2
        start_y = max(start_y, miny + y_width)
        end_y = start_y + range

    if maxz - minz < range:
        z_width = (range - (maxz - minz)) // 2
        start_z = max(start_z, minz - z_width)
        if not start_z + range > end_z:
            end_z = start_z + range

        else:
            end_z = z
            start_z = end_z - range
    else:
        z_width = ((maxz - minz) - range) // 2
        start_z = max(start_z, minz + z_width)
        end_z = start_z + range
    X = np.copy(image[start_x: end_x, start_y: end_y, start_z: end_z])
    y = np.copy(label[start_x: end_x, start_y: end_y, start_z: end_z])

    coordinate = [start_x, end_x, start_y, end_y, start_z, end_z]
    return X, y, coordinate


def load_case(image_nifty_file, label_nifty_file):
    # load the image and label file, get the image content and return a numpy array for each
    image_ = nib.load(image_nifty_file)
    image_affine = image_.affine
    image = image_.get_fdata()
    label_ = nib.load(label_nifty_file)
    label = label_.get_fdata()
    label_affine = label_.affine

    return image, label, image_affine, label_affine