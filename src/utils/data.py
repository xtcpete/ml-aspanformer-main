import numpy as np
import cv2
import json
import torch
import h5py
from .camera import create_intrinsics, create_camera_pose

def read_aws_color(path,
                         resize=None,
                         df=None,
                         padding=False,
                         augment_fn=None,
                         rotation=0):
    """
    Args:
        resize (int, optional): the longer edge of resized images. None for no resize.
        padding (bool): If set to 'True', zero-pad resized images to squared size.
        augment_fn (callable, optional): augments images with pre-defined visual effects
    Returns:
        image (torch.tensor): (3, h, w)
        mask (torch.tensor): (h, w)
        scale (torch.tensor): [w/w_new, h/h_new]
    """
    # read image
    image = imread_color(path, augment_fn, client=None)
    if rotation != 0:
        image = np.rot90(image, k=rotation).copy()

    # resize image
    w, h = image.shape[1], image.shape[0]
    w_new, h_new = get_resized_wh(w, h, resize)
    w_new, h_new = get_divisible_wh(w_new, h_new, df)

    image = cv2.resize(image, (w_new, h_new))
    scale = torch.tensor([w / w_new, h / h_new], dtype=torch.float)
    scale_wh = torch.tensor([w_new, h_new], dtype=torch.float)

    if padding:  # padding
        pad_to = max(h_new, w_new)
        image, mask = pad_bottom_right(image, pad_to, ret_mask=True)
    else:
        mask = None

    image = (torch.from_numpy(image).float() / 255
             )  # (3, h, w) -> (3, h, w) and normalized
    mask = torch.from_numpy(mask)

    return image, mask, scale, scale_wh
    
def load_image_depth_camera(imagePath, depthPath, cameraPath, resize=None, df=None, padding=None, source="AWS", color=True):
    if color:
        img, mask, scale, _ = read_aws_color(imagePath, resize, df, padding)
    else:
        img, mask, scale, _ = read_aws_gray(imagePath, resize, df, padding)
    depth = read_depth(depthPath, 2000)

    config = None
    with open(cameraPath, 'r') as file:
        config = json.load(file)

    k = torch.from_numpy(create_intrinsics(config["intrinsics"], source)).float()
    p = torch.from_numpy(create_camera_pose(config["extrinsics"], source)).float()

    return img, mask, scale, depth, k, p, _
    
def imread_color(path, augment_fn=None, client=None):
    cv_type = cv2.IMREAD_COLOR
    # if str(path).startswith('s3://'):
    #     image = load_array_from_s3(str(path), client, cv_type)
    # else:
    #     image = cv2.imread(str(path), cv_type)

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if augment_fn is not None:
        image = augment_fn(image)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image  # (h, w)
    
def read_aws_gray(path, resize=None, df=None, padding=None):
    cv_type = cv2.IMREAD_GRAYSCALE
    image = cv2.imread(str(path), cv_type) # (h, w)

    w, h = image.shape[1], image.shape[0]
    w_new, h_new = get_resized_wh(w, h, resize) # 840, 472
    w_new, h_new = get_divisible_wh(w_new, h_new, df)

    image = cv2.resize(image, (w_new, h_new)) # [h, w]
    scale = torch.tensor([w / w_new, h / h_new], dtype=torch.float)
    pad = [0, 0]

    if padding:  # padding
        pad_to = max(h_new, w_new) # pad to 840
        image, mask = pad_bottom_right(image, pad_to, ret_mask=True)
        mask = torch.from_numpy(mask)
        pad = [pad_to - w_new, pad_to - h_new]
    else:
        mask = None

    image = torch.from_numpy(image).float()[None] / 255 # (h,w) -> (1,h,w) and normalized

    return image, mask, scale, pad

def getImageSize(path):
    cv_type = cv2.IMREAD_GRAYSCALE
    image = cv2.imread(str(path), cv_type) # (h, w)

    w, h = image.shape[1], image.shape[0]

    return [h, w]

def read_depth(path, pad_to=None):

    depth = np.load(path, allow_pickle=True)

    if (path.split('.')[1] == 'npz'):
        depth = depth['arr_0']
    if pad_to is not None:
        depth, _ = pad_bottom_right(depth, pad_to, ret_mask=False)
    depth = torch.from_numpy(depth).float() # (h, w)
    return depth

def get_resized_wh(w, h, resize=None):
    if resize is not None:  # resize the longer edge
        scale = resize / max(h, w) # 840 / 1920 = 0.4375
        w_new, h_new = int(round(w*scale)), int(round(h*scale)) # 840, 472
    else:
        w_new, h_new = w, h
    return w_new, h_new

def get_divisible_wh(w, h, df=None):
    if df is not None:
        w_new, h_new = map(lambda x: int(x // df * df), [w, h])
    else:
        w_new, h_new = w, h
    return w_new, h_new

def pad_bottom_right(inp, pad_size, ret_mask=False):
    assert isinstance(pad_size, int) and pad_size >= max(
        inp.shape[-2:]), f'{pad_size} < {max(inp.shape[-2:])}'
    mask = None
    if inp.ndim == 2:
        padded = np.zeros((pad_size, pad_size), dtype=inp.dtype)
        padded[:inp.shape[0], :inp.shape[1]] = inp
        if ret_mask:
            mask = np.zeros((pad_size, pad_size), dtype=bool)
            mask[:inp.shape[0], :inp.shape[1]] = True
    elif inp.ndim == 3:
        padded = np.zeros((inp.shape[2], pad_size, pad_size), dtype=inp.dtype)
        padded[:, :inp.shape[0], :inp.shape[1]] = inp.transpose(2, 0, 1)
        if ret_mask:
            mask = np.zeros((pad_size, pad_size), dtype=bool)
            mask[:inp.shape[0], :inp.shape[1]] = True
    else:
        raise NotImplementedError()
    return padded, mask

def load_color_image(path, resize=None, df=None, padding=None):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # [h, w 3]

    w, h = image.shape[1], image.shape[0]
    w_new, h_new = get_resized_wh(w, h, resize)
    w_new, h_new = get_divisible_wh(w_new, h_new, df)

    image = cv2.resize(image, (w_new, h_new)) # [h, w, 3]

    if padding:  # padding
        pad_to = max(h_new, w_new) # pad to 840
        padded = np.zeros((3, pad_to, pad_to), dtype=image.dtype)
        image = image.transpose(2, 0, 1) #[3, h, w]
        padded[:, :image.shape[1], :image.shape[2]] = image
        image = padded # [3, h, w]
    else:
        image = image.transpose(2, 0, 1)

    image_tensor = torch.from_numpy(image).float() / 255
    image = image.transpose(1, 2, 0) # [h, w, 3]

    return image, image_tensor

# ------------------------------- Megadepth ---------------------------------------------------------

def read_megadepth_depth(path, pad_to=None):
    depth = np.array(h5py.File(path, 'r')['depth'])
    if pad_to is not None:
        depth, _ = pad_bottom_right(depth, pad_to, ret_mask=False)
    depth = torch.from_numpy(depth).float()  # (h, w)
    return depth