from torch.utils.data import (Dataset, DataLoader, ConcatDataset, SubsetRandomSampler)
import os
import torch
import torch.nn.functional as F
import numpy as np
from src.utils.data import load_image_depth_camera

class DataModule():

    def __init__(self, args, config):
        super().__init__()

        self.data_dir = args.data_path
        self.config = config

        train_index_path = f"{self.data_dir}/train_index.txt"
        validate_index_path = f"{self.data_dir}/validate_index.txt"

        self.train_dataset = self.setup(train_index_path)
        self.validate_dataset = self.setup(validate_index_path)

        self.train_loader_params = {
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'pin_memory': getattr(args, 'pin_memory', True)
        }

    def setup(self, index_path):

        datasets = []
        dataset = AWSDataset(self.data_dir, index_path, self.config)

        datasets.append(dataset)

        return ConcatDataset(datasets)
    
    def train_dataloader(self, subset=False, sample_size = 50000):
        tb = len(self.train_dataset)
        if subset:
            indices = torch.randperm(tb)[:tb if tb <sample_size else sample_size]
            random_test_sampler = SubsetRandomSampler(indices)
        else:
            random_test_sampler = None
        dataloader = DataLoader(self.train_dataset, sampler=random_test_sampler, **self.train_loader_params)
        return dataloader
    
    def validation_dataloader(self, subset=True):
        tb = len(self.validate_dataset)
        if subset:
            indices = torch.randperm(tb)[:tb if tb <200 else 200]
            random_test_sampler = SubsetRandomSampler(indices)
        else:
            random_test_sampler = None
        dataloader = DataLoader(self.validate_dataset, sampler=random_test_sampler, **self.train_loader_params)
        return dataloader
    
    def test_dataloader(self):
        dataloader = DataLoader(self.test_dataloader, sampler=None, **self.train_loader_params)
        return dataloader


class AWSDataset(Dataset):

    def __init__(self, data_dir, index_path, config):
        super().__init__()

        if type(config) == dict:
            self.coarse_scale = 1 / config['resolution'][0] if config is not None else 1/8
        else:
            self.coarse_scale = 1 / config.RESOLUTION[0] if config is not None else 1/8
        

        self.pair_count = 0

        self.data_dir = data_dir # data
        self.index_path = index_path # data/train_index.txt

        with open(self.index_path, 'r') as file:
            self.pairs = [line.strip().split(',') for line in file]
        
        self.pair_count = len(self.pairs) 

    def __len__(self):
        return self.pair_count
    
    def __getitem__(self, idx):
        image_path = f"{self.data_dir}/raw"
    
        png_path_0, depth_path_0, camera_path_0 = get_train_files_path(image_path, self.pairs[idx][0])
        png_path_1, depth_path_1, camera_path_1 = get_train_files_path(image_path, self.pairs[idx][1])
        resize = 832
        df = 8
        padding = True
        image0, mask0, scale0, d0, k0, t0, scale_wh0 = load_image_depth_camera(png_path_0, depth_path_0, camera_path_0, resize, df, padding, source='carla', color=False)
        image1, mask1, scale1, d1, k1, t1, scale_wh1 = load_image_depth_camera(png_path_1, depth_path_1, camera_path_1, resize, df, padding, source='carla', color=False)
        
        f = np.array([[ 0,  1,  0 ], [ 0,  0, -1 ], [ 1,  0,  0 ]], dtype=np.float32)
        f = np.linalg.inv(f)
        k0 = (torch.tensor(np.linalg.inv(f)).to(k0.device) @ k0.inverse()).inverse()
        k1 = (torch.tensor(np.linalg.inv(f)).to(k1.device) @ k1.inverse()).inverse()
        
        # convert coordinates from unreal to opencv format for calculating pose error
        transform = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        transform2 = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        t0_cv = t0 @ transform @ transform2
        t1_cv = t1 @ transform @ transform2
        T_0to1 = t1_cv.inverse() @ t0_cv
        T_1to0 = T_0to1.inverse()

        data = {
            'image0': image0, # (1, h, w)
            'depth0': d0,     # (h, w)
            'image1': image1,
            'depth1': d1,
            'T_0to1': T_0to1, # (4, 4)
            'T_1to0': T_1to0,
            'K0': k0,         # (3, 3)
            'K1': k1,
            'scale0': scale0, # [scale_w, scale_h]
            'scale1': scale1,
            'scale_wh0': scale_wh0,
            'scale_wh1': scale_wh1,
            'scale_wh0': scale_wh0,
            'scale_wh1': scale_wh1,
            'scene_id': -1,
            'pair_id': -1,
            'pairs_name': self.pairs[idx]
        }
    
    
        # for training
        if mask0 is not None: # img_padding is true
            if self.coarse_scale:
                [ts_mask_0, ts_mask_1] = F.interpolate(torch.stack([mask0, mask1], dim=0)[None].float(),
                                                           scale_factor = self.coarse_scale,
                                                           mode='nearest',
                                                           recompute_scale_factor=False)[0].bool()
                data.update({'mask0': ts_mask_0, 'mask1': ts_mask_1})
        
        
        return data

def get_train_files_path(root_dir, filename):
    png_path = os.path.join(root_dir, filename + '.png')
    depth_path = os.path.join(root_dir, filename + '.npy')
    camera_path = os.path.join(root_dir, filename + '.json')
    return png_path, depth_path, camera_path


def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False
    
# if __name__ == '__main__':
#     aws = AWSDataset("data", "train_index.txt", None)