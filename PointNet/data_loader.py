import os
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset

# --- Data Augmentation Functions  ---
def rotate_point_cloud_z(pc):
    """ Randomly rotate the point clouds around the Z axis. """
    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, -sinval, 0],
                                [sinval, cosval, 0],
                                [0, 0, 1]], dtype=np.float32)
    return np.dot(pc, rotation_matrix)

def jitter_point_cloud(pc, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point. """
    N, C = pc.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return pc + jittered_data

def random_scale_point_cloud(pc, scale_low=0.8, scale_high=1.25):
    """ Randomly scale the point cloud. Scale is per shape. """
    scale = np.random.uniform(scale_low, scale_high)
    return pc * scale


class ShapeNetPartH5(Dataset):

    # It accepts individual flags for each augmentation technique.
    def __init__(self, data_path, split='train', n_points=2048, 
                 augment_rotation=True, augment_jitter=True, augment_scale=True):
        self.root = data_path
        self.n_points = n_points
        self.split = split
        # Store the individual augmentation flags
        self.augment_rotation = augment_rotation
        self.augment_jitter = augment_jitter
        self.augment_scale = augment_scale
        # A flag to check if we are in training mode
        self.is_training = (split == 'train')

        self.all_points = []
        self.all_seg_labels = []
        self.all_cls_labels = []

        h5_files = [f for f in os.listdir(self.root) if f.endswith('.h5') and self.split in f]
        if not h5_files:
            raise FileNotFoundError(f"No H5 files found for split '{self.split}' in '{self.root}'")

        print(f"Loading H5 files for '{self.split}' split: {h5_files}")

        for h5_filename in sorted(h5_files):
            f = h5py.File(os.path.join(self.root, h5_filename), 'r')
            self.all_points.append(f['data'][:])
            self.all_seg_labels.append(f.get('pid', f.get('seg'))[:])
            self.all_cls_labels.append(f['label'][:])
            f.close()
            
        self.all_points = np.concatenate(self.all_points, axis=0)
        self.all_seg_labels = np.concatenate(self.all_seg_labels, axis=0)
        self.all_cls_labels = np.concatenate(self.all_cls_labels, axis=0).squeeze()

        print(f'The size of {self.split} data is {len(self.all_points)}')

    def __len__(self):
        return len(self.all_points)

    def __getitem__(self, index):
        points = self.all_points[index].copy()
        seg_labels = self.all_seg_labels[index].copy()
        cls_label = self.all_cls_labels[index].copy()

        # This ensures the --n_points argument works correctly.
        num_current_points = points.shape[0]
        if num_current_points > self.n_points:
            choice = np.random.choice(num_current_points, self.n_points, replace=False)
        else:
            choice = np.random.choice(num_current_points, self.n_points, replace=True)
        
        points = points[choice, :]
        seg_labels = seg_labels[choice]

        points = self.pc_normalize(points)
        
        # Now we check each flag to apply augmentations individually.
        if self.is_training:
            if self.augment_rotation:
                points = rotate_point_cloud_z(points)
            if self.augment_jitter:
                points = jitter_point_cloud(points)
            if self.augment_scale:
                points = random_scale_point_cloud(points)
        
        return (
            torch.from_numpy(points).float(),
            torch.from_numpy(np.array([cls_label])).long(),
            torch.from_numpy(seg_labels).long()
        )

    @staticmethod
    def pc_normalize(pc):
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / (m + 1e-9)
        return pc
