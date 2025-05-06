import torch
import numpy as np
import torch.utils.data as data

import os

import copy
import sys

import glob
import open3d

from pointnet2.util import load_h5_data

sys.path.insert(0, os.path.dirname(__file__))
from dataset_utils import augment_cloud
from PIL import Image
from torchvision import transforms
import h5py

class ViPCDataLoaderTest(data.Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.key = [0] * 100
        self.R = 4
        # self.labels = np.full(shape=(len(self.key),), fill_value=self.R-1, dtype=np.int64)
        self.image_transforms = transforms.Compose([
            transforms.Resize(100),
            transforms.ToTensor()
        ])
    
    def __getitem__(self, index):
        # generate dummy data for testing purposes
        result = {}
        result['partial'] = torch.zeros(256, 3)
        result['complete'] = torch.zeros(1024, 3) # R=4
        pil_image = Image.open("../chair.jpg").convert("RGB")
        result['image'] = self.image_transforms(pil_image)
        result['label'] = self.R - 1
        return result
    
    def __len__(self):
        return len(self.key)

class ModelNet10(data.Dataset):
    def __init__(
            self,
            data_dir,
            train=True,
            scale=1,
            npoints=1024,
            augmentation=False,
            return_augmentation_params=False,
            R=4,
            debug=False,
    ):
        self.return_augmentation_params = return_augmentation_params
        self.class_names = ['bathtub', 'bed', 'chair', 'desk', 'dresser', 
                            'monitor', 'night_stand', 'sofa', 'table', 'toilet']
        if debug:
            self.input_data,self.gt_data,self.class_indices,self.obj_path = self.load_custom_h5(os.path.join(data_dir,"ModelNet10_test_1024_256.h5"))
            total_samples = 20
            self.input_data = self.input_data[:total_samples]
            self.gt_data = self.gt_data[:total_samples]
            self.class_indices = self.class_indices[:total_samples]
            self.obj_path = self.obj_path[:total_samples]
            
        elif train:
            self.input_data,self.gt_data,self.class_indices,self.obj_path = self.load_custom_h5(os.path.join(data_dir,"ModelNet10_train_1024_256.h5"))
        else:
            self.input_data,self.gt_data,self.class_indices,self.obj_path = self.load_custom_h5(os.path.join(data_dir,"ModelNet10_test_1024_256.h5"))

        # # ---- condition ----
        # plys = glob.glob(os.path.join(self.input_path, "*.xyz"))
        # input_data = []
        # for ply in plys:
        #     pc = open3d.io.read_point_cloud(ply)
        #     points = np.asarray(pc.points, dtype=np.float32)
        #     input_data.append(points)
        # self.input_data = np.stack(input_data, axis=0)
        # # ---- condition ----

        # # ---- gt ----
        # plys = glob.glob(os.path.join(self.gt_path, "*.xyz"))
        # gt_data = []
        # for ply in plys:
        #     pc = open3d.io.read_point_cloud(ply)
        #     points = np.asarray(pc.points, dtype=np.float32)
        #     gt_data.append(points)
        # self.gt_data = np.stack(gt_data, axis=0)
        # # ---- gt ----

        # # ---- name ----
        # self.plys = [ply.split("/")[-1][:-4] for ply in plys]
            # ---- name ----

        self.train = train  # controls the trainset and testset
        # self.benchmark = benchmark
        self.augmentation = augmentation  # augmentation could be a dict or False

        # ---- label ----
        self.labels = np.full(shape=(self.input_data.shape[0],), fill_value=R-1, dtype=np.int64)
        # ---- label ----

        self.scale = scale
        self.input_data = self.input_data * scale
        self.gt_data = self.gt_data * scale

        print('partial point clouds:', self.input_data.shape)
        print('gt complete point clouds:', self.gt_data.shape)
        print('labels', self.labels.shape)
        self.labels = self.labels.astype(int)
        self.class_indices = self.class_indices.astype(int)
        self.len = self.input_data.shape[0]

    def load_custom_h5(self, path):
        with h5py.File(path, 'r') as f:
            input = np.array(f['data_sparse'][:])
            gt = np.array(f['data_dense'][:])
            class_indices = np.array(f["class_idx"][:])
            obj_path = f["obj_path"][:]
        # the center point of input
        input_centroid = np.mean(input, axis=1, keepdims=True)
        input = input - input_centroid
        # (b, 1)
        input_furthest_distance = np.amax(np.sqrt(np.sum(input ** 2, axis=-1)), axis=1, keepdims=True)
        # normalize to a unit sphere
        input = input / np.expand_dims(input_furthest_distance, axis=-1)
        gt = gt - input_centroid
        gt = gt / np.expand_dims(input_furthest_distance, axis=-1)

        return input, gt, class_indices, obj_path
        
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        # deepcopy is necessary here, because we may alter partial and complete for data augmentation
        # it will change the original data if we do not deep copy
        result = {}
        result['partial'] = copy.deepcopy(self.input_data[index]).astype(np.float32)
        result['complete'] = copy.deepcopy(self.gt_data[index]).astype(np.float32)
        
        # augment the point clouds
        if (isinstance(self.augmentation, dict) and self.train):
            result_list = list(result.values())
            if self.return_augmentation_params:
                result_list, augmentation_params = augment_cloud(
                    result_list,
                    self.augmentation,
                    return_augmentation_params=True
                )
            else:
                result_list = augment_cloud(
                    result_list,
                    self.augmentation,
                    return_augmentation_params=False
                )
            for idx, key in enumerate(result.keys()):
                result[key] = result_list[idx]

        if self.return_augmentation_params:
            for key in augmentation_params.keys():
                result[key] = augmentation_params[key]
        for key in result.keys():
            result[key] = torch.from_numpy(result[key])

        result['label'] = self.labels[index]
        result['class_index'] = self.class_indices[index]

        if(not self.train):
            result['name'] = self.obj_path[index]

        return result

class PU1K(data.Dataset):
    def __init__(
            self,
            data_dir,
            train=True,
            scale=1,
            npoints=2048,
            augmentation=False,
            return_augmentation_params=False,
            R=8,
            debug=False,
    ):
        self.return_augmentation_params = return_augmentation_params

        if debug:
            self.input_data,self.gt_data = load_h5_data(os.path.join(data_dir,"train","pu1k_poisson_256_poisson_1024_pc_2500_patch50_addpugan.h5"))
            total_samples = 20
            self.input_data = self.input_data[:total_samples]
            self.gt_data = self.gt_data[:total_samples]

        elif train:
            self.input_data,self.gt_data = load_h5_data(os.path.join(data_dir,"train","pu1k_poisson_256_poisson_1024_pc_2500_patch50_addpugan.h5"))
        else:
            # self.input_path = f"{data_dir}/test/input_{npoints}_{R}X/input_{npoints}"
            # self.gt_path = f"{data_dir}/test/input_{npoints}_{R}X/gt_{npoints*R}"
            self.input_data,self.gt_data = load_h5_data(os.path.join(data_dir,"train","pu1k_poisson_256_poisson_1024_pc_2500_patch50_addpugan.h5"))

            # ---- condition ----
            plys = glob.glob(os.path.join(self.input_path, "*.xyz"))
            input_data = []
            for ply in plys:
                pc = open3d.io.read_point_cloud(ply)
                points = np.asarray(pc.points, dtype=np.float32)
                input_data.append(points)
            self.input_data = np.stack(input_data, axis=0)
            # ---- condition ----

            # ---- gt ----
            plys = glob.glob(os.path.join(self.gt_path, "*.xyz"))
            gt_data = []
            for ply in plys:
                pc = open3d.io.read_point_cloud(ply)
                points = np.asarray(pc.points, dtype=np.float32)
                gt_data.append(points)
            self.gt_data = np.stack(gt_data, axis=0)
            # ---- gt ----

            # ---- name ----
            self.plys = [ply.split("/")[-1][:-4] for ply in plys]
            # ---- name ----

        self.train = train  # controls the trainset and testset
        # self.benchmark = benchmark
        self.augmentation = augmentation  # augmentation could be a dict or False

        # ---- label ----
        self.labels = np.full(shape=(self.input_data.shape[0],), fill_value=R-1, dtype=np.int64)
        # ---- label ----

        self.scale = scale
        self.input_data = self.input_data * scale
        self.gt_data = self.gt_data * scale

        print('partial point clouds:', self.input_data.shape)
        print('gt complete point clouds:', self.gt_data.shape)
        print('labels', self.labels.shape)
        self.labels = self.labels.astype(int)

        self.len = self.input_data.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        # deepcopy is necessary here, because we may alter partial and complete for data augmentation
        # it will change the original data if we do not deep copy
        result = {}
        result['partial'] = copy.deepcopy(self.input_data[index])
        result['complete'] = copy.deepcopy(self.gt_data[index])
        result['label'] = self.labels[index]
        result['class_index'] = self.labels[index]
        # augment the point clouds
        if (isinstance(self.augmentation, dict) and self.train):
            result_list = list(result.values())
            if self.return_augmentation_params:
                result_list, augmentation_params = augment_cloud(
                    result_list,
                    self.augmentation,
                    return_augmentation_params=True
                )
            else:
                result_list = augment_cloud(
                    result_list,
                    self.augmentation,
                    return_augmentation_params=False
                )
            for idx, key in enumerate(result.keys()):
                result[key] = result_list[idx]

        if self.return_augmentation_params:
            for key in augmentation_params.keys():
                result[key] = augmentation_params[key]
        for key in result.keys():
            result[key] = torch.from_numpy(result[key])
        if(not self.train):
            result['name'] = copy.deepcopy(self.plys[index])

        return result

class PUGAN(data.Dataset):
    def __init__(
            self,
            data_dir,
            train=True,
            scale=1,
            npoints=2048,
            augmentation=False,
            return_augmentation_params=False,
            R=8,
            debug=False,
    ):
        self.return_augmentation_params = return_augmentation_params
        if debug:
            self.input_data,self.gt_data = load_h5_data(os.path.join(data_dir,"train","PUGAN_poisson_256_poisson_1024.h5"))
            total_samples = 24
            self.input_data = self.input_data[:total_samples]
            self.gt_data = self.gt_data[:total_samples]
            self.plys = [str(i) for i in range(total_samples)]

        elif train:
            self.input_data,self.gt_data = load_h5_data(os.path.join(data_dir,"train","PUGAN_poisson_256_poisson_1024.h5"))
        else:
            # self.input_path = f"{data_dir}/test/input_{npoints}_{R}X/input_{npoints}"
            # self.gt_path = f"{data_dir}/test/input_{npoints}_{R}X/gt_{npoints*R}"
            self.input_data,self.gt_data = load_h5_data(os.path.join(data_dir,"train","PUGAN_poisson_256_poisson_1024.h5"))

            # ---- condition ----
            plys = glob.glob(os.path.join(self.input_path, "*.xyz"))
            input_data = []
            for ply in plys:
                pc = open3d.io.read_point_cloud(ply)
                points = np.asarray(pc.points, dtype=np.float32)
                input_data.append(points)
            self.input_data = np.stack(input_data, axis=0)
            # ---- condition ----

            # ---- gt ----
            plys = glob.glob(os.path.join(self.gt_path, "*.xyz"))
            gt_data = []
            for ply in plys:
                pc = open3d.io.read_point_cloud(ply)
                points = np.asarray(pc.points, dtype=np.float32)
                gt_data.append(points)
            self.gt_data = np.stack(gt_data, axis=0)
            # ---- gt ----

            # ---- name ----
            self.plys = [ply.split("/")[-1][:-4] for ply in plys]
            # ---- name ----

        self.train = train  # controls the trainset and testset
        self.augmentation = augmentation  # augmentation could be a dict or False

        # ---- label ----
        self.labels = np.full(shape=(self.input_data.shape[0],), fill_value=R-1, dtype=np.int64)
        # ---- label ----

        self.scale = scale
        self.input_data = self.input_data * scale
        self.gt_data = self.gt_data * scale

        print('partial point clouds:', self.input_data.shape)
        # if not benchmark:
        print('gt complete point clouds:', self.gt_data.shape)
        print('labels', self.labels.shape)
        self.labels = self.labels.astype(int)

        self.len = self.input_data.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        # deepcopy is necessary here, because we may alter partial and complete for data augmentation
        # it will change the original data if we do not deep copy
        result = {}
        result['partial'] = copy.deepcopy(self.input_data[index])
        result['complete'] = copy.deepcopy(self.gt_data[index])
        result['class_index'] = np.array(0)
        result['label'] = np.array(self.labels[index])
        # augment the point clouds
        if (isinstance(self.augmentation, dict) and self.train):
            result_list = list(result.values())
            if self.return_augmentation_params:
                result_list, augmentation_params = augment_cloud(
                    result_list,
                    self.augmentation,
                    return_augmentation_params=True
                )
            else:
                result_list = augment_cloud(
                    result_list,
                    self.augmentation,
                    return_augmentation_params=False
                )
            for idx, key in enumerate(result.keys()):
                result[key] = result_list[idx]

        if self.return_augmentation_params:
            for key in augmentation_params.keys():
                result[key] = augmentation_params[key]
        for key in result.keys():
            result[key] = torch.from_numpy(result[key])
        if(not self.train):
            result['name'] = copy.deepcopy(self.plys[index])

        return result



if __name__ == '__main__':
    pass