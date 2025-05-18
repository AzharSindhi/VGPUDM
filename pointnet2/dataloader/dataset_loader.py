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
import numpy as np
import pickle
import random
import math 
from tqdm import tqdm
import open3d as o3d
import time
import pandas as pd

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
            total_samples = 10
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


class ViPCDataLoader(data.Dataset):
    def __init__(self, data_path, status, pc_input_num=3500, R=4, scale=1, image_size=480, 
                 augmentation=False, return_augmentation_params=False, debug=False, 
                 view_align=False, category='plane', mini=True):
        
        super(ViPCDataLoader,self).__init__()
        self.pc_input_num = pc_input_num
        self.status = status
        self.filelist = []
        self.cat = []
        self.key = []
        self.category = category
        self.view_align = view_align
        self.cat_map = {
            'plane':'02691156',
            'bench': '02828884', 
            'cabinet':'02933112', 
            'car':'02958343',
            'chair':'03001627',
            'monitor': '03211117',
            'lamp':'03636649',
            'speaker': '03691459', 
            'firearm': '04090263', 
            'couch':'04256520',
            'table':'04379243',
            'cellphone': '04401088', 
            'watercraft':'04530566'
        }

        filepath = f'{status}_list.txt'
        with open(os.path.join(data_path, filepath),'r') as f:
            self.filelist = f.read().strip().splitlines()
        
        assert len(self.filelist) > 0, f"No data found in {filepath}"
        
        self.imcomplete_path = os.path.join(data_path,'ShapeNetViPC-Partial')
        self.gt_path = os.path.join(data_path,'ShapeNetViPC-GT')
        self.rendering_path = os.path.join(data_path,'ShapeNetViPC-View')
        
        for key in self.filelist:
            if category !='all':
                if key.split('/')[0]!= self.cat_map[category]:
                    continue
            self.cat.append(key.split(';')[0])
            self.key.append(key)
        
        num_samples = 2000 if status == "train" else 700
        
        if debug:
            nsamples = 10
            self.key = self.key[:nsamples]
        elif mini:
            self.key = random.sample(self.key, num_samples)
        
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])

        print(f'{status} data num: {len(self.key)}')

        self.train = status == "train"
        self.augmentation = augmentation  # augmentation could be a dict or False
        self.return_augmentation_params = return_augmentation_params

        # ---- label ----
        self.labels = np.full(shape=(len(self.key),), fill_value=R-1, dtype=np.int64)
        # ---- label ----

        self.scale = scale
        # self.input_data = self.input_data * scale
        # self.gt_data = self.gt_data * scale

        print('partial point clouds:', len(self.key))
        # if not benchmark:
        print('gt complete point clouds:', len(self.key))
        print('labels', len(self.labels))
        self.labels = self.labels.astype(int)
        self.R = R
        self.image_size = image_size

    def __getitem__(self, idx):

        key = self.key[idx]
       
        pc_part_path = os.path.join(self.imcomplete_path,key.split('/')[0]+'/'+ key.split('/')[1]+'/'+key.split('/')[-1].replace('\n', '')+'.dat')
        # view_align = True, means the view of image equal to the view of partial points
        # view_align = False, means the view of image is different from the view of partial points
        if self.view_align:
            ran_key = key        
        else:
            ran_key = key[:-3]+str(random.randint(0,23)).rjust(2,'0')
       
        pc_path = os.path.join(self.gt_path, ran_key.split('/')[0]+'/'+ ran_key.split('/')[1]+'/'+ran_key.split('/')[-1].replace('\n', '')+'.dat')
        view_path = os.path.join(self.rendering_path,ran_key.split('/')[0]+'/'+ran_key.split('/')[1]+'/rendering/'+ran_key.split('/')[-1].replace('\n','')+'.png')
        
        #Inserted to correct a bug in the splitting for some lines 
        if(len(ran_key.split('/')[-1])>3):
            print("bug")
            print(ran_key.split('/')[-1])
            fin = ran_key.split('/')[-1][-2:]
            interm = ran_key.split('/')[-1][:-2]
            
            pc_path = os.path.join(self.gt_path, ran_key.split('/')[0]+'/'+ interm +'/'+ fin.replace('\n', '')+'.dat')
            view_path = os.path.join(self.rendering_path,ran_key.split('/')[0]+ '/' + interm + '/rendering/' + fin.replace('\n','')+'.png')

        views = Image.open(view_path).resize((self.image_size, self.image_size))#self.transform(Image.open(view_path))
        # views = views[:3,:,:]
        # load partial points
        with open(pc_path,'rb') as f:
            pc = pickle.load(f).astype(np.float32)
        # load gt
        with open(pc_part_path,'rb') as f:
            pc_part = pickle.load(f).astype(np.float32)
        # incase some item point number less than 3500 
        if pc_part.shape[0]<self.pc_input_num:
            pc_part = np.repeat(pc_part,(self.pc_input_num//pc_part.shape[0])+1,axis=0)[0:self.pc_input_num]


        # time in seconds
        # start_time = time.time()


        # load the view metadata
        image_view_id = view_path.split('.')[0].split('/')[-1]
        part_view_id = pc_part_path.split('.')[0].split('/')[-1]
        
        view_metadata = np.loadtxt(view_path[:-6]+'rendering_metadata.txt')

        theta_part = math.radians(view_metadata[int(part_view_id),0])
        phi_part = math.radians(view_metadata[int(part_view_id),1])

        theta_img = math.radians(view_metadata[int(image_view_id),0])
        phi_img = math.radians(view_metadata[int(image_view_id),1])

        pc_part = self.rotation_y(self.rotation_x(pc_part, - phi_part),np.pi + theta_part)
        pc_part = self.rotation_x(self.rotation_y(pc_part, np.pi - theta_img), phi_img)

        pcd_part = o3d.geometry.PointCloud()
        pcd_part.points = o3d.utility.Vector3dVector(pc_part)
        pcd_part_downsampled = pcd_part.farthest_point_down_sample(num_samples=self.pc_input_num//self.R)
        pc_part = np.asarray(pcd_part_downsampled.points)

        return {
            'name': copy.deepcopy(self.key[idx]).replace("/", "_"),
            'view': views,
            'partial': pc_part,
            'complete': pc,
            'category': self.category
        }

        # normalize partial point cloud and GT to the same scale
        gt_mean = pc.mean(axis=0) 
        pc = pc -gt_mean
        pc_L_max = np.max(np.sqrt(np.sum(abs(pc ** 2), axis=-1)))
        pc = pc/pc_L_max

        pc_part = pc_part-gt_mean
        pc_part = pc_part/pc_L_max

        result = {}
        result['partial'] = (pc_part * self.scale).astype(np.float32)
        result['complete'] = (pc * self.scale).astype(np.float32)
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
            result['name'] = copy.deepcopy(self.key[idx]).replace("/", "_")

        result['label'] = np.array(self.labels[idx])
        result['class_index'] = views.float().numpy()
        return result

    def __len__(self):
        return len(self.key)

    def rotation_z(self, pts, theta):
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        rotation_matrix = np.array([[cos_theta, -sin_theta, 0.0],
                                    [sin_theta, cos_theta, 0.0],
                                    [0.0, 0.0, 1.0]])
        return pts @ rotation_matrix.T


    def rotation_y(self, pts, theta):
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        rotation_matrix = np.array([[cos_theta, 0.0, -sin_theta],
                                    [0.0, 1.0, 0.0],
                                    [sin_theta, 0.0, cos_theta]])
        return pts @ rotation_matrix.T


    def rotation_x(self, pts, theta):
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        rotation_matrix = np.array([[1.0, 0.0, 0.0],
                                    [0.0, cos_theta, -sin_theta],
                                    [0.0, sin_theta, cos_theta]])
        return pts @ rotation_matrix.T



class ViPCDataLoaderMemory(data.Dataset):
    def __init__(self, data_path, status, pc_input_num=3500, R=4, scale=1, image_size=480, 
                 augmentation=False, return_augmentation_params=False, debug=False, 
                 view_align=False, category='plane', mini=True):
        
        super(ViPCDataLoaderMemory,self).__init__()
        self.pc_input_num = pc_input_num
        self.status = status
        self.filelist = []
        self.cat = []
        self.key = []
        self.category = category
        self.view_align = view_align
        self.cat_map = {
            'plane':'02691156',
            'bench': '02828884', 
            'cabinet':'02933112', 
            'car':'02958343',
            'chair':'03001627',
            'monitor': '03211117',
            'lamp':'03636649',
            'speaker': '03691459', 
            'firearm': '04090263', 
            'couch':'04256520',
            'table':'04379243',
            'cellphone': '04401088', 
            'watercraft':'04530566'
        }

        filepath = f"shapenet_vipc_processed_fps4096_R4_{category}_{status}.h5"
        df = pd.read_hdf(os.path.join(data_path, filepath), key='df')
        self.key = df['name'].tolist()
        self.views = df['view'].tolist()
        self.pcs = df['complete'].tolist()
        self.pc_parts = df['partial'].tolist()
        self.categories = df['category'].tolist()
        
        assert len(self.key) > 0, f"No data found in {filepath}"
        
        num_samples = 2000 if status == "train" else 700
        
        if debug:
            num_samples = 10

        self.key = self.key[:num_samples]
        self.views = self.views[:num_samples]
        self.pcs = self.pcs[:num_samples]
        self.pc_parts = self.pc_parts[:num_samples]
        self.categories = self.categories[:num_samples]
        
        
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])

        print(f'{status} data num: {len(self.key)}')

        self.train = status == "train"
        self.augmentation = augmentation  # augmentation could be a dict or False
        self.return_augmentation_params = return_augmentation_params

        # ---- label ----
        self.labels = np.full(shape=(len(self.key),), fill_value=R-1, dtype=np.int64)
        # ---- label ----

        self.scale = scale
        # self.input_data = self.input_data * scale
        # self.gt_data = self.gt_data * scale

        print('partial point clouds:', len(self.key))
        # if not benchmark:
        print('gt complete point clouds:', len(self.key))
        print('labels', len(self.labels))
        self.labels = self.labels.astype(int)
        self.R = R

    def __getitem__(self, idx):

        # key = self.key[idx]
       
        # pc_part_path = os.path.join(self.imcomplete_path,key.split('/')[0]+'/'+ key.split('/')[1]+'/'+key.split('/')[-1].replace('\n', '')+'.dat')
        # # view_align = True, means the view of image equal to the view of partial points
        # # view_align = False, means the view of image is different from the view of partial points
        # if self.view_align:
        #     ran_key = key        
        # else:
        #     ran_key = key[:-3]+str(random.randint(0,23)).rjust(2,'0')
       
        # pc_path = os.path.join(self.gt_path, ran_key.split('/')[0]+'/'+ ran_key.split('/')[1]+'/'+ran_key.split('/')[-1].replace('\n', '')+'.dat')
        # view_path = os.path.join(self.rendering_path,ran_key.split('/')[0]+'/'+ran_key.split('/')[1]+'/rendering/'+ran_key.split('/')[-1].replace('\n','')+'.png')
        
        # #Inserted to correct a bug in the splitting for some lines 
        # if(len(ran_key.split('/')[-1])>3):
        #     print("bug")
        #     print(ran_key.split('/')[-1])
        #     fin = ran_key.split('/')[-1][-2:]
        #     interm = ran_key.split('/')[-1][:-2]
            
        #     pc_path = os.path.join(self.gt_path, ran_key.split('/')[0]+'/'+ interm +'/'+ fin.replace('\n', '')+'.dat')
        #     view_path = os.path.join(self.rendering_path,ran_key.split('/')[0]+ '/' + interm + '/rendering/' + fin.replace('\n','')+'.png')

        views = self.transform(self.views[idx])
        views = views[:3,:,:]
        pc = self.pcs[idx]
        pc_part = self.pc_parts[idx]

        # # load partial points
        # with open(pc_path,'rb') as f:
        #     pc = pickle.load(f).astype(np.float32)
        # # load gt
        # with open(pc_part_path,'rb') as f:
        #     pc_part = pickle.load(f).astype(np.float32)
        # # incase some item point number less than 3500 
        # if pc_part.shape[0]<self.pc_input_num:
        #     pc_part = np.repeat(pc_part,(self.pc_input_num//pc_part.shape[0])+1,axis=0)[0:self.pc_input_num]

        # # load the view metadata
        # image_view_id = view_path.split('.')[0].split('/')[-1]
        # part_view_id = pc_part_path.split('.')[0].split('/')[-1]
        
        # view_metadata = np.loadtxt(view_path[:-6]+'rendering_metadata.txt')

        # theta_part = math.radians(view_metadata[int(part_view_id),0])
        # phi_part = math.radians(view_metadata[int(part_view_id),1])

        # theta_img = math.radians(view_metadata[int(image_view_id),0])
        # phi_img = math.radians(view_metadata[int(image_view_id),1])

        # pc_part = self.rotation_y(self.rotation_x(pc_part, - phi_part),np.pi + theta_part)
        # pc_part = self.rotation_x(self.rotation_y(pc_part, np.pi - theta_img), phi_img)

        # normalize partial point cloud and GT to the same scale
        gt_mean = pc.mean(axis=0) 
        pc = pc -gt_mean
        pc_L_max = np.max(np.sqrt(np.sum(abs(pc ** 2), axis=-1)))
        pc = pc/pc_L_max

        pc_part = pc_part-gt_mean
        pc_part = pc_part/pc_L_max

        result = {}
        result['partial'] = (pc_part * self.scale).astype(np.float32)
        result['complete'] = (pc * self.scale).astype(np.float32)
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
        
        # if(not self.train):
        result['name'] = copy.deepcopy(self.key[idx]).replace("/", "_")

        result['label'] = np.array(self.labels[idx])
        result['class_index'] = views.float().numpy()
        return result

    def __len__(self):
        return len(self.key)

    def rotation_z(self, pts, theta):
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        rotation_matrix = np.array([[cos_theta, -sin_theta, 0.0],
                                    [sin_theta, cos_theta, 0.0],
                                    [0.0, 0.0, 1.0]])
        return pts @ rotation_matrix.T


    def rotation_y(self, pts, theta):
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        rotation_matrix = np.array([[cos_theta, 0.0, -sin_theta],
                                    [0.0, 1.0, 0.0],
                                    [sin_theta, 0.0, cos_theta]])
        return pts @ rotation_matrix.T


    def rotation_x(self, pts, theta):
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        rotation_matrix = np.array([[1.0, 0.0, 0.0],
                                    [0.0, cos_theta, -sin_theta],
                                    [0.0, sin_theta, cos_theta]])
        return pts @ rotation_matrix.T



# if __name__ == '__main__':
#     import tqdm
#     import os
#     import pandas as pd
#     config = {
#         'data_path': '~/Documents/datasets/ShapeNetViPC-Dataset',
#         'status': 'train',
#         'pc_input_num': 3500,
#         'R': 4,
#         'scale': 1,
#         'image_size': 480,
#         'augmentation': False,
#         'return_augmentation_params': False,
#         'debug': False,
#         'view_align': True,
#         'category': 'plane',
#         'mini': True
#     }
#     config['data_path'] = os.path.expanduser(config['data_path'])
#     vipc_data = ViPCDataLoaderMemory(**config)
#     for i in range(len(vipc_data)):
#         data = vipc_data[i]
#         print(data["partial"].shape, data["complete"].shape, data["class_index"].shape)

    # names = []
    # views = []
    # partial = []
    # complete = []
    # categories = []

    # test_dir = "data_processed_test/"
    # # os.makedirs(test_dir, exist_ok=True)
    # outdir = config['data_path']
    # outname = f"shapenet_vipc_processed_fps4096_R4_{config['category']}_{config['status']}.h5"
    

    # for i in tqdm.tqdm(range(len(vipc_data))):
    #     data = vipc_data[i]
    #     names.append(data['name'])
    #     views.append(data['view'])
    #     partial.append(data['partial'])
    #     complete.append(data['complete'])
    #     categories.append(data['category'])
    #     # print(data["partial"].shape)
    #     # print(data["complete"].shape)
    #     # save partial and complete as .xyz
    #     # np.savetxt(os.path.join(test_dir, data['name'].replace('/', '_') + '_partial.xyz'), data['partial'])
    #     # np.savetxt(os.path.join(test_dir, data['name'].replace('/', '_') + '_complete.xyz'), data['complete'])

    # out_df = pd.DataFrame({
    #     'name': names,
    #     'view': views,
    #     'partial': partial,
    #     'complete': complete,
    #     'category': categories
    # })
    # out_df.to_hdf(os.path.join(outdir, outname), key='df', mode='w')    

    