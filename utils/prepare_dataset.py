import os
import torch
import random
import numpy as np
import pandas as pd
from torch import nn
import torch.optim as optim
import matplotlib.pyplot as plt
from skimage import io, transform
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import h5py

class Select_Points(object):
    def __init__(self, output_size):
        assert isinstance(output_size, int)
        self.output_size = output_size

    def __call__(self, mesh):
        verts, faces = mesh
        verts = np.array(verts)
        areas = np.zeros((len(faces)))

        for i in range(len(areas)):
            a = np.linalg.norm(verts[faces[i][0]]-verts[faces[i][1]])
            b = np.linalg.norm(verts[faces[i][1]]-verts[faces[i][2]])
            c = np.linalg.norm(verts[faces[i][2]]-verts[faces[i][0]])
            s = 0.5*(a+b+c)
            areas[i] = max(s*(s-a)*(s-b)*(s-c),0)**0.5
            
        sampled_faces = (random.choices(faces, weights=areas, cum_weights=None, k=self.output_size))
        sampled_points = np.zeros((self.output_size, 3))

        for i in range(len(sampled_faces)):
            point1, point2, point3 = (verts[sampled_faces[i][0]],verts[sampled_faces[i][1]], verts[sampled_faces[i][2]])
            s, t = sorted([random.random(), random.random()])
            func = lambda i: s * point1[i] + (t-s)*point2[i] + (1-t)*point3[i]
            sampled_points[i] = (func(0), func(1), func(2))
        return sampled_points
    
class Normalize(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2
        norm_pointcloud = pointcloud - np.mean(pointcloud, axis=0) 
        norm_pointcloud /= np.max(np.linalg.norm(norm_pointcloud, axis=1))
        return  norm_pointcloud
    
class ToTensor(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2
        a = torch.from_numpy(pointcloud)
        return a.type(torch.DoubleTensor)


class ModelNet_Dataset(Dataset):
    def __init__(self, root_dir,  folder=None, transform_dense=None, transform_sparse=None):
        self.transform_dense = transform_dense
        self.transform_sparse = transform_sparse
        self.class_names = [dir for dir in sorted(os.listdir(root_dir)) if os.path.isdir(root_dir+"/"+dir)]
        self.class2idx = {class_name: i for i, class_name in enumerate(self.class_names)}
        self.files = []
        for cl in self.class_names:
            new_dir = root_dir+"/"+cl+"/"+folder
            for file in os.listdir(new_dir):
                if file.endswith('.off'):
                    sample = {}
                    sample['obj_path'] = new_dir+"/"+file
                    sample['class'] = cl
                    self.files.append(sample)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):            
        obj_path = self.files[idx]['obj_path']
        class_name = self.files[idx]['class']
        datapoint_dense = None
        datapoint_sparse = None
        with open(obj_path, 'r') as file:
            off_header = file.readline().strip()
            if 'OFF' == off_header:
                data, n_faces, __ = tuple([int(s) for s in file.readline().strip().split(' ')])
            else:
                data, n_faces, __  = tuple([int(s) for s in off_header[3:].split(' ')])
            data = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(data)]
            faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
            if self.transform_dense:
                datapoint_dense = self.transform_dense((data, faces))
            if self.transform_sparse:
                datapoint_sparse = self.transform_sparse((data, faces))
        return {
            'data_dense': datapoint_dense, 
            'data_sparse': datapoint_sparse,
            'class_idx': self.class2idx[class_name],
            'class_name': class_name
        }

def get_data_triplets(data_loader):
    data_dense = []
    data_sparse = []
    class_idx = []
    for data_batch in tqdm(data_loader):
        # print(data_batch['data_dense'].shape)
        data_dense.extend(data_batch['data_dense'].numpy().tolist())
        data_sparse.extend(data_batch['data_sparse'].numpy().tolist())
        class_idx.extend(data_batch['class_idx'].numpy().tolist())
    
    return data_dense, data_sparse, class_idx


def prepare_ViPC_mini_dataset(category, file_list_path, outpath, num_samples=2000):
    cat_map = {
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
    
    cat_id = cat_map[category]
    with open(file_list_path,'r') as f:
        lines = f.readlines()
    
    filtered_lines = [line for line in lines if line.split('/')[0] == cat_id]
    # randomly sample num_samples lines
    sampled_lines = random.sample(filtered_lines, num_samples)
    
    with open(outpath,'w') as f:
        f.writelines(sampled_lines) 


def prepare_modenet():

    dense_tra = transforms.Compose([
    Select_Points(1024),
    # Normalize(),
    # ToTensor()
    ])
    sparse_tra = transforms.Compose([
    Select_Points(256),
    # Normalize(),
    # ToTensor()
    ])
    dataset_path = "/home/ez48awud/Documents/datasets/ModelNet10/ModelNet10"

    train_dataset = ModelNet_Dataset(dataset_path, "train", dense_tra, sparse_tra)
    test_dataset = ModelNet_Dataset(dataset_path, "test", dense_tra, sparse_tra)

    train_loader = DataLoader(train_dataset, shuffle=True, num_workers=2, batch_size=1)
    test_loader = DataLoader(test_dataset, shuffle=True, num_workers=2, batch_size=1)

    print('Train DataSet len:',len(train_dataset))
    print('Train DataSet Classes:',train_dataset.class_names)
    print('Test DataSet len:',len(test_dataset))
    print('Test DataSet Classes:',test_dataset.class_names)

    
    # save as h5 file of sparse and dense pointclouds with class index
    
    # train_data_dense, train_data_sparse, train_class_idx = get_data_triplets(train_loader)
    # outpath = "/home/ez48awud/Documents/datasets/ModelNet10/ModelNet10_train_1024_256.h5"
    # h5f = h5py.File(outpath, 'w')
    # h5f.create_dataset('data_dense', data=train_data_dense)
    # h5f.create_dataset('data_sparse', data=train_data_sparse)
    # h5f.create_dataset('class_idx', data=train_class_idx)
    # h5f.close()

    # test_data_dense, test_data_sparse, test_class_idx = get_data_triplets(test_loader)
    # outpath = "/home/ez48awud/Documents/datasets/ModelNet10/ModelNet10_test_1024_256.h5"
    # h5f = h5py.File(outpath, 'w')
    # h5f.create_dataset('data_dense', data=test_data_dense)
    # h5f.create_dataset('data_sparse', data=test_data_sparse)
    # h5f.create_dataset('class_idx', data=test_class_idx)
    # h5f.close()

    # # save class names
    # class_names = train_dataset.class_names
    # outpath = "/home/ez48awud/Documents/datasets/ModelNet10/ModelNet10_class_names.txt"
    # with open(outpath, 'w') as f:
    #     for class_name in class_names:
    #         f.write(class_name + '\n')

    # print("Done")

    # visualize 1 pointcloud for each class
    for pointcloud in train_loader:
        # print(pointcloud['class_idx'])
        # print(pointcloud['data_dense'].shape)
        # print(pointcloud['data_sparse'].shape)
        # pointcloud = pointcloud_batch[0]
        print(pointcloud['class_idx'], pointcloud['class_name'])
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(pointcloud['data_dense'][:, 0], pointcloud['data_dense'][:, 1], pointcloud['data_dense'][:, 2], c='r')
        ax.scatter(pointcloud['data_sparse'][0][:, 0], pointcloud['data_sparse'][0][:, 1], pointcloud['data_sparse'][0][:, 2], c='b')
        ax.set_axis_off()
        plt.show()
    


if __name__ == '__main__':

    category="table"
    train_file_path = "train_list.txt"
    test_file_path = "test_list.txt"
    
    prepare_ViPC_mini_dataset(category, train_file_path, f"train_list_mini_{category}.txt", num_samples=2000)
    prepare_ViPC_mini_dataset(category, test_file_path, f"test_list_mini_{category}.txt", num_samples=1000)
    