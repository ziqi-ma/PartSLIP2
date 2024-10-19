import os
import torch
import json
from pytorch3d.io import IO
import numpy as np
from src.utils import normalize_pc
from src.render_pc import render_pc
from src.gen_superpoint import gen_superpoint
import time
import glob
import h5py


def rotate_pts(pts, angles, device=None): # list of points as a tensor, N*3

    roll = angles[0].reshape(1)
    yaw = angles[1].reshape(1)
    pitch = angles[2].reshape(1)

    tensor_0 = torch.zeros(1).to(device)
    tensor_1 = torch.ones(1).to(device)

    RX = torch.stack([
                    torch.stack([tensor_1, tensor_0, tensor_0]),
                    torch.stack([tensor_0, torch.cos(roll), -torch.sin(roll)]),
                    torch.stack([tensor_0, torch.sin(roll), torch.cos(roll)])]).reshape(3,3)

    RY = torch.stack([
                    torch.stack([torch.cos(yaw), tensor_0, torch.sin(yaw)]),
                    torch.stack([tensor_0, tensor_1, tensor_0]),
                    torch.stack([-torch.sin(yaw), tensor_0, torch.cos(yaw)])]).reshape(3,3)

    RZ = torch.stack([
                    torch.stack([torch.cos(pitch), -torch.sin(pitch), tensor_0]),
                    torch.stack([torch.sin(pitch), torch.cos(pitch), tensor_0]),
                    torch.stack([tensor_0, tensor_0, tensor_1])]).reshape(3,3)

    R = torch.mm(RZ, RY)
    R = torch.mm(R, RX)
    if device == "cuda":
        R = R.cuda()
    pts_new = torch.mm(pts, R.T)
    return pts_new

def load_data_partseg_subset(data_path, class_choice):
    all_data = []
    all_label = []
    all_seg = []
    cat2id = {'airplane': 0, 'bag': 1, 'cap': 2, 'car': 3, 'chair': 4, 
                'earphone': 5, 'guitar': 6, 'knife': 7, 'lamp': 8, 'laptop': 9, 
                'motorbike': 10, 'mug': 11, 'pistol': 12, 'rocket': 13, 'skateboard': 14, 'table': 15}
    index_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]

    file = glob.glob(os.path.join(data_path, 'hdf5_data', '*test*.h5'))
    for h5_name in file:
        f = h5py.File(h5_name, 'r+')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        seg = f['pid'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
        all_seg.append(seg)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    all_seg = np.concatenate(all_seg, axis=0)
    #print(all_data.shape)
    # get random rotation
    # first time, generate random rotation
    id_choice = cat2id[class_choice]
    indices = (all_label == id_choice).squeeze()
    all_data = all_data[indices]
    all_seg = all_seg[indices]
    seg_start_index = index_start[id_choice]
    all_rotation = torch.load(f"{data_path}/random_rotation_test.pt")[indices]

    # get subset
    subset_idxs = np.loadtxt(f"/data/ziqi/shapenetpart/{class_choice}_subsample.txt").astype(int)
    sub_data = all_data[subset_idxs]
    sub_seg = all_seg[subset_idxs]
    sub_rotation = all_rotation[subset_idxs]
    sub_seg -= seg_start_index # labels start from 0

    return sub_data, sub_seg, sub_rotation, subset_idxs.tolist()
    

def Infer(xyz, rot, apply_rotation=False, save_dir="tmp"):
    
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    os.makedirs(save_dir, exist_ok=True)
    
    rgb = torch.ones(xyz.shape)*0.5
    
    if apply_rotation:
        # apply rotation
        rotated_pts = rotate_pts(torch.tensor(xyz).float(), rot)
        img_dir, pc_idx, screen_coords, num_views = render_pc(rotated_pts, rgb, save_dir, device)
        superpoint = gen_superpoint(rotated_pts, rgb, visualize=True, save_dir=save_dir)
    else:
        img_dir, pc_idx, screen_coords, num_views = render_pc(torch.tensor(xyz), rgb, save_dir, device)
        superpoint = gen_superpoint(torch.tensor(xyz), rgb, visualize=True, save_dir=save_dir)

    
if __name__ == "__main__":
    
    categories_list = {'airplane': 0, 'bag': 1, 'cap': 2, 'car': 3, 'chair': 4, 
                       'earphone': 5, 'guitar': 6, 'knife': 7, 'lamp': 8, 'laptop': 9, 
                       'motorbike': 10, 'mug': 11, 'pistol': 12, 'rocket': 13, 'skateboard': 14, 'table': 15}

    categories = ['airplane', 'bag', 'cap', 'car', 'chair','earphone','guitar','knife','lamp','laptop','motorbike',
                  'mug','pistol','rocket','skateboard', 'table']
    for category in categories:  
        stime = time.time()
        xyz, label, rotation, sample_idxs = load_data_partseg_subset('/data/ziqi/shapenetpart', category)
        for i in range(10):
            Infer(xyz[i,:,:], rotation[i,:], apply_rotation=False, save_dir=f"./data/img_sp/{category}/{sample_idxs[i]}")
        etime = time.time()
        print(category)
        print(etime-stime)