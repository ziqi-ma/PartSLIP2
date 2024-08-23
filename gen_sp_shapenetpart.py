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

def load_data_partseg(data_path, class_choice):
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

    # take 5 random (but randomness is fixed)
    random_indices = torch.randint(0, all_data.shape[0], (10,))
    torch.save(random_indices, f"./data/img_sp/{class_choice}/rand_idxs.pt")
    sub_data = all_data[random_indices]
    sub_seg = all_seg[random_indices]
    sub_rotation = all_rotation[random_indices]
    sub_seg -= seg_start_index # labels start from 0

    return sub_data, sub_seg, sub_rotation, random_indices
    

def Infer(xyz, rot, save_dir="tmp"):
    
    print("[creating tmp dir...]")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    io = IO()
    os.makedirs(save_dir, exist_ok=True)
    
    print("[normalizing input point cloud...]")
    rgb = torch.ones(xyz.shape)*0.5
    # apply rotation
    rotated_pts = rotate_pts(torch.tensor(xyz).float(), rot)
    
    print("[rendering input point cloud...]")
    img_dir, pc_idx, screen_coords, num_views = render_pc(rotated_pts, rgb, save_dir, device)
    
    # print('[generating superpoints...]')
    superpoint = gen_superpoint(rotated_pts, rgb, visualize=False, save_dir=save_dir)
    
    print("[finish!]")
    
if __name__ == "__main__":
    stime = time.time()
    categories_list = {'airplane': 0, 'bag': 1, 'cap': 2, 'car': 3, 'chair': 4, 
                       'earphone': 5, 'guitar': 6, 'knife': 7, 'lamp': 8, 'laptop': 9, 
                       'motorbike': 10, 'mug': 11, 'pistol': 12, 'rocket': 13, 'skateboard': 14, 'table': 15}

    # categories = ["Camera", "Cart", "Dispenser", "Kettle"]
    # categories = ["Bottle", "Chair", "Display", "Door"]
    # categories = ["Knife", "Lamp", "StorageFurniture", "Table"]
    # categories = ["KitchenPot", "Oven", "Suitcase", "Toaster"]
    categories = ["airplane"]
    for category in categories:  
        xyz, label, rotation, sample_idxs = load_data_partseg('/data/ziqi/shapenetpart', category)
        for i in range(10):
            Infer(xyz[i,:,:], category, rotation[i,:], save_dir=f"./data/img_sp/{category}/{sample_idxs[i].item()}")
    etime = time.time()
    print(etime-stime)