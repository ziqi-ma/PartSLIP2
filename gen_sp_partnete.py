import os
import torch
import json
from pytorch3d.io import IO
import numpy as np
from src.utils import normalize_pc
from src.render_pc import render_pc
from src.gen_superpoint import gen_superpoint
import time

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

def Infer(input_pc_file, apply_rotation=False, save_dir="tmp"):
    
    #print("[creating tmp dir...]")
    obj_path = "/".join(input_pc_file.split("/")[:-1])
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    io = IO()
    os.makedirs(save_dir, exist_ok=True)
    
    #print("[normalizing input point cloud...]")
    xyz, rgb = normalize_pc(input_pc_file, save_dir, io, device)
    # apply rotation
    rot = torch.load(f"{obj_path}/rand_rotation.pt")

    if apply_rotation:
        rotated_pts = rotate_pts(torch.tensor(xyz).float(), rot)
    else:
        rotated_pts = torch.tensor(xyz).float()
    
    img_dir, pc_idx, screen_coords, num_views = render_pc(rotated_pts, rgb, save_dir, device)
    
    superpoint = gen_superpoint(rotated_pts, rgb, visualize=True, save_dir=save_dir)
    
    #print("[finish!]")
    
if __name__ == "__main__":
    
    partnete_meta = json.load(open("PartNetE_meta.json")) 
    categories = partnete_meta.keys()

    for category in categories:  
        stime = time.time()
        with open(f"/data/ziqi/partnet-mobility/test/{category}/subsampled_ids.txt", 'r') as f:
            data_paths = f.read().splitlines()
        for path in data_paths:
            model = path.split("/")[-1]
            Infer(f"{path}/pc.ply", apply_rotation=False, save_dir=f"./data/img_sp/{category}/{model}")
        etime = time.time()
        print(category)
        print(etime-stime)
