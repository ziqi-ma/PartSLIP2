import os
import torch
import json
from pytorch3d.io import IO
import numpy as np
from src.utils import normalize_pc
from src.render_pc import render_pc
from src.gen_superpoint import gen_superpoint
import time
import open3d as o3d


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

def Infer(obj_dir, save_dir="tmp"):
    
    #print("[creating tmp dir...]")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    io = IO()
    os.makedirs(save_dir, exist_ok=True)

    #print("[normalizing input point cloud...]")
    pcd = o3d.io.read_point_cloud(f"{obj_dir}/points5000.pcd")
    xyz = np.asarray(pcd.points)
    xyz = xyz - xyz.mean(axis=0)
    xyz = xyz / np.linalg.norm(xyz, ord=2, axis=1).max().item()
    rgb = np.asarray(pcd.colors)
    
    # no need to apply rotation because the point cloud comes pre-rotated already for objaverse sets
    #print("[rendering input point cloud...]")
    img_dir, pc_idx, screen_coords, num_views = render_pc(torch.tensor(xyz).float(), rgb, save_dir, device)
    
    # print('[generating superpoints...]')
    superpoint = gen_superpoint(torch.tensor(xyz).float(), rgb, visualize=True, save_dir=save_dir)
    
    #print("[finish!]")
    
if __name__ == "__main__":
    stime = time.time()
    split = "shapenetpart"#"unseen"#"seenclass"#
    data_path = '/data/ziqi/objaverse/holdout'
    class_uids = [uid for uid in sorted(os.listdir(f"{data_path}/{split}"))]
    #class_uids = [class_uids[i] for i in [2,3,4,8,23,25,29,31]]
    print(len(class_uids))
    
    for class_uid in class_uids:  
        obj_dir = f"{data_path}/{split}/{class_uid}"
        Infer(obj_dir, save_dir=f"./data/img_sp/{class_uid}")
    etime = time.time()
    print(etime-stime)