import os
import torch
import json
from pytorch3d.io import IO
import numpy as np
from src.utils import normalize_pc
from src.render_pc import render_pc
from src.glip_inference import glip_inference, load_model
from src.bbox2seg import bbox2seg
from gen_sp_partnete import rotate_pts
import time
from segment_anything import sam_model_registry, SamPredictor

def compute_iou(pred, gt):
    num_parts = int(gt.max()+1)
    ious = []
    for i in range(num_parts):
        I = np.logical_and(pred==i, gt==i).sum()
        U = np.logical_or(pred==i, gt==i).sum()
        if U == 0:
            pass
        else:
            iou = I / U
            ious.append(iou)
    mean_iou = np.mean(ious)
    return mean_iou

def Infer(input_pc_file, category, model, part_names, apply_rotation=False, zero_shot=True, save_dir="tmp"):
    if zero_shot:
        config ="GLIP/configs/glip_Swin_L.yaml"
        weight_path = "/data/ziqi/checkpoints/semseg3d/glip_large_model.pth"
        #print("-----Zero-shot inference of %s-----" % input_pc_file)
    else:
        config ="GLIP/configs/glip_Swin_L_pt.yaml"
        weight_path = "./models/%s.pth" % category
        print("-----Few-shot inference of %s-----" % input_pc_file)
        
    #print("[loading GLIP model...]")
    glip_demo = load_model(config, weight_path)
    obj_path = "/".join(input_pc_file.split("/")[:-1])

    #print("[creating tmp dir...]")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    io = IO()
    os.makedirs(save_dir, exist_ok=True)
    #print(save_dir)
    
    xyz, rgb = normalize_pc(input_pc_file, save_dir, io, device)
    # apply rotation
    rot = torch.load(f"{obj_path}/rand_rotation.pt")
    if apply_rotation:
        rotated_pts = rotate_pts(torch.tensor(xyz).float(), rot)
    else:
        rotated_pts = torch.tensor(xyz).float()
    
    #print("[rendering input point cloud...]")
    img_dir, pc_idx, screen_coords, num_views = render_pc(rotated_pts, rgb, save_dir, device)
    
    #print("[glip infrence...]")
    SAM_ENCODER_VERSION = "vit_h"
    SAM_CHECKPOINT_PATH = "/data/ziqi/checkpoints/semseg3d/sam_vit_h_4b8939.pth"
    sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
    sam.to(device=torch.device("cuda:0"))
    sam_predictor = SamPredictor(sam)
    masks = glip_inference(glip_demo, save_dir, part_names, sam_predictor, num_views=num_views)
    
    #print('[generating superpoints...]')
    superpoint = np.load(f"/home/ziqi/Repos/PartSLIP2/data/img_sp/{category}/{model}/sp.npy", allow_pickle=True)
    
    #print('[converting bbox to 3D segmentation...]')
    sem_seg, _ = bbox2seg(rotated_pts, superpoint, masks, screen_coords, pc_idx, part_names, save_dir, solve_instance_seg=False, num_view=num_views)
    gt = np.load(f"/data/ziqi/partnet-mobility/test/{category}/{model}/label.npy", allow_pickle=True).item()["semantic_seg"]
    acc = np.sum(sem_seg==gt)/gt.shape[0]
    iou = compute_iou(sem_seg, gt)
    
    print(f"[finish!], acc {acc}, iou {iou}")
    return acc, iou
    
if __name__ == "__main__":
    
    partnete_meta = json.load(open("PartNetE_meta.json")) 
    categories = partnete_meta.keys()
    all_mean_ious = []
    all_stime = time.time()
    for category in categories:
        stime = time.time()
        accs = []
        ious = []
        models = os.listdir(f"/data/ziqi/partnet-mobility/test/{category}") # list of models
        with open(f"/data/ziqi/partnet-mobility/test/{category}/subsampled_ids.txt", 'r') as f:
            data_paths = f.read().splitlines()
        for path in data_paths:
            model = path.split("/")[-1]
            part_names = partnete_meta[category]
            part_names = [f"{part} of a {category}" for part in partnete_meta[category]]
            acc, iou = Infer(f"/data/ziqi/partnet-mobility/test/{category}/{model}/pc.ply", category, model, part_names, apply_rotation=False, zero_shot=True, save_dir=f"./result_ps/{category}/{model}")
            accs.append(acc)
            if not np.isnan(iou):
                ious.append(iou)
        mean_acc = np.mean(accs)
        mean_iou = np.mean(ious)
        print(f"{category} acc: {mean_acc}, iou: {mean_iou}")
        etime = time.time()
        print(f"time {etime-stime}")
        all_mean_ious.append(mean_iou)
    all_etime = time.time()
    overall_mean_iou = np.mean(all_mean_ious)
    print(f"overall mean iou: {overall_mean_iou}")
    print(f"overall time: {all_etime-all_stime}")
        