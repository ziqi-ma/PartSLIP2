import os
import torch
import json
from pytorch3d.io import IO
import numpy as np
from src.utils import normalize_pc
from src.render_pc import render_pc
from src.glip_inference import glip_inference, load_model
from src.bbox2seg import bbox2seg
from gen_sp_shapenetpart import rotate_pts
import time
from segment_anything import sam_model_registry, SamPredictor
import glob
import h5py

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

    # get subset
    subset_idxs = np.loadtxt(f"/data/ziqi/shapenetpart/{class_choice}_subsample.txt").astype(int)
    sub_data = all_data[subset_idxs]
    sub_seg = all_seg[subset_idxs]
    sub_rotation = all_rotation[subset_idxs]
    sub_seg -= seg_start_index # labels start from 0

    return sub_data, sub_seg, sub_rotation, subset_idxs.tolist()


def Infer(category, model, xyz, rot, gt, part_names, apply_rotation=False, zero_shot=True, save_dir="tmp"):
    if zero_shot:
        config ="GLIP/configs/glip_Swin_L.yaml"
        weight_path = "/data/ziqi/checkpoints/semseg3d/glip_large_model.pth"
        print(f"-----Zero-shot inference of -----{category}{model}")
    else:
        config ="GLIP/configs/glip_Swin_L_pt.yaml"
        weight_path = "./models/%s.pth" % category
        print(f"-----Few-shot inference of -----{category}{model}")
        
    #print("[loading GLIP model...]")
    glip_demo = load_model(config, weight_path)

    #print("[creating tmp dir...]")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    io = IO()
    os.makedirs(save_dir, exist_ok=True)
    #print(save_dir)
    
    #print("[normalizing input point cloud...]")
    rgb = torch.ones(xyz.shape)*0.5
    # apply rotation
    if apply_rotation:
        rotated_pts = rotate_pts(torch.tensor(xyz).float(), rot)
    else:
        rotated_pts = torch.tensor(xyz)
    
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
    acc = np.sum(sem_seg==gt)/gt.shape[0]
    iou = compute_iou(sem_seg, gt)
    
    #print(f"[finish!], acc {acc}, iou {iou}")
    return acc, iou
    
if __name__ == "__main__":
    stime = time.time()
    categories = ['airplane', 'bag', 'cap', 'car', 'chair','earphone','guitar','knife','lamp','laptop','motorbike',
                  'mug','pistol','rocket','skateboard', 'table']
    cat2part = {'airplane': ['body','wing','tail','engine or frame'], 'bag': ['handle','body'], 'cap': ['panels or crown','visor or peak'], 
            'car': ['roof','hood','wheel or tire','body'],
            'chair': ['back','seat pad','leg','armrest'], 'earphone': ['earcup','headband','data wire'], 
            'guitar': ['head or tuners','neck','body'], 
            'knife': ['blade', 'handle'], 'lamp': ['leg or wire','lampshade'], 
            'laptop': ['keyboard','screen or monitor'], 
            'motorbike': ['gas tank','seat','wheel','handles or handlebars','light','engine or frame'], 'mug': ['handle', 'cup'], 
            'pistol': ['barrel', 'handle', 'trigger and guard'], 
            'rocket': ['body','fin','nose cone'], 'skateboard': ['wheel','deck','belt for foot'], 'table': ['desktop','leg or support','drawer']}
    all_mious = []
    for category in categories:
        accs = []
        ious = []
        part_names = cat2part[category]
        part_names = [f"{part} of a {category}" for part in part_names]
        xyz, label, rotation, sample_idxs = load_data_partseg('/data/ziqi/shapenetpart', category)
        for i in range(10):
            acc, iou = Infer(category, sample_idxs[i], xyz[i,:,:], rotation[i,:], label[i,:], part_names, apply_rotation=True, zero_shot=True, save_dir=f"./data/img_sp/{category}/{sample_idxs[i]}")
            accs.append(acc)
            ious.append(iou)
        mean_acc = np.mean(accs)
        mean_iou = np.mean(ious)
        print(f"{category} acc: {mean_acc}, iou: {mean_iou}")
        all_mious.append(mean_iou)
    all_mean_iou = np.mean(all_mious)
    etime = time.time()
    print(etime-stime)
    print(all_mean_iou)