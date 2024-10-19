import os
import torch
import json
from pytorch3d.io import IO
import numpy as np
from src.utils import normalize_pc
from src.render_pc import render_pc
from src.glip_inference import glip_inference, load_model
from src.bbox2seg import bbox2seg
import time
import torch.nn.functional as F
from segment_anything import sam_model_registry, SamPredictor
import open3d as o3d

def visualize_pts(pts, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors.numpy())
    o3d.visualization.draw_plotly([pcd])
    
def visualize_pt_labels(pts, labels): # pts is n*3, colors is n, 0 - n-1 where 0 is unlabeled
    part_num = labels.max()
    cmap_matrix = torch.tensor([[1,1,1], [1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,0,1],
                [0,1,1], [0.5,0.5,0.5], [0.5,0.5,0], [0.5,0,0.5],[0,0.5,0.5],
                [0.1,0.2,0.3],[0.2,0.5,0.3], [0.6,0.3,0.2], [0.5,0.3,0.5],
                [0.6,0.7,0.2],[0.5,0.8,0.3]])[:part_num+1,:]
    colors = ["white", "red", "green", "blue", "yellow", "magenta", "cyan","grey", "olive",
                "purple", "teal", "navy", "darkgreen", "brown", "pinkpurple", "yellowgreen", "limegreen"]
    caption_list=[f"{i}:{colors[i]}" for i in range(part_num+1)]
    onehot = F.one_hot(labels.long(), num_classes=part_num+1) * 1.0 # n_pts, part_num+1, each row 00.010.0, first place is unlabeled (0 originally)
    pts_rgb = torch.matmul(onehot, cmap_matrix) # n_pts,3
    visualize_pts(pts, pts_rgb)
    print(caption_list)

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

def Infer(obj_dir, class_uid, zero_shot=True, save_dir="tmp", visualize=False):
    if zero_shot:
        config ="GLIP/configs/glip_Swin_L.yaml"
        weight_path = "/data/ziqi/checkpoints/semseg3d/glip_large_model.pth"
        #print("-----Zero-shot inference of %s-----" % class_uid)
    else:
        config ="GLIP/configs/glip_Swin_L_pt.yaml"
        weight_path = "./models/%s.pth" % category
        print("-----Few-shot inference of %s-----" % class_uid)
        
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
    print(save_dir)
    
    #print("[normalizing input point cloud...]")
    pcd = o3d.io.read_point_cloud(f"{obj_dir}/points5000.pcd")
    xyz = np.asarray(pcd.points)
    xyz = xyz - xyz.mean(axis=0)
    xyz = xyz / np.linalg.norm(xyz, ord=2, axis=1).max().item()
    rgb = np.asarray(pcd.colors)
    
    print("[rendering input point cloud...]")
    img_dir, pc_idx, screen_coords, num_views = render_pc(torch.tensor(xyz).float(), rgb, save_dir, device)

    with open(f"{obj_dir}/label_map.json") as f:
        mapping = json.load(f)
    part_names = []
    for i in range(len(mapping)):
        part_names.append(mapping[str(i+1)]) # label starts from 1

    # decorate
    cat = " ".join(class_uid.split("_")[:-1])
    part_names = [f"{part} of a {cat}" for part in part_names]
    # the default is -1 so no need to provide an "other" label
    
    #print("[glip infrence...]")
    SAM_ENCODER_VERSION = "vit_h"
    SAM_CHECKPOINT_PATH = "/data/ziqi/checkpoints/semseg3d/sam_vit_h_4b8939.pth"
    sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
    sam.to(device=torch.device("cuda:0"))
    sam_predictor = SamPredictor(sam)
    masks = glip_inference(glip_demo, save_dir, part_names, sam_predictor, num_views=num_views)
    
    #print('[generating superpoints...]')
    superpoint = np.load(f"/home/ziqi/Repos/PartSLIP2/data/img_sp/{class_uid}/sp.npy", allow_pickle=True)
    
    #print('[converting bbox to 3D segmentation...]')
    sem_seg, _ = bbox2seg(torch.tensor(xyz).float(), superpoint, masks, screen_coords, pc_idx, part_names, save_dir, solve_instance_seg=False, num_view=num_views)
    gt = np.load(f"{obj_dir}/labels.npy") - 1 # now -1 becomes unlabeled, 0-k-1 are classes, save as prediction
    acc = np.sum(sem_seg==gt)/gt.shape[0]

    if visualize:
        visualize_pt_labels(xyz, torch.tensor(gt)+1)
        visualize_pt_labels(xyz, torch.tensor(sem_seg)+1)
    iou = compute_iou(sem_seg, gt)
    
    print(f"[finish!], acc {acc}, iou {iou}")
    return acc, iou
    
if __name__ == "__main__":
    stime = time.time()
    data_path = '/data/ziqi/objaverse/holdout'
    split = "shapenetpart"#"unseen"#"seenclass"#
    visualization = False
    class_uids = [uid for uid in sorted(os.listdir(f"{data_path}/{split}"))]
    if visualization:
        class_uids = [class_uids[i] for i in [2,3,4,8,23,25,29,31]]
    iou_list = []
    acc_list = []
    for class_uid in class_uids:  
        obj_dir = f"{data_path}/{split}/{class_uid}"
        cat = " ".join(class_uid.split("_")[:-1])
        acc, iou = Infer(obj_dir, class_uid, zero_shot=True, save_dir=f"./result_ps/{class_uid}", visualize=visualization)
        iou_list += [iou]
        acc_list += [acc]
    miou = np.mean(iou_list)
    macc = np.mean(acc_list)
    print(f"instance mean iou: {miou}, instance mean acc: {macc}")
    etime = time.time()
    print(etime-stime)
        