import numpy as np
import json
import os
from scipy.spatial.transform import Rotation as R


# from Minghua's notebook
def save_colored_pc(file_name, xyz, rgb):
    # rgb is [0, 1]
    n = xyz.shape[0]
    f = open(file_name, "w")
    f.write("ply\n")
    f.write("format ascii 1.0\n")
    f.write("element vertex %d\n" % n)
    f.write("property float x\n")
    f.write("property float y\n")
    f.write("property float z\n")
    f.write("property uchar red\n")
    f.write("property uchar green\n")
    f.write("property uchar blue\n")
    f.write("end_header\n")
    rgb = rgb * 255
    for i in range(n):
        if rgb.shape[1] == 3:
            f.write(
                "%f %f %f %d %d %d\n"
                % (xyz[i][0], xyz[i][1], xyz[i][2], rgb[i][0], rgb[i][1], rgb[i][2])
            )
        else:
            f.write(
                "%f %f %f %d %d %d\n"
                % (xyz[i][0], xyz[i][1], xyz[i][2], rgb[i][0], rgb[i][0], rgb[i][0])
            )


# refer to kornia
def unproject_points(
    point_2d,
    depth,
    camera_matrix,
    opengl=False,
):
    r"""Unproject a 2d point in 3d.
    Transform coordinates in the pixel frame to the camera frame.
    Args:
        point2d: tensor containing the 2d to be projected to
            world coordinates. The shape of the tensor can be :math:`(*, 2)`.
        depth: tensor containing the depth value of each 2d
            points. The tensor shape must be equal to point2d :math:`(*, 1)`.
        camera_matrix: tensor containing the intrinsics camera
            matrix. The tensor shape must be :math:`(*, 3, 3)`.
        normalize: whether to normalize the pointcloud. This
            must be set to `True` when the depth is represented as the Euclidean
            ray length from the camera position.
    Returns:
        tensor of (x, y, z) world coordinates with shape :math:`(*, 3)`.
    Example:
        >>> _ = torch.manual_seed(0)
        >>> x = torch.rand(1, 2)
        >>> depth = torch.ones(1, 1)
        >>> K = torch.eye(3)[None]
        >>> unproject_points(x, depth, K)
        tensor([[0.4963, 0.7682, 1.0000]])
    """

    # projection eq. K_inv * [u v 1]'
    # x = (u - cx) * Z / fx
    # y = (v - cy) * Z / fy

    # unpack coordinates
    u_coord = point_2d[..., 0]
    v_coord = point_2d[..., 1]

    # unpack intrinsics
    fx = camera_matrix[..., 0, 0]
    fy = camera_matrix[..., 1, 1]
    cx = camera_matrix[..., 0, 2]
    cy = camera_matrix[..., 1, 2]

    # projective
    x_coord = (u_coord - cx) / fx
    y_coord = (v_coord - cy) / fy

    xyz = np.stack([x_coord, y_coord, np.ones_like(x_coord)], axis=-1)

    if opengl:
        xyz = xyz * np.array([1, -1, -1])[None, :]

    return xyz * depth


def project_points(points, cam2world, K, resolution):
    world2cam = np.linalg.inv(cam2world)

    # Transform points into camera frame
    points = np.concatenate((points, np.ones_like(points[:, :1])), -1)
    points_cam = (world2cam @ points.T).T
    points_cam[..., 2] *= -1
    depth = points_cam[..., -2]

    # Project 3D points into 2D
    points_2d = (K @ points_cam[:, :3].T).T
    points_2d /= points_2d[:, 2:]
    points_2d = points_2d[:, :2]

    # Flip y axis
    points_2d[..., 1] = (resolution - 1) - points_2d[..., 1]
    return points_2d, depth


def cal_iou_point(a, b):
    I = (a & b).sum()
    U = (a | b).sum()
    return I / U


def get_union(f, x):  # union-find
    if f[x] == x:
        return x
    f[x] = get_union(f, f[x])
    return f[x]


r = R.from_euler("xyz", [-90, 90, 0], degrees=True)


def fuse_point_cloud(files: str, num_views, num_points):
    for file in files:
        fused_points = []
        fused_rgb = []
        fused_label = []
        for i in range(num_views):
            f = np.load(f"{file}/{i:04d}.npy", allow_pickle=True).item()
            rgb = np.array(f["colors"])
            depth = np.array(f["depth"])

            v, u = np.nonzero(depth < 100)
            points_3d_cam = unproject_points(
                np.stack([u, v], axis=-1),
                depth[v, u][..., None],
                np.array(f["cam_K"]),
                opengl=True,
            )
            # R @ x + t
            # 3x3 3x1 + 3x1
            cam2world = np.array(f["cam2world"])
            rot, trans = cam2world[:3, :3], cam2world[:3, -1]
            points_3d = np.matmul(rot, points_3d_cam[..., None]) + trans[..., None]

            fused_points.append(points_3d)
            fused_rgb.append(rgb[v, u] / 255.0)

        fused_xyz = np.concatenate(fused_points)[..., 0]
        fused_rgb = np.concatenate(fused_rgb)
        fused_label = np.concatenate(fused_label)
