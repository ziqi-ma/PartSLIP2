import argparse
import json
from pathlib import Path

import numpy as np
import sapien.core as sapien
import imageio.v3 as imageio
from transforms3d.quaternions import mat2quat
import numpy as np


def normalize_vector(x, eps=1e-6):
    x = np.asarray(x)
    assert x.ndim == 1, x.ndim
    norm = np.linalg.norm(x)
    if norm < eps:
        return np.zeros_like(x)
    else:
        return x / norm


def look_at(eye, target, up=(0, 0, 1)) -> sapien.Pose:
    """Get the camera pose in SAPIEN by the Look-At method.

    Note:
        https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/lookat-function/framing-lookat-function.html
        The SAPIEN camera follows the convention: (forward, right, up) = (x, -y, z)
        while the OpenGL camera follows (forward, right, up) = (-z, x, y)
        Note that the camera coordinate system (OpenGL) is left-hand.

    Args:
        eye: camera location
        target: looking-at location
        up: a general direction of "up" from the camera.

    Returns:
        sapien.Pose: camera pose
    """
    forward = normalize_vector(np.array(target) - np.array(eye))
    up = normalize_vector(up)
    left = np.cross(up, forward)
    if np.linalg.norm(left) < 1e-6:
        print("Corner case for LookAt")
        # https://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/
        left = normalize_vector(np.random.randn(3))
    up = np.cross(forward, left)
    rotation = np.stack([forward, left, up], axis=1)
    return sapien.Pose(p=eye, q=mat2quat(rotation))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--scale", type=float, default=0.8)
    parser.add_argument("--random-qpos", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--data-camera-mode", type=str, default="laptop")
    args = parser.parse_args()

    # ---------------------------------------------------------------------------- #
    # Init
    # ---------------------------------------------------------------------------- #
    np.random.seed(args.seed)
    urdf_file = Path(args.urdf_file)
    model_dir = urdf_file.parent
    model_id = model_dir.name

    img_dir = Path(args.output_dir) / "img" / f"{model_id}"
    cam_dir = Path(args.output_dir) / "camera" / f"{model_id}"
    img_dir.mkdir(parents=True, exist_ok=True)
    cam_dir.mkdir(parents=True, exist_ok=True)
    print("Images will be saved in", img_dir)
    print("Cameras will be saved in", cam_dir)

    # NOTE(jigu): This bounding box seems to be in Blender convention
    with open(model_dir / "bounding_box.json", "r") as f:
        bounding_box = json.load(f)
    bbox_size = np.array(bounding_box["max"]) - np.array(bounding_box["min"])
    scale_factor = args.scale / np.max(bbox_size)

    # ---------------------------------------------------------------------------- #
    # Create scene
    # ---------------------------------------------------------------------------- #
    engine = sapien.Engine()
    renderer = sapien.SapienRenderer()
    # Disable SAPIEN warning for convex hull
    engine.set_log_level("off")
    engine.set_renderer(renderer)

    scene = engine.create_scene()
    scene.set_timestep(1 / 100.0)

    loader = scene.create_urdf_loader()
    loader.load_multiple_collisions_from_file = True
    loader.fix_root_link = True
    loader.scale = scale_factor
    # NOTE: Kinematic articulation has an incorrect configuration unless scene.step()
    articulation = loader.load(args.urdf_file)
    # NOTE: Need to set pose explicitly
    articulation.set_pose(sapien.Pose())
    assert articulation, "URDF ({}) is not loaded.".format(args.urdf_file)
    # print(urdf.get_qpos())

    # Lighting
    # scene.set_ambient_light([0.5] * 3)
    scene.add_directional_light([1, 1, -1], [1.0, 1.0, 1.0])
    scene.add_directional_light([1, -1, -1], [1.0, 1.0, 1.0])
    scene.add_directional_light([-1, 1, -1], [1.0, 1.0, 1.0])

    # ---------------------------------------------------------------------------- #
    # Visualization
    # ---------------------------------------------------------------------------- #
    # from sapien.utils.viewer import Viewer
    # viewer = Viewer(renderer)
    # viewer.set_scene(scene)
    # while not viewer.closed:
    #     # scene.step()
    #     scene.update_render()
    #     viewer.render()

    # ---------------------------------------------------------------------------- #
    # Rendering
    # ---------------------------------------------------------------------------- #
    # Add cameras
    near, far = 0.1, 100
    width, height = args.resolution, args.resolution
    fovy = np.arctan(32 / 2 / 35) * 2
    camera = scene.add_camera(
        name="camera",
        width=width,
        height=height,
        fovy=fovy,
        near=near,
        far=far,
    )

    aabb = [[-args.scale / 2] * 3, [args.scale / 2] * 3]
    meta = {"camera_angle_x": fovy, "aabb": aabb}
    frames = []
    azimuths = []
    elevations = []

    if args.random_qpos:
        qlimits = np.ascontiguousarray(articulation.get_qlimits())  # [N, 2]
        if not np.any(np.isinf(qlimits)):
            qpos = np.random.uniform(qlimits[:, 0], qlimits[:, 1])
            articulation.set_qpos(qpos)
            joint_positions = qpos.tolist()
        else:
            qpos = np.random.uniform(-np.pi, np.pi, len(qlimits))
            articulation.set_qpos(qpos)
            joint_positions = qpos.tolist()

    poi = np.zeros(3)
    views = [
        [30, 0],
        [30, 60],
        [30, 120],
        [30, 180],
        [30, 240],
        [30, 300],
        [-30, 30],
        [-30, 90],
        [-30, 150],
        [-30, 210],
        [-30, 270],
        [-30, 330],
    ]
    for i in range(len(views)):
        # Sample random camera location above objects
        azimuth = views[i][1] * np.pi / 180
        elevation = views[i][0] * np.pi / 180
        x = np.cos(azimuth) * np.cos(elevation)
        y = np.sin(azimuth) * np.cos(elevation)
        z = np.sin(elevation)
        location = np.array([x, y, z]) * 1.2

        # Set camera pose
        cam_pose = look_at(location, poi)
        camera.set_pose(cam_pose)

        # Update for render and take picture
        scene.update_render()
        camera.take_picture()

        # NOTE: camera matrix is valid after scene.update_render()
        cam2world_matrix = camera.get_model_matrix()

        frame = {
            "file_path": f"{i:04d}.png",
            "transform_matrix": cam2world_matrix.tolist(),
        }
        if args.random_qpos:
            frame["joint_positions"] = joint_positions
        frames.append(frame)

        azimuths.append(azimuth)
        elevations.append(elevation)

        rgba = camera.get_float_texture("Color")  # [H, W, 4]
        rgba = (rgba * 255).clip(0, 255).astype("uint8")
        seg = camera.get_uint32_texture("Segmentation")  # [H, W, 4]
        # Use actor-level seg to replace alpha channel
        rgba[..., -1] = (seg[..., 1] != 0).astype("uint8") * 255
        imageio.imwrite(str(img_dir / frames[i]["file_path"]), rgba)

        seg_map = seg[..., 0]
        position = camera.get_float_texture("Position")
        depth = -position[..., 2]
        intrinsic = camera.get_intrinsic_matrix()
        cam2world = camera.get_model_matrix()
        data = {
            "colors": rgba[..., :3],
            "depth": depth,
            "cam_K": intrinsic,
            "cam2world": cam2world,
            "seg": seg_map,
        }
        np.save(str(img_dir / frames[i]["file_path"]).replace(".png", ".npy"), data)

    # Save meta info
    meta["frames"] = frames
    with open(img_dir / "transforms.json", "w") as f:
        json.dump(meta, f, indent=4)

    # Save camera poses
    np.save(cam_dir / "rotation", np.rad2deg(azimuths))
    np.save(cam_dir / "elevation", np.rad2deg(elevations))


if __name__ == "__main__":
    main()
