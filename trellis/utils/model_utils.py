import os, os.path as osp
import glob
import json
import numpy as np
import torch
import utils3d
from PIL import Image
from pathlib import Path


def project_and_normalize(ref_grid, src_proj, length):
    """

    @param ref_grid: b 3 n
    @param src_proj: b 4 4
    @param length:   int
    @return:  b, n, 2
    """
    src_grid = src_proj[:, :3, :3] @ ref_grid + src_proj[:, :3, 3:]  # b 3 n
    div_val = src_grid[:, -1:]
    div_val[div_val < 1e-4] = 1e-4
    src_grid = src_grid[:, :2] / div_val  # divide by depth (b, 2, n)
    src_grid[:, 0] = src_grid[:, 0] / ((length - 1) / 2) - 1  # scale to -1~1
    src_grid[:, 1] = src_grid[:, 1] / ((length - 1) / 2) - 1  # scale to -1~1
    src_grid = src_grid.permute(0, 2, 1)  # (b, n, 2)
    return src_grid


def construct_project_matrix(x_ratio, y_ratio, Ks, poses):
    """
    @param x_ratio: float
    @param y_ratio: float
    @param Ks:      b,3,3
    @param poses:   b,3,4
    @return:
    """
    rfn = Ks.shape[0]
    scale_m = torch.tensor([x_ratio, y_ratio, 1.0], dtype=torch.float32, device=Ks.device)
    scale_m = torch.diag(scale_m)
    ref_prj = scale_m[None, :, :] @ Ks @ poses  # rfn,3,4
    pad_vals = torch.zeros([rfn, 1, 4], dtype=torch.float32, device=ref_prj.device)
    pad_vals[:, :, 3] = 1.0
    ref_prj = torch.cat([ref_prj, pad_vals], 1)  # rfn,4,4
    return ref_prj


def get_warp_coordinates(volume_xyz, warp_size, input_size, Ks, warp_pose):
    B, _, D, H, W = volume_xyz.shape
    ratio = warp_size / input_size
    warp_proj = construct_project_matrix(ratio, ratio, Ks, warp_pose)  # B,4,4
    warp_coords = project_and_normalize(volume_xyz.view(B, 3, D * H * W), warp_proj, warp_size).view(B, D, H, W, 2)
    return warp_coords


def intrinsics_to_projection(
    intrinsics: np.ndarray,
    near: float,
    far: float,
) -> np.ndarray:
    """
    OpenCV intrinsics to OpenGL perspective matrix

    Args:
        intrinsics (np.ndarray): [3, 3] OpenCV intrinsics matrix
        near (float): near plane to clip
        far (float): far plane to clip
    Returns:
        (np.ndarray): [4, 4] OpenGL perspective matrix
    """
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    ret = np.zeros((4, 4), dtype=intrinsics.dtype, device=intrinsics.device)
    ret[0, 0] = 2 * fx
    ret[1, 1] = 2 * fy
    ret[0, 2] = 2 * cx - 1
    ret[1, 2] = -2 * cy + 1
    ret[2, 2] = far / (far - near)
    ret[2, 3] = near * far / (near - far)
    ret[3, 2] = 1.
    return ret


def ensure_list_type(data):
    return [x.tolist() if isinstance(x, np.ndarray) else x for x in data]


def make_rel_path(path, par):
    rel_path = Path(path).relative_to(Path(par))
    # remove extension
    return osp.join(str(rel_path.parent), osp.splitext(rel_path.name)[0])


def load_trellis_data(image_dir: str, camera_info_path: str):
    image_paths = glob.glob(os.path.join(image_dir, "*.png"))
    camera_info = json.load(open(camera_info_path, "r"))
    assert "extrinsics" in camera_info and "intrinsics" in camera_info, "Can not find extrinsics or intrinsics"
    extrinsics, intrinsics = camera_info['extrinsics'], camera_info['intrinsics']
    assert len(image_paths) == len(extrinsics) and len(image_paths) == len(
        intrinsics), "# of Reference image should be the same with # of extrinsics."
    # sort and read the images
    image_paths = sorted(image_paths, key=lambda x: int(osp.splitext(osp.basename(x))[0].split('_')[-1]))
    return image_paths, extrinsics, intrinsics


def fovx_to_fovy(fovx, aspect):
    return np.arctan(np.tan(fovx / 2) / aspect) * 2.0


def focal_length_to_fovy(focal_length, sensor_height):
    return 2 * np.arctan(0.5 * sensor_height / focal_length)


# Reworked so this matches gluPerspective / glm::perspective, using fovy
def perspective(fovy=0.7854, aspect=1.0, n=0.1, f=1000.0, device=None):
    y = np.tan(fovy / 2)
    return torch.tensor([[1 / (y * aspect), 0, 0, 0], [0, 1 / -y, 0, 0],
                         [0, 0, -(f + n) / (f - n), -(2 * f * n) / (f - n)], [0, 0, -1, 0]],
                        dtype=torch.float32,
                        device=device)


def rotate_x(a, device=None):
    s, c = np.sin(a), np.cos(a)
    return torch.tensor([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]], dtype=torch.float32, device=device)


def parse_trellis_frame(indir, frame: dict):
    image = Image.open(osp.join(indir, frame['file_path']))
    image = image.resize((518, 518), Image.Resampling.LANCZOS)
    image = np.array(image).astype(np.float32) / 255
    image = image[:, :, :3] * image[:, :, 3:]
    image = torch.from_numpy(image).permute(2, 0, 1).float()

    c2w = torch.tensor(frame['transform_matrix'])
    c2w[:3, 1:3] *= -1
    extrinsics = torch.inverse(c2w)
    fov = frame['camera_angle_x']
    intrinsics = utils3d.torch.intrinsics_from_fov_xy(torch.tensor(fov), torch.tensor(fov))

    return {'image': image, 'extrinsics': extrinsics, 'intrinsics': intrinsics}
