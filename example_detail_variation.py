import os
import sys
# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'  # Can be 'native' or 'auto', default is 'auto'.
# 'auto' is faster but will do benchmarking at the beginning.
# Recommended to set to 'native' if run only once.

import imageio
import numpy as np
import open3d as o3d
import torch
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils
"""
Image-version detail variation (Sec3.4 of the paper)
1. Voxelize a GIVEN mesh into the form of Sparse Structure.
2. Run ONLY the second stage with the image prompt.
"""


def voxelize(mesh_path: str, resolution: int = 64):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    # clamp vertices to the range [-0.5, 0.5]
    vertices = np.clip(np.asarray(mesh.vertices), -0.5 + 1e-6, 0.5 - 1e-6)
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(mesh,
                                                                                voxel_size=1 / resolution,
                                                                                min_bound=(-0.5, -0.5, -0.5),
                                                                                max_bound=(0.5, 0.5, 0.5))
    vertices = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
    binary_voxel = np.zeros((resolution, resolution, resolution), dtype=bool)
    binary_voxel[vertices[:, 0], vertices[:, 1], vertices[:, 2]] = True
    return binary_voxel


# Load a pipeline from a model folder or a Hugging Face model hub.
pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
pipeline.cuda()

# Test image path and saving directory
saveroot = "results/texgen"

example_mesh_image_pairs = [
    ("assets/example_mesh/typical_creature_dragon.obj", "assets/example_image/typical_creature_elephant.png"),
    ("assets/example_mesh/typical_creature_dragon.obj", "assets/example_image/typical_creature_furry.png"),
    ("assets/example_mesh/typical_creature_dragon.obj", "assets/example_image/typical_creature_robot_dinosour.png"),
    ("assets/example_mesh/typical_creature_dragon.obj", "assets/example_image/typical_creature_robot_crab.png"),
    ("assets/example_mesh/typical_humanoid_block_robot.obj", "assets/example_image/typical_building_mushroom.png"),
    ("assets/example_mesh/typical_humanoid_block_robot.obj", "assets/example_image/typical_humanoid_mech.png")
]

for mesh_image_pair in example_mesh_image_pairs:
    mesh_path, image_path = mesh_image_pair
    instance_name = f"{os.path.splitext(os.path.basename(mesh_path))[0]}+{os.path.splitext(os.path.basename(image_path))[0]}"
    savedir = os.path.join(saveroot, instance_name)
    os.makedirs(savedir, exist_ok=True)

    # Load the image
    image = Image.open(image_path)

    binary_voxel = voxelize(mesh_path)

    # Run the pipeline
    outputs = pipeline.run_detail_variation(
        binary_voxel,
        image,
        seed=1,
        # more steps, larger cfg
        slat_sampler_params={
            "steps": 35,
            "cfg_strength": 6.0,
        },
    )

    torch.cuda.empty_cache()
    # Render the outputs
    video = render_utils.render_video(outputs['gaussian'][0])['color']
    imageio.mimsave(os.path.join(savedir, f"{instance_name}_gs.mp4"), video, fps=30)
    video = render_utils.render_video(outputs['radiance_field'][0])['color']
    imageio.mimsave(os.path.join(savedir, f"{instance_name}_rf.mp4"), video, fps=30)
    video = render_utils.render_video(outputs['mesh'][0])['normal']
    imageio.mimsave(os.path.join(savedir, f"{instance_name}_mesh.mp4"), video, fps=30)

    # GLB files can be extracted from the outputs
    glb = postprocessing_utils.to_trimesh(
        outputs['gaussian'][0],
        outputs['mesh'][0],
        # Optional parameters
        simplify=0.95,  # Ratio of triangles to remove in the simplification process
        texture_size=1024,  # Size of the texture used for the GLB
        debug=False,
        verbose=True)
