import os

# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ["SPCONV_ALGO"] = "native"  # Can be 'native' or 'auto', default is 'auto'.
# 'auto' is faster but will do benchmarking at the beginning.
# Recommended to set to 'native' if run only once.

import imageio
import numpy as np
import torch
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

use_gpu = True
# Low VRAM mode, NOTICE that if the model is intended to be run on CPU, set to False
low_vram = False
# Load a pipeline from a model folder or a Hugging Face model hub.
pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
pipeline.low_vram = low_vram
if not low_vram and use_gpu:
    # Move to GPU immediately
    pipeline.cuda()

# Load an image
# image = Image.open("assets/example_image/T.png")
image = Image.open("assets/example_image/typical_humanoid_mech.png")
image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255
image = image[None].repeat(2, 1, 1, 1)[:, :3, :518, :518]

# Run the pipeline
# The pipeline can takes BOTH a PIL.Image | B x 3 x 518 x 518 image tensor as input
outputs = pipeline.run(
    image,
    seed=1,
    num_samples=2,
    preprocess_image=not isinstance(image, torch.Tensor),
    # Optional parameters
    # sparse_structure_sampler_params={
    #     "steps": 12,
    #     "cfg_strength": 7.5,
    # },
    # slat_sampler_params={
    #     "steps": 12,
    #     "cfg_strength": 3,
    # },
)
# outputs is a dictionary containing generated 3D assets in different formats:
# - outputs['gaussian']: a list of 3D Gaussians
# - outputs['radiance_field']: a list of radiance fields
# - outputs['mesh']: a list of meshes

if low_vram:
    del pipeline
    torch.cuda.empty_cache()

# Render the outputs
video = render_utils.render_video(outputs["gaussian"][0])["color"]
imageio.mimsave("sample_gs.mp4", video, fps=30)
video = render_utils.render_video(outputs["radiance_field"][0])["color"]
imageio.mimsave("sample_rf.mp4", video, fps=30)
video = render_utils.render_video(outputs["mesh"][0])["normal"]
imageio.mimsave("sample_mesh.mp4", video, fps=30)

# GLB files can be extracted from the outputs
glb = postprocessing_utils.to_trimesh(
    outputs["gaussian"][0],
    outputs["mesh"][0],
    # Optional parameters
    texture_bake_mode=(
        "fast" if low_vram else "opt"
    ),  # Low VRAM mode defaults baking to fast mode
    # Postprocessing methods
    postprocess_mode="simplify",
    remesh_iters=10,
    subdivision_times=1,
    simplify=0.95,  # Ratio of triangles to remove in the simplification process
    texture_size=512 if low_vram else 1024,  # Size of the texture used for the GLB
    debug=False,
    verbose=True,
)
glb.export("sample.glb")

# Save Gaussians as PLY files
outputs["gaussian"][0].save_ply("sample.ply")
