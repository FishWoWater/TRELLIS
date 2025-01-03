import os
import glob 
import argparse 
import json
import time 
# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.

import imageio
import numpy as np 
import torch
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

# Setup the input image paths
parser = argparse.ArgumentParser()
parser.add_argument("--infile", type=str, default='')
parser.add_argument("--indir", type=str, default='./assets/example_image')
args = parser.parse_args()

assert args.infile or args.indir, "Input file or input directory must be specified"
image_paths = [args.infile] if args.infile else glob.glob(os.path.join(args.indir, '*.png'))

# Repeatedly loading pipelines to avoid OOM
# Load a pipeline from a model folder or a Hugging Face model hub.
pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
pipeline.cuda()
    
# Test image path and saving directory
debug = True 
save_root = "results"
simplify_ratio = 0.95 
texture_size = 1024
texture_bake_mode = "opt"

for image_path in image_paths:
    tik = time.time()
    print('processing the image', image_path)
    instance_name = os.path.splitext(os.path.basename(image_path))[0]
    save_dir = os.path.join(save_root, instance_name)
    if os.path.exists(save_dir):
        continue 
    os.makedirs(save_dir, exist_ok=True)

    # Load the image 
    image = Image.open(image_path)

    # Pipeline inference does not require grads 
    # torch.set_grad_enabled(False)
    # Run the pipeline
    outputs = pipeline.run(
        image,
        seed=1,
    )

    torch.cuda.empty_cache()
    # Render the outputs
    video = render_utils.render_video(outputs['gaussian'][0])['color']
    imageio.mimsave(os.path.join(save_dir, f"{instance_name}_gs.mp4"), video, fps=30)
    video = render_utils.render_video(outputs['radiance_field'][0])['color']
    imageio.mimsave(os.path.join(save_dir, f"{instance_name}_rf.mp4"), video, fps=30)
    video = render_utils.render_video(outputs['mesh'][0])['normal']
    imageio.mimsave(os.path.join(save_dir, f"{instance_name}_mesh.mp4"), video, fps=30)

    # Texture baking requires grads
    # torch.set_grad_enabled(True)
    # Extract the mesh 
    mesh_extraction_results = postprocessing_utils.to_trimesh(
        outputs['gaussian'][0],
        outputs['mesh'][0],
        # Optional parameters
        simplify=simplify_ratio,          # Ratio of triangles to remove in the simplification process
        texture_size=texture_size,        # Size of the texture used for the GLB
        get_srgb_texture=True,
        texture_bake_mode=texture_bake_mode,
        debug=debug, 
        verbose=True
    )
    if isinstance(mesh_extraction_results, dict):
        # in the debug mode, save all outputs  
        trimesh_mesh = mesh_extraction_results['mesh']
        trimesh_mesh_yup = mesh_extraction_results['mesh_yup']
        texture = mesh_extraction_results['texture']
        texture_raw = mesh_extraction_results['texture_raw']
        obs = mesh_extraction_results['observations']
        exs, ins = mesh_extraction_results['extrinsics'], mesh_extraction_results['intrinsics']
        if texture is not None:
            texture.save(os.path.join(save_dir, f"texture.png"))
        if texture_raw is not None:
            texture_raw.save(os.path.join(save_dir, f"textuare_raw.png"))
        
        # export raw mesh 
        trimesh_rawmesh = mesh_extraction_results['mesh_raw']
        trimesh_rawmesh.export(os.path.join(save_dir, f"{instance_name}_raw.obj"))
        
        # export gaussian rendered images 
        obs_savedir = os.path.join(save_dir, "observations")
        os.makedirs(obs_savedir, exist_ok=True)
        for image_id, image in enumerate(obs):
            image = Image.fromarray(image)
            image.save(os.path.join(obs_savedir, f"image_{image_id:03d}.png"))
            
        # export camera parameters 
        json.dump({
            "extrinsics": [extrinsic.tolist() for extrinsic in exs],
            "intrinsics": [intrinsic.tolist() for intrinsic in ins]
        }, open(os.path.join(save_dir, f"{instance_name}_caminfo.json"), 'w'), indent=2)
    else:
        # in the non-debug mode, only having the mesh (z-up)
        trimesh_mesh = mesh_extraction_results
        trimesh_mesh_yup = trimesh_mesh.copy()
        trimesh_mesh_yup.vertices = trimesh_mesh_yup.vertices @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    
    # export the mesh in two formats 
    trimesh_mesh.export(os.path.join(save_dir, f"{instance_name}.glb"))
    trimesh_mesh.export(os.path.join(save_dir, f"{instance_name}.obj"))
    # also export the y-axis up version 
    trimesh_mesh_yup.export(os.path.join(save_dir, f"{instance_name}_yup.glb"))
    
    del outputs, mesh_extraction_results
    torch.cuda.empty_cache()
    print(f'image processing of {image_path} finished in {time.time() - tik}s')
