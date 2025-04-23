# Prediction interface for Cog ⚙️
# https://cog.run/python

import os

os.environ["ATTN_BACKEND"] = "xformers"
import logging
import numpy as np
import imageio
import open3d as o3d
from typing import Literal, Optional
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils
from cog import BasePredictor, Input, Path, BaseModel

MAX_SEED = np.iinfo(np.int32).max

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

class PredictOutput(BaseModel):
    no_background_images: list[Path] | None = None
    color_video: Path | None = None
    normal_video: Path | None = None
    combined_video: Path | None = None
    model_file: Path | None = None
    gaussian_ply: Path | None = None


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.logger.info("Setting up environment variables...")
        os.environ["SPCONV_ALGO"] = "native"
        os.environ["ATTN_BACKEND"] = "xformers"

        self.logger.info("Loading TRELLIS pipeline...")
        self.pipeline = TrellisImageTo3DPipeline.from_pretrained(
            "JeffreyXiang/TRELLIS-image-large"
        )
        self.pipeline.cuda()

        self.logger.info("Preloading rembg...")
        try:
            self.pipeline.preprocess_image(
                Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8))
            )
        except Exception as e:
            self.logger.warning(
                f"Rembg preload warning (this is usually fine): {str(e)}"
            )

        self.logger.info("Setup complete!")

    def predict(
        self,
        images: list[Path] = Input(
            description="List of input images to generate 3D asset from"
        ),
        tex_images: list[Path] = Input(
            description="List of images to generate the texture of 3D asset from",
            default=None,
        ),
        mesh: Optional[Path] = Input(
            description="Input mesh (Z-UP, ONLY required for the detail variation mode)",
            default=None,
        ),
        seed: int = Input(description="Random seed for generation", default=0),
        randomize_seed: bool = Input(description="Randomize seed", default=True),
        generate_color: bool = Input(
            description="Generate color video render", default=True
        ),
        generate_normal: bool = Input(
            description="Generate normal video render", default=False
        ),
        generate_model: bool = Input(
            description="Generate 3D model file (GLB)", default=False
        ),
        save_gaussian_ply: bool = Input(
            description="Save Gaussian point cloud as PLY file", default=False
        ),
        return_no_background: bool = Input(
            description="Return the preprocessed images without background",
            default=False,
        ),
        ss_guidance_strength: float = Input(
            description="Stage 1: Sparse Structure Generation - Guidance Strength",
            default=7.5,
            ge=0.0,
            le=10.0,
        ),
        ss_sampling_steps: int = Input(
            description="Stage 1: Sparse Structure Generation - Sampling Steps",
            default=12,
            ge=1,
            le=50,
        ),
        slat_guidance_strength: float = Input(
            description="Stage 2: Structured Latent Generation - Guidance Strength",
            default=3.0,
            ge=0.0,
            le=10.0,
        ),
        slat_sampling_steps: int = Input(
            description="Stage 2: Structured Latent Generation - Sampling Steps",
            default=12,
            ge=1,
            le=50,
        ),
        mesh_simplify: float = Input(
            description="GLB Extraction - Mesh Simplification (only used if generate_model=True)",
            default=0.95,
            ge=0.9,
            le=0.98,
        ),
        texture_size: int = Input(
            description="GLB Extraction - Texture Size (only used if generate_model=True)",
            default=1024,
            ge=512,
            le=2048,
        ),
        tex_bake_mode: Literal["rgb", "srgb"] = Input(
            description="Mode for baking the texture (defaults to rgb, which will be darker)",
            default="rgb",
        ),
    ) -> PredictOutput:
        """Run a single prediction on the model"""

        # Load and process images
        self.logger.info("Loading and preprocessing input images...")
        input_images = [Image.open(str(image)) for image in images]
        processed_images = [self.pipeline.preprocess_image(img) for img in input_images]
        tex_processed_images = None
        if tex_images is not None:
            tex_images = [Image.open(str(image)) for image in tex_images]
            tex_processed_images = [
                self.pipeline.preprocess_image(img) for img in tex_images
            ]

        # Save the processed images (without background)
        no_bg_paths = []
        if return_no_background:
            for idx, processed_image in enumerate(processed_images):
                no_bg_path = Path(f"output_no_background_{idx}.png")
                processed_image.save(str(no_bg_path))
                no_bg_paths.append(no_bg_path)
            self.logger.info("Saved images without background")

        # Randomize seed if requested
        if randomize_seed:
            seed = np.random.randint(0, MAX_SEED)
            self.logger.info(f"Using randomized seed: {seed}")
        else:
            self.logger.info(f"Using provided seed: {seed}")

        # Generate 3D asset
        self.logger.info("Running TRELLIS pipeline...")
        if mesh is None:
            if len(processed_images) > 1:
                outputs = self.pipeline.run_multi_image(
                    processed_images,
                    tex_images=tex_processed_images,
                    seed=seed,
                    formats=["gaussian", "mesh"],
                    preprocess_image=False,
                    sparse_structure_sampler_params={
                        "steps": ss_sampling_steps,
                        "cfg_strength": ss_guidance_strength,
                    },
                    slat_sampler_params={
                        "steps": slat_sampling_steps,
                        "cfg_strength": slat_guidance_strength,
                    },
                )
            else:
                outputs = self.pipeline.run(
                    processed_images[0],
                    tex_images=(
                        tex_processed_images[0]
                        if tex_processed_images is not None
                        else None
                    ),
                    seed=seed,
                    formats=["gaussian", "mesh"],
                    preprocess_image=False,
                    sparse_structure_sampler_params={
                        "steps": ss_sampling_steps,
                        "cfg_strength": ss_guidance_strength,
                    },
                    slat_sampler_params={
                        "steps": slat_sampling_steps,
                        "cfg_strength": slat_guidance_strength,
                    },
                )
        else:
            binary_voxel = voxelize(mesh)
            # Tex images have higher priority if both are specified
            dv_images = (
                tex_processed_images
                if tex_processed_images is not None
                else processed_images
            )
            # addressing the detail variation mode
            if len(dv_images) > 1:
                outputs = self.pipeline.run_detail_variation_multi_image(
                    binary_voxel,
                    dv_images,
                    seed=seed,
                    formats=["gaussian", "mesh"],
                    preprocess_image=False,
                    slat_sampler_params={
                        "steps": slat_sampling_steps,
                        "cfg_strength": slat_guidance_strength,
                    },
                )
            else:
                outputs = self.pipeline.run_detail_variation(
                    binary_voxel,
                    dv_images[0],
                    seed=seed,
                    formats=["gaussian", "mesh"],
                    preprocess_image=False,
                    slat_sampler_params={
                        "steps": slat_sampling_steps,
                        "cfg_strength": slat_guidance_strength,
                    },
                )
        self.logger.info("TRELLIS pipeline complete!")
        self.logger.info(f"Available output formats: {outputs.keys()}")

        # Initialize output paths as None
        color_path = None
        normal_path = None
        combined_path = None
        model_path = None
        gaussian_path = None

        # Render videos if requested
        if generate_color or generate_normal:
            self.logger.info("Starting video rendering...")

            if generate_color and generate_normal:
                # Generate both videos and combine them side by side
                self.logger.info("Generating color video from gaussian output...")
                color_renders = render_utils.render_video(
                    outputs["gaussian"][0], num_frames=120
                )
                self.logger.info(
                    f"Available gaussian render types: {list(color_renders.keys())}"
                )

                self.logger.info("Generating normal video from mesh output...")
                normal_renders = render_utils.render_video(
                    outputs["mesh"][0], num_frames=120
                )
                self.logger.info(
                    f"Available mesh render types: {list(normal_renders.keys())}"
                )

                if "color" in color_renders and "normal" in normal_renders:
                    self.logger.info(
                        "Combining color and normal videos side by side..."
                    )
                    color_video = color_renders["color"]
                    normal_video = normal_renders["normal"]
                    combined_video = [
                        np.concatenate([color_video[i], normal_video[i]], axis=1)
                        for i in range(len(color_video))
                    ]

                    # Save combined video only
                    combined_path = Path("output_combined.mp4")
                    imageio.mimsave(str(combined_path), combined_video, fps=15)
                    self.logger.info("Generated combined video successfully")
                else:
                    self.logger.warning("Missing required render types!")
                    if "color" not in color_renders:
                        self.logger.warning(
                            "No color render type found in gaussian output!"
                        )
                    if "normal" not in normal_renders:
                        self.logger.warning(
                            "No normal render type found in mesh output!"
                        )
            else:
                if generate_color:
                    self.logger.info("Generating color video from gaussian output...")
                    color_renders = render_utils.render_video(
                        outputs["gaussian"][0], num_frames=120
                    )
                    self.logger.info(
                        f"Available gaussian render types: {list(color_renders.keys())}"
                    )
                    if "color" in color_renders:
                        color_path = Path("output_color.mp4")
                        imageio.mimsave(str(color_path), color_renders["color"], fps=15)
                        self.logger.info("Generated color video successfully")
                    else:
                        self.logger.warning(
                            "No color render type found in gaussian output!"
                        )

                if generate_normal:
                    self.logger.info("Generating normal video from mesh output...")
                    normal_renders = render_utils.render_video(
                        outputs["mesh"][0], num_frames=120
                    )
                    self.logger.info(
                        f"Available mesh render types: {list(normal_renders.keys())}"
                    )
                    if "normal" in normal_renders:
                        normal_path = Path("output_normal.mp4")
                        imageio.mimsave(
                            str(normal_path), normal_renders["normal"], fps=15
                        )
                        self.logger.info("Generated normal video successfully")
                    else:
                        self.logger.warning(
                            "No normal render type found in mesh output!"
                        )

            self.logger.info("Video rendering complete!")

        # Generate GLB only if requested
        if generate_model:
            self.logger.info("Generating GLB model...")
            glb = postprocessing_utils.to_trimesh(
                outputs["gaussian"][0],
                outputs["mesh"][0],
                simplify=mesh_simplify,
                texture_size=texture_size,
                get_srgb_texture=tex_bake_mode == "srgb",
                verbose=False,
                debug=False,
            )
            model_path = Path("output.glb")
            glb.export(str(model_path))
            self.logger.info("GLB model generation complete!")

        # Save Gaussian PLY if requested
        if save_gaussian_ply:
            self.logger.info("Saving Gaussian point cloud as PLY...")
            gaussian_path = Path("output_gaussian.ply")
            outputs["gaussian"][0].save_ply(str(gaussian_path))
            self.logger.info("Gaussian PLY file saved successfully!")

        self.logger.info("Prediction complete! Returning results...")
        return PredictOutput(
            no_background_images=no_bg_paths if return_no_background else None,
            color_video=(
                color_path if (generate_color and not generate_normal) else None
            ),
            normal_video=(
                normal_path if (generate_normal and not generate_color) else None
            ),
            combined_video=(
                combined_path if (generate_color and generate_normal) else None
            ),
            model_file=model_path,
            gaussian_ply=gaussian_path if save_gaussian_ply else None,
        )
