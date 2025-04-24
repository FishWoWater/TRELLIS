import os
from typing import *
from contextlib import contextmanager
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from typing import List, Optional
from easydict import EasyDict as edict
from torchvision import transforms
from PIL import Image
import rembg
from .base import Pipeline
from . import samplers
from ..modules import sparse as sp


class TrellisImageTo3DPipeline(Pipeline):
    """
    Pipeline for inferring Trellis image-to-3D models.

    Args:
        models (dict[str, nn.Module]): The models to use in the pipeline.
        sparse_structure_sampler (samplers.Sampler): The sampler for the sparse structure.
        slat_sampler (samplers.Sampler): The sampler for the structured latent.
        slat_normalization (dict): The normalization parameters for the structured latent.
        image_cond_model (str): The name of the image conditioning model.
        low_vram (bool): Run the pipeline in the low memory mode.
    """

    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        slat_sampler: samplers.Sampler = None,
        slat_normalization: dict = None,
        image_cond_model: str = None,
        low_vram: bool = False,
    ):
        if models is None:
            return
        super().__init__(models, low_vram=low_vram)
        self.sparse_structure_sampler = sparse_structure_sampler
        self.slat_sampler = slat_sampler
        self.sparse_structure_sampler_params = {}
        self.slat_sampler_params = {}
        self.slat_normalization = slat_normalization
        self.rembg_session = None
        self.low_vram = low_vram
        self._init_image_cond_model(image_cond_model)

    @staticmethod
    def from_pretrained(path: str) -> "TrellisImageTo3DPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        pipeline = super(
            TrellisImageTo3DPipeline, TrellisImageTo3DPipeline
        ).from_pretrained(path)
        new_pipeline = TrellisImageTo3DPipeline()
        new_pipeline.__dict__ = pipeline.__dict__
        args = pipeline._pretrained_args

        new_pipeline.sparse_structure_sampler = getattr(
            samplers, args["sparse_structure_sampler"]["name"]
        )(**args["sparse_structure_sampler"]["args"])
        new_pipeline.sparse_structure_sampler_params = args["sparse_structure_sampler"][
            "params"
        ]

        new_pipeline.slat_sampler = getattr(samplers, args["slat_sampler"]["name"])(
            **args["slat_sampler"]["args"]
        )
        new_pipeline.slat_sampler_params = args["slat_sampler"]["params"]

        new_pipeline.slat_normalization = args["slat_normalization"]

        new_pipeline._init_image_cond_model(args["image_cond_model"])

        return new_pipeline

    def _init_image_cond_model(self, name: str):
        """
        Initialize the image conditioning model.
        """
        # dinov2_model = torch.hub.load("facebookresearch/dinov2", name, pretrained=True)
        # dinov2_model = torch.hub.load("/usr/local/app/.cache/torch/hub/facebookresearch_dinov2_main", "dinov2_vitl14_reg", source='local', pretrained=True)
        dinov2_model = torch.hub.load(
            "./local_torch_hub_cache/facebookresearch_dinov2_main",
            "dinov2_vitl14_reg",
            source="local",
            pretrained=True,
        )
        dinov2_model.eval()
        self.models["image_cond_model"] = dinov2_model
        transform = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.image_cond_model_transform = transform

    def preprocess_voxel(
        self, binary_voxel: np.ndarray, voxel_res: int = 64
    ) -> torch.Tensor:
        """
        Preprocess(read / voxelize) the given 3D object.
        """
        assert all(
            [s == voxel_res for s in binary_voxel.shape]
        ), "Input voxels have incompatible resolution {}".format(binary_voxel.shape)
        # Active voxels (N_p x 3)
        x, y, z = np.nonzero(binary_voxel)
        values_sum = x * voxel_res * voxel_res + y * voxel_res + z
        active_voxels = np.stack([x, y, z], axis=1)[np.argsort(values_sum)]
        active_voxels = np.concatenate(
            (np.zeros((len(active_voxels), 1), dtype=np.int32), active_voxels), axis=1
        )
        # Pad with the batch dimension
        return torch.from_numpy(active_voxels).int().cuda()

    def preprocess_image(
        self,
        input: Image.Image,
        rembg_model: Literal["u2net", "birefnet-general"] = "birefnet-general",
    ) -> Image.Image:
        """
        Preprocess the input image.
        """
        # if has alpha channel, use it directly; otherwise, remove background
        has_alpha = False
        if input.mode == "RGBA":
            alpha = np.array(input)[:, :, 3]
            if not np.all(alpha == 255):
                has_alpha = True
        if has_alpha:
            output = input
        else:
            input = input.convert("RGB")
            max_size = max(input.size)
            scale = min(1, 1024 / max_size)
            if scale < 1:
                input = input.resize(
                    (int(input.width * scale), int(input.height * scale)),
                    Image.Resampling.LANCZOS,
                )
            if getattr(self, "rembg_session", None) is None:
                self.rembg_session = rembg.new_session(rembg_model)
            output = rembg.remove(input, session=self.rembg_session)
        output_np = np.array(output)
        alpha = output_np[:, :, 3]
        bbox = np.argwhere(alpha > 0.8 * 255)
        bbox = (
            np.min(bbox[:, 1]),
            np.min(bbox[:, 0]),
            np.max(bbox[:, 1]),
            np.max(bbox[:, 0]),
        )
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        size = int(size * 1.2)
        bbox = (
            center[0] - size // 2,
            center[1] - size // 2,
            center[0] + size // 2,
            center[1] + size // 2,
        )
        output = output.crop(bbox)  # type: ignore
        output = output.resize((518, 518), Image.Resampling.LANCZOS)
        output = np.array(output).astype(np.float32) / 255
        output = output[:, :, :3] * output[:, :, 3:4]
        output = Image.fromarray((output * 255).astype(np.uint8))
        return output

    @torch.no_grad()
    def encode_image(
        self, image: Union[torch.Tensor, list[Image.Image]]
    ) -> torch.Tensor:
        """
        Encode the image.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image to encode

        Returns:
            torch.Tensor: The encoded features.
        """
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        elif isinstance(image, list):
            if all(isinstance(i, torch.Tensor) for i in image):
                image = image[0]
            else:
                assert all(
                    isinstance(i, Image.Image) for i in image
                ), "Image list should be list of PIL images"
                image = [i.resize((518, 518), Image.LANCZOS) for i in image]
                image = [
                    np.array(i.convert("RGB")).astype(np.float32) / 255 for i in image
                ]
                image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
                image = torch.stack(image).to(self.device)
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")

        image = self.image_cond_model_transform(image).to(self.device)
        if self.low_vram:
            features = self.load_model("image_cond_model")(image, is_training=True)[
                "x_prenorm"
            ]
        else:
            features = self.models["image_cond_model"](image, is_training=True)[
                "x_prenorm"
            ]
        patchtokens = F.layer_norm(features, features.shape[-1:])

        if self.low_vram:
            self.unload_models(["image_cond_model"])
        return patchtokens

    def get_cond(self, image: Union[torch.Tensor, list[Image.Image]]) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image prompts.

        Returns:
            dict: The conditioning information
        """
        cond = self.encode_image(image)
        neg_cond = torch.zeros_like(cond)
        return {
            "cond": cond,
            "neg_cond": neg_cond,
        }

    def sample_sparse_structure(
        self,
        cond: dict,
        num_samples: int = 1,
        sampler_params: dict = {},
        verbose: bool = True,
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.

        Args:
            cond (dict): The conditioning information.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample occupancy latent
        flow_model = (
            self.models["sparse_structure_flow_model"]
            if not self.low_vram
            else self.load_model("sparse_structure_flow_model")
        )
        reso = flow_model.resolution
        noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(
            self.device
        )
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        z_s = self.sparse_structure_sampler.sample(
            flow_model, noise, **cond, **sampler_params, verbose=verbose
        ).samples
        if self.low_vram:
            self.unload_models(["sparse_structure_flow_model"])

        # Decode occupancy latent
        decoder = (
            self.models["sparse_structure_decoder"]
            if not self.low_vram
            else self.load_model("sparse_structure_decoder")
        )
        coords = torch.argwhere(decoder(z_s) > 0)[:, [0, 2, 3, 4]].int()
        if self.low_vram:
            self.unload_models(["sparse_structure_decoder"])

        return coords

    @torch.no_grad()
    def decode_slat(
        self,
        slat: sp.SparseTensor,
        formats: List[str] = ["mesh", "gaussian", "radiance_field"],
    ) -> dict:
        """
        Decode the structured latent.

        Args:
            slat (sp.SparseTensor): The structured latent.
            formats (List[str]): The formats to decode the structured latent to.

        Returns:
            dict: The decoded structured latent.
        """
        ret = {}
        if "mesh" in formats:
            ret["mesh"] = (
                self.models["slat_decoder_mesh"](slat)
                if not self.low_vram
                else self.load_model("slat_decoder_mesh")(slat)
            )
            if self.low_vram:
                self.unload_models(["slat_decoder_mesh"])
        if "gaussian" in formats:
            ret["gaussian"] = (
                self.models["slat_decoder_gs"](slat)
                if not self.low_vram
                else self.load_model("slat_decoder_gs")(slat)
            )
            if self.low_vram:
                self.unload_models(["slat_decoder_gs"])
        if "radiance_field" in formats:
            ret["radiance_field"] = (
                self.models["slat_decoder_rf"](slat)
                if not self.low_vram
                else self.load_model("slat_decoder_rf")(slat)
            )
            if self.low_vram:
                self.unload_models(["slat_decoder_rf"])
        return ret

    def sample_slat(
        self,
        cond: dict,
        coords: torch.Tensor,
        sampler_params: dict = {},
        verbose: bool = True,
    ) -> sp.SparseTensor:
        """
        Sample structured latent with the given conditioning.

        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample structured latent
        flow_model = (
            self.models["slat_flow_model"]
            if not self.low_vram
            else self.load_model("slat_flow_model")
        )
        noise = sp.SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.slat_sampler_params, **sampler_params}
        slat = self.slat_sampler.sample(
            flow_model, noise, **cond, **sampler_params, verbose=verbose
        ).samples

        std = torch.tensor(self.slat_normalization["std"])[None].to(slat.device)
        mean = torch.tensor(self.slat_normalization["mean"])[None].to(slat.device)
        slat = slat * std + mean
        if self.low_vram:
            self.unload_models(["slat_flow_model"])

        return slat

    @torch.no_grad()
    def run_detail_variation(
        self,
        binary_voxel: np.ndarray,
        image: Image.Image,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ["mesh", "gaussian", "radiance_field"],
        preprocess_image: bool = True,
        **kwargs,
    ) -> dict:
        """
        Run the texture generation(2nd stage) pipeline.

        Args:
            binary_voxel (np.ndarray): The input binary voxel.
            image (Image.Image or a list of Image.Image): The image prompt(s).
            num_samples (int): The number of samples to generate.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            preprocess_image (bool): Whether to preprocess the image.
        """
        if self.low_vram:
            self.verify_model_low_vram_devices()

        if preprocess_image:
            image = self.preprocess_image(image)
        cond = self.get_cond([image])

        torch.manual_seed(seed)
        coords = self.preprocess_voxel(binary_voxel)
        slat = self.sample_slat(cond, coords, slat_sampler_params)
        return self.decode_slat(slat, formats)

    @torch.no_grad()
    def run_detail_variation_multi_image(
        self,
        binary_voxel: np.ndarray,
        images: List[Image.Image],
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ["mesh", "gaussian", "radiance_field"],
        preprocess_image: bool = True,
        mode: Literal["stochastic", "multidiffusion"] = "stochastic",
        **kwargs,
    ) -> dict:
        """
        Run the texture generation(2nd stage) pipeline.

        Args:
            binary_voxel (np.ndarray): The input binary voxel.
            image (Image.Image or a list of Image.Image): The image prompt(s).
            num_samples (int): The number of samples to generate.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            preprocess_image (bool): Whether to preprocess the image.
        """
        if self.low_vram:
            self.verify_model_low_vram_devices()

        if preprocess_image:
            images = [self.preprocess_image(image) for image in images]
        cond = self.get_cond(images)
        cond["neg_cond"] = cond["neg_cond"][:1]
        torch.manual_seed(seed)
        coords = self.preprocess_voxel(binary_voxel)
        slat_steps = {**self.slat_sampler_params, **slat_sampler_params}.get("steps")
        with self.inject_sampler_multi_image(
            "slat_sampler", len(images), slat_steps, mode=mode
        ):
            slat = self.sample_slat(cond, coords, slat_sampler_params)
        return self.decode_slat(slat, formats)

    @torch.no_grad()
    def run(
        self,
        image: Image.Image,
        tex_image: Optional[Image.Image] = None,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ["mesh", "gaussian", "radiance_field"],
        preprocess_image: bool = True,
        verbose: bool = True,
        **kwargs,
    ) -> dict:
        """
        Run the pipeline.

        Args:
            image (Image.Image): The image prompt.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            formats (List[str]): The formats to decode the structured latent to.
            preprocess_image (bool): Whether to preprocess the image.
        """
        if self.low_vram:
            self.verify_model_low_vram_devices()

        if preprocess_image:
            image = self.preprocess_image(image)
            tex_image = (
                self.preprocess_image(tex_image) if tex_image is not None else None
            )
        cond = self.get_cond([image])
        tex_cond = self.get_cond([tex_image]) if tex_image is not None else None
        # When the image is already a tensor, NO NEED to duplicate it
        if not isinstance(image, torch.Tensor) and num_samples > 1:
            cond["cond"] = cond["cond"].repeat(num_samples, 1, 1)
            cond["neg_cond"] = cond["neg_cond"].repeat(num_samples, 1, 1)

        torch.manual_seed(seed)
        coords = self.sample_sparse_structure(
            cond, num_samples, sparse_structure_sampler_params, verbose=verbose
        )
        slat = self.sample_slat(
            tex_cond or cond, coords, slat_sampler_params, verbose=verbose
        )
        return self.decode_slat(slat, formats)

    @contextmanager
    def inject_sampler_multi_image(
        self,
        sampler_name: str,
        num_images: int,
        num_steps: int,
        mode: Literal["stochastic", "multidiffusion"] = "stochastic",
    ):
        """
        Inject a sampler with multiple images as condition.

        Args:
            sampler_name (str): The name of the sampler to inject.
            num_images (int): The number of images to condition on.
            num_steps (int): The number of steps to run the sampler for.
        """
        sampler = getattr(self, sampler_name)
        setattr(sampler, f"_old_inference_model", sampler._inference_model)

        if mode == "stochastic":
            if num_images > num_steps:
                print(
                    f"\033[93mWarning: number of conditioning images is greater than number of steps for {sampler_name}. "
                    "This may lead to performance degradation.\033[0m"
                )

            cond_indices = (np.arange(num_steps) % num_images).tolist()

            def _new_inference_model(self, model, x_t, t, cond, **kwargs):
                cond_idx = cond_indices.pop(0)
                cond_i = cond[cond_idx : cond_idx + 1]
                return self._old_inference_model(model, x_t, t, cond=cond_i, **kwargs)

        elif mode == "multidiffusion":
            from .samplers import FlowEulerSampler

            def _new_inference_model(
                self,
                model,
                x_t,
                t,
                cond,
                neg_cond,
                cfg_strength,
                cfg_interval,
                **kwargs,
            ):
                if cfg_interval[0] <= t <= cfg_interval[1]:
                    preds = []
                    for i in range(len(cond)):
                        preds.append(
                            FlowEulerSampler._inference_model(
                                self, model, x_t, t, cond[i : i + 1], **kwargs
                            )
                        )
                    pred = sum(preds) / len(preds)
                    neg_pred = FlowEulerSampler._inference_model(
                        self, model, x_t, t, neg_cond, **kwargs
                    )
                    return (1 + cfg_strength) * pred - cfg_strength * neg_pred
                else:
                    preds = []
                    for i in range(len(cond)):
                        preds.append(
                            FlowEulerSampler._inference_model(
                                self, model, x_t, t, cond[i : i + 1], **kwargs
                            )
                        )
                    pred = sum(preds) / len(preds)
                    return pred

        else:
            raise ValueError(f"Unsupported mode: {mode}")

        sampler._inference_model = _new_inference_model.__get__(sampler, type(sampler))

        yield

        sampler._inference_model = sampler._old_inference_model
        delattr(sampler, f"_old_inference_model")

    @torch.no_grad()
    def run_multi_image(
        self,
        images: List[Image.Image],
        tex_images: Optional[List[Image.Image]] = None,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ["mesh", "gaussian", "radiance_field"],
        preprocess_image: bool = True,
        mode: Literal["stochastic", "multidiffusion"] = "stochastic",
    ) -> dict:
        """
        Run the pipeline with multiple images as condition

        Args:
            images (List[Image.Image]): The multi-view images of the assets
            num_samples (int): The number of samples to generate.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            preprocess_image (bool): Whether to preprocess the image.
        """
        if self.low_vram:
            self.verify_model_low_vram_devices()

        if preprocess_image:
            images = [self.preprocess_image(image) for image in images]
            tex_images = (
                [self.preprocess_image(tex_image) for tex_image in tex_images]
                if tex_images is not None
                else None
            )
        cond = self.get_cond(images)
        cond["neg_cond"] = cond["neg_cond"][:1]
        tex_cond = self.get_cond(tex_images) if tex_images is not None else None
        if tex_cond is not None:
            tex_cond["neg_cond"] = tex_cond["neg_cond"][:1]

        torch.manual_seed(seed)
        ss_steps = {
            **self.sparse_structure_sampler_params,
            **sparse_structure_sampler_params,
        }.get("steps")
        with self.inject_sampler_multi_image(
            "sparse_structure_sampler", len(images), ss_steps, mode=mode
        ):
            coords = self.sample_sparse_structure(
                cond, num_samples, sparse_structure_sampler_params
            )
        slat_steps = {**self.slat_sampler_params, **slat_sampler_params}.get("steps")
        with self.inject_sampler_multi_image(
            "slat_sampler", len(tex_images or images), slat_steps, mode=mode
        ):
            slat = self.sample_slat(tex_cond or cond, coords, slat_sampler_params)
        return self.decode_slat(slat, formats)
