from typing import *

import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import trimesh
from PIL import Image
from transformers import AutoTokenizer, CLIPTextModel

from ..modules import sparse as sp
from . import samplers
from .base import Pipeline


class TrellisTextTo3DPipeline(Pipeline):
    """
    Pipeline for inferring Trellis text-to-3D models.

    Args:
        models (dict[str, nn.Module]): The models to use in the pipeline.
        sparse_structure_sampler (samplers.Sampler): The sampler for the sparse structure.
        slat_sampler (samplers.Sampler): The sampler for the structured latent.
        slat_normalization (dict): The normalization parameters for the structured latent.
        text_cond_model (str): The name of the text conditioning model.
    """

    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        slat_sampler: samplers.Sampler = None,
        slat_normalization: dict = None,
        text_cond_model: str = None,
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
        self._init_text_cond_model(text_cond_model)

    @staticmethod
    def from_pretrained(
        path: str, cache_dir: str = "", skip_models: List = []
    ) -> "TrellisTextTo3DPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        pipeline = super(
            TrellisTextTo3DPipeline, TrellisTextTo3DPipeline
        ).from_pretrained(path, cache_dir=cache_dir, skip_models=skip_models)
        new_pipeline = TrellisTextTo3DPipeline()
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

        new_pipeline._init_text_cond_model(args["text_cond_model"])

        return new_pipeline

    def _init_text_cond_model(self, name: str):
        """
        Initialize the text conditioning model.
        """
        # load model
        model = CLIPTextModel.from_pretrained(name)
        tokenizer = AutoTokenizer.from_pretrained(name)
        model.eval()
        if not self.low_vram:
            model = model.cuda()
        self.text_cond_model = {
            "model": model,
            "tokenizer": tokenizer,
        }
        self.models["text_cond_model"] = self.text_cond_model
        self.text_cond_model["null_cond"] = self.encode_text([""])

    @torch.no_grad()
    def encode_text(self, text: List[str]) -> torch.Tensor:
        """
        Encode the text.
        """
        assert isinstance(text, list) and all(isinstance(t, str) for t in text), (
            "text must be a list of strings"
        )
        if self.low_vram:
            self.load_model("text_cond_model")
            encoding = self.models["text_cond_model"]["tokenizer"](
                text,
                max_length=77,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            tokens = encoding["input_ids"].cuda()
            embeddings = self.models["text_cond_model"]["model"](
                input_ids=tokens
            ).last_hidden_state
            self.unload_models(["text_cond_model"])
        else:
            encoding = self.models["text_cond_model"]["tokenizer"](
                text,
                max_length=77,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            tokens = encoding["input_ids"].cuda()
            embeddings = self.models["text_cond_model"]["model"](
                input_ids=tokens
            ).last_hidden_state

        return embeddings

    def get_cond(self, prompt: List[str], negative_prompt: List[str] = []) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            prompt (List[str]): The text prompt.
            negative_prompt (List[str]): The negative text prompt.

        Returns:
            dict: The conditioning information
        """
        cond = self.encode_text(prompt)
        neg_cond = (
            self.encode_text(negative_prompt)
            if negative_prompt and negative_prompt[0]
            else self.text_cond_model["null_cond"]
        )
        return {
            "cond": cond,
            "neg_cond": neg_cond,
        }

    def sample_sparse_structure(
        self,
        cond: dict,
        num_samples: int = 1,
        sampler_params: dict = {},
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
            flow_model, noise, **cond, **sampler_params, verbose=True
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

    def sample_sparse_structure_repaint(
        self,
        cond: dict,
        ss_x0: torch.Tensor,
        ss_mask: torch.Tensor,
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
        flow_model = self.models["sparse_structure_flow_model"]

        reso = flow_model.resolution
        noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(
            self.device
        )
        sampler_params = {
            **self.sparse_structure_repaint_sampler_params,
            **sampler_params,
        }
        z_s = self.sparse_structure_repaint_sampler.sample(
            flow_model, noise, ss_mask, ss_x0, **cond, **sampler_params, verbose=verbose
        ).samples

        # Decode occupancy latent
        decoder = self.models["sparse_structure_decoder"]
        coords = torch.argwhere(decoder(z_s) > 0)[:, [0, 2, 3, 4]].int()

        return coords

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
            flow_model, noise, **cond, **sampler_params, verbose=True
        ).samples

        std = torch.tensor(self.slat_normalization["std"])[None].to(slat.device)
        mean = torch.tensor(self.slat_normalization["mean"])[None].to(slat.device)
        slat = slat * std + mean

        if self.low_vram:
            self.unload_models(["slat_flow_model"])

        return slat

    @torch.no_grad()
    def run(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ["mesh", "gaussian", "radiance_field"],
        **kwargs,
    ) -> dict:
        """
        Run the pipeline.

        Args:
            prompt (str): The text prompt.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            formats (List[str]): The formats to decode the structured latent to.
        """
        if self.low_vram:
            self.verify_model_low_vram_devices()
        cond = self.get_cond([prompt], [negative_prompt])
        torch.manual_seed(seed)
        coords = self.sample_sparse_structure(
            cond, num_samples, sparse_structure_sampler_params
        )
        slat = self.sample_slat(cond, coords, slat_sampler_params)
        return self.decode_slat(slat, formats)

    def sample_slat_repaint(
        self,
        cond: dict,
        coords: torch.Tensor,
        ref_slat: sp.SparseTensor,
        sampler_params: dict = {},
        mask_object: Optional[o3d.geometry.TriangleMesh] = None,
        post_mask_transform: Optional[dict] = None,
        verbose: bool = True,
    ) -> sp.SparseTensor:
        """
        Sample structured latent with the given conditioning.

        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        from ..utils.edit_utils import map_slat_sp

        std = torch.tensor(self.slat_normalization["std"])[None].to(coords.device)
        mean = torch.tensor(self.slat_normalization["mean"])[None].to(coords.device)
        # normalise the reference
        # ref_slat.feats = (ref_slat.feats - mean) / std
        ref_slat = (ref_slat - mean) / std

        # Sample structured latent
        flow_model = self.models["slat_flow_model"]

        noise = sp.SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        # Build slat_mask at runtime
        # Build slat_x0 with query to ref_slat & noise
        slat_x0, slat_mask = map_slat_sp(
            ref_slat,
            noise,
            mask_object,
            transform=post_mask_transform,
            verbose=True,
            debug=False,
        )
        noise = sp.SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.slat_repaint_sampler_params, **sampler_params}
        slat = self.slat_repaint_sampler.sample(
            flow_model,
            noise,
            slat_mask,
            slat_x0,
            **cond,
            **sampler_params,
            verbose=verbose,
        ).samples

        # slat_feats = slat.feats
        # slat_feats[slat_mask.bool().view(-1)] = slat_feats[slat_mask.bool().view(-1)] / 4
        # slat.replace(slat_feats)

        # return ref_slat * std + mean
        slat = slat * std + mean
        return slat

    @torch.no_grad()
    def run_local_editing(
        self,
        ss_x0: torch.Tensor,  # data point x0 of the sparse structure
        ref_slat: torch.Tensor,  # data point x0 of the structured latent
        ss_mask: torch.Tensor,  # mask of the sparse structure, notice that the mask of the slat will be built at runtime
        image: Image.Image,
        num_samples: int = 1,
        seed: int = 42,
        post_mask_transform: dict = {},
        mask_object: Optional[o3d.geometry.TriangleMesh] = None,
        min_bound: Optional[np.ndarray] = None,
        max_bound: Optional[np.ndarray] = None,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        enable_slat_repaint: bool = True,
        formats: List[str] = ["mesh", "gaussian", "radiance_field"],
        preprocess_image: bool = True,
        return_all: bool = False,
    ):
        assert mask_object is not None or (
            min_bound is not None and max_bound is not None
        ), "At least one of mask_object and (min_bound and max_bound) must be provided"

        if preprocess_image:
            image = self.preprocess_image(image)
        cond = self.get_cond([image])

        torch.manual_seed(seed)

        ss = self.sample_sparse_structure_repaint(
            cond,
            ss_x0,
            ss_mask,
            num_samples=num_samples,
            sampler_params=sparse_structure_sampler_params,
        )
        if enable_slat_repaint:
            slat = self.sample_slat_repaint(
                cond,
                ss,
                ref_slat,
                slat_sampler_params,
                mask_object=mask_object,
                post_mask_transform=post_mask_transform,
            )
        else:
            slat = self.sample_slat(cond, ss, slat_sampler_params)

        if return_all:
            return {
                "ss_coords": ss.cpu().numpy(),
                "slat_coords": slat.coords.cpu().numpy(),
                "slat_feats": slat.feats.cpu().numpy(),
                "outputs": self.decode_slat(slat, formats),
            }
        return self.decode_slat(slat, formats)

    @torch.no_grad()
    def run_variant(
        self,
        mesh: Union[o3d.geometry.TriangleMesh, trimesh.Trimesh],
        prompt: str,
        negative_prompt: str = "",
        binary_voxel: Optional[np.ndarray] = None,
        num_samples: int = 1,
        seed: int = 42,
        slat_sampler_params: dict = {},
        formats: List[str] = ["mesh", "gaussian", "radiance_field"],
        **kwargs,
    ) -> dict:
        """
        Run the pipeline for making variants of an asset.

        Args:
            mesh (o3d.geometry.TriangleMesh or trimesh.Trimesh): The base mesh.
            prompt (str): The text prompt.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            formats (List[str]): The formats to decode the structured latent to.
        """
        if self.low_vram:
            self.verify_model_low_vram_devices()

        cond = self.get_cond([prompt], [negative_prompt])
        coords = (
            self.voxelize(mesh)
            if binary_voxel is None
            else self.preprocess_voxel(binary_voxel, concat_dim=False)
        )
        coords = torch.cat(
            [
                torch.arange(num_samples)
                .repeat_interleave(coords.shape[0], 0)[:, None]
                .int()
                .cuda(),
                coords.repeat(num_samples, 1),
            ],
            1,
        )
        torch.manual_seed(seed)
        slat = self.sample_slat(cond, coords, slat_sampler_params)
        return self.decode_slat(slat, formats)
