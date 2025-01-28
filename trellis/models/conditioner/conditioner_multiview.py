import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file
from .blocks import VolumeFeatureExtractionNetwork
from ...utils.model_utils import get_warp_coordinates


class MultiviewConditioner(nn.Module):

    def __init__(self,
                 encoder_name: str = "dinov2_vitl14_reg",
                 input_image_size: int = 518,
                 spatial_volume_size: int = 16,
                 spatial_volume_length: float = 0.5):
        super(MultiviewConditioner, self).__init__()
        self.target_encoder = torch.hub.load("facebookresearch/dinov2", encoder_name).cuda()
        self.input_image_size = input_image_size
        self.spatial_volume_size = spatial_volume_size
        self.spatial_volume_length = spatial_volume_length
        self.n_patch = input_image_size // 14
        self._config_dict = self._get_init_params(locals())
        self.downsampler_net = VolumeFeatureExtractionNetwork(input_dim=1024, dims=(512, 1024))

    def _get_init_params(self, local_params):
        """
        Extracts and returns the initialization parameters from locals(),
        excluding 'self' and any non-serializable entries.
        """
        init_params = local_params.copy()
        init_params.pop('self', None)  # Remove 'self' from the parameters
        # Optionally remove other non-serializable parameters or sensitive information
        return init_params

    @torch.no_grad()
    def construct_spatial_volume(self, x, extrinsics, intrinsics):
        """
        @param x:            B,N,4,H,W, N = # of views
        @param extrinsics:   B,N,3,4
        @param intrinsics:   B,N,3,3
        @return:
        """
        B, N, _, H, W = x.shape
        V = self.spatial_volume_size
        device = x.device

        spatial_volume_verts = torch.linspace(-self.spatial_volume_length,
                                              self.spatial_volume_length,
                                              V,
                                              dtype=torch.float32,
                                              device=device)
        spatial_volume_verts = torch.stack(
            torch.meshgrid(spatial_volume_verts, spatial_volume_verts, spatial_volume_verts), -1)
        spatial_volume_verts = spatial_volume_verts.reshape(1, V**3, 3)[:, :, (2, 1, 0)]
        spatial_volume_verts = spatial_volume_verts.view(1, V, V, V, 3).permute(0, 4, 1, 2, 3).repeat(B, 1, 1, 1, 1)

        if intrinsics.ndim == 3:
            intrinsics = intrinsics.unsqueeze(0).repeat(B, 1, 1, 1)
            extrinsics = extrinsics.unsqueeze(0).repeat(B, 1, 1, 1)

        # extract 2D image features
        spatial_volume_feats = []
        # project source features
        for ni in range(0, N):
            pose_source_ = extrinsics[:, ni]
            K_source_ = intrinsics[:, ni]
            # get dinov2 features
            x_ = self.target_encoder(x[:, ni], is_training=True)['x_prenorm']
            x_ = x_[:, self.target_encoder.num_register_tokens + 1:].permute(0, 2, 1).reshape(
                B, 1024, self.n_patch, self.n_patch)
            C = x_.shape[1]

            coords_source = get_warp_coordinates(spatial_volume_verts, x_.shape[-1], self.input_image_size, K_source_,
                                                 pose_source_).view(B, V, V * V, 2)
            unproj_feats_ = F.grid_sample(x_, coords_source, mode='bilinear', padding_mode='zeros', align_corners=True)
            unproj_feats_ = unproj_feats_.view(B, C, V, V, V)
            spatial_volume_feats.append(unproj_feats_)

        spatial_volume_feats = torch.stack(spatial_volume_feats, 1)  # B,N,C,V,V,V
        N = spatial_volume_feats.shape[1]
        spatial_volume_feats = spatial_volume_feats.mean(dim=1).view(B, C, V, V, V)

        return spatial_volume_feats

    def forward(self, images: torch.Tensor, extrinsics: torch.Tensor, intrinsics: torch.Tensor):
        # aggregate to spatial volume
        spatial_volume = self.construct_spatial_volume(images, extrinsics[..., :3, :], intrinsics)
        downsampled_volume = self.downsampler_net(spatial_volume)
        B, C, = downsampled_volume.shape[:2]
        # flatten
        flattend_volume = downsampled_volume.view(B, C, -1).permute(0, 2, 1)
        return flattend_volume

    def save_pretrained(self, save_dir: str):
        # TODO: build string here
        # Create the directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        # Save the configuration parameters to a JSON file
        config_file = os.path.join(save_dir, 'config.json')
        with open(config_file, 'w') as f:
            json.dump(self._config_dict, f, indent=2)

        # Save the model's state_dict (weights) to a safetensors file
        weights_file = os.path.join(save_dir, 'model.safetensors')
        state_dict = self.state_dict()
        save_file(state_dict, weights_file)
