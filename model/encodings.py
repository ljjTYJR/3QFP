import torch
import torch.nn as nn
import numpy as np
import tinycudann as tcnn
from math import pi


class FrequencyEncoder(nn.Module):
    """
    This code is modified from the following link:
    https://github.com/PRBonn/LocNDF/blob/main/src/loc_ndf/models/models.py#L233C1-L256C65
    """
    def __init__(self, freq, num_bands=5, dimensionality=3, base=2):
        super().__init__()
        self.freq, self.num_bands = torch.tensor(freq), num_bands
        self.dimensionality, self.base = dimensionality, torch.tensor(base)

    def forward(self, x):
        x = x[..., :self.dimensionality, None]
        device, dtype, = x.device, x.dtype

        scales = torch.logspace(0., self.freq, self.num_bands, base=self.base, device=device, dtype=dtype)
        # Fancy reshaping
        scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

        x = x * scales * pi
        x = torch.cat([x.sin(), x.cos()], dim=-1)
        x = x.flatten(-2, -1)
        return x

    def fea_dim(self):
        return self.num_bands * 2 * self.dimensionality

class FFEncoder(nn.Module):
    """
    follow the paper : Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains
    Randomly sample isotropic frequencies to generate features
    """
    def __init__(self, scale, embedding_size, dim=3):
        super().__init__()
        self.scale = scale
        self.embedding_size = embedding_size
        self.dim = dim
        # TODO more elastic implementation
        self.bval = (torch.randn(self.embedding_size, 1) * self.scale * torch.pi * 2).to(device=torch.device('cuda'))
        self.bval = self.bval.squeeze()

    def forward(self, x):
        x = x[..., :self.dim, None]
        device, dtype, = x.device, x.dtype
        bval = self.bval[(*((None,) * (len(x.shape) - 1)), Ellipsis)]
        x = x * bval
        x = torch.cat([x.sin(), x.cos()], dim=-1)
        x = x.flatten(-2, -1)
        return x

    def fea_dim(self):
        return self.embedding_size * 2 * self.dim



def tiny_cuda_nn_get_encoder(encoding, input_dim=3,
                degree=4, n_bins=16, n_frequencies=12,
                n_levels=16, level_dim=2,
                base_resolution=16, log2_hashmap_size=19,
                desired_resolution=512):

    # Dense grid encoding
    if 'dense' in encoding.lower():
        n_levels = 4
        per_level_scale = np.exp2(np.log2(desired_resolution  / base_resolution) / (n_levels - 1))
        embed = tcnn.Encoding(
            n_input_dims=input_dim,
            encoding_config={
                    "otype": "Grid",
                    "type": "Dense",
                    "n_levels": n_levels,
                    "n_features_per_level": level_dim,
                    "base_resolution": base_resolution,
                    "per_level_scale": per_level_scale,
                    "interpolation": "Linear"},
                dtype=torch.float
        )
        out_dim = embed.n_output_dims

    # Sparse grid encoding
    elif 'hash' in encoding.lower() or 'tiled' in encoding.lower():
        print('Hash size', log2_hashmap_size)
        per_level_scale = np.exp2(np.log2(desired_resolution  / base_resolution) / (n_levels - 1))
        embed = tcnn.Encoding(
            n_input_dims=input_dim,
            encoding_config={
                "otype": 'HashGrid',
                "n_levels": n_levels,
                "n_features_per_level": level_dim,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_resolution,
                "per_level_scale": per_level_scale
            },
            dtype=torch.float
        )
        out_dim = embed.n_output_dims

    # Spherical harmonics encoding
    elif 'spherical' in encoding.lower():
        embed = tcnn.Encoding(
                n_input_dims=input_dim,
                encoding_config={
                "otype": "SphericalHarmonics",
                "degree": degree,
                },
                dtype=torch.float
            )
        out_dim = embed.n_output_dims

    # OneBlob encoding
    elif 'blob' in encoding.lower():
        print('Use blob')
        embed = tcnn.Encoding(
                n_input_dims=input_dim,
                encoding_config={
                "otype": "OneBlob", #Component type.
	            "n_bins": n_bins
                },
                dtype=torch.float
            )
        out_dim = embed.n_output_dims

    # Frequency encoding
    elif 'freq' in encoding.lower():
        print('Use frequency')
        embed = tcnn.Encoding(
                n_input_dims=input_dim,
                encoding_config={
                "otype": "Frequency",
                "n_frequencies": n_frequencies
                },
                dtype=torch.float
            )
        out_dim = embed.n_output_dims

    # Identity encoding
    elif 'identity' in encoding.lower():
        embed = tcnn.Encoding(
                n_input_dims=input_dim,
                encoding_config={
                "otype": "Identity"
                },
                dtype=torch.float
            )
        out_dim = embed.n_output_dims

    return embed, out_dim