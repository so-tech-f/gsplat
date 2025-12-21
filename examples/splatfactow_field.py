from typing import Literal
import torch
from torch import Tensor, nn
from gsplat.cuda._wrapper import spherical_harmonics
# directionsとの扱いだけ変更必要

class BGField(nn.Module):
    def __init__(
        self,
        appearance_embedding_dim: int,
        sh_levels: int = 4,
        layer_width: int = 128,
        num_layers: int = 3,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__()
        self.sh_dim = (sh_levels + 1) ** 2
        layers = []
        in_dim = appearance_embedding_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, layer_width))
            layers.append(nn.ReLU())
            in_dim = layer_width

        self.encoder = nn.Sequential(*layers).to(device)
        self.sh_base_head = nn.Linear(layer_width, 3).to(device)
        self.sh_rest_head = nn.Linear(layer_width, (self.sh_dim - 1) * 3).to(device)
        # zero initialization
        self.sh_rest_head.weight.data.zero_()
        self.sh_rest_head.bias.data.zero_()

    def forward(self, appearance_embedding=None) -> Tensor:
        x = self.encoder(appearance_embedding)
        base_color = self.sh_base_head(x)
        sh_rest = self.sh_rest_head(x)
        sh_coeffs = torch.cat([base_color, sh_rest], dim=-1).view(-1, self.sh_dim, 3)  
        return sh_coeffs


class SplatfactoWField(nn.Module):
    def __init__(
        self,
        appearance_embed_dim: int,
        appearance_features_dim: int,
        sh_levels: int = 4,
        num_layers: int = 3,
        layer_width: int = 256,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__()

        layers = []
        in_dim = appearance_embed_dim + appearance_features_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, layer_width))
            layers.append(nn.ReLU())
            in_dim = layer_width

        self.encoder = nn.Sequential(*layers).to(device)
        self.sh_dim = (sh_levels + 1) ** 2
        self.sh_base_head = nn.Linear(layer_width, 3).to(device)
        self.sh_rest_head = nn.Linear(layer_width, (self.sh_dim - 1) * 3).to(device)
        # zero initialization
        self.sh_rest_head.weight.data.zero_()
        self.sh_rest_head.bias.data.zero_()

    def forward(
        self,
        appearance_embed: Tensor,
        appearance_features: Tensor,
    ) -> Tensor:
        x = self.encoder(
            torch.cat((appearance_embed, appearance_features), dim=-1)
        ).float()
        base_color = self.sh_base_head(x)
        sh_rest = self.sh_rest_head(x)
        sh_coeffs = torch.cat([base_color, sh_rest], dim=-1).view(-1, self.sh_dim, 3)  
        return sh_coeffs
