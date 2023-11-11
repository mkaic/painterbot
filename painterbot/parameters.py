from pathlib import Path
from typing import List

import torch
import torch.nn as nn


class StrokeParameters(nn.Module):
    def __init__(self, n_strokes: int, width_to_height_ratio: float):
        super().__init__()

        self.register_buffer("n_strokes", torch.tensor(n_strokes))
        self.register_buffer(
            "width_to_height_ratio", torch.tensor(width_to_height_ratio)
        )

        self.center_x = torch.ones(n_strokes, 1) * width_to_height_ratio * 0.5
        self.center_x = nn.Parameter(self.center_x)

        self.center_y = torch.ones(n_strokes, 1) * 0.5
        self.center_y = nn.Parameter(self.center_y)

        self.rotation = (torch.rand(n_strokes, 1) - 0.5) * 2 * torch.pi
        self.rotation = nn.Parameter(self.rotation)

        self.mu_r = torch.ones(n_strokes, 1) * 0.2
        self.mu_r = nn.Parameter(self.mu_r)

        self.sigma_r = torch.ones(n_strokes, 1) * 0.1
        self.sigma_r = nn.Parameter(self.sigma_r)

        self.sigma_theta = torch.ones(n_strokes, 1) * 0.1
        self.sigma_theta = nn.Parameter(self.sigma_theta)

        self.color = torch.ones(n_strokes, 3, 1, 1)
        self.color = self.color * torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        self.color = self.color + torch.randn_like(self.color) * 0.1
        self.color = nn.Parameter(self.color)

        self.alpha = torch.ones(n_strokes) * 0.01
        self.alpha = self.alpha.view(n_strokes, 1)
        self.alpha = nn.Parameter(self.alpha)

    def clamp_parameters(self):
        self.center_x.data.clamp_(0, self.width_to_height_ratio)
        self.center_y.data.clamp_(0, 1)

        self.rotation.data.clamp_(-torch.pi, torch.pi)

        # Guassian gets weirdly pinched if it gets too close to the pole
        self.mu_r.data.clamp_(0.05, 2)
        self.sigma_r.data.clamp_(0.0001, 0.2)

        # Strokes can only ever curve a certain amount to avoid a bunch of
        # obvious perfect circles showing up in the painting
        self.sigma_theta.data.clamp_(0.001, torch.pi / 4)

        self.alpha.data.clamp_(0.01, 1)
        self.color.data.clamp_(0, 1)

    def smart_init(self, target: torch.Tensor, canvas: torch.Tensor):
        device = target.device
        height, width = target.shape[-2:]

        error_map = torch.mean(torch.abs(target - canvas), dim=0)  # (3, H, W) -> (H, W)

        # Unravel the error map, then softmax to get probability distribution over
        # the pixel locations, then sample from it
        flat_error_pdf = nn.functional.softmax(
            error_map.view(-1), dim=0
        )  # (H, W) -> (H*W)

        n_samples = self.n_strokes

        coordinates = torch.multinomial(flat_error_pdf, n_samples)  # (N, 1)

        new_coords = torch.zeros(2, len(coordinates), dtype=torch.long, device=device)

        # x coordinates (which column the flat index is in)
        # are found by moduloing it by the length of a row
        new_coords[0, :] = coordinates % width

        # y coordinates (which row the flat index is in) are
        # found by integer-dividing it by the length of a column
        new_coords[1, :] = torch.div(coordinates, height, rounding_mode="floor")

        coordinates = new_coords.unsqueeze(-1)  # (N, 1) -> (2, N, 1)

        center_x, center_y = coordinates

        color = target[:, center_y, center_x].view(self.n_strokes, 3, 1, 1)

        center_x = center_x / height
        center_x = center_x.detach().clone()
        self.center_x.data = center_x

        center_y = center_y / height
        center_y = center_y.detach().clone()
        self.center_y.data = center_y

        color = color.detach().clone()
        self.color.data = color

        self.clamp_parameters()

    def save(self, path: Path):
        state_dict = self.state_dict()
        bfloat16_keys = [
            "width_to_height_ratio",
            "center_x",
            "center_y",
            "rotation",
            "mu_r",
            "sigma_r",
            "sigma_theta",
        ]
        uint8_keys = ["color", "alpha"]

        for key in bfloat16_keys:
            state_dict[key] = state_dict[key].to(torch.bfloat16)
        for key in uint8_keys:
            state_dict[key] = state_dict[key] * 255
            state_dict[key] = state_dict[key].to(torch.uint8)

        torch.save(state_dict, path)

    @staticmethod
    def from_file(path: Path):
        state_dict = torch.load(path)

        bfloat16_keys = [
            "width_to_height_ratio",
            "center_x",
            "center_y",
            "rotation",
            "mu_r",
            "sigma_r",
            "sigma_theta",
        ]
        uint8_keys = ["color", "alpha"]

        for key in bfloat16_keys:
            state_dict[key] = state_dict[key].to(torch.float32)
        for key in uint8_keys:
            state_dict[key] = state_dict[key].to(torch.float32) / 255

        n_strokes = state_dict["n_strokes"].item()
        width_to_height_ratio = state_dict["width_to_height_ratio"].item()

        params = StrokeParameters(
            n_strokes=n_strokes, width_to_height_ratio=width_to_height_ratio
        )
        params.load_state_dict(state_dict)

        return params


def concat_stroke_parameters(
    stroke_parameters: List[StrokeParameters],
) -> StrokeParameters:
    concatted = stroke_parameters[0]

    if len(stroke_parameters) > 1:
        for param in stroke_parameters[1:]:
            concatted.center_x.data = torch.cat(
                [concatted.center_x.data, param.center_x.data], dim=0
            )
            concatted.center_y.data = torch.cat(
                [concatted.center_y.data, param.center_y.data], dim=0
            )
            concatted.rotation.data = torch.cat(
                [concatted.rotation.data, param.rotation.data], dim=0
            )
            concatted.mu_r.data = torch.cat(
                [concatted.mu_r.data, param.mu_r.data], dim=0
            )
            concatted.sigma_r.data = torch.cat(
                [concatted.sigma_r.data, param.sigma_r.data], dim=0
            )
            concatted.sigma_theta.data = torch.cat(
                [concatted.sigma_theta.data, param.sigma_theta.data], dim=0
            )
            concatted.color.data = torch.cat(
                [concatted.color.data, param.color.data], dim=0
            )
            concatted.alpha.data = torch.cat(
                [concatted.alpha.data, param.alpha.data], dim=0
            )
            concatted.n_strokes += param.n_strokes

    return concatted


def split_stroke_parameters(
    parameters: StrokeParameters, block_size: int = 128
) -> List[StrokeParameters]:
    n_strokes = parameters.n_strokes.item()
    n_blocks = n_strokes // block_size
    n_leftover = n_strokes % block_size

    if n_leftover > 0:
        n_blocks += 1

    split_parameters = []

    for i in range(n_blocks):
        start = i * block_size
        end = min(start + block_size, n_strokes)

        parameter_block = StrokeParameters(
            n_strokes=end - start,
            width_to_height_ratio=parameters.width_to_height_ratio.item(),
        )

        parameter_block.center_x.data = parameters.center_x.data[start:end]
        parameter_block.center_y.data = parameters.center_y.data[start:end]
        parameter_block.rotation.data = parameters.rotation.data[start:end]
        parameter_block.mu_r.data = parameters.mu_r.data[start:end]
        parameter_block.sigma_r.data = parameters.sigma_r.data[start:end]
        parameter_block.sigma_theta.data = parameters.sigma_theta.data[start:end]
        parameter_block.color.data = parameters.color.data[start:end]
        parameter_block.alpha.data = parameters.alpha.data[start:end]

        split_parameters.append(parameter_block)

    return split_parameters
