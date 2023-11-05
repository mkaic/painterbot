# +
import shutil
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torchvision.transforms as T
from tqdm.auto import tqdm

EPSILON = 1e-8

# +


class StrokeParameters(nn.Module):
    def __init__(self, n_strokes: int, width_to_height_ratio: float):
        super().__init__()

        self.register_buffer("n_strokes", torch.tensor(n_strokes))
        self.register_buffer(
            "width_to_height_ratio", torch.tensor(width_to_height_ratio)
        )

        self.center_x = torch.ones(n_strokes, 1) * self.width_to_height_ratio * 0.5
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
        self.alpha = self.alpha.view(n_strokes, 1, 1, 1)
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

    def smart_init(
        self, target: torch.Tensor, canvas: torch.Tensor, temperature: float = 1.0
    ):
        device = target.device
        height, width = target.shape[-2:]

        error_map = torch.mean(torch.abs(target - canvas), dim=0)  # (3, H, W) -> (H, W)

        # Unravel the error map, then softmax to get probability distribution over
        # the pixel locations, then sample from it
        flat_error_pdf = nn.functional.softmax(
            error_map.view(-1) / temperature, dim=0
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


def loss_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(x - y))


def make_coordinates(canvas: torch.Tensor, n_strokes: int) -> torch.Tensor:
    # Aspect ratio should always be >1:1, never 1:<1
    height, width = canvas.shape[-2:]

    w = torch.linspace(0, width - 1, width, device=canvas.device) / height
    h = torch.linspace(0, height - 1, height, device=canvas.device) / height

    x, y = torch.meshgrid(w, h, indexing="xy")

    # Reshape coordinates into a list of XY pairs, then duplicate it
    # once for each stroke in the group
    cartesian_coordinates = torch.stack([x, y], dim=0).view(1, 2, -1)
    cartesian_coordinates = cartesian_coordinates.repeat(
        n_strokes, 1, 1
    )  # (1 x 2 x M) -> (N x 2 x M)

    return cartesian_coordinates


def evaluate_pdf(
    cartesian_coordinates: torch.Tensor,
    parameters: StrokeParameters,
    n_strokes: int,
) -> torch.Tensor:
    cos_rot = torch.cos(parameters.rotation)
    sin_rot = torch.sin(parameters.rotation)

    # Offset coordinates to change where the center of the
    # polar coordinates is placed
    offset_x = parameters.center_x - (cos_rot * parameters.mu_r)
    # the direction of the y-axis is inverted, so we add rather than subtract
    offset_y = parameters.center_y + (sin_rot * parameters.mu_r)
    cartesian_offset = torch.stack(
        [offset_x, offset_y], dim=1
    )  # 2x (N x 1) -> (N x 2 x 1)
    cartesian_coordinates = cartesian_coordinates - cartesian_offset

    rotation_matrices = torch.cat([cos_rot, -sin_rot, sin_rot, cos_rot], dim=-1)
    rotation_matrices = rotation_matrices.view(n_strokes, 2, 2)
    cartesian_coordinates = torch.bmm(rotation_matrices, cartesian_coordinates)

    # Convert to polar coordinates and apply polar-space offsets
    # (N x 2 x M) -> (N x M)
    r = torch.linalg.norm(cartesian_coordinates, dim=1)
    r = r - parameters.mu_r

    theta = torch.atan2(
        cartesian_coordinates[:, 1] + EPSILON,
        cartesian_coordinates[:, 0] + EPSILON,
    )  # -> (N x M), ranges from -pi to pi

    # Simplified Gaussian PDF function:
    # e ^ (
    #    -1
    #     * (
    #         ((x - mu_x) ^ 2 / (2 * sigma_x ^ 2))
    #         + ((y - mu_y) ^ 2 / (2 * sigma_y ^ 2))
    #     )
    # )

    r = torch.square(r)
    theta = torch.square(theta)

    # sigma_r is expressed as a fraction of the radius instead of
    # an absolute quantity
    sigma_r = parameters.sigma_r * parameters.mu_r
    sigmas = torch.stack([sigma_r, parameters.sigma_theta], dim=1).view(n_strokes, 2, 1)

    polar_coordinates = torch.stack([r, theta], dim=1)
    polar_coordinates = polar_coordinates / (2 * torch.square(sigmas) + EPSILON)
    pdf = torch.exp(-1 * torch.sum(polar_coordinates, dim=1))

    return pdf


def calculate_strokes(
    canvas: torch.Tensor,
    parameters: StrokeParameters,
    n_strokes: int,
) -> torch.Tensor:
    height, width = canvas.shape[-2:]
    centers = torch.stack(
        [parameters.center_x, parameters.center_y], dim=1
    )  # (N x 2 x 1)

    stroke_maxes = evaluate_pdf(
        cartesian_coordinates=centers, parameters=parameters, n_strokes=n_strokes
    )  # (N x 1)

    coordinates = make_coordinates(
        canvas=canvas,
        n_strokes=n_strokes,
    )  # (N x 2 x M)

    strokes = evaluate_pdf(
        cartesian_coordinates=coordinates, parameters=parameters, n_strokes=n_strokes
    )  # (N x M)

    strokes = strokes / (stroke_maxes + EPSILON)
    strokes = strokes.view(n_strokes, 1, height, width)
    strokes = strokes * parameters.alpha

    return strokes


def render_stroke(
    stroke: torch.Tensor, color: torch.Tensor, canvas: torch.Tensor
) -> Tuple[torch.Tensor]:
    # Use the stroke as an alpha to blend between the canvas and a color
    canvas = ((torch.ones_like(stroke) - stroke) * canvas) + (stroke * color)
    # Clamp canvas to valid RGB color space.
    canvas = canvas.clamp(0, 1)

    return canvas


def render(
    canvas: torch.Tensor,
    parameters: StrokeParameters,
    target: torch.Tensor,
    n_strokes: int,
) -> torch.Tensor:
    strokes = calculate_strokes(canvas, parameters, n_strokes=n_strokes)

    loss = 0
    for i in range(n_strokes):
        stroke = strokes[i]
        color = parameters.color[i]
        canvas = render_stroke(stroke=stroke, color=color, canvas=canvas)
        loss += loss_fn(canvas.unsqueeze(0), target.unsqueeze(0))

    loss = loss / n_strokes

    return (canvas.detach(), loss)


def render_timelapse_frames(
    canvas: torch.Tensor, parameters: StrokeParameters, output_path: Path
) -> torch.Tensor:
    output_path = Path(output_path)
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir()

    with torch.no_grad():
        strokes = calculate_strokes(canvas, parameters, n_strokes=parameters.n_strokes)

        height, width = canvas.shape[-2:]

        for i, (stroke, color, center_x, center_y) in enumerate(
            zip(strokes, parameters.color, parameters.center_x, parameters.center_y)
        ):
            canvas = render_stroke(stroke=stroke, color=color, canvas=canvas)

            to_save = canvas.clone().cpu()

            # center_x = int(center_x * width)
            # center_y = int(center_y * height)
            # to_save[
            #     :,
            #     max(center_y - 3, 0) : min(center_y + 3, height),
            #     max(center_x - 3, 0) : min(center_x + 3, width),
            # ] = torch.tensor([1.0, 0.0, 0.0]).view(3, 1, 1)

            T.functional.to_pil_image(to_save).save(output_path / f"{i:05}.jpg")

        return canvas


def optimize(
    target: torch.Tensor,
    n_groups: int = 10,
    n_strokes_per_group: int = 50,
    iterations: int = 300,
    lr: float = 0.01,
    show_inner_pbar: bool = True,
    error_map_temperature: float = 1.0,
    log_every: int = 15,
):
    if n_groups == 1:
        show_inner_pbar = True

    device = target.device
    width, height = target.shape[-2:]
    width_to_height_ratio = width / height

    canvas = torch.zeros_like(target)

    frozen_params: StrokeParameters = None

    output_path = Path("optimization_timelapse_frames")
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir()

    loss_history = []
    mae_history = []

    outer_pbar = tqdm(range(n_groups))
    outer_pbar.set_description(f"Loss=?, MAE=? | Group 1/{n_groups}")

    for i in outer_pbar:
        active_params = StrokeParameters(
            n_strokes=n_strokes_per_group,
            width_to_height_ratio=width_to_height_ratio,
        ).to(device)

        active_params.smart_init(
            target=target,
            canvas=canvas,
            temperature=error_map_temperature,
        )

        optimizer = torch.optim.Adam(
            active_params.parameters(), lr=lr, betas=(0.8, 0.9)
        )

        if show_inner_pbar:
            inner_iterator = tqdm(range(iterations), leave=False)
        else:
            inner_iterator = range(iterations)

        for j in inner_iterator:
            result, loss = render(
                canvas=canvas,
                parameters=active_params,
                target=target,
                n_strokes=n_strokes_per_group,
            )

            loss.backward()

            optimizer.step()

            optimizer.zero_grad(set_to_none=True)

            active_params.clamp_parameters()

            mae = torch.mean(torch.abs(result - target))

            if show_inner_pbar:
                description = f"Loss={loss:.5f}, MAE={mae:.5f}"
                inner_iterator.set_description(description)

            if log_every is not None and j % log_every == 0:
                T.functional.to_pil_image(result).save(
                    output_path / f"{i:05}_{j:06}.jpg"
                )

            loss_history.append(loss.detach().cpu().item())
            mae_history.append(mae.detach().cpu().item())

        canvas, _ = render(
            canvas=canvas,
            parameters=active_params,
            target=target,
            n_strokes=n_strokes_per_group,
        )
        if i == 0:
            frozen_params = active_params
        else:
            frozen_params = concat_stroke_parameters([frozen_params, active_params])

        print(len(frozen_params.alpha), frozen_params.n_strokes)

        outer_pbar.set_description(
            f"Loss={loss:.5f}, MAE={mae:.5f} | Group {i+1}/{n_groups}"
        )
    return frozen_params, loss_history, mae_history
