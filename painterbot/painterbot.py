# +
import shutil
from pathlib import Path
from typing import Tuple

import torch
import torchvision.transforms as T
from tqdm.auto import tqdm

from .triton_render_kernel import triton_pdf_forwards
from .parameters import StrokeParameters, concat_stroke_parameters


def loss_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(x - y))


def evaluate_pdf(
    coordinates: torch.Tensor,
    parameters: StrokeParameters,
    EPSILON: float = 1e-8,
) -> torch.Tensor:
    n_strokes = parameters.n_strokes

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
    coordinates = coordinates - cartesian_offset

    rotation_matrices = torch.cat([cos_rot, -sin_rot, sin_rot, cos_rot], dim=-1)
    rotation_matrices = rotation_matrices.view(n_strokes, 2, 2)
    coordinates = torch.bmm(rotation_matrices, coordinates)

    # Convert to polar coordinates and apply polar-space offsets
    # (N x 2 x HW) -> (N x HW)
    r = torch.linalg.norm(coordinates, dim=1)
    r = r - parameters.mu_r

    theta = torch.atan2(
        coordinates[:, 1] + EPSILON,
        coordinates[:, 0] + EPSILON,
    )  # -> (N x HW), ranges from -pi to pi

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

    pdf = pdf * parameters.alpha

    return pdf


def calculate_strokes(
    canvas: torch.Tensor,
    parameters: StrokeParameters,
    triton: bool = False,
) -> torch.Tensor:
    height, width = canvas.shape[-2:]
    device = canvas.device
    dtype = canvas.dtype

    n_strokes = parameters.n_strokes

    if triton:
        pdf_func = triton_pdf_forwards
    else:
        pdf_func = evaluate_pdf

    w = torch.linspace(0, width - 1, width, device=device, dtype=dtype) / height
    h = torch.linspace(0, height - 1, height, device=device, dtype=dtype) / height

    coordinates = torch.cartesian_prod(w, h).permute(1, 0)  # (2 x HW)
    coordinates = coordinates.unsqueeze(0)  # (1 x 2 x HW)
    coordinates = coordinates.repeat(n_strokes, 1, 1)  # (N x 2 x HW)

    strokes = pdf_func(coordinates=coordinates, parameters=parameters)  # (N x HW)

    strokes = strokes.view(n_strokes, 1, height, width)

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
    target: torch.Tensor = None,
    triton: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
    n_strokes = parameters.n_strokes

    strokes = calculate_strokes(
        canvas=canvas,
        parameters=parameters,
        triton=triton,
    )

    loss = 0
    for i in range(n_strokes):
        stroke = strokes[i]
        color = parameters.color[i]
        canvas = render_stroke(stroke=stroke, color=color, canvas=canvas)

        if target is not None:
            loss += loss_fn(canvas.unsqueeze(0), target.unsqueeze(0))

    loss = loss / n_strokes

    if target is not None:
        return (canvas.detach(), loss)
    else:
        return canvas.detach()


def render_timelapse_frames(
    canvas: torch.Tensor,
    parameters: StrokeParameters,
    output_path: Path,
    triton: bool = False,
) -> torch.Tensor:
    output_path = Path(output_path)
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir()

    with torch.no_grad():
        strokes = calculate_strokes(
            canvas=canvas,
            parameters=parameters,
            triton=triton,
        )

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
    triton: bool = False,
):
    if n_groups == 1:
        show_inner_pbar = True

    device = target.device
    width, height = target.shape[-2:]
    width_to_height_ratio = width / height

    canvas = torch.zeros_like(target)

    frozen_params: StrokeParameters = None

    output_path = Path("timelapse_frames_optimization")
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
                triton=triton,
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
            triton=triton,
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
