# +

from typing import Tuple

import torch

from .parameters import StrokeParameters

EPSILON: float = 1e-8


def torch_pdf(
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    parameters: StrokeParameters,
) -> torch.Tensor:
    n_strokes = parameters.n_strokes

    w = torch.linspace(0, width - 1, width, device=device, dtype=dtype) / height
    h = torch.linspace(0, height - 1, height, device=device, dtype=dtype) / height

    coordinates = torch.cartesian_prod(h, w).permute(1, 0)  # (2 x HW)
    coordinates = coordinates.unsqueeze(0)  # (1 x 2 x HW)
    coordinates = coordinates.repeat(n_strokes, 1, 1)  # (N x 2 x HW)

    cos_rot = torch.cos(parameters.rotation)
    sin_rot = torch.sin(parameters.rotation)

    # Offset coordinates to change where the center of the
    # polar coordinates is placed
    offset_x = parameters.center_x - (cos_rot * parameters.mu_r)
    # the direction of the y-axis is inverted, so we add rather than subtract
    offset_y = parameters.center_y + (sin_rot * parameters.mu_r)

    x_coordinates, y_coordinates = coordinates[:, 1, :], coordinates[:, 0, :]

    x_coordinates = x_coordinates - offset_x
    y_coordinates = y_coordinates - offset_y

    x_coordinates = (x_coordinates * cos_rot) - (y_coordinates * sin_rot)
    y_coordinates = (x_coordinates * sin_rot) + (y_coordinates * cos_rot)

    # Convert to polar coordinates and apply polar-space offsets
    # (N x 2 x HW) -> (N x HW)
    r = torch.sqrt(torch.square(x_coordinates) + torch.square(y_coordinates))
    r = r - parameters.mu_r

    theta = torch.atan2(
        y_coordinates + EPSILON,
        x_coordinates + EPSILON,
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

    r = r / (2 * torch.square(sigma_r) + EPSILON)
    theta = theta / (2 * torch.square(parameters.sigma_theta) + EPSILON)
    pdf = torch.exp(-1 * (r + theta))

    pdf = pdf * parameters.alpha

    strokes = pdf.view(n_strokes, 1, height, width)

    return strokes


def torch_blend(
    canvas: torch.Tensor,
    strokes: torch.Tensor,
    parameters: StrokeParameters,
    KEEP_HISTORY: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if KEEP_HISTORY:
        canvas_history = []
    else:
        canvas_history = None

    for i in range(parameters.n_strokes):
        if KEEP_HISTORY:
            canvas_history.append(canvas.clone())
        stroke = strokes[i]
        color = parameters.color[i]
        canvas = ((torch.ones_like(stroke) - stroke) * canvas) + (stroke * color)

    if KEEP_HISTORY:
        canvas_history = torch.stack(canvas_history, dim=0)
    return canvas, canvas_history


def torch_render(
    canvas: torch.Tensor,
    parameters: StrokeParameters,
    KEEP_HISTORY: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
    height, width = canvas.shape[-2:]
    device = canvas.device
    dtype = canvas.dtype

    strokes = torch_pdf(
        height=height,
        width=width,
        device=device,
        dtype=dtype,
        parameters=parameters,
    )

    canvas, canvas_history = torch_blend(
        canvas=canvas,
        strokes=strokes,
        parameters=parameters,
        KEEP_HISTORY=KEEP_HISTORY,
    )

    if KEEP_HISTORY:
        return (canvas, canvas_history)
    else:
        return canvas
