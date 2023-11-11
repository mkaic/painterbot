# +

from typing import Tuple

import torch

from .parameters import StrokeParameters

EPSILON: float = 1e-8


def pdf(
    center_x: torch.Tensor,
    center_y: torch.Tensor,
    rotation: torch.Tensor,
    mu_r: torch.Tensor,
    sigma_r: torch.Tensor,
    sigma_theta: torch.Tensor,
    alpha: torch.Tensor,
    n_strokes: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    n_strokes = n_strokes

    w = torch.linspace(0, width - 1, width, device=device, dtype=dtype) / height
    h = torch.linspace(0, height - 1, height, device=device, dtype=dtype) / height

    coordinates = torch.cartesian_prod(h, w).permute(1, 0)  # (2 x HW)
    coordinates = coordinates.unsqueeze(0)  # (1 x 2 x HW)
    coordinates = coordinates.repeat(n_strokes, 1, 1)  # (N x 2 x HW)

    cos_rot = torch.cos(rotation)
    sin_rot = torch.sin(rotation)

    # Offset coordinates to change where the center of the
    # polar coordinates is placed
    offset_x = center_x - (cos_rot * mu_r)
    # the direction of the y-axis is inverted, so we add rather than subtract
    offset_y = center_y + (sin_rot * mu_r)

    x_coordinates, y_coordinates = coordinates[:, 1, :], coordinates[:, 0, :]

    x_coordinates = x_coordinates - offset_x
    y_coordinates = y_coordinates - offset_y

    x_coordinates = (x_coordinates * cos_rot) - (y_coordinates * sin_rot)
    y_coordinates = (x_coordinates * sin_rot) + (y_coordinates * cos_rot)

    # Convert to polar coordinates and apply polar-space offsets
    # (N x 2 x HW) -> (N x HW)
    r = torch.sqrt(torch.square(x_coordinates) + torch.square(y_coordinates))
    r = r - mu_r

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
    sigma_r = sigma_r * mu_r

    r = r / (2 * torch.square(sigma_r) + EPSILON)
    theta = theta / (2 * torch.square(sigma_theta) + EPSILON)
    pdf = torch.exp(-1 * (r + theta))

    pdf = pdf * alpha

    strokes = pdf.view(n_strokes, 1, height, width)

    return strokes


pdf_compiled = torch.compile(pdf)


def blend(
    canvas: torch.Tensor,
    strokes: torch.Tensor,
    n_strokes: int,
    colors: torch.Tensor,
    KEEP_HISTORY: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if KEEP_HISTORY:
        canvas_history = []
    else:
        canvas_history = None

    for i in range(n_strokes):
        if KEEP_HISTORY:
            canvas_history.append(canvas.clone())
        stroke = strokes[i]
        color = colors[i]
        canvas = ((torch.ones_like(stroke) - stroke) * canvas) + (stroke * color)

    if KEEP_HISTORY:
        canvas_history = torch.stack(canvas_history, dim=0)
    return canvas, canvas_history


blend_compiled = torch.compile(blend)


def render(
    canvas: torch.Tensor,
    parameters: StrokeParameters,
    KEEP_HISTORY: bool = True,
    COMPILED: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
    height, width = canvas.shape[-2:]
    device = canvas.device
    dtype = canvas.dtype

    pdf_fn = pdf_compiled if COMPILED else pdf
    blend_fn = blend_compiled if COMPILED else blend

    strokes = pdf_fn(
        center_x=parameters.center_x,
        center_y=parameters.center_y,
        rotation=parameters.rotation,
        mu_r=parameters.mu_r,
        sigma_r=parameters.sigma_r,
        sigma_theta=parameters.sigma_theta,
        alpha=parameters.alpha,
        n_strokes=parameters.n_strokes,
        height=height,
        width=width,
        device=device,
        dtype=dtype,
    )

    canvas, canvas_history = blend_fn(
        canvas=canvas,
        strokes=strokes,
        n_strokes=parameters.n_strokes,
        colors=parameters.color,
        KEEP_HISTORY=KEEP_HISTORY,
    )

    if KEEP_HISTORY:
        return (canvas, canvas_history)
    else:
        return canvas
