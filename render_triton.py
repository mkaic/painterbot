import torch
import triton
import triton.language as tl

from painterbot import StrokeParameters


@triton.jit
def _render_forwards(
    canvas_ptr,
    target_ptr,
    center_x_ptr,
    center_y_ptr,
    rotation_ptr,
    mu_r_ptr,
    sigma_r_ptr,
    sigma_theta_ptr,
    color_ptr,
    alpha_ptr,
    N_STROKES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pass


@triton.jit
def _render_backwards(
    canvas_ptr,
    target_ptr,
    center_x_ptr,
    center_y_ptr,
    rotation_ptr,
    mu_r_ptr,
    sigma_r_ptr,
    sigma_theta_ptr,
    color_ptr,
    alpha_ptr,
    N_STROKES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pass


def triton_render_forwards(
    canvas: torch.Tensor,
    params: StrokeParameters,
    target: torch.Tensor,
    n_strokes: int,
) -> torch.Tensor:
    pass


def triton_render_backwards(
    canvas: torch.Tensor,
    params: StrokeParameters,
    target: torch.Tensor,
    n_strokes: int,
) -> torch.Tensor:
    pass
