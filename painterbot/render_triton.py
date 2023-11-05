import torch
import triton
import triton.language as tl

from .painterbot import StrokeParameters

BLOCK_SIZE = 512


@triton.jit
def _pdf_forwards(
    coordinates_ptr,
    center_x_ptr,
    center_y_ptr,
    rotation_ptr,
    mu_r_ptr,
    sigma_r_ptr,
    sigma_theta_ptr,
    color_ptr,
    alpha_ptr,
    maxes_ptr,
    N_STROKES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)


def triton_pdf_forwards(
    canvas: torch.Tensor,
    params: StrokeParameters,
    target: torch.Tensor,
    n_strokes: int,
) -> torch.Tensor:
    assert (
        canvas.is_cuda and target.is_cuda and params.alpha.is_cuda
    ), "all tensors must be on cuda"

    canvas = canvas.contiguous()
    target = target.contiguous()
    center_x = params.center_x.contiguous()
    center_y = params.center_y.contiguous()
    rotation = params.rotation.contiguous()
    mu_r = params.mu_r.contiguous()
    sigma_r = params.sigma_r.contiguous()
    sigma_theta = params.sigma_theta.contiguous()
    color = params.color.contiguous()
    alpha = params.alpha.contiguous()

    image_shape = target.shape

    canvas = canvas.view(-1)
    target = target.view(-1)
    center_x = center_x.view(-1)
    center_y = center_y.view(-1)
    rotation = rotation.view(-1)
    mu_r = mu_r.view(-1)
    sigma_r = sigma_r.view(-1)
    sigma_theta = sigma_theta.view(-1)
    color = color.view(-1)
    alpha = alpha.view(-1)

    canvas_output = torch.empty_like(canvas)
