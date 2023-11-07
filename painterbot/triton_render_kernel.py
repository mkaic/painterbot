import torch
import triton
import triton.language as tl

from .painterbot import StrokeParameters

BLOCK_SIZE = 128


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
    N_STROKES: int,
    N_COORDINATES: int,
    BLOCK_SIZE: tl.constexpr,
):
    stroke_id = tl.program_id(0)
    block_id = tl.program_id(1)

    start_ptr = (
        coordinates_ptr + (stroke_id * N_COORDINATES * 2) + (block_id * BLOCK_SIZE * 2)
    )
    x_coordinate_offsets = tl.arange(0, BLOCK_SIZE * 2, 2)
    y_coordinate_offsets = x_coordinate_offsets + 1
    x_coordinate_pointers = start_ptr + x_coordinate_offsets
    y_coordinate_pointers = start_ptr + y_coordinate_offsets


def triton_pdf_forwards(
    coordinates: torch.Tensor,
    params: StrokeParameters,
) -> torch.Tensor:
    assert coordinates.is_cuda and params.alpha.is_cuda, "all tensors must be on cuda"

    coordinates = coordinates.contiguous()
    center_x = params.center_x.contiguous()
    center_y = params.center_y.contiguous()
    rotation = params.rotation.contiguous()
    mu_r = params.mu_r.contiguous()
    sigma_r = params.sigma_r.contiguous()
    sigma_theta = params.sigma_theta.contiguous()
    color = params.color.contiguous()
    alpha = params.alpha.contiguous()

    n_strokes, _, n_coordinates = coordinates.shape

    coordinates = coordinates.view(-1)
    center_x = center_x.view(-1)
    center_y = center_y.view(-1)
    rotation = rotation.view(-1)
    mu_r = mu_r.view(-1)
    sigma_r = sigma_r.view(-1)
    sigma_theta = sigma_theta.view(-1)
    color = color.view(-1)
    alpha = alpha.view(-1)

    output = torch.empty(n_strokes, 1, n_coordinates, device=coordinates.device)
    output = output.view(-1)

    grid = (
        n_strokes,
        triton.cdiv(n_coordinates, BLOCK_SIZE),
    )

    _pdf_forwards[grid](
        coordinates_ptr=coordinates,
        center_x_ptr=center_x,
        center_y_ptr=center_y,
        rotation_ptr=rotation,
        mu_r_ptr=mu_r,
        sigma_r_ptr=sigma_r,
        sigma_theta_ptr=sigma_theta,
        color_ptr=color,
        alpha_ptr=alpha,
        maxes_ptr=output,
        N_STROKES=n_strokes,
        N_COORDINATES=n_coordinates,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    output = output.view(n_strokes, 1, n_coordinates)
