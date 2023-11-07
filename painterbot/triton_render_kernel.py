import torch
import triton
import triton.language as tl

from .parameters import StrokeParameters

BLOCK_SIZE = 128


@triton.jit
def _pdf_forwards(
    coordinates_x_ptr,
    coordinates_y_ptr,
    center_x_ptr,
    center_y_ptr,
    rotation_ptr,
    mu_r_ptr,
    sigma_r_ptr,
    sigma_theta_ptr,
    alpha_ptr,
    output_ptr,
    N_COORDINATES: int,
    BLOCK_SIZE: tl.constexpr,
    EPSILON: float = 1e-8,
):
    stroke_id = tl.program_id(0)
    block_id = tl.program_id(1)

    stroke_offset = stroke_id * N_COORDINATES

    coord_offsets = (block_id * BLOCK_SIZE) + tl.arange(0, BLOCK_SIZE)
    x_coord_pointers = coordinates_x_ptr + stroke_offset + coord_offsets
    y_coord_pointers = coordinates_y_ptr + stroke_offset + coord_offsets

    coords_mask = coord_offsets < N_COORDINATES

    x_coords = tl.load(x_coord_pointers, mask=coords_mask)
    y_coords = tl.load(y_coord_pointers, mask=coords_mask)

    center_x = tl.load(center_x_ptr + stroke_id)
    center_y = tl.load(center_y_ptr + stroke_id)
    rotation = tl.load(rotation_ptr + stroke_id)
    mu_r = tl.load(mu_r_ptr + stroke_id)
    sigma_r = tl.load(sigma_r_ptr + stroke_id)
    sigma_theta = tl.load(sigma_theta_ptr + stroke_id)
    alpha = tl.load(alpha_ptr + stroke_id)

    cos = tl.cos(rotation)
    sin = tl.sin(rotation)

    offset_x = center_x - (cos * mu_r)
    # the direction of the y-axis is inverted, so we add rather than subtract
    offset_y = center_y + (sin * mu_r)

    x_coords = x_coords - offset_x
    y_coords = y_coords - offset_y

    # rotate coordinates
    x_coords = (x_coords * cos) - (y_coords * sin)
    y_coords = (x_coords * sin) + (y_coords * cos)

    r_coords = tl.sqrt(x_coords * x_coords + y_coords * y_coords)
    r_coords = r_coords - mu_r
    r_coords = r_coords * r_coords

    theta_coords = tl.math.atan2(y_coords, x_coords)
    theta_coords = theta_coords * theta_coords

    sigma_r = sigma_r * mu_r
    sigma_r = sigma_r * sigma_r * 2

    sigma_theta = sigma_theta * sigma_theta * 2

    r_coords = r_coords / (sigma_r + EPSILON)
    theta_coords = theta_coords / (sigma_theta + EPSILON)

    pdf = tl.exp(-(r_coords + theta_coords))
    pdf = pdf * alpha

    tl.store(output_ptr + stroke_offset + coord_offsets, pdf, mask=coords_mask)


def triton_pdf_forwards(
    coordinates: torch.Tensor,
    parameters: StrokeParameters,
) -> torch.Tensor:
    assert (
        coordinates.is_cuda and parameters.alpha.is_cuda
    ), "all tensors must be on cuda"

    n_strokes, _, n_coordinates = coordinates.shape

    x_coordinates = coordinates[:, 0]  # (N, 2, HW) -> (N, HW
    y_coordinates = coordinates[:, 1]  # (N, 2, HW) -> (N, HW)
    x_coordinates = x_coordinates.contiguous()
    y_coordinates = y_coordinates.contiguous()
    center_x = parameters.center_x.contiguous()
    center_y = parameters.center_y.contiguous()
    rotation = parameters.rotation.contiguous()
    mu_r = parameters.mu_r.contiguous()
    sigma_r = parameters.sigma_r.contiguous()
    sigma_theta = parameters.sigma_theta.contiguous()
    color = parameters.color.contiguous()
    alpha = parameters.alpha.contiguous()

    x_coordinates = x_coordinates.view(-1)
    y_coordinates = y_coordinates.view(-1)
    center_x = center_x.view(-1)
    center_y = center_y.view(-1)
    rotation = rotation.view(-1)
    mu_r = mu_r.view(-1)
    sigma_r = sigma_r.view(-1)
    sigma_theta = sigma_theta.view(-1)
    color = color.view(-1)
    alpha = alpha.view(-1)

    output = torch.empty(n_strokes, n_coordinates, device=coordinates.device)
    output = output.view(-1)

    grid = (
        n_strokes,
        triton.cdiv(n_coordinates, BLOCK_SIZE),
    )

    print(grid)

    _pdf_forwards[grid](
        coordinates_x_ptr=x_coordinates,
        coordinates_y_ptr=y_coordinates,
        center_x_ptr=center_x,
        center_y_ptr=center_y,
        rotation_ptr=rotation,
        mu_r_ptr=mu_r,
        sigma_r_ptr=sigma_r,
        sigma_theta_ptr=sigma_theta,
        alpha_ptr=alpha,
        output_ptr=output,
        N_COORDINATES=n_coordinates,
        BLOCK_SIZE=BLOCK_SIZE,
        EPSILON=1e-8,
    )

    output = output.view(n_strokes, 1, n_coordinates)
    return output
