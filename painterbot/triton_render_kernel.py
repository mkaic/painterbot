import torch
import triton
import triton.language as tl

from .parameters import StrokeParameters

BLOCK_SIZE = 32
EPSILON = (1e-8,)


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
    N_COORDINATES,
    BLOCK_SIZE: tl.constexpr,
):
    stroke_id = tl.program_id(0)
    block_id = tl.program_id(1)

    stroke_offset = stroke_id * N_COORDINATES

    coord_offsets = (block_id * BLOCK_SIZE) + tl.arange(0, BLOCK_SIZE)
    coords_mask = coord_offsets < N_COORDINATES

    x_coord_pointers = coordinates_x_ptr + stroke_offset + coord_offsets
    y_coord_pointers = coordinates_y_ptr + stroke_offset + coord_offsets

    x_coords = tl.load(x_coord_pointers, mask=coords_mask)
    y_coords = tl.load(y_coord_pointers, mask=coords_mask)

    center_x = tl.load(center_x_ptr + stroke_id)
    center_y = tl.load(center_y_ptr + stroke_id)
    rotation = tl.load(rotation_ptr + stroke_id)
    mu_r = tl.load(mu_r_ptr + stroke_id)
    sigma_r = tl.load(sigma_r_ptr + stroke_id)
    sigma_theta = tl.load(sigma_theta_ptr + stroke_id)
    alpha = tl.load(alpha_ptr + stroke_id)

    cos_rot = tl.math.cos(rotation)
    sin_rot = tl.math.sin(rotation)

    offset_x = center_x - (cos_rot * mu_r)
    # the direction of the y-axis is inverted, so we add rather than subtract
    offset_y = center_y + (sin_rot * mu_r)

    x_coords = x_coords - offset_x
    y_coords = y_coords - offset_y

    # rotate coordinates
    x_coords = (x_coords * cos_rot) - (y_coords * sin_rot) + EPSILON
    y_coords = (x_coords * sin_rot) + (y_coords * cos_rot) + EPSILON

    r_coords = tl.sqrt((x_coords * x_coords) + (y_coords * y_coords))
    r_coords = r_coords - mu_r
    r_coords = r_coords * r_coords

    theta_coords = tl.math.atan2(y_coords, x_coords)
    theta_coords = theta_coords * theta_coords

    sigma_r = sigma_r * mu_r
    sigma_r = sigma_r * sigma_r * 2.0

    sigma_theta = sigma_theta * sigma_theta * 2.0

    r_coords = r_coords / (sigma_r + EPSILON)
    theta_coords = theta_coords / (sigma_theta + EPSILON)

    pdf = r_coords + theta_coords

    pdf = tl.math.exp(-1.0 * pdf)
    pdf = pdf * alpha

    tl.store(output_ptr + stroke_offset + coord_offsets, pdf, mask=coords_mask)


@triton.jit
def _blend_forwards(
    strokes_ptr,
    color_ptr,
    canvas_ptr,
    N_STROKES,
    N_COORDINATES,
    RETURN_LOSS: tl.constexpr,
    target_ptr=None,
    loss_ptr=None,
):
    pixel_id = tl.program_id(0)

    alpha_map_offsets = (tl.arange(N_STROKES) * N_COORDINATES) + pixel_id
    alpha_map_pointer_mask = alpha_map_offsets < (N_STROKES * N_COORDINATES)
    alpha_map_pointers = strokes_ptr + alpha_map_offsets
    alpha_map_values = tl.load(alpha_map_pointers, mask=alpha_map_pointer_mask)

    red_offsets = (3 * N_COORDINATES) + (pixel_id * 3) + 0
    red_pointer_mask = red_offsets < (3 * N_COORDINATES)
    canvas_red_pointers = canvas_ptr + red_offsets
    canvas_red_value = tl.load(canvas_red_pointers, mask=red_pointer_mask)

    green_offsets = (3 * N_COORDINATES) + (pixel_id * 3) + 1
    green_pointer_mask = green_offsets < (3 * N_COORDINATES)
    canvas_green_pointers = canvas_ptr + green_offsets
    canvas_green_value = tl.load(canvas_green_pointers, mask=green_pointer_mask)

    blue_offsets = (3 * N_COORDINATES) + (pixel_id * 3) + 2
    blue_pointer_mask = blue_offsets < (3 * N_COORDINATES)
    canvas_blue_pointers = canvas_ptr + blue_offsets
    canvas_blue_value = tl.load(canvas_blue_pointers, mask=blue_pointer_mask)

    if RETURN_LOSS:
        target_red_pointers = target_ptr + red_offsets
        target_red_value = tl.load(target_red_pointers, mask=red_pointer_mask)
        target_green_pointers = target_ptr + green_offsets
        target_green_value = tl.load(target_green_pointers, mask=green_pointer_mask)
        target_blue_pointers = target_ptr + blue_offsets
        target_blue_value = tl.load(target_blue_pointers, mask=blue_pointer_mask)

    for stroke_id in range(N_STROKES):
        color_offsets = (stroke_id * 3) + tl.arange(0, 3)
        color_pointers = color_ptr + color_offsets
        stroke_red_value, stroke_green_value, stroke_blue_value = tl.load(
            color_pointers
        )

        alpha_map_value = alpha_map_values[stroke_id]

        canvas_red_value = ((1.0 - alpha_map_value) * canvas_red_value) + (
            alpha_map_value * stroke_red_value
        )
        canvas_green_value = ((1.0 - alpha_map_value) * canvas_green_value) + (
            alpha_map_value * stroke_green_value
        )
        canvas_blue_value = ((1.0 - alpha_map_value) * canvas_blue_value) + (
            alpha_map_value * stroke_blue_value
        )

        if RETURN_LOSS:
            stroke_loss_offset = (stroke_id * N_COORDINATES) + pixel_id
            stroke_loss_pointer = loss_ptr + stroke_loss_offset
            stroke_loss_value = (
                tl.abs(canvas_red_value - target_red_value)
                + tl.abs(canvas_green_value - target_green_value)
                + tl.abs(canvas_blue_value - target_blue_value)
            )
            tl.store(stroke_loss_pointer, stroke_loss_value)

    tl.store(canvas_red_pointers, canvas_red_value, mask=red_pointer_mask)
    tl.store(canvas_green_pointers, canvas_green_value, mask=green_pointer_mask)
    tl.store(canvas_blue_pointers, canvas_blue_value, mask=blue_pointer_mask)


def triton_render_forward(
    coordinates: torch.Tensor,
    parameters: StrokeParameters,
    canvas: torch.Tensor,
    target: torch.Tensor = None,
) -> torch.Tensor:
    assert (
        coordinates.is_cuda and parameters.alpha.is_cuda
    ), "all tensors must be on cuda"

    RETURN_LOSS = target is not None

    n_strokes, _, n_coordinates = coordinates.shape

    x_coordinates = coordinates[:, 0]  # (N, 2, HW) -> (N, HW)
    y_coordinates = coordinates[:, 1]  # (N, 2, HW) -> (N, HW)
    x_coordinates = x_coordinates.contiguous().view(-1)
    y_coordinates = y_coordinates.contiguous().view(-1)
    center_x = parameters.center_x.contiguous().view(-1)
    center_y = parameters.center_y.contiguous().view(-1)
    rotation = parameters.rotation.contiguous().view(-1)
    mu_r = parameters.mu_r.contiguous().view(-1)
    sigma_r = parameters.sigma_r.contiguous().view(-1)
    sigma_theta = parameters.sigma_theta.contiguous().view(-1)
    alpha = parameters.alpha.contiguous().view(-1)

    strokes = torch.empty(n_strokes, n_coordinates, device=coordinates.device).view(-1)

    pdf_grid = (
        n_strokes,
        triton.cdiv(n_coordinates, BLOCK_SIZE),
    )

    _pdf_forwards[pdf_grid](
        coordinates_x_ptr=x_coordinates,
        coordinates_y_ptr=y_coordinates,
        center_x_ptr=center_x,
        center_y_ptr=center_y,
        rotation_ptr=rotation,
        mu_r_ptr=mu_r,
        sigma_r_ptr=sigma_r,
        sigma_theta_ptr=sigma_theta,
        alpha_ptr=alpha,
        output_ptr=strokes,
        N_COORDINATES=n_coordinates,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    canvas_shape = canvas.shape
    canvas = canvas.contiguous().view(-1)
    target = target.contiguous().view(-1)

    blend_grid = (n_coordinates,)

    if RETURN_LOSS:
        loss = torch.empty(n_strokes, n_coordinates, device=coordinates.device).view(-1)
    else:
        loss = None

    _blend_forwards[blend_grid](
        strokes_ptr=strokes,
        color_ptr=parameters.color,
        canvas_ptr=canvas,
        target_ptr=target,
        loss_ptr=loss,
        N_STROKES=n_strokes,
        N_COORDINATES=n_coordinates,
        RETURN_LOSS=RETURN_LOSS,
    )

    canvas = canvas.view(canvas_shape)

    if RETURN_LOSS:
        loss = loss.view(n_strokes, n_coordinates)
        loss = loss.mean(dim=1)

        return canvas.detach(), loss

    else:
        return canvas.detach()
