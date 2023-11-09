import torch
import triton
import triton.language as tl

from .parameters import StrokeParameters

BLOCK_SIZE = 32
EPSILON = 1e-8


@triton.jit
def _pdf_forwards(
    center_x_ptr,
    center_y_ptr,
    rotation_ptr,
    mu_r_ptr,
    sigma_r_ptr,
    sigma_theta_ptr,
    alpha_ptr,
    output_ptr,
    HEIGHT: tl.constexpr,
    WIDTH: tl.constexpr,
    N_STROKES: tl.constexpr,
    N_STROKES_POW2: tl.constexpr,
):
    N_COORDINATES = HEIGHT * WIDTH

    stroke_offsets = tl.arange(0, N_STROKES_POW2)
    layer_offsets = stroke_offsets * N_COORDINATES
    layer_mask = stroke_offsets < N_STROKES

    row_id = tl.program_id(0)
    column_id = tl.program_id(1)

    row_offset = row_id * WIDTH
    column_offset = column_id

    coord_offset = row_offset + column_offset

    x_coord = column_id / HEIGHT
    y_coord = row_id / HEIGHT

    center_x = tl.load(center_x_ptr + stroke_offsets, mask=layer_mask)
    center_y = tl.load(center_y_ptr + stroke_offsets, mask=layer_mask)
    rotation = tl.load(rotation_ptr + stroke_offsets, mask=layer_mask)
    mu_r = tl.load(mu_r_ptr + stroke_offsets, mask=layer_mask)
    sigma_r = tl.load(sigma_r_ptr + stroke_offsets, mask=layer_mask)
    sigma_theta = tl.load(sigma_theta_ptr + stroke_offsets, mask=layer_mask)
    alpha = tl.load(alpha_ptr + stroke_offsets, mask=layer_mask)

    cos_rot = tl.math.cos(rotation)
    sin_rot = tl.math.sin(rotation)

    offset_x = center_x - (cos_rot * mu_r)
    # the direction of the y-axis is inverted, so we add rather than subtract
    offset_y = center_y + (sin_rot * mu_r)

    x_coord = x_coord - offset_x
    y_coord = y_coord - offset_y

    # rotate coordinates
    x_coord = (x_coord * cos_rot) - (y_coord * sin_rot) + EPSILON
    y_coord = (x_coord * sin_rot) + (y_coord * cos_rot) + EPSILON

    r_coord = tl.sqrt((x_coord * x_coord) + (y_coord * y_coord))
    r_coord = r_coord - mu_r
    r_coord = r_coord * r_coord

    theta_coord = tl.math.atan2(y_coord, x_coord)
    theta_coord = theta_coord * theta_coord

    sigma_r = sigma_r * mu_r
    sigma_r = sigma_r * sigma_r * 2.0

    sigma_theta = sigma_theta * sigma_theta * 2.0

    r_coord = r_coord / (sigma_r + EPSILON)
    theta_coord = theta_coord / (sigma_theta + EPSILON)

    pdf = r_coord + theta_coord

    pdf = tl.math.exp(-1.0 * pdf)
    pdf = pdf * alpha

    store_pointers = output_ptr + layer_offsets

    store_pointers = store_pointers + coord_offset

    tl.store(store_pointers, pdf, mask=layer_mask)


@triton.jit
def _blend_forwards(
    strokes_ptr,
    color_ptr,
    canvas_ptr,
    target_ptr,
    loss_ptr,
    N_STROKES: tl.constexpr,
    HEIGHT: tl.constexpr,
    WIDTH: tl.constexpr,
    RETURN_LOSS: tl.constexpr,
):
    N_COORDINATES = HEIGHT * WIDTH

    row_id = tl.program_id(0)
    column_id = tl.program_id(1)

    row_offset = row_id * WIDTH
    column_offset = column_id

    pixel_id = row_offset + column_offset

    red_offsets = pixel_id + (0 * N_COORDINATES)
    red_pointer_mask = red_offsets < (3 * N_COORDINATES)
    canvas_red_pointer = canvas_ptr + red_offsets
    canvas_red_value = tl.load(canvas_red_pointer, mask=red_pointer_mask)

    green_offsets = pixel_id + (1 * N_COORDINATES)
    green_pointer_mask = green_offsets < (3 * N_COORDINATES)
    canvas_green_pointer = canvas_ptr + green_offsets
    canvas_green_value = tl.load(canvas_green_pointer, mask=green_pointer_mask)

    blue_offsets = pixel_id + (2 * N_COORDINATES)
    blue_pointer_mask = blue_offsets < (3 * N_COORDINATES)
    canvas_blue_pointer = canvas_ptr + blue_offsets
    canvas_blue_value = tl.load(canvas_blue_pointer, mask=blue_pointer_mask)

    if RETURN_LOSS:
        target_red_pointers = target_ptr + red_offsets
        target_red_value = tl.load(target_red_pointers, mask=red_pointer_mask)
        target_green_pointers = target_ptr + green_offsets
        target_green_value = tl.load(target_green_pointers, mask=green_pointer_mask)
        target_blue_pointers = target_ptr + blue_offsets
        target_blue_value = tl.load(target_blue_pointers, mask=blue_pointer_mask)

    for stroke_id in range(N_STROKES):
        color_offset = stroke_id * 3
        stroke_red_pointer = color_offset + 0
        stroke_green_pointer = color_offset + 1
        stroke_blue_pointer = color_offset + 2

        stroke_red_value = tl.load(color_ptr + stroke_red_pointer)
        stroke_green_value = tl.load(color_ptr + stroke_green_pointer)
        stroke_blue_value = tl.load(color_ptr + stroke_blue_pointer)

        alpha_map_offset = (stroke_id * N_COORDINATES) + pixel_id
        alpha_map_pointer_mask = alpha_map_offset < (N_STROKES * N_COORDINATES)
        alpha_map_pointer = strokes_ptr + alpha_map_offset
        alpha_map_value = tl.load(alpha_map_pointer, mask=alpha_map_pointer_mask)

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

    tl.store(canvas_red_pointer, canvas_red_value, mask=red_pointer_mask)
    tl.store(canvas_green_pointer, canvas_green_value, mask=green_pointer_mask)
    tl.store(canvas_blue_pointer, canvas_blue_value, mask=blue_pointer_mask)


def triton_pdf_forward(
    parameters: StrokeParameters,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    n_strokes = parameters.n_strokes.item()
    n_strokes_pow2 = triton.next_power_of_2(n_strokes)

    center_x = parameters.center_x.contiguous()
    center_y = parameters.center_y.contiguous()
    rotation = parameters.rotation.contiguous()
    mu_r = parameters.mu_r.contiguous()
    sigma_r = parameters.sigma_r.contiguous()
    sigma_theta = parameters.sigma_theta.contiguous()
    alpha = parameters.alpha.contiguous()

    strokes = torch.empty(n_strokes, 1, height, width, device=device, dtype=dtype)

    pdf_grid = (
        height,
        width,
    )

    _pdf_forwards[pdf_grid](
        center_x_ptr=center_x,
        center_y_ptr=center_y,
        rotation_ptr=rotation,
        mu_r_ptr=mu_r,
        sigma_r_ptr=sigma_r,
        sigma_theta_ptr=sigma_theta,
        alpha_ptr=alpha,
        output_ptr=strokes,
        HEIGHT=height,
        WIDTH=width,
        N_STROKES=n_strokes,
        N_STROKES_POW2=n_strokes_pow2,
    )

    return strokes


def triton_blend_forward(
    canvas: torch.Tensor,
    target: torch.Tensor,
    strokes: torch.Tensor,
    parameters: StrokeParameters,
) -> torch.Tensor:
    n_strokes = parameters.n_strokes.item()
    _, height, width = canvas.shape

    canvas = canvas.contiguous()
    strokes = strokes.contiguous()
    color = parameters.color.contiguous()
    if target is not None:
        target = target.contiguous()
        loss = torch.empty_like(strokes)
    else:
        loss = None

    blend_grid = (height, width)

    _blend_forwards[blend_grid](
        strokes_ptr=strokes,
        color_ptr=color,
        canvas_ptr=canvas,
        target_ptr=target,
        loss_ptr=loss,
        N_STROKES=n_strokes,
        HEIGHT=height,
        WIDTH=width,
        RETURN_LOSS=target is not None,
    )

    canvas = canvas.view(3, height, width)
    return canvas, loss


def triton_render_forward(
    parameters: StrokeParameters,
    canvas: torch.Tensor,
    target: torch.Tensor = None,
) -> torch.Tensor:
    assert parameters.alpha.is_cuda, "parameter tensors must be on cuda"
    n_strokes = parameters.n_strokes.item()

    height, width = canvas.shape[-2:]
    device = canvas.device
    dtype = canvas.dtype

    strokes = triton_pdf_forward(
        parameters=parameters,
        height=height,
        width=width,
        device=device,
        dtype=dtype,
    )

    canvas, loss = triton_blend_forward(
        canvas=canvas,
        target=target,
        strokes=strokes,
        parameters=parameters,
    )

    intermediates = {
        "loss_unreduced": loss,
        "strokes": strokes,
        "canvas_untouched": canvas,
    }

    if target is not None:
        loss = loss.mean() / n_strokes
        return (canvas.detach(), loss, intermediates)
    else:
        return canvas.detach()
