import torch
import triton
import triton.language as tl

from .parameters import StrokeParameters

BLOCK_SIZE = 32
EPSILON = 1e-8


@triton.jit
def _pdf_forward(
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
def _blend_forward(
    strokes_ptr,
    color_ptr,
    canvas_ptr,
    canvas_history_ptr,
    N_STROKES: tl.constexpr,
    HEIGHT: tl.constexpr,
    WIDTH: tl.constexpr,
    KEEP_HISTORY: tl.constexpr,
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

    for stroke_id in range(N_STROKES):
        if KEEP_HISTORY:  # save canvas history
            canvas_history_red_pointer = (
                canvas_history_ptr
                + (stroke_id * N_COORDINATES * 3)
                + (0 * N_COORDINATES)
                + pixel_id
            )
            canvas_history_green_pointer = (
                canvas_history_ptr
                + (stroke_id * N_COORDINATES * 3)
                + (1 * N_COORDINATES)
                + pixel_id
            )
            canvas_history_blue_pointer = (
                canvas_history_ptr
                + (stroke_id * N_COORDINATES * 3)
                + (2 * N_COORDINATES)
                + pixel_id
            )
            tl.store(
                canvas_history_red_pointer, canvas_red_value, mask=red_pointer_mask
            )
            tl.store(
                canvas_history_green_pointer,
                canvas_green_value,
                mask=green_pointer_mask,
            )
            tl.store(
                canvas_history_blue_pointer, canvas_blue_value, mask=blue_pointer_mask
            )

        stroke_color_offset = stroke_id * 3
        stroke_red_pointer = stroke_color_offset + 0
        stroke_green_pointer = stroke_color_offset + 1
        stroke_blue_pointer = stroke_color_offset + 2

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

    tl.store(canvas_red_pointer, canvas_red_value, mask=red_pointer_mask)
    tl.store(canvas_green_pointer, canvas_green_value, mask=green_pointer_mask)
    tl.store(canvas_blue_pointer, canvas_blue_value, mask=blue_pointer_mask)
