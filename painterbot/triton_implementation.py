import torch
import triton

from .triton_forward_kernels import _pdf_forward, _blend_forward
from .parameters import StrokeParameters
from typing import Tuple


class triton_pdf(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
        parameters: StrokeParameters,
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

        _pdf_forward[pdf_grid](
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

        ctx.save_for_backward(parameters)
        ctx.height = height
        ctx.width = width

        return strokes

    @staticmethod
    def backward(ctx, grad_output):
        pass


class triton_blend(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        canvas: torch.Tensor,
        strokes: torch.Tensor,
        parameters: StrokeParameters,
        KEEP_HISTORY: bool = True,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        n_strokes = parameters.n_strokes.item()
        _, height, width = canvas.shape

        canvas = canvas.contiguous()
        strokes = strokes.contiguous()
        color = parameters.color.contiguous()

        if KEEP_HISTORY:
            canvas_history = torch.empty(
                n_strokes, 3, height, width, device=canvas.device, dtype=canvas.dtype
            ).contiguous()
        else:
            canvas_history = None

        blend_grid = (height, width)

        _blend_forward[blend_grid](
            strokes_ptr=strokes,
            color_ptr=color,
            canvas_ptr=canvas,
            canvas_history_ptr=canvas_history,
            N_STROKES=n_strokes,
            HEIGHT=height,
            WIDTH=width,
            KEEP_HISTORY=KEEP_HISTORY,
        )

        canvas = canvas.view(3, height, width)

        ctx.save_for_backward(canvas_history)
        ctx.parameters = parameters
        ctx.strokes = strokes

        if KEEP_HISTORY:
            return canvas, canvas_history
        else:
            return canvas

    @staticmethod
    def backward(ctx, grad_output):
        pass


def triton_render(
    canvas: torch.Tensor, parameters: StrokeParameters, KEEP_HISTORY: bool = True
) -> Tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
    assert parameters.alpha.is_cuda, "parameter tensors must be on cuda"

    height, width = canvas.shape[-2:]
    device = canvas.device
    dtype = canvas.dtype

    strokes = triton_pdf(
        parameters=parameters,
        height=height,
        width=width,
        device=device,
        dtype=dtype,
    )

    canvas, canvas_history = triton_blend(
        canvas=canvas,
        strokes=strokes,
        parameters=parameters,
        KEEP_HISTORY=KEEP_HISTORY,
    )

    if KEEP_HISTORY:
        return (canvas.detach(), canvas_history)
    else:
        return canvas.detach()
