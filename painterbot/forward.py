import shutil
from pathlib import Path
from typing import Callable

import torch
from torchvision.transforms.functional import to_pil_image

from .parameters import StrokeParameters, split_stroke_parameters


def elementwise_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    abs_error = torch.abs(x - y)
    return abs_error


def forward(
    canvas: torch.Tensor,
    parameters: StrokeParameters,
    render_fn: Callable,
    target: torch.Tensor = None,
    make_timelapse: Path = None,
):
    if target is not None:
        canvas, canvas_history = render_fn(
            canvas=canvas,
            parameters=parameters,
            KEEP_HISTORY=True,
        )

        target = target.unsqueeze(0)
        target = target.repeat(parameters.n_strokes, 1, 1, 1)
        loss = elementwise_loss(canvas_history, target)
        loss = torch.mean(loss)

        return canvas.detach(), loss

    else:  # means we're in eval mode, so we don't need to compute the loss or have gradients
        with torch.no_grad():
            parameter_blocks_list = split_stroke_parameters(parameters, block_size=128)

            if make_timelapse is None:
                for parameter_block in parameter_blocks_list:
                    canvas, canvas_history = render_fn(
                        canvas=canvas,
                        parameters=parameter_block,
                        KEEP_HISTORY=False,
                    )

            if make_timelapse is not None:
                make_timelapse = Path(make_timelapse)
                if make_timelapse.exists():
                    shutil.rmtree(make_timelapse)
                make_timelapse.mkdir()

                for i, parameter_block in enumerate(parameter_blocks_list):
                    canvas, canvas_history = render_fn(
                        canvas=canvas,
                        parameters=parameter_block,
                        KEEP_HISTORY=True,
                    )
                    for j, canvas_frame in enumerate(canvas_history):
                        to_pil_image(canvas_frame).save(
                            make_timelapse / f"{i:05}_{j:05}.jpg"
                        )

            return canvas.detach()
