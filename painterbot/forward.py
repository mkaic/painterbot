import shutil
from pathlib import Path

import torch
from torchvision.transforms.functional import to_pil_image

from .parameters import StrokeParameters, split_stroke_parameters
from .render import render
from .ms_ssim import MS_SSIM

EPSILON = 1e-8
ms_ssim = MS_SSIM(data_range=1.0, size_average=True, channel=3)


def rgb_to_oklab(rgb: torch.Tensor) -> torch.Tensor:
    # https://bottosson.github.io/posts/oklab/

    batch, _, height, width = rgb.shape
    rgb = rgb.reshape(3, -1)
    rgb = rgb.clamp(0, 1) + EPSILON

    # sRGB to Linear sRGB
    rgb = torch.where(
        rgb > 0.04045,
        ((rgb + 0.055) / 1.055).pow(2.4),
        rgb / 12.92,
    )

    m1 = torch.tensor(
        [
            [0.4122214708, 0.5363325363, -0.051445992],
            [0.2119034982, 0.6806995451, 0.1073969566],
            [0.0883024619, 0.2817188376, 0.6299787005],
        ],
        device=rgb.device,
        dtype=rgb.dtype,
    )

    # Approximate cone response
    # (BHW x 3 x 3) @ (BHW x 3 x 1) -> (BHW x 3)
    lms = torch.matmul(
        m1,
        rgb,
    )
    # torch produces NaNs when you try to cube-root a negative number
    # so this sign workaround is necessary. and an epsilon, just in case
    lms = lms.sign() * (lms.abs() + EPSILON).pow(1 / 3)

    m2 = torch.tensor(
        [
            [0.2104542553, 0.7936177850, -0.0040720468],
            [1.9779984951, -2.4285922050, 0.4505937099],
            [0.0259040371, 0.7827717662, -0.8086757660],
        ],
        device=rgb.device,
        dtype=rgb.dtype,
    )

    oklab = torch.matmul(
        m2,
        lms,
    )

    oklab = oklab.view(batch, 3, height, width)

    return oklab


# import colour
# print(rgb_to_oklab(torch.tensor([1.0, 1.0, 0.0]).view(1, 3, 1, 1)))
# print(colour.convert([1.0, 1.0, 0.0], "sRGB", "Oklab"))


def loss_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # convert to perceptually uniform color space
    x = rgb_to_oklab(x)
    y = rgb_to_oklab(y)

    # ||L_r - L_t|| + distance(C_r, C_t)
    luminance_l2 = torch.mean(torch.square(x[:, 0, :, :] - y[:, 0, :, :]))
    chroma_squared_euclidean_distance = torch.mean(
        torch.sum(torch.square(x[:, 1:, :, :] - y[:, 1:, :, :]), dim=1)
    )

    ms_ssim_loss = -ms_ssim(x, y)

    return luminance_l2 + chroma_squared_euclidean_distance + ms_ssim_loss


loss_fn_compiled = torch.compile(loss_fn)


def forward(
    canvas: torch.Tensor,
    parameters: StrokeParameters,
    target: torch.Tensor = None,
    make_timelapse: Path = None,
):
    if target is not None:
        canvas, canvas_history = render(
            canvas=canvas,
            parameters=parameters,
            KEEP_HISTORY=True,
            COMPILED=True,
        )

        target = target.unsqueeze(0).expand(
            parameters.n_strokes, -1, -1, -1
        )  # (3 x H x W) -> (N x 3 x H x W)
        loss = loss_fn_compiled(canvas_history, target)

        return canvas.detach(), loss

    else:  # means we're in eval mode, so we don't need to compute the loss or have gradients
        with torch.no_grad():
            parameter_blocks_list = split_stroke_parameters(parameters, block_size=128)

            if make_timelapse is None:
                for parameter_block in parameter_blocks_list:
                    canvas = render(
                        canvas=canvas,
                        parameters=parameter_block,
                        KEEP_HISTORY=False,
                        COMPILED=False,
                    )
            else:
                make_timelapse = Path(make_timelapse)
                if make_timelapse.exists():
                    shutil.rmtree(make_timelapse)
                make_timelapse.mkdir()

                for i, parameter_block in enumerate(parameter_blocks_list):
                    canvas, canvas_history = render(
                        canvas=canvas,
                        parameters=parameter_block,
                        KEEP_HISTORY=True,
                        COMPILED=False,
                    )
                    for j, canvas_frame in enumerate(canvas_history):
                        to_pil_image(canvas_frame).save(
                            make_timelapse / f"{i:05}_{j:05}.jpg"
                        )

            return canvas.detach()
