import os
import shutil
from pathlib import Path

import torch
from torchvision.transforms.functional import to_pil_image
from tqdm.auto import tqdm

from .forward import forward
from .parameters import StrokeParameters, concat_stroke_parameters
from .preprocessing import load_image


def optimize(
    target: torch.Tensor,
    n_groups: int = 10,
    n_strokes_per_group: int = 50,
    iterations: int = 300,
    lr: float = 0.01,
    show_inner_pbar: bool = True,
    log_every: int = 15,
):
    if n_groups == 1:
        show_inner_pbar = True

    device = target.device
    width, height = target.shape[-2:]
    width_to_height_ratio = width / height

    canvas = torch.zeros_like(target)

    frozen_params: StrokeParameters = None

    output_path = Path("timelapse_frames_optimization")
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir()

    loss_history = []
    mae_history = []

    outer_pbar = tqdm(range(n_groups))
    outer_pbar.set_description(f"Loss=?, MAE=? | Group 1/{n_groups}")

    for i in outer_pbar:
        active_params = StrokeParameters(
            n_strokes=n_strokes_per_group,
            width_to_height_ratio=width_to_height_ratio,
        ).to(device)

        active_params.smart_init(
            target=target,
            canvas=canvas,
        )

        optimizer = torch.optim.Adam(
            active_params.parameters(), lr=lr, betas=(0.8, 0.9)
        )

        if show_inner_pbar:
            inner_iterator = tqdm(range(iterations), leave=False)
        else:
            inner_iterator = range(iterations)

        for j in inner_iterator:
            result, loss = forward(
                canvas=canvas,
                parameters=active_params,
                target=target,
            )

            loss.backward()

            optimizer.step()

            optimizer.zero_grad(set_to_none=True)

            active_params.clamp_parameters()

            mae = torch.mean(torch.abs(result - target))

            if show_inner_pbar:
                description = f"Loss={loss:.5f}, MAE={mae:.5f}"
                inner_iterator.set_description(description)

            if log_every is not None and j % log_every == 0:
                to_pil_image(result).save(output_path / f"{i:05}_{j:06}.jpg")

            loss_history.append(loss.detach().cpu().item())
            mae_history.append(mae.detach().cpu().item())

        canvas = forward(
            canvas=canvas,
            parameters=active_params,
        )
        if i == 0:
            frozen_params = active_params
        else:
            frozen_params = concat_stroke_parameters([frozen_params, active_params])

        outer_pbar.set_description(
            f"Loss={loss:.5f}, MAE={mae:.5f} | Group {i+1}/{n_groups}"
        )
    return frozen_params, loss_history, mae_history


if __name__ == "__main__":
    device = "cuda:0"
    image_path = Path("source_images/lisa.jpg")

    target = load_image(
        image_path=image_path,
        image_size=512,
        device=device,
        dtype=torch.bfloat16,
    )

    n_groups = 16
    n_strokes_per_group = 64
    iterations = 300

    parameters, loss_history, mae_history = optimize(
        target,
        n_groups=n_groups,
        n_strokes_per_group=n_strokes_per_group,
        iterations=iterations,
        show_inner_pbar=True,
        log_every=30,
    )

    canvas = torch.zeros_like(target, device=device)
    forward(
        canvas=canvas,
        parameters=parameters,
        make_timelapse=Path("timelapse_frames_painting"),
    )

    canvas = torch.zeros_like(target, device=device)
    result = forward(canvas=canvas, parameters=parameters)
    result = to_pil_image(result)
    result.save("result.jpg")

    # run script to convert frames to video
    os.system("./make_timelapses.sh")

    parameters = parameters.cpu()
    saved_params = Path("saved_params")
    if not saved_params.exists():
        saved_params.mkdir()

    parameters.save(
        saved_params
        / f"{image_path.stem}_{n_groups}_{n_strokes_per_group}_{iterations}.pt",
    )
