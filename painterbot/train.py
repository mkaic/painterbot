import os
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image

from .painterbot import optimize, render_timelapse_frames

device = "cuda:0"
image_size = 512
image_path = Path("source_images/lisa.jpg")
target = Image.open(image_path).convert("RGB")

preprocessing = T.Compose(
    [
        T.PILToTensor(),
        T.Resize(image_size, antialias=True),
        T.CenterCrop(image_size),
    ]
)
target = preprocessing(target)
target = target.to(device)
target = target / 255

n_groups = 10
n_strokes_per_group = 50
iterations = 300

params, loss_history, mae_history = optimize(
    target,
    n_groups=n_groups,
    n_strokes_per_group=n_strokes_per_group,
    iterations=iterations,
    show_inner_pbar=True,
    error_map_temperature=1.0,
    log_every=30,
)

canvas = torch.zeros(3, 512, 512, device=device)
result = render_timelapse_frames(
    canvas, params, Path("timelapse_frames_painting"))

T.functional.to_pil_image(result).save("result.jpg")

# run script to convert frames to video
os.system("./make_timelapses.sh")

params = params.cpu()

saved_params = Path("saved_params")
if not saved_params.exists():
    saved_params.mkdir()

params.save(
    saved_params
    / f"{image_path.stem}_{n_groups}_{n_strokes_per_group}_{iterations}.pt",
)
