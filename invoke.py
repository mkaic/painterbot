import os
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image

from painterbot import optimize

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

params, renderer, loss_history, mae_history = optimize(
    target,
    n_groups=n_groups,
    n_strokes_per_group=n_strokes_per_group,
    iterations=iterations,
    show_inner_pbar=True,
    error_map_temperature=1.0,
)

canvas = torch.zeros(3, 512, 512, device=device)
result = renderer.render_timelapse_frames(
    canvas, params, Path("painting_timelapse_frames")
)

# run script to convert frames to video
os.system("./make_timelapses.sh")

params = params.cpu()

saved_params = Path("saved_params")
if not saved_params.exists():
    saved_params.mkdir()
torch.save(
    params,
    saved_params
    / f"{image_path.stem}_{n_groups}_{n_strokes_per_group}_{iterations}.pt",
)

# +
# 0.03159 for direct optimization of MAE. WAY faster than using MSSIM. 10 groups of 50, 200 iterations
# 0.03218 achieved by swapping MAE for MSE. okay, back to MAE
# 0.03178 achieved by going for 5 groups of 100 strokes instead of 10 groups of 50
# 0.03228 achieved with 1 group of 500 strokes
# 0.03193 achieved with 500 groups of 1 stroke
# -
