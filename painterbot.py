# +
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

torch.cuda.empty_cache()

from PIL import Image
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from pathlib import Path
import shutil
import numpy as np
import time
import math

from torchmetrics.functional.image import image_gradients


# +
def batch_unravel_index_xy(indices, shape, device):
    output = torch.zeros(2, len(indices), dtype=torch.long, device=device)

    # x coordinates (which column the flat index is in)
    # are found by moduloing it by the length of a row
    output[0, :] = indices % shape[-1]

    # y coordinates (which row the flat index is in) are
    # found by integer-dividing it by the length of a column
    output[1, :] = torch.div(indices, shape[-2], rounding_mode="floor")

    return output


class StrokeGroup(nn.Module):
    def __init__(self, group_number, n_strokes, dtype):
        super().__init__()

        self.group_number = group_number

        self.offset_x = torch.ones(n_strokes, 1) * 0.5
        self.offset_x = nn.Parameter(self.offset_x)

        self.offset_y = torch.ones(n_strokes, 1) * 0.5
        self.offset_y = nn.Parameter(self.offset_y)

        self.mu_r = torch.ones(n_strokes, 1) * 0.2
        self.mu_r = nn.Parameter(self.mu_r)

        self.mu_theta = torch.randn(n_strokes, 1)
        self.mu_theta = nn.Parameter(self.mu_theta)

        self.sigma_r = torch.ones(n_strokes, 1) * 0.1
        self.sigma_r = nn.Parameter(self.sigma_r)

        self.sigma_theta = torch.ones(n_strokes, 1) * 0.1
        self.sigma_theta = nn.Parameter(self.sigma_theta)

        self.color = torch.ones(n_strokes, 3, 1, 1)
        self.color = self.color * torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        self.color = self.color + torch.randn_like(self.color) * 0.1
        self.color = nn.Parameter(self.color)

        self.alpha = torch.ones(n_strokes) * 0.01
        self.alpha = self.alpha.view(n_strokes, 1, 1, 1)
        self.alpha = nn.Parameter(self.alpha)

        self.epsilon = 1e-6

        self.n_strokes = n_strokes
                
        self.mae = nn.L1Loss().to(dtype)

    def loss_fn(self, x, y):
        
        l2 = self.mae(x,y)
        
        loss = l2
        
        return loss

    def calculate_strokes(self, canvas):
        device = canvas.device

        # Aspect ratio should always be >1:1, never 1:<1
        height, width = canvas.shape[-2:]
        divisor = min(height, width)

        w = torch.linspace(0, width - 1, width, device=device) / divisor
        h = torch.linspace(0, height - 1, height, device=device) / divisor

        x, y = torch.meshgrid(w, h, indexing="xy")

        # Reshape coordinates into a list of XY pairs, then duplicate it
        # once for each stroke in the group
        cartesian_coordinates = torch.stack([x, y], dim=0).view(1, 2, -1)
        cartesian_coordinates = cartesian_coordinates.repeat(
            self.n_strokes, 1, 1
        )  # (1 x 2 x M) -> (N x 2 x M)

        # Offset coordinates to change where the center of the
        # polar coordinates is placed
        cartesian_offset = torch.stack(
            [self.offset_x, self.offset_y], dim=1
        )  # 2x (N x 1) -> (N x 2 x 1)
        cartesian_coordinates = cartesian_coordinates - cartesian_offset

        # Convert to polar coordinates and apply polar-space offsets
        r = torch.sqrt(
            torch.sum(
                torch.square(cartesian_coordinates + self.epsilon),
                dim=1,  # (N x 2 x M) -> (N x M)
            )
        )
        r = r - self.mu_r

        theta = torch.atan2(
            cartesian_coordinates[:, 1] + self.epsilon,
            cartesian_coordinates[:, 0] + self.epsilon,
        )  # -> (N x M)
        theta = theta + torch.pi
        theta = theta - (self.mu_theta * torch.pi)

        # Simplified Gaussian PDF function:
        # e ^ (
        #    -1
        #     * (
        #         ((x - mu_x) ^ 2 / (2 * sigma_x ^ 2))
        #         + ((y - mu_y) ^ 2 / (2 * sigma_y ^ 2))
        #     )
        # )

        r = torch.square(r)

        # These 3 coordinate grids together cover the full range of theta
        # values I expect to evaluate the PDF on, so by evaluating it
        # separately on each grid and then averaging the results, I get
        # smooth strokes with no discontinuities!

        theta_top = torch.square(theta + (2 * torch.pi))
        theta_middle = torch.square(theta)
        theta_bottom = torch.square(theta - (2 * torch.pi))

        # sigma_r is expressed as a fraction of the radius instead of
        # an absolute quantity
        sigma_r = self.sigma_r * self.mu_r
        sigma_theta = self.sigma_theta * torch.pi
        sigmas = torch.stack([sigma_r, sigma_theta], dim=1).view(self.n_strokes, 2, 1)

        # And finally, I'll finalize each of the 3 layers of the PDF
        pdfs = []
        for theta_layer in [theta_bottom, theta_middle, theta_top]:
            polar_coordinates = torch.stack([r, theta_layer], dim=1)
            polar_coordinates = polar_coordinates / (
                2 * torch.square(sigmas) + self.epsilon
            )
            pdf = torch.exp(-1 * torch.sum(polar_coordinates, dim=1))
            pdfs.append(pdf)

        # Stack all three theta-slices of the PDF
        pdf = torch.stack(pdfs, dim=0)  # 3 x (N x HW) -> (3 x N x HW)
        # And then average them
        pdf = pdf.mean(dim=0)  # (3 x N x HW) -> (N x HW)

        # Now map the PDF to the range [0, alpha]
        maxes, _ = pdf.max(dim=-1)
        maxes = maxes.view(self.n_strokes, 1)
        pdf = pdf / (maxes + self.epsilon)

        pdf = pdf.view(self.n_strokes, 1, height, width)
        pdf = pdf * self.alpha
        return pdf

    def render_stroke(self, stroke, color, canvas):
        # Use the stroke as an alpha to blend between the canvas and a color
        canvas = ((torch.ones_like(stroke) - stroke) * canvas) + (stroke * color)
        # Clamp canvas to valid RGB color space.
        canvas = canvas.clamp(0, 1)

        return canvas

    def forward(self, canvas, target):
        strokes = self.calculate_strokes(canvas)  # (N x 1 x H x W)

        loss = 0.0
        for stroke, color in zip(strokes, self.color):
            canvas = self.render_stroke(stroke=stroke, color=color, canvas=canvas)

            loss += self.loss_fn(canvas.unsqueeze(0), target.unsqueeze(0))

        loss = loss / self.n_strokes

        return canvas.detach(), loss

    def clamp_parameters(self, canvas):
        # Aspect ratio should always be >1:1, never 1:<1
        height, width = canvas.shape[-2:]
        divisor = min(height, width)
        height_max = height / divisor
        width_max = width / divisor

        # This offset needs to be able to be bigger than the size of the canvas
        # because it's not describing the position of the stroke, it's describing
        # the point around which the stroke curves. So if we want the algorithm
        # to be able to have straight-ish strokes sometimes, it needs to be able
        # to expand the stroke curvature radius until it's big enough that the
        # curvature becomes negligible.
        self.offset_x.data.clamp_(-2.0 * width_max, 2.0 * width_max)
        self.offset_y.data.clamp_(-2.0 * height_max, 2.0 * height_max)

        self.mu_r.data.clamp_(0.05, 2.0)
        self.mu_theta.data.clamp_(0.0, 1.0)

        # Guassian gets weirdly pinched if it gets too close to the pole
        self.sigma_r.data.clamp_(0.001, 0.2)

        # Strokes can only ever curve a certain amount to avoid a bunch of
        # obvious perfect circles showing up in the painting
        self.sigma_theta.data.clamp_(0.001, 0.15)

        self.alpha.data.clamp_(0.01, 1.0)
        self.color.data.clamp_(0.0, 1.0)

    def set_parameters(
        self,
        canvas,
        offset_x=None,
        offset_y=None,
        mu_r=None,
        mu_theta=None,
        sigma_r=None,
        sigma_theta=None,
        color=None,
        alpha=None,
    ):
        if offset_x is not None:
            offset_x = offset_x.detach().clone()
            self.offset_x.data = offset_x
        if offset_y is not None:
            offset_y = offset_y.detach().clone()
            self.offset_y.data = offset_y

        if mu_r is not None:
            mu_r = mu_r.detach().clone()
            self.mu_r.data = mu_r
        if mu_theta is not None:
            mu_theta = mu_theta.detach().clone()
            self.mu_theta.data = mu_theta

        if sigma_r is not None:
            sigma_r = sigma_r.detach().clone()
            self.sigma_r.data = sigma_r
        if sigma_theta is not None:
            sigma_theta = sigma_theta.detach().clone()
            self.sigma_theta.data = sigma_theta

        if color is not None:
            color = color.detach().clone()
            self.color.data = color
        if alpha is not None:
            alpha = alpha.detach().clone()
            self.alpha.data = alpha

        self.clamp_parameters(canvas)

    def smart_initialize(self, target, canvas, temperature=1.0):
        device = target.device
        height, width = target.shape[-2:]
        divisor = min(height, width)

        error_map = torch.mean(torch.abs(target - canvas), dim=0)  # (3, H, W) -> (H, W)

        # Unravel the error map, then softmax to get probability distribution over
        # the pixel locations, then sample from it
        flat_error_pdf = nn.functional.softmax(
            error_map.view(-1) / temperature, dim=0
        )  # (H, W) -> (H*W)

        n_samples = self.n_strokes

        coordinates = torch.multinomial(flat_error_pdf, n_samples)  # (N, 1)

        coordinates = batch_unravel_index_xy(
            coordinates, (height, width), device=device
        ).unsqueeze(
            -1
        )  # (N, 1) -> (2, N, 1)

        # As the error of the reconstruction decreases, so does the
        # starting stroke size.
        mae = torch.mean(error_map)
        sigma_r = mae.expand(self.n_strokes, 1)
        sigma_theta = mae.expand(self.n_strokes, 1)

        offset_x, offset_y = coordinates

        color = target[:, offset_y, offset_x].view(self.n_strokes, 3, 1, 1)

        self.set_parameters(
            offset_x=offset_x / divisor,
            offset_y=offset_y / divisor,
            sigma_r=sigma_r,
            sigma_theta=sigma_theta,
            color=color,
            canvas=canvas,
        )

    def render_timelapse_frames(self, canvas, output_path):
        assert isinstance(output_path, Path)

        with torch.no_grad():
            strokes = self.calculate_strokes(canvas)

            for i, (stroke, color) in enumerate(zip(strokes, self.color)):
                canvas = self.render_stroke(stroke=stroke, color=color, canvas=canvas)
                T.functional.to_pil_image(canvas.cpu()).save(
                    output_path / f"{self.group_number:05}_{i:05}.jpg"
                )

            return canvas


class Painting(nn.Module):
    def __init__(self, target, n_groups):
        super().__init__()

        self.n_groups = n_groups
        self.stroke_groups = nn.ModuleList()
        self.active_group = None
        self.target = target
        self.mae = nn.L1Loss()

    def forward(self, canvas):
        if self.active_group is not None:
            active_group = self.stroke_groups[self.active_group]
            canvas, loss = active_group(canvas, self.target)
        else:
            for group in self.stroke_groups:
                canvas, loss = group(canvas)

        mae = self.mae(self.target.unsqueeze(0), canvas.unsqueeze(0)).detach()

        return canvas, loss, mae

    def render_timelapse_frames(self, canvas, output_path):
        output_path = Path(output_path)
        if output_path.exists():
            shutil.rmtree(output_path)
        output_path.mkdir()

        with torch.no_grad():
            for i, group in enumerate(self.stroke_groups):
                canvas = group.render_timelapse_frames(canvas, output_path)

            plt.figure(figsize=(10, 10))
            plt.imshow(canvas.cpu().permute(1, 2, 0))

            return canvas


def optimize_grouped_painting(
    target,
    n_groups,
    n_strokes_per_group,
    iterations,
    lr,
    show_inner_pbar=True,
    error_map_temperature=1.0,
    log_every=10,
    dtype=torch.float,
):
    if n_groups == 1:
        show_inner_pbar = True

    device = target.device

    painting = Painting(
        n_groups=n_groups,
        target=target,
    ).to(device).to(dtype)

    canvas = torch.zeros_like(target)

    output_path = Path("optimization_timelapse_frames")
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir()

    error_map_path = Path("error_map_timelapse_frames")
    if error_map_path.exists():
        shutil.rmtree(error_map_path)
    error_map_path.mkdir()

    loss = 0.0
    mae = 1.0

    loss_history = []
    mae_history = []

    outer_pbar = tqdm(range(n_groups))
    outer_pbar.set_description(
        f"MAE = 1.0 | Loss = 0.0 | Optimizing Stroke Group: 1 of {n_groups}"
    )

    for group_number in outer_pbar:
        painting.active_group = group_number
        new_stroke_group = StrokeGroup(group_number=group_number, n_strokes=n_strokes_per_group, dtype=dtype).to(device).to(dtype)
        new_stroke_group.smart_initialize(
            target, canvas, error_map_temperature
        )
        painting.stroke_groups.append(
            new_stroke_group
        )

        optimizer = torch.optim.Adam(painting.parameters(), lr=lr, betas=(0.8, 0.9))

        adaptive_iterations = max(iterations - (50 * group_number), 50)
        if show_inner_pbar:
            inner_iterator = tqdm(range(adaptive_iterations), leave=False)
        else:
            inner_iterator = range(adaptive_iterations)

        for i in inner_iterator:
            result, loss, mae = painting(canvas)

            loss.backward()

            optimizer.step()

            optimizer.zero_grad(set_to_none=True)

            painting.stroke_groups[group_number].clamp_parameters(canvas)

            if show_inner_pbar:
                description = f"MAE = {mae:.5f} | Loss = {loss:.5f} "
                inner_iterator.set_description(description)

            if i % log_every == 0:
                T.functional.to_pil_image(result).save(
                    output_path / f"{group_number:05}_{i:06}.jpg"
                )
                T.functional.to_pil_image((target - result).abs()).save(
                    error_map_path / f"{group_number:05}_{i:06}.jpg"
                )

            loss_history.append(loss.detach().cpu().item())
            mae_history.append(mae.detach().cpu().item())

        canvas, _, _ = painting(canvas)

        outer_pbar.set_description(
            f"MAE = {mae:.5f} | Loss = {loss:.5f} | Optimizing Stroke Group: {group_number+1} of {n_groups}"
        )

    painting.active_group = None
    return painting, loss_history, mae_history


# print("format")
# pg = PolarGaussian().to("cuda:1")
# canvas = torch.zeros(3, 256, 256, device="cuda:1")
# result = pg(canvas)
# plt.figure(figsize=(10, 10))
# plt.imshow(result.detach().cpu().permute(1, 2, 0))
"format"
# +
# torch.autograd.set_detect_anomaly(True)

device = "cuda:1"
dtype = torch.bfloat16

image_size = 512

target = Image.open("source_images/lisa.jpg").convert("RGB")


preprocessing = T.Compose(
    [
        T.PILToTensor(),
        T.Resize(image_size),
        T.CenterCrop(image_size),
    ]
)
target = preprocessing(target)
target = target.to(device)
target = target / 255

target = target.to(dtype)

painting, loss_history, mae_history = optimize_grouped_painting(
    target,
    n_groups=10,
    n_strokes_per_group=50,
    iterations=300,
    lr=0.01,
    show_inner_pbar=True,
    error_map_temperature=1.0,
    log_every=10,
    dtype=dtype
)

canvas = torch.zeros(3, 512, 512, device=device)
result = painting.render_timelapse_frames(canvas, "painting_timelapse_frames")

# +
# 0.03159 for direct optimization of MAE. WAY faster than using MSSIM. 10 groups of 50, 200 iterations
# 0.03218 achieved by swapping MAE for MSE. okay, back to MAE
# 0.03178 achieved by going for 5 groups of 100 strokes instead of 10 groups of 50
# 0.03228 achieved with 1 group of 500 strokes
# 0.03193 achieved with 500 groups of 1 stroke
# Multi scale does nothing
# what if i use LPIPs now?

# bfloat16: took 2 minutes, got MAE of 0.03088. Memory usage: 
# float16: instant nan loss
# float32. memory usage: 2.9GB

# +
loss_history_arr, mae_history_arr = np.array(loss_history), np.array(mae_history)

mae_range = np.max(mae_history_arr) - np.min(mae_history_arr)

loss_history_arr = loss_history_arr - np.min(loss_history_arr)
loss_history_arr = loss_history_arr / np.max(loss_history_arr)

loss_history_arr = loss_history_arr * mae_range
loss_history_arr = loss_history_arr + np.min(mae_history_arr)

plt.plot(range(len(loss_history_arr)), loss_history_arr, label="Loss")
plt.plot(range(len(mae_history_arr)), mae_history_arr, label="MAE")
plt.legend()
# -


