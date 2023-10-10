# +
import torch
import torch.nn as nn
import torchvision.transforms as T

from PIL import Image
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from pathlib import Path
import shutil
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure as MS_SSIM
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
import numpy as np


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
    def __init__(self, group_number, n_strokes):
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

        self.ms_ssim = MS_SSIM(data_range=1.0)
        self.psnr = PSNR()

    def loss_fn(self, x, y):
        psnr = self.psnr(x, y) / 20  # Peak Signal-to-Noise Ratio
        ms_ssim = self.ms_ssim(x, y)  # Multiscale Structural Similarity Index Measure

        loss = psnr + ms_ssim
        loss = loss * -1
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
        self.sigma_theta.data.clamp_(0.001, 0.25)

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
):
    if n_groups == 1:
        show_inner_pbar = True

    device = target.device

    painting = Painting(
        n_groups=n_groups,
        target=target,
    ).to(device)

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
        painting.stroke_groups.append(
            StrokeGroup(group_number=group_number, n_strokes=n_strokes_per_group).to(
                device
            )
        )
        painting.stroke_groups[group_number].smart_initialize(
            target, canvas, error_map_temperature
        )

        optimizer = torch.optim.Adam(painting.parameters(), lr=lr, betas=(0.8, 0.9))

        if show_inner_pbar:
            inner_iterator = tqdm(range(iterations), leave=False)
        else:
            inner_iterator = range(iterations)

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

image_size = 256

target = Image.open("branos.jpg").convert("RGB")


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
target = target

painting, loss_history, mae_history = optimize_grouped_painting(
    target,
    n_groups=20,
    n_strokes_per_group=50,
    iterations=300,
    lr=0.01,
    show_inner_pbar=True,
    error_map_temperature=1.0,
    log_every=10,
)

canvas = torch.zeros(3, 1024, 1024, device=device)
result = painting.render_timelapse_frames(canvas, "painting_timelapse_frames")

# +
# EXPERIMENTS LOG

# Baseline: SSIM + PSNR, 300 strokes, 300 iterations, LR=0.05 * 0.99^step.
# Takes about 5 minutes. reaches MAE=0.045

# Experiment 1: does adding L1 into the mix help?
# reaches MAE=0.044. not statistically significant.

# Experiment 2: does adding L2 into the mix help?
# reaches MAE=0.044. not statistically significant.

# Experiment 3: NEW BASELINE, LR=0.01, no decay, 50 strokes for 400 steps.
# Reaches MAE=0.0738.

# Experiment 4: does decay help? setting lr_gamma to 0.995.
# Reaches MAE=0.885. decay hurts.

# Experiment 5: higher LR=0.02. try decay again.
# Reaches MAE = 0.080

# Experiment 6: higher LR=0.02 with no decay.
# Reaches MAE = 0.072. LR is inconsequential with Adam perhaps?

# Experiment 7: AdamW instead of Adam.
# Reaches MAE = 0.076. Will stick with the simpler Adam.

# Experiment 8: does iteratively adding strokes once every 1 step help?
# Reaches MAE = 0.0685. It does!

# Experiment 9: what if we do it every 2 steps?
# Reaches MAE = 0.0716. Inconclusive, need to compensate for difference in computing power between the approaches.

# Experiment 10: 50 strokes with one introduced every 2 steps means the
# first 100 steps only take up as much compute as 50 all-strokes steps. This means 50 additional steps can be
# added to the total to compensate. Thus, the step count is now (total + (stroke_count*ramp_stride/2)).
# In this case, that's 450 steps.
# Reaches MAE = 0.0706. Inconclusive.

# Experiment 11: let's ramp up to ramp_stride = 8 for the heck of it.
# Reaches MAE = 0.0685. I think this strategy can work, just need to once again reset the baseline.

# Experiment 12: 50 strokes, 1000 iterations. No ramping. Let's reset the baseline.
# Reaches MAE = 0.0637

# Experiment 13: same as new baseline, but with ramping and ramp_stride set to 20. This means
# all strokes won't be active until step 1000, and to keep compute parity we'll need to set
# total steps to 1500.
# Reaches MAE = 0.0636. seems to make no real difference, but I like it so let's keep it.

# Experiment 14: scaling up the baseline to 300 strokes, 1000 steps.
# Reaches MAE = 0.0354 and Loss = -1.751

# Experiment 15: 300 strokes, ramp_stride=3, 1450 steps (to compensate for compute disparity)
# Reaches MAE = 0.03651 and Loss = -1.762. No difference. is learning rate to blame?

# Experiment 16: same as #14, but lower the learning rate to 0.003.
# Reaches MAE = 0.0413 and Loss = -1.675. Lower learning rate is bad.

# Experiment 17: totally reworked the system. Now the painting optimizes one group of strokes at a time, then
# freezes them. When the number of strokes in a group is 1, this is equivalent to my original approach, just
# using SSIM and PSNR as metrics. So, for this run, to establish a baseline,
# I am setting strokes and groups to 300. Each group is optimized for 100 steps.
# Reaches MAE = 0.0875 and Loss = -1.457. This approach still sucks!

# Experiment 18: trying a group size of 10 strokes per group, so 30 groups. Since there are 10x
# fewer groups, I optimize each for 10x as long, 1000 steps. Same computational load as #17, but
# hopefully far superior quality.
# Reaches MAE = 0.050 and Loss = -1.76. This approaches the quality of optimizing all 300 in parallel!
# This doesn't really show definite *improvement* yet though since we still used the same compute
# budget as the parallel optimization.

# Experiment 19: Let's set another baseline. 300 strokes, 5 groups, 500 iterations per group. Much
# lower budget than previous experiment, but bigger group size.
# Reached MAE = 0.0579 and Loss = -1.67. hmmm.

# Experiment 20: It would seem that 1000 iterations is pretty necessary for convergence. Let's go
# with 300 strokes, 6 groups of 50 each, and 1000 iterations per group.
# Reached MAE = 0.045 and Loss = -1.80. Pretty good, but less efficient than just optimizing in
# one big group.

# Experiment 21: 300 strokes, 30 groups of 10 each, 300 iterations per group.
# Reached MAE = 0.0608 and Loss = -1.666. ugh.

# Experiment 22: Let's make things faster. Reduce steps by a factor of 3 (1000 to 300)
# Raise the learning rate a factor of 3 (0.01 to 0.03)
# Reached MAE = 0.0581 and Loss = -1.653.  Not great, I think learning rate decay may actually be useful here?

# Experiment 23: Cosine decay on the big 0.03 LR? Otherwise same as #22.
# Reached MAE = 0.0598 and Loss = -1.646. No change.

# Experiment 24: Multiplicative (gamma = 0.99) decay on the big 0.03 LR. Otherwise same as #22.
# Reached MAE = 0.0624 and Loss = -1.633. No change.

# Experiment 25: Okay, once and for all, does learning rate or LR decay matter?
# Same as #22, but with LR=0.01. So LR=0.01, strokes = 300, steps = 300.
# MAE: 0.05807 | Loss: -1.67026. Meh. I'm not messing with decay any more.

# Experiment 26: what if I ditch PSNR and only use SSIM?
# Loss: -0.66356 | MAE: 0.09006. meeeehhhhhh weird bad color accuracy.

# Experiment 27: what if i go back to the old loss but divide PSNR by 30 instead of 20.
# MAE: 0.06551 | Loss: -1.28067

# Experiment 28: What if instead of reducing PSNR's influence compared to 22, I double it?
# MAE = 0.05987 | Loss = -2.67799. Insigificant difference from 22.

# Experiment 29: 500 strokes, 50 groups, 300 iterations.
# MAE = 0.05246 | Loss = -1.75182

# Experiment 30: 300 strokes, 6 groups, 300 iterations. LR = 0.01. New baseline.
# MAE = 0.05699 | Loss = -1.68893 

# Experiment 31: 300 strokes, 6 groups, 300 iterations. Trying out a new loss where I compare against the
# change each stroke *should* have made instead of just directly comparing against the image.
# MAE = 0.06615 | Loss = -1.65992. No real change.

# Experiment 32: Same as #30, but with the old loss thrown in too.
# MAE = 0.05845 | Loss = -1.63180 Nope, just get rid of the new loss. Doesn't do anything.

# Experiment 33: same as #30, but with multi-scale SSIM instead of regular SSIM
# MAE = 0.05276 | Loss = -1.86894. Ayyy finally an improvement!

# Experiment 34: same as #33, but tacking on a Sobel edge loss to see if that does anything.
# MAE = 0.06488 | Loss = -1.45236. Made it considerably worse.

# Experiment 35: same as 34, but divide sobel maps by 8 before 
# comparing as suggested by ptrblck on the PyTorch forums.
# MAE = 0.05722 | Loss = -1.76451. Doesn't help. Sobel loss bad.

# Experiment 35: 1000 strokes, 10 groups, 500 iterations, LR=0.01
# MAE = 0.03450 | Loss = -2.13276

# Experiment 36: Better initialization. I'm setting the color based on the actual color of
# the target image, the position is sampled based on where the reconstruction error is highest,
# and the stroke size is made proportional to the MAE. All these things can still be optimized
# if algorithm wants to, but they should be starting out way less random now.
# 300 strokes, 6 groups, 300 iterations. Also, it would appear that by setting theta to always be 1,
# I removed the optimizer's ability to rotate the strokes. whoops.
# MAE = 0.03173 | Loss = -2.14367. Output is blocky but accurate.

# Experiment 37: same as 36, but went back to randomly initializing the rotation of the gaussians.
# I want a smoother output this time, as the blockiness looks jarring. For some reason, the rotation
# of the gaussians is not being optimized?
# forgot what the numbers were here but it was marginally better than 36.

# Experiment 38: fixed theta optimization. Apparently it NEVER WORKED??? LIKE AT ALL???
# Now that theta can ACTUALLY be optimized I'm really really excited about the potential for
# improvement. Fingers crossed!
# MAE = 0.02872 | Loss = -2.21096 OMG OMG OMG OMG OMG OMG

# Experiment 39: 300 strokes, 300 groups, 100 iterations.
# MAE = 0.05021 | Loss = -1.90338. Bad!

# Experiment 40: 500 strokes, 10 groups, 300 iterations.
# MAE = 0.02323 | Loss = -2.32215 WOOOOOOOOO

# SWITCHING TO THE MONA LISA FOR TESTING

# Mona Lisa 1: 500 strokes, 10 groups, 100 iterations. LR=0.01.
# MAE = 0.03102 | Loss = -2.27161

# Mona Lisa 2: 500 strokes, 10 groups, 100 iterations. LR=0.03 BIG LR
# MAE = 0.03430 | Loss = -2.19209

# Mona Lisa 3: 500 strokes, 10 groups, 100 iterations. LR=0.01. 
# lowered adam betas to (0.8, 0.9) bc i'm only optimizing for 100 iterations.
# MAE = 0.030

# Mona Lisa 4: same as 3, but with even lower betas=(0.66, 0.66)
# MAE = 0.031

# Mona Lisa 5: Same as 4, but with betas=(0.8,0.9) and MAE directly included in the loss
# MAE = 0.03035 | Loss = -2.25097. Removed MAE after this

# Mona Lisa 6: same as 2, but with 20 groups instead of 10. and the newest betas as well.
# LR = 0.03, because I'm not ready to give up on a big learning rate yet. 100 iterations
# should be plenty for this, but I gotta figure out the hyperparams to make that work!
# MAE = 0.035, not great. I think big learning rate is just not gonna happen.

# Mona Lisa 7: back to LR=0.01, but still 100 iterations and betas=(0.8,0.9)
# Same setup as 3, but this time I'm setting the softmax temperature to 0.5
# MAE = 0.03172 | Loss = -2.24758

# Mona Lisa 8: Same as 7, but now softmax temp is 0.25.
# MAE = 0.03138 | Loss = -2.25816 basically exactly the same. no change.

# Mona Lisa 9: keepign low softmax temp and bumping groups from 10 to 50. I have a theory
# that more frequent sampling of the error map combined with more agressive error-targeting
# could result in better reconstruction.
# MAE = 0.03196 | Loss = -2.25030. No difference. seems like the temperature and group numbers
# simply don't matter that much.

# NEW BASELINE IN PREPARATION FOR CURVED STROKE TESTING
# Mona Lisa, 100 strokes, 200 iterations, 3 groups.
# MAE = 0.04143 | Loss = -2.02877

# WITH POLAR COORDINATE WARPED GAUSSIANS????
# Mona Lisa, 100 strokes, 200 iterations, 3 groups, POLAR WARPING ENABLED
# MAE = 0.04146 | Loss = -2.00878. Nearly identical loss but i don't care, i worked too hard on
# getting the polar warping to work.

# Polar coords activated, 500 strokes, 10 groups, 300 iterations.
# shooting for a new lowest loss on Mona Lisa.
# 

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
