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

from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure as MS_SSIM
from torchmetrics.image import PeakSignalNoiseRatio as PSNR


# +
class Painting(nn.Module):
    def __init__(self, n_strokes, dtype):
        super().__init__()

        self.n_strokes = n_strokes
        self.dtype=dtype

        self.offset = torch.randn(self.n_strokes, 2, 1) * 0.5 + 0.5
        self.offset = nn.Parameter(self.offset)

        self.sigma = torch.randn(self.n_strokes, 2, 1) * 0.1 + 0.5
        self.sigma = nn.Parameter(self.sigma)
        
        self.theta = torch.rand(self.n_strokes, 1) * 2 * torch.pi
        self.theta = nn.Parameter(self.theta)

        self.color = torch.rand(self.n_strokes, 3, 1, 1)
        self.color = nn.Parameter(self.color)

        self.alpha = torch.zeros(self.n_strokes, 1, 1, 1)
        self.alpha = nn.Parameter(self.alpha)
        
        self.epsilon = 1e-6
        
        self.psnr = PSNR()

    def loss_fn(self, x, y):
        
        # l1 = torch.mean(torch.abs(x - y))
        l2 = torch.mean(torch.square(x - y))
        
        sharpness = torch.mean(torch.abs(1 - (self.sigma[:, 0] / self.sigma[:, 1]))) * 0.001
        transparency = -1 * torch.mean(torch.abs(self.alpha)) * 0.001
        psnr = -1 * self.psnr(y, x) / 30
        
        loss = l2 + transparency + sharpness + psnr
        
        return loss

    def calculate_strokes(self, canvas):
        device = canvas.device

        height, width = canvas.shape[-2:]
        width_to_height_ratio = width/height

        w = torch.linspace(0, width - 1, width, device=device) / width
        h = torch.linspace(0, height - 1, height, device=device) / height

        x, y = torch.meshgrid(w, h, indexing="xy")

        # Reshape coordinates into a list of XY pairs, then duplicate it
        # once for each stroke in the group
        coordinates = torch.stack([x, y], dim=0).view(1, 2, -1)
        coordinates = coordinates.repeat(
            self.n_strokes, 1, 1
        )  # (1 x 2 x M) -> (N x 2 x M)

        # Simplified Gaussian PDF function:
        # e ^ (
        #    -1
        #     * (
        #         ((x - mu_x) ^ 2 / (2 * sigma_x ^ 2))
        #         + 
        #         ((y - mu_y) ^ 2 / (2 * sigma_y ^ 2))
        #     )
        # )

        coordinates = coordinates - self.offset
        coordinates = coordinates.to(self.dtype)
        
        # Rotate coordinates theta radians
        cos_theta = torch.cos(self.theta)
        sin_theta = torch.sin(self.theta)
        rotation_matrices = torch.cat(
            [cos_theta, -sin_theta, sin_theta, -cos_theta], dim=-1
        )
        rotation_matrices = rotation_matrices.view(self.n_strokes, 2, 2)
        
        coordinates = torch.matmul(rotation_matrices, coordinates)
        
        # Rescale sigma_x to avoid stretching in non-square images
        sigma = self.sigma.clone()
        sigma[:, 0] = sigma[:, 0] / width_to_height_ratio
        
        coordinates = coordinates.square()
        coordinates = coordinates / (2 * sigma.square())
        coordinates = coordinates.sum(dim=1) # (N x 2 x M) -> (N x M)
        
        pdf = torch.exp(-1 * coordinates)

        # Now map the PDF to the range [0, alpha]
        maxes, _ = pdf.max(dim=-1)
        maxes = maxes.view(self.n_strokes, 1)
        pdf = pdf / (maxes + self.epsilon)

        pdf = pdf.view(self.n_strokes, 1, height, width)
        pdf = pdf * self.alpha
        
        return pdf

    def forward(self, canvas, target):
        strokes = self.calculate_strokes(canvas)  # (N x 1 x H x W)
        strokes = strokes * self.color
        canvas = canvas + strokes.sum(dim=0)
        canvas = torch.sigmoid(canvas)
        loss = self.loss_fn(canvas.unsqueeze(0), target.unsqueeze(0))

        return canvas.detach(), loss

    def clamp_parameters(self):

        self.offset.data.clamp_(0.0, 1.0)
        self.sigma.data.clamp_(0.001, 1)
        self.theta.data.clamp_(0, 2*torch.pi)
        self.alpha.data.clamp_(-5.0, 5.0)
        self.color.data.clamp_(0.0, 1.0)
        
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

        offset = torch.zeros(self.n_strokes, 2, dtype=torch.long, device=device)

        # X coordinates (which column the flat index is in)
        # are found by moduloing it by the length of a row
        offset[:, 0] = coordinates % width

        # Y coordinates (which row the flat index is in) are
        # found by integer-dividing it by the length of a column
        offset[:, 1] = torch.div(coordinates, height, rounding_mode="floor")

        # Y coordinates first, then X, because tensors are indexed backwards
        color = target[:, offset[:, 1], offset[:, 0]].view(self.n_strokes, 3, 1, 1)
        
        offset = offset.to(self.dtype)
        
        offset[:, 0] = offset[:, 0] / width
        offset[:, 1] = offset[:, 1] / height
        
        offset = offset.unsqueeze(-1)

        self.offset.data = offset.detach().clone()
        self.color.data = color.detach().clone()
            
        self.clamp_parameters()
        
    def run_pruning(self):
        
        device = self.offset.device
        
        _offset = torch.randn_like(self.offset) * 0.5 + 0.5
        _sigma = torch.randn_like(self.sigma) * 0.1 + 0.5
        _theta = torch.rand_like(self.theta) * 2 * torch.pi
        _color = torch.rand_like(self.color)
        _alpha = torch.zeros_like(self.alpha)
        
        small_area_reinit = 0
        big_area_reinit = 0
        alpha_reinit = 0
        grad_reinit = 0
        
        reinit_total = 0
        
        for i in range(self.n_strokes):
            
            mask = torch.tensor(False, device=device)
            
            area = self.sigma[i,0].squeeze() * self.sigma[i,1].squeeze()
            
            small_area_mask = (area < 0.001)
            small_area_reinit += small_area_mask
            big_area_mask = (area > 0.5)
            big_area_reinit += big_area_mask
            
            alpha_mask = torch.abs(self.alpha[i].squeeze()) < 0.001
            alpha_reinit += alpha_mask
            
            mask = torch.logical_or(mask, small_area_mask)
            mask = torch.logical_or(mask, big_area_mask)
            mask = torch.logical_or(mask, alpha_mask)
            
            offset_grad = self.offset.grad[i].squeeze()
            sigma_grad = self.sigma.grad[i].squeeze()
            theta_grad = self.theta.grad[i].squeeze().unsqueeze(-1)
            color_grad = self.color.grad[i].squeeze()
            alpha_grad = self.alpha.grad[i].squeeze().unsqueeze(-1)
            
            all_grads = torch.cat([offset_grad, sigma_grad, theta_grad, color_grad, alpha_grad], dim=0)
            all_grads = all_grads.abs()
            grad_mask = torch.any(all_grads > 1.0)
            grad_reinit += grad_mask
            
            mask = torch.logical_or(mask, alpha_mask)
            
            if mask:
                self.color.data[i] = _color[i]
                self.theta.data[i] = _theta[i]
                self.sigma.data[i] = _sigma[i]
                self.offset.data[i] = _offset[i]
                self.alpha.data[i] = _alpha[i]
                
                reinit_total += 1
                
        #print(f"Reinitialized {reinit_total} gaussians")
        # print(f"{small_area_reinit.item()}, {big_area_reinit.item()}, {alpha_reinit.item()}, {grad_reinit.item()}")


def optimize(
    target,
    n_strokes,
    iterations,
    lr,
    error_map_temperature=1.0,
    log_every=10,
    prune_every=None,
    dtype=torch.float,
):

    device = target.device

    canvas = torch.zeros_like(target)

    timelapse_path = Path("optimization_timelapse_frames")
    if timelapse_path.exists():
        shutil.rmtree(timelapse_path)
    timelapse_path.mkdir()

    loss = 0.0
    mae = 1.0

    loss_history = []
    mae_history = []

    painting = Painting(n_strokes=n_strokes, dtype=dtype).to(device).to(dtype)
    
    painting.smart_initialize(
        target, canvas, error_map_temperature
    )

    optimizer = torch.optim.AdamW(painting.parameters(), lr=lr)

    pbar = tqdm(range(iterations))
    pbar.set_description(
        f"MAE = 1.0 | Loss = 0.0"
    )
    for i in pbar:
        
        result, loss = painting(canvas=canvas, target=target)
        mae = torch.mean(torch.abs(result - target))

        loss.backward()         
        optimizer.step()
        
        if prune_every is not None and (i+1) % prune_every == 0:
            painting.run_pruning()
            
        optimizer.zero_grad()
        painting.clamp_parameters()
        
        if (i+1) % log_every == 0:
            T.functional.to_pil_image(result).save(
                timelapse_path / f"{i:06}.jpg"
            )
            

        loss_history.append(loss.detach().cpu().item())
        mae_history.append(mae.detach().cpu().item())
        
        pbar.set_description(
            f"MAE = {mae:.5f} | Loss = {loss:.5f}"
        )

    canvas, loss = painting(canvas=canvas, target=target)
    
    return canvas.cpu(), loss_history, mae_history

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

canvas, loss_history, mae_history = optimize(
    target,
    n_strokes=100,
    iterations=2000,
    lr=0.01,
    error_map_temperature=1.0,
    log_every=25,
    prune_every = None,
    dtype=dtype
)

plt.figure(figsize=(10, 10))
plt.imshow(canvas.float().permute(1,2,0))

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


