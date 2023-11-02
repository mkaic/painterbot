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
from typing import List


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


class StrokeParameters(nn.Module):
    def __init__(self, n_strokes: int, width_to_height_ratio: float):
        super().__init__()

        self.width_to_height_ratio = width_to_height_ratio

        self.center_x = torch.ones(n_strokes, 1) * self.width_to_height_ratio * 0.5
        self.center_x = nn.Parameter(self.center_x)

        self.center_y = torch.ones(n_strokes, 1) * 0.5
        self.center_y = nn.Parameter(self.center_y)

        self.rotation = (torch.rand(n_strokes, 1) - 0.5) * 2 * torch.pi
        self.rotation = nn.Parameter(self.rotation)

        self.mu_r = torch.ones(n_strokes, 1) * 0.2
        self.mu_r = nn.Parameter(self.mu_r)

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

    def clamp_parameters(self):
        self.center_x.data.clamp_(0, self.width_to_height_ratio)
        self.center_y.data.clamp_(0, 1)

        # Guassian gets weirdly pinched if it gets too close to the pole
        self.mu_r.data.clamp_(0.05, 2)
        self.sigma_r.data.clamp_(0.0001, 0.2)

        # Strokes can only ever curve a certain amount to avoid a bunch of
        # obvious perfect circles showing up in the painting
        self.sigma_theta.data.clamp_(0.001, torch.pi / 4)

        self.alpha.data.clamp_(0.01, 1)
        self.color.data.clamp_(0, 1)

    def smart_init(
        self, target: torch.Tensor, canvas: torch.Tensor, temperature: float = 1.0
    ):
        device = target.device
        height, width = target.shape[-2:]

        error_map = torch.mean(torch.abs(target - canvas), dim=0)  # (3, H, W) -> (H, W)

        # Unravel the error map, then softmax to get probability distribution over
        # the pixel locations, then sample from it
        flat_error_pdf = nn.functional.softmax(
            error_map.view(-1) / temperature, dim=0
        )  # (H, W) -> (H*W)

        n_samples = len(self.alpha)

        coordinates = torch.multinomial(flat_error_pdf, n_samples)  # (N, 1)

        coordinates = batch_unravel_index_xy(
            coordinates, (height, width), device=device
        ).unsqueeze(
            -1
        )  # (N, 1) -> (2, N, 1)

        center_x, center_y = coordinates

        color = target[:, center_y, center_x].view(len(self.alpha), 3, 1, 1)

        center_x = center_x / height
        center_x = center_x.detach().clone()
        self.center_x.data = center_x

        center_y = center_y / height
        center_y = center_y.detach().clone()
        self.center_y.data = center_y

        color = color.detach().clone()
        self.color.data = color

        self.clamp_parameters()


def concat_stroke_parameters(stroke_parameters: List[StrokeParameters]):
    concatted = stroke_parameters[0]

    if len(stroke_parameters) > 1:
        for param in stroke_parameters[1:]:
            concatted.center_x.data = torch.cat(
                [concatted.center_x.data, param.center_x.data], dim=0
            )
            concatted.center_y.data = torch.cat(
                [concatted.center_y.data, param.center_y.data], dim=0
            )
            concatted.rotation.data = torch.cat(
                [concatted.rotation.data, param.rotation.data], dim=0
            )
            concatted.mu_r.data = torch.cat(
                [concatted.mu_r.data, param.mu_r.data], dim=0
            )
            concatted.sigma_r.data = torch.cat(
                [concatted.sigma_r.data, param.sigma_r.data], dim=0
            )
            concatted.sigma_theta.data = torch.cat(
                [concatted.sigma_theta.data, param.sigma_theta.data], dim=0
            )
            concatted.color.data = torch.cat(
                [concatted.color.data, param.color.data], dim=0
            )
            concatted.alpha.data = torch.cat(
                [concatted.alpha.data, param.alpha.data], dim=0
            )

    return concatted


class Renderer(nn.Module):
    def __init__(self):
        super().__init__()
        self.epsilon = 1e-8

    def loss_fn(self, x, y):
        return torch.mean(torch.abs(x - y))

    def calculate_strokes(self, canvas: torch.Tensor, parameters: StrokeParameters):
        device = canvas.device

        n_strokes = len(parameters.alpha)

        # Aspect ratio should always be >1:1, never 1:<1
        height, width = canvas.shape[-2:]

        w = torch.linspace(0, width - 1, width, device=device) / height
        h = torch.linspace(0, height - 1, height, device=device) / height

        x, y = torch.meshgrid(w, h, indexing="xy")

        # Reshape coordinates into a list of XY pairs, then duplicate it
        # once for each stroke in the group
        cartesian_coordinates = torch.stack([x, y], dim=0).view(1, 2, -1)
        cartesian_coordinates = cartesian_coordinates.repeat(
            n_strokes, 1, 1
        )  # (1 x 2 x M) -> (N x 2 x M)

        cos_rot = torch.cos(parameters.rotation)
        sin_rot = torch.sin(parameters.rotation)

        # Offset coordinates to change where the center of the
        # polar coordinates is placed
        offset_x = parameters.center_x - cos_rot * parameters.mu_r
        offset_y = parameters.center_y - sin_rot * parameters.mu_r
        cartesian_offset = torch.stack(
            [offset_x, offset_y], dim=1
        )  # 2x (N x 1) -> (N x 2 x 1)
        cartesian_coordinates = cartesian_coordinates - cartesian_offset

        rotation_matrices = torch.cat([cos_rot, -sin_rot, sin_rot, -cos_rot], dim=-1)
        rotation_matrices = rotation_matrices.view(n_strokes, 2, 2)
        cartesian_coordinates = torch.matmul(rotation_matrices, cartesian_coordinates)

        # Convert to polar coordinates and apply polar-space offsets
        r = torch.sqrt(
            torch.sum(
                torch.square(cartesian_coordinates + self.epsilon),
                dim=1,  # (N x 2 x M) -> (N x M)
            )
        )
        r = r - parameters.mu_r

        theta = torch.atan2(
            cartesian_coordinates[:, 1] + self.epsilon,
            cartesian_coordinates[:, 0] + self.epsilon,
        )  # -> (N x M), ranges from -pi to pi

        # Simplified Gaussian PDF function:
        # e ^ (
        #    -1
        #     * (
        #         ((x - mu_x) ^ 2 / (2 * sigma_x ^ 2))
        #         + ((y - mu_y) ^ 2 / (2 * sigma_y ^ 2))
        #     )
        # )

        r = torch.square(r)
        theta = torch.square(theta)

        # sigma_r is expressed as a fraction of the radius instead of
        # an absolute quantity
        sigma_r = parameters.sigma_r * parameters.mu_r
        sigmas = torch.stack([sigma_r, parameters.sigma_theta], dim=1).view(
            n_strokes, 2, 1
        )

        polar_coordinates = torch.stack([r, theta], dim=1)
        polar_coordinates = polar_coordinates / (
            2 * torch.square(sigmas) + self.epsilon
        )
        pdf = torch.exp(-1 * torch.sum(polar_coordinates, dim=1))

        # Now map the PDF to the range [0, alpha]
        maxes, _ = pdf.max(dim=-1)
        maxes = maxes.view(n_strokes, 1)
        pdf = pdf / (maxes + self.epsilon)

        pdf = pdf.view(n_strokes, 1, height, width)
        pdf = pdf * parameters.alpha
        return pdf

    def render_stroke(
        self, stroke: torch.Tensor, color: torch.Tensor, canvas: torch.Tensor
    ):
        # Use the stroke as an alpha to blend between the canvas and a color
        canvas = ((torch.ones_like(stroke) - stroke) * canvas) + (stroke * color)
        # Clamp canvas to valid RGB color space.
        canvas = canvas.clamp(0, 1)

        return canvas

    def forward(
        self, canvas: torch.Tensor, parameters: StrokeParameters, target: torch.Tensor
    ):
        strokes = self.calculate_strokes(canvas, parameters)  # (N x 1 x H x W)

        loss = 0
        for stroke, color in zip(strokes, parameters.color):
            canvas = self.render_stroke(stroke=stroke, color=color, canvas=canvas)
            loss += self.loss_fn(canvas.unsqueeze(0), target.unsqueeze(0))

        loss = loss / len(strokes)

        return canvas.detach(), loss

    def render_timelapse_frames(
        self, canvas: torch.Tensor, parameters: StrokeParameters, output_path: Path
    ):
        output_path = Path(output_path)
        if output_path.exists():
            shutil.rmtree(output_path)
        output_path.mkdir()

        with torch.no_grad():
            strokes = self.calculate_strokes(canvas, parameters)

            for i, (stroke, color) in enumerate(zip(strokes, parameters.color)):
                canvas = self.render_stroke(stroke=stroke, color=color, canvas=canvas)
                T.functional.to_pil_image(canvas.cpu()).save(
                    output_path / f"{i:05}.jpg"
                )

            return canvas


def optimize(
    target: torch.Tensor,
    n_groups: int,
    n_strokes_per_group: int,
    iterations: int,
    lr: float,
    show_inner_pbar: bool = True,
    error_map_temperature: float = 1.0,
    log_every: int = 30,
):
    if n_groups == 1:
        show_inner_pbar = True

    device = target.device
    width, height = target.shape[-2:]
    width_to_height_ratio = width / height

    canvas = torch.zeros_like(target)

    renderer = Renderer().to(device)

    frozen_params: StrokeParameters = None

    output_path = Path("optimization_timelapse_frames")
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir()

    loss = 0.0

    loss_history = []
    mae_history = []

    outer_pbar = tqdm(range(n_groups))
    outer_pbar.set_description(f"Loss = 0.0 | Optimizing Stroke Group: 1 of {n_groups}")

    for i in outer_pbar:
        active_params = StrokeParameters(
            n_strokes=n_strokes_per_group,
            width_to_height_ratio=width_to_height_ratio,
        ).to(device)

        active_params.smart_init(
            target=target,
            canvas=canvas,
            temperature=error_map_temperature,
        )

        optimizer = torch.optim.Adam(
            active_params.parameters(), lr=lr, betas=(0.8, 0.9)
        )

        if show_inner_pbar:
            inner_iterator = tqdm(range(iterations), leave=False)
        else:
            inner_iterator = range(iterations)

        for j in inner_iterator:
            result, loss = renderer(canvas, active_params, target)

            loss.backward()

            optimizer.step()

            optimizer.zero_grad(set_to_none=True)

            active_params.clamp_parameters()

            if show_inner_pbar:
                description = f"Loss = {loss:.5f} "
                inner_iterator.set_description(description)

            if j % log_every == 0:
                T.functional.to_pil_image(result).save(
                    output_path / f"{i:05}_{j:06}.jpg"
                )

            loss_history.append(loss.detach().cpu().item())

        canvas, _ = renderer(canvas, active_params, target)
        if i == 0:
            frozen_params = active_params
        else:
            frozen_params = concat_stroke_parameters([frozen_params, active_params])

        outer_pbar.set_description(
            f"Loss = {loss:.5f} | Optimizing Stroke Group: {i+1} of {n_groups}"
        )
    return frozen_params, renderer, loss_history, mae_history


if __name__ == "__main__":
    device = "cuda:0"
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

    params, renderer, loss_history, mae_history = optimize(
        target,
        n_groups=10,
        n_strokes_per_group=50,
        iterations=300,
        lr=0.01,
        show_inner_pbar=True,
        error_map_temperature=1.0,
    )

    canvas = torch.zeros(3, 512, 512, device=device)
    result = renderer.render_timelapse_frames(
        canvas, params, Path("painting_timelapse_frames")
    )

# +
# 0.03159 for direct optimization of MAE. WAY faster than using MSSIM. 10 groups of 50, 200 iterations
# 0.03218 achieved by swapping MAE for MSE. okay, back to MAE
# 0.03178 achieved by going for 5 groups of 100 strokes instead of 10 groups of 50
# 0.03228 achieved with 1 group of 500 strokes
# 0.03193 achieved with 500 groups of 1 stroke
# -
