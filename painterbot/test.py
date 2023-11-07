from .painterbot import StrokeParameters, render

import torch
from PIL import Image


def compare(a: torch.Tensor, b: torch.Tensor, path: str):
    both_results = torch.cat([a, b], dim=2)
    print(both_results.shape)
    both_results = (both_results.cpu().permute(
        1, 2, 0) * 255).to(torch.uint8).numpy()

    as_pil = Image.fromarray(both_results)
    as_pil.save(path)


if __name__ == "__main__":
    device = torch.device("cuda:0")
    canvas = torch.zeros(3, 256, 256, device=device)
    parameters = StrokeParameters.from_file(
        "checkpoints/lisa_5_50_300.pt").to(device)

    triton_result = render(canvas=canvas, parameters=parameters, triton=True)
    torch_result = render(canvas=canvas, parameters=parameters)
    compare(torch_result, triton_result, "test_result.png")
