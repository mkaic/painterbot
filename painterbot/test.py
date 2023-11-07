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
    device = torch.device("cuda:1")
    dtype = torch.float16
    canvas = torch.zeros(3, 256, 256, device=device, dtype=dtype)
    parameters = (
        StrokeParameters.from_file(
            "checkpoints/lisa_5_10_100.pt").to(device).to(dtype)
    )

    print("running PyTorch render")
    torch_result = render(canvas=canvas, parameters=parameters)
    print("running Triton render")
    try:
        triton_result = render(
            canvas=canvas, parameters=parameters, triton=True)
    except Exception as e:
        print("Triton render failed")
        triton_result = torch.zeros_like(torch_result)
        raise (e)
    finally:
        compare(torch_result, triton_result, "test_result.png")
