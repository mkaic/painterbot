from .painterbot import (
    StrokeParameters,
    render,
    render_timelapse_frames,
)

import torch
from PIL import Image
import os
from pathlib import Path
import shutil
from torchvision.transforms.functional import to_tensor, to_pil_image


def compare(a: torch.Tensor, b: torch.Tensor, path: str):
    both_results = torch.cat([a, b], dim=2)
    both_results = (both_results.cpu().permute(
        1, 2, 0) * 255).to(torch.uint8).numpy()

    as_pil = Image.fromarray(both_results)
    as_pil.save(path)


if __name__ == "__main__":
    device = torch.device("cuda:1")
    torch.cuda.set_device(device.index)
    dtype = torch.float32
    canvas = torch.zeros(3, 256, 256, device=device, dtype=dtype)
    parameters = (
        StrokeParameters.from_file(
            "checkpoints/lisa_5_10_100.pt").to(device).to(dtype)
    )

    print("running PyTorch timelapse")
    torch_timelapse = Path("test_timelapse_torch")
    _ = render_timelapse_frames(
        canvas=canvas,
        parameters=parameters,
        output_path=torch_timelapse,
    )
    print("running Triton timelapse")
    triton_timelapse = Path("test_timelapse_triton")
    _ = render_timelapse_frames(
        canvas=canvas,
        parameters=parameters,
        output_path=triton_timelapse,
        triton=True,
    )

    out = Path("torch_triton_timelapse")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir()

    torch_frames = sorted([x.absolute()
                          for x in torch_timelapse.glob("*.jpg")])
    triton_frames = sorted([x.absolute()
                           for x in triton_timelapse.glob("*.jpg")])
    for i, (torch_frame, triton_frame) in enumerate(zip(torch_frames, triton_frames)):
        torch_frame = to_tensor(Image.open(torch_frame))
        triton_frame = to_tensor(Image.open(triton_frame))
        both_frames = torch.cat([torch_frame, triton_frame], dim=1)
        both_frames = to_pil_image(both_frames)
        both_frames.save(out / f"{i:05}.jpg")

    print("running PyTorch render")
    torch_result = render(canvas=canvas, parameters=parameters)
    print("running Triton render")
    try:
        triton_result = render(
            canvas=canvas, parameters=parameters, triton=True)
    except Exception as e:
        print("Triton render failed")
        triton_result = torch.zeros_like(torch_result)
        print(e)
    finally:
        compare(torch_result, triton_result, "test_result.png")
