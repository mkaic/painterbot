from .torch_implementation import (
    StrokeParameters,
    torch_pdf,
    torch_blend,
    torch_render,
)
from .triton_implementation import (
    triton_pdf,
    triton_blend,
    triton_render,
)

from .forward import forward

import torch
from PIL import Image
from tqdm import tqdm
import time


def compare(a: torch.Tensor, b: torch.Tensor, path: str):
    both_results = torch.cat([a, b], dim=1)
    both_results = (both_results.cpu().permute(2, 1, 0) * 255).to(torch.uint8).numpy()

    as_pil = Image.fromarray(both_results)
    as_pil.save(path)


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device.index)
    dtype = torch.float32
    parameters = (
        StrokeParameters.from_file("checkpoints/lisa_16_32_256.pt").to(device).to(dtype)
    )

    height, width = 300, 300

    with torch.no_grad():
        print("benchmarking PyTorch calculate_strokes")
        start = time.time()
        for i in tqdm(range(100)):
            strokes = torch_pdf(
                height=height,
                width=width,
                device=device,
                dtype=dtype,
                parameters=parameters,
            )
        print(f"PyTorch calculate_strokes took {time.time() - start} seconds")

        print("benchmarking PyTorch blend")
        start = time.time()
        for i in tqdm(range(100)):
            canvas = torch.zeros(3, height, width, device=device, dtype=dtype)
            _ = torch_blend(
                canvas=canvas,
                strokes=strokes,
                parameters=parameters,
            )
        print(f"PyTorch blend took {time.time() - start} seconds")

        print("benchmarking triton_pdf_forward")
        start = time.time()
        for i in tqdm(range(100)):
            strokes = triton_pdf(
                height=height,
                width=width,
                device=device,
                dtype=dtype,
                parameters=parameters,
            )
        print(f"Triton triton_pdf_forward took {time.time() - start} seconds")

        print("benchmarking triton_blend_forward")
        start = time.time()
        for i in tqdm(range(100)):
            canvas = torch.zeros(3, height, width, device=device, dtype=dtype)
            _ = triton_blend(
                canvas=canvas,
                strokes=strokes,
                parameters=parameters,
            )
        print(f"Triton triton_blend_forward took {time.time() - start} seconds")

        canvas = torch.zeros(3, 300, 300, device=device, dtype=dtype)
        torch_result = forward(
            canvas=canvas,
            parameters=parameters,
            render_fn=torch_render,
        )

        canvas = torch.zeros(3, 300, 300, device=device, dtype=dtype)
        triton_result = forward(
            canvas=canvas,
            parameters=parameters,
            render_fn=triton_render,
        )

        compare(torch_result, triton_result, "test_result.png")
