import time

import torch
from PIL import Image
from tqdm import tqdm

from .forward import forward
from .torch_implementation import StrokeParameters, torch_blend, torch_pdf, torch_render
from .triton_implementation import triton_blend, triton_pdf, triton_render


def compare(a: torch.Tensor, b: torch.Tensor, path: str):
    both_results = torch.cat([a, b], dim=2)
    both_results = (both_results.cpu().permute(1, 2, 0) * 255).to(torch.uint8).numpy()

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
        # start = time.time()
        for i in tqdm(range(100), leave=False):
            strokes = torch_pdf(
                height=height,
                width=width,
                device=device,
                dtype=dtype,
                parameters=parameters,
            )
            if i == 0:
                start = time.time()
        print(f"torch_pdf: {(time.time() - start):.3f}")

        # start = time.time()
        for i in tqdm(range(100), leave=False):
            canvas = torch.zeros(3, height, width, device=device, dtype=dtype)
            _ = torch_blend(
                canvas=canvas,
                strokes=strokes,
                parameters=parameters,
            )
            if i == 0:
                start = time.time()
        print(f"torch_blend: {(time.time() - start):.3f}")

        # start = time.time()
        for i in tqdm(range(100), leave=False):
            strokes = triton_pdf.apply(
                parameters,
                height,
                width,
                device,
                dtype,
            )
            if i == 0:
                start = time.time()
        print(f"triton_pdf: {(time.time() - start):.3f}")

        # start = time.time()
        for i in tqdm(range(100), leave=False):
            canvas = torch.zeros(3, height, width, device=device, dtype=dtype)
            _ = triton_blend.apply(
                canvas,
                strokes,
                parameters,
                False,
            )
            if i == 0:
                start = time.time()
        print(f"triton_blend: {(time.time() - start):.3f}")

        canvas = torch.zeros(3, height, width, device=device, dtype=dtype)
        torch_result = forward(
            canvas=canvas,
            parameters=parameters,
            render_fn=torch_render,
        )

        canvas = torch.zeros(3, height, width, device=device, dtype=dtype)
        triton_result = forward(
            canvas=canvas,
            parameters=parameters,
            render_fn=triton_render,
        )

        compare(torch_result, triton_result, "test_result.png")
