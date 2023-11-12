from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image


def load_image(
    image_path: Path,
    device: torch.device,
    image_size: int = 512,
    crop: bool = False,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    image_path = Path(image_path)
    target = Image.open(image_path).convert("RGB")

    preprocessing = T.Compose(
        [
            T.PILToTensor(),
            T.Resize(image_size, antialias=True),
            T.CenterCrop(image_size) if crop else T.Lambda(lambda x: x),
        ]
    )
    target: torch.Tensor = preprocessing(target)
    target = target / 255
    target = target.to(device)
    target = target.to(dtype)

    return target
