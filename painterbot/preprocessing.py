from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image


def load_image(
    image_path: Path, device: torch.device, image_size: int = 512
) -> torch.Tensor:
    image_path = Path(image_path)
    target = Image.open(image_path).convert("RGB")

    preprocessing = T.Compose(
        [
            T.PILToTensor(),
            T.Resize(image_size, antialias=True),
            T.CenterCrop(image_size),
        ]
    )
    target: torch.Tensor = preprocessing(target)
    target = target.to(device)
    target = target / 255

    return target
