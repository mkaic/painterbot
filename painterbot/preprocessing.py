from pathlib import Path
from PIL import Image
import torchvision.transforms as T
import torch


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
