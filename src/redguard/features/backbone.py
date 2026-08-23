import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import (
    ResNet18_Weights,
    resnet18,
)


class VisionBackbone:
    """Shared visual feature extractor for RedGuard AI."""

    def __init__(
        self,
        pretrained: bool = True,
        device: str | None = None,
    ) -> None:
        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        weights = (
            ResNet18_Weights.DEFAULT
            if pretrained
            else None
        )

        self.model = resnet18(
            weights=weights
        ).to(self.device)

        self.model.eval()

        for parameter in self.model.parameters():
            parameter.requires_grad = False

        self.pretrained = pretrained

    def _prepare(
        self,
        image: np.ndarray,
    ) -> torch.Tensor:
        if not isinstance(image, np.ndarray):
            raise ValueError(
                "image must be a NumPy array."
            )

        if image.size == 0:
            raise ValueError(
                "image cannot be empty."
            )

        if image.ndim == 2:
            image = cv2.cvtColor(
                image,
                cv2.COLOR_GRAY2RGB,
            )
        elif image.shape[2] == 3:
            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB,
            )
        elif image.shape[2] == 4:
            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGRA2RGB,
            )
        else:
            raise ValueError(
                "unsupported image channels."
            )

        image = cv2.resize(
            image,
            (224, 224),
            interpolation=cv2.INTER_AREA,
        )

        tensor = torch.from_numpy(
            image
        ).float()

        tensor = (
            tensor
            .permute(2, 0, 1)
            .unsqueeze(0)
            / 255.0
        )

        mean = torch.tensor(
            [0.485, 0.456, 0.406]
        ).view(1, 3, 1, 1)

        std = torch.tensor(
            [0.229, 0.224, 0.225]
        ).view(1, 3, 1, 1)

        tensor = (
            tensor - mean
        ) / std

        return tensor.to(
            self.device
        )

    @torch.inference_mode()
    def global_embedding(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        x = self._prepare(image)

        model = self.model

        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)

        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)

        x = model.avgpool(x)
        x = torch.flatten(x, 1)

        x = F.normalize(
            x,
            p=2,
            dim=1,
        )

        return (
            x[0]
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    @torch.inference_mode()
    def patch_embeddings(
        self,
        image: np.ndarray,
    ) -> tuple[np.ndarray, tuple[int, int]]:
        x = self._prepare(image)

        model = self.model

        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)

        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)

        _, channels, height, width = x.shape

        patches = (
            x[0]
            .permute(1, 2, 0)
            .reshape(-1, channels)
        )

        patches = F.normalize(
            patches,
            p=2,
            dim=1,
        )

        return (
            patches.cpu().numpy().astype(
                np.float32
            ),
            (height, width),
        )
