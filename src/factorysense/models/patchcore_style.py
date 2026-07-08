from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def select_device(device: str = "auto") -> torch.device:
    if device == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    return torch.device(device)


def load_image_tensor(path: str | Path, image_size: int = 256) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    image = image.resize((image_size, image_size))

    array = np.asarray(image).astype("float32") / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)

    mean = IMAGENET_MEAN
    std = IMAGENET_STD

    return (tensor - mean) / std


class ResNet18Layer2Extractor(nn.Module):
    """
    Lightweight feature extractor using ResNet18 up to layer2.

    For 256x256 input, the feature map is usually 32x32.
    Each spatial position is treated as a patch embedding.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        weights = None

        if pretrained:
            try:
                weights = ResNet18_Weights.DEFAULT
                backbone = resnet18(weights=weights)
            except Exception as exc:
                print("Warning: could not load pretrained ResNet18 weights.")
                print(f"Reason: {exc}")
                print("Falling back to random weights. Results will be less meaningful.")
                backbone = resnet18(weights=None)
        else:
            backbone = resnet18(weights=None)

        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class PatchCoreStyleAnomalyDetector:
    """
    A lightweight PatchCore-style anomaly detector.

    This model is designed for learning:
    - extract pretrained CNN patch features
    - build a memory bank from normal images
    - score test patches using nearest-neighbor distance
    - create anomaly maps and Pass/Reject decisions
    """

    def __init__(
        self,
        image_size: int = 256,
        device: str = "auto",
        pretrained: bool = True,
    ):
        self.image_size = image_size
        self.device = select_device(device)
        self.pretrained = pretrained

        self.extractor = ResNet18Layer2Extractor(pretrained=pretrained)
        self.extractor.to(self.device)
        self.extractor.eval()

        self.memory_bank: torch.Tensor | None = None
        self.threshold: float | None = None
        self.base_threshold: float | None = None
        self.threshold_multiplier: float = 1.0
        self.threshold_margin: float = 0.0

    @torch.no_grad()
    def extract_feature_map(self, image_path: str | Path) -> torch.Tensor:
        tensor = load_image_tensor(image_path, image_size=self.image_size).to(self.device)
        features = self.extractor(tensor)
        features = F.normalize(features, p=2, dim=1)
        return features.squeeze(0)

    @torch.no_grad()
    def extract_patch_features(self, image_path: str | Path) -> torch.Tensor:
        feature_map = self.extract_feature_map(image_path)
        channels, height, width = feature_map.shape

        patches = feature_map.permute(1, 2, 0).reshape(height * width, channels)
        patches = F.normalize(patches, p=2, dim=1)

        return patches.detach().cpu()

    def fit(
        self,
        normal_image_paths: Iterable[str | Path],
        max_memory_patches: int = 5000,
        random_state: int = 42,
    ) -> None:
        paths = list(normal_image_paths)

        if not paths:
            raise ValueError("No normal images were provided for fitting.")

        all_patches = []

        for path in paths:
            patches = self.extract_patch_features(path)
            all_patches.append(patches)

        memory_bank = torch.cat(all_patches, dim=0)

        if max_memory_patches > 0 and memory_bank.shape[0] > max_memory_patches:
            generator = torch.Generator().manual_seed(random_state)
            indices = torch.randperm(memory_bank.shape[0], generator=generator)
            indices = indices[:max_memory_patches]
            memory_bank = memory_bank[indices]

        self.memory_bank = F.normalize(memory_bank, p=2, dim=1).to(self.device)

    @torch.no_grad()
    def anomaly_map(self, image_path: str | Path) -> np.ndarray:
        if self.memory_bank is None:
            raise ValueError("Memory bank is not fitted yet.")

        feature_map = self.extract_feature_map(image_path)
        channels, height, width = feature_map.shape

        patches = feature_map.permute(1, 2, 0).reshape(height * width, channels)
        patches = F.normalize(patches, p=2, dim=1)

        distances = torch.cdist(patches, self.memory_bank)
        min_distances = distances.min(dim=1).values

        low_res_map = min_distances.reshape(1, 1, height, width)

        high_res_map = F.interpolate(
            low_res_map,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        return high_res_map.squeeze().detach().cpu().numpy().astype("float32")

    def score(self, image_path: str | Path) -> float:
        amap = self.anomaly_map(image_path)

        return float(np.quantile(amap, 0.99))

    def calibrate_threshold(
        self,
        normal_image_paths: Iterable[str | Path],
        quantile: float = 0.95,
        multiplier: float = 1.2,
        margin: float = 0.0,
    ) -> float:
        if not 0 < quantile < 1:
            raise ValueError("quantile must be between 0 and 1")

        if multiplier <= 0:
            raise ValueError("multiplier must be positive")

        scores = [self.score(path) for path in normal_image_paths]

        if not scores:
            raise ValueError("No normal images were provided for threshold calibration.")

        self.base_threshold = float(np.quantile(scores, quantile))
        self.threshold_multiplier = float(multiplier)
        self.threshold_margin = float(margin)
        self.threshold = float(self.base_threshold * multiplier + margin)

        return self.threshold

    def predict(self, image_path: str | Path) -> dict:
        if self.threshold is None:
            raise ValueError("Threshold is not calibrated yet.")

        score = self.score(image_path)
        decision = "Reject" if score >= self.threshold else "Pass"

        return {
            "image_path": str(image_path),
            "anomaly_score": score,
            "threshold": self.threshold,
            "decision": decision,
        }

    def binary_mask(self, image_path: str | Path, threshold: float | None = None) -> np.ndarray:
        amap = self.anomaly_map(image_path)
        used_threshold = self.threshold if threshold is None else threshold

        if used_threshold is None:
            raise ValueError("Threshold is not available.")

        return (amap >= used_threshold).astype("uint8")

    def save(self, path: str | Path) -> None:
        if self.memory_bank is None:
            raise ValueError("Memory bank is not fitted yet.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            path,
            image_size=self.image_size,
            memory_bank=self.memory_bank.detach().cpu().numpy(),
            threshold=-1.0 if self.threshold is None else self.threshold,
            base_threshold=-1.0 if self.base_threshold is None else self.base_threshold,
            threshold_multiplier=self.threshold_multiplier,
            threshold_margin=self.threshold_margin,
            pretrained=int(self.pretrained),
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        device: str = "auto",
    ) -> "PatchCoreStyleAnomalyDetector":
        data = np.load(path)
        image_size = int(data["image_size"])
        pretrained = bool(int(data["pretrained"])) if "pretrained" in data.files else True

        model = cls(
            image_size=image_size,
            device=device,
            pretrained=pretrained,
        )

        memory_bank = torch.from_numpy(data["memory_bank"]).float()
        model.memory_bank = F.normalize(memory_bank, p=2, dim=1).to(model.device)

        threshold = float(data["threshold"])
        model.threshold = None if threshold < 0 else threshold

        if "base_threshold" in data.files:
            base_threshold = float(data["base_threshold"])
            model.base_threshold = None if base_threshold < 0 else base_threshold

        if "threshold_multiplier" in data.files:
            model.threshold_multiplier = float(data["threshold_multiplier"])

        if "threshold_margin" in data.files:
            model.threshold_margin = float(data["threshold_margin"])

        return model
