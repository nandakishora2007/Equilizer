from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "checkpoints" / "deepfake_efficientnet_b0.pth"

IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Detector:
    """
    EfficientNet-B0 binary deepfake detector.

    Required class mapping:
        index 0 -> Real
        index 1 -> Fake

    The checkpoint must contain weights for an EfficientNet-B0
    whose classifier has 2 output neurons.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()

        return cls._instance

    def _initialize(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Deepfake model checkpoint not found:\n"
                f"{MODEL_PATH}\n\n"
                "Place the trained EfficientNet-B0 checkpoint there."
            )

        self.device = DEVICE

        self.model = models.efficientnet_b0(weights=None)

        in_features = self.model.classifier[1].in_features

        self.model.classifier[1] = nn.Linear(
            in_features,
            2
        )

        checkpoint = torch.load(
            MODEL_PATH,
            map_location=self.device
        )

        state_dict = self._extract_state_dict(checkpoint)

        cleaned_state_dict = {}

        for key, value in state_dict.items():
            # Handle common wrappers from DataParallel / compiled models.
            for prefix in (
                "module.",
                "model.",
                "_orig_mod."
            ):
                if key.startswith(prefix):
                    key = key[len(prefix):]

            cleaned_state_dict[key] = value

        try:
            self.model.load_state_dict(
                cleaned_state_dict,
                strict=True
            )
        except RuntimeError as exc:
            raise RuntimeError(
                "The checkpoint is incompatible with the expected "
                "EfficientNet-B0 architecture with 2 output classes "
                "(0=Real, 1=Fake).\n\n"
                f"Original error:\n{exc}"
            ) from exc

        self.model.to(self.device)
        self.model.eval()

        # Standard ImageNet preprocessing.
        # This must match the preprocessing used during training.
        self.transform = transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]
        )

    @staticmethod
    def _extract_state_dict(checkpoint):
        """
        Accept common PyTorch checkpoint formats.
        """

        if isinstance(checkpoint, dict):

            if "state_dict" in checkpoint:
                return checkpoint["state_dict"]

            if "model_state_dict" in checkpoint:
                return checkpoint["model_state_dict"]

            if "model" in checkpoint and isinstance(
                checkpoint["model"],
                dict
            ):
                return checkpoint["model"]

            # Raw state_dict.
            if all(
                isinstance(value, torch.Tensor)
                for value in checkpoint.values()
            ):
                return checkpoint

        raise RuntimeError(
            "Could not find a valid PyTorch state_dict in the checkpoint."
        )

    @torch.inference_mode()
    def predict(self, face_image) -> Dict:
        """
        Predict one face crop.

        Returns:
            {
                "prediction": "Real" or "Fake",
                "confidence": 0-1,
                "fake_probability": 0-1,
                "real_probability": 0-1
            }
        """

        if isinstance(face_image, Image.Image):
            image = face_image.convert("RGB")
        else:
            image = Image.fromarray(face_image).convert("RGB")

        tensor = self.transform(image)

        # [C, H, W] -> [1, C, H, W]
        tensor = tensor.unsqueeze(0).to(self.device)

        logits = self.model(tensor)

        if logits.ndim != 2 or logits.shape[0] != 1 or logits.shape[1] != 2:
            raise RuntimeError(
                "Detector returned an unexpected tensor shape: "
                f"{tuple(logits.shape)}. Expected [1, 2]."
            )

        probabilities = torch.softmax(
            logits,
            dim=1
        )[0]

        real_probability = float(probabilities[0].item())
        fake_probability = float(probabilities[1].item())

        if fake_probability >= real_probability:
            prediction = "Fake"
            confidence = fake_probability
        else:
            prediction = "Real"
            confidence = real_probability

        return {
            "prediction": prediction,
            "confidence": confidence,
            "fake_probability": fake_probability,
            "real_probability": real_probability
        }


def get_detector():
    """
    Return the shared detector instance.
    """
    return Detector()