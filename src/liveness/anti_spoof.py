from pathlib import Path

import torch

from .vendor.MiniFASNet import MiniFASNetV2


PROJECT_ROOT = Path(__file__).resolve().parents[2]

WEIGHTS_PATH = (
    PROJECT_ROOT
    / "models"
    / "antispoof"
    / "2.7_80x80_MiniFASNetV2.pth"
)


def load_model():
    if not WEIGHTS_PATH.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {WEIGHTS_PATH}")

    model = MiniFASNetV2(conv6_kernel=(5, 5), num_classes=3)

    weights = torch.load(
        WEIGHTS_PATH,
        map_location="cpu",
        weights_only=True,
    )

    weights = {
        name.removeprefix("module."): value
        for name, value in weights.items()
    }

    model.load_state_dict(weights, strict=True)
    model.eval()

    return model


if __name__ == "__main__":
    model = load_model()

    dummy_image = torch.zeros(1, 3, 80, 80)

    with torch.inference_mode():
        output = model(dummy_image)

    assert output.shape == (1, 3), "Unexpected output shape"
    assert torch.isfinite(output).all().item(), "Invalid output values"

    print("Model loaded successfully.")
    print("Device:", next(model.parameters()).device)
    print("Output shape:", tuple(output.shape))
    print("Smoke test passed.")