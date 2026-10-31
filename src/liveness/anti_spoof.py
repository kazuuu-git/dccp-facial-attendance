from pathlib import Path

import torch

from .vendor.MiniFASNet import MiniFASNetV2, MiniFASNetV1SE


PROJECT_ROOT = Path(__file__).resolve().parents[2]

WEIGHTS_DIR = PROJECT_ROOT / "models" / "antispoof"

MODEL_CONFIGS = {
    "2.7_80x80_MiniFASNetV2.pth": {
        "model_fn": MiniFASNetV2,
        "conv6_kernel": (5, 5),
        "num_classes": 3,
        "crop_scale": 2.7,
        "input_size": (80, 80),
    },
    "4_0_0_80x80_MiniFASNetV1SE.pth": {
        "model_fn": MiniFASNetV1SE,
        "conv6_kernel": (5, 5),
        "num_classes": 3,
        "crop_scale": 4.0,
        "input_size": (80, 80),
    },
}


def _clean_state_dict(weights):
    return {
        name.removeprefix("module."): value
        for name, value in weights.items()
    }


def _load_checkpoint(checkpoint_path):
    config = MODEL_CONFIGS[checkpoint_path.name]

    model = config["model_fn"](
        conv6_kernel=config["conv6_kernel"],
        num_classes=config["num_classes"],
    )

    weights = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(_clean_state_dict(weights), strict=True)
    model.eval()

    return {
        "name": checkpoint_path.name,
        "model": model,
        "crop_scale": config["crop_scale"],
        "input_size": config["input_size"],
        "num_classes": config["num_classes"],
    }


def load_model():
    """Backward-compatible loader for the original MiniFASNetV2 model."""
    checkpoint_path = WEIGHTS_DIR / "2.7_80x80_MiniFASNetV2.pth"

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    return _load_checkpoint(checkpoint_path)["model"]


def load_models():
    """Load every supported anti-spoofing checkpoint in the model folder."""
    loaded_models = []

    for checkpoint_path in sorted(WEIGHTS_DIR.glob("*.pth")):
        if checkpoint_path.name not in MODEL_CONFIGS:
            print(f"Skipping unsupported checkpoint: {checkpoint_path.name}")
            continue

        loaded_models.append(_load_checkpoint(checkpoint_path))

    if not loaded_models:
        raise FileNotFoundError(
            f"No supported anti-spoofing checkpoints found in: {WEIGHTS_DIR}"
        )

    return loaded_models


if __name__ == "__main__":
    models = load_models()

    print(f"Loaded {len(models)} anti-spoofing model(s).")

    for item in models:
        height, width = item["input_size"]
        dummy_image = torch.zeros(1, 3, height, width)

        with torch.inference_mode():
            output = item["model"](dummy_image)

        assert output.shape == (1, item["num_classes"]), (
            f"Unexpected output shape for {item['name']}"
        )
        assert torch.isfinite(output).all().item(), (
            f"Invalid output values for {item['name']}"
        )

        print(f"{item['name']} loaded successfully.")
        print("Device:", next(item["model"].parameters()).device)
        print("Output shape:", tuple(output.shape))

    print("Smoke test passed.")