import argparse

import cv2
import torch

from .anti_spoof import load_model
from .vendor.generate_patches import CropImage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    image = cv2.imread(args.image)

    if image is None:
        raise FileNotFoundError(f"Cannot read image: {args.image}")

    # Select the face manually until the detector is integrated.
    bbox = cv2.selectROI(
        "Select face, then press Enter",
        image,
        showCrosshair=True,
        fromCenter=False,
    )
    cv2.destroyAllWindows()

    x, y, width, height = map(int, bbox)

    if width <= 0 or height <= 0:
        print("No face selected. Test cancelled.")
        return

    crop = CropImage().crop(
        org_img=image,
        bbox=[x, y, width, height],
        scale=2.7,
        out_w=80,
        out_h=80,
        crop=True,
    )

    # Preserve OpenCV BGR order and the reference 0–255 range.
    tensor = (
        torch.from_numpy(crop.transpose(2, 0, 1).copy())
        .float()
        .unsqueeze(0)
    )

    model = load_model()

    with torch.inference_mode():
        logits = model(tensor)
        scores = torch.softmax(logits, dim=1)[0]

    predicted_class = int(scores.argmax().item())

    print("Class scores:", scores.tolist())
    print("Live score:", round(scores[1].item(), 4))
    print(
        "Single-model prediction:",
        "LIVE" if predicted_class == 1 else "SPOOF",
    )
    print("Diagnostic result only; not a validated security decision.")


if __name__ == "__main__":
    main()