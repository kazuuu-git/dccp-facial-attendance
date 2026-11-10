import argparse
import math

import cv2
import torch

from .anti_spoof import load_models
from .threshold import evaluate_liveness
from .vendor.generate_patches import CropImage


def main():
    parser = argparse.ArgumentParser(
        description="Test MiniFASNet ensemble scores with a liveness threshold."
    )
    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--threshold",
        required=True,
        type=float,
        help="Minimum average live score from 0 to 1; requires validation.",
    )
    args = parser.parse_args()

    if not math.isfinite(args.threshold) or not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be a finite number from 0 to 1.")

    image = cv2.imread(args.image)

    if image is None:
        raise FileNotFoundError(f"Cannot read image: {args.image}")

    # Select the face manually until the detector is integrated.
    try:
        bbox = cv2.selectROI(
            "Select face, then press Enter",
            image,
            showCrosshair=True,
            fromCenter=False,
        )
    finally:
        cv2.destroyAllWindows()

    x, y, width, height = map(int, bbox)

    if width <= 0 or height <= 0:
        print("No face selected. Test cancelled.")
        return

    models = load_models()
    cropper = CropImage()
    model_results = []

    for item in models:
        input_height, input_width = item["input_size"]

        crop = cropper.crop(
            org_img=image,
            bbox=[x, y, width, height],
            scale=item["crop_scale"],
            out_w=input_width,
            out_h=input_height,
            crop=True,
        )

        if crop is None or crop.size == 0:
            print("Decision: INVALID")
            print(f"Reason: Empty crop for {item['name']}.")
            return

        # Preserve OpenCV BGR order and the reference 0-255 range.
        tensor = (
            torch.from_numpy(crop.transpose(2, 0, 1).copy())
            .float()
            .unsqueeze(0)
        )

        with torch.inference_mode():
            logits = item["model"](tensor)
            scores = torch.softmax(logits, dim=1)[0]

        model_results.append(
            {
                "name": item["name"],
                "crop_scale": item["crop_scale"],
                "scores": scores.tolist(),
            }
        )

    decision = evaluate_liveness(
        [item["scores"] for item in model_results],
        threshold=args.threshold,
    )

    print("\nIndividual model results:")

    for item in model_results:
        print(f"- {item['name']} (crop scale {item['crop_scale']}):")
        print("  Scores:", item["scores"])
        print("  Live score:", round(item["scores"][1], 4))

    print("\nModels evaluated:", len(model_results))
    print("Average class scores:", decision["average_scores"])
    print("Average live score:", decision["live_score"])
    print("Threshold:", decision["threshold"])
    print("Decision:", decision["status"])
    print("Reason:", decision["reason"])
    print("Diagnostic result only; threshold is not yet validated.")


if __name__ == "__main__":
    main()