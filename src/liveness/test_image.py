import argparse

import cv2
import torch

from .anti_spoof import load_models
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

    models = load_models()
    cropper = CropImage()

    combined_scores = torch.zeros(3)
    model_results = []

    for item in models:
        crop = cropper.crop(
            org_img=image,
            bbox=[x, y, width, height],
            scale=item["crop_scale"],
            out_w=item["input_size"][0],
            out_h=item["input_size"][1],
            crop=True,
        )

        # Preserve OpenCV BGR order and the reference 0-255 range.
        tensor = (
            torch.from_numpy(crop.transpose(2, 0, 1).copy())
            .float()
            .unsqueeze(0)
        )

        with torch.inference_mode():
            logits = item["model"](tensor)
            scores = torch.softmax(logits, dim=1)[0]

        combined_scores += scores

        model_results.append(
            {
                "name": item["name"],
                "crop_scale": item["crop_scale"],
                "scores": scores,
                "prediction": "LIVE" if int(scores.argmax().item()) == 1 else "SPOOF",
            }
        )

    predicted_class = int(combined_scores.argmax().item())

    print("\nIndividual model results:")
    for result in model_results:
        print(f"- {result['name']} (crop scale {result['crop_scale']}):")
        print(f"  Scores: {result['scores'].tolist()}")
        print(f"  Live score: {round(result['scores'][1].item(), 4)}")
        print(f"  Prediction: {result['prediction']}")

    print("\nCombined scores:", combined_scores.tolist())
    print("Combined live score:", round(combined_scores[1].item(), 4))
    print(
        "Ensemble prediction:",
        "LIVE" if predicted_class == 1 else "SPOOF",
    )
    print("Diagnostic result only; not a validated security decision.")


if __name__ == "__main__":
    main()