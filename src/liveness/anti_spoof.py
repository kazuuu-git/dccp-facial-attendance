"""
Milestone 2: Passive Liveness & Anti-Spoofing Detection
Owner: Corliss (Implementation & Callable Engine Interface)

Uses MiniFASNetV2 to detect presentation attacks (printed photos, replay screens).
Integrates directly with app_scanner.py via PassiveLivenessEngine.
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch

from .vendor.generate_patches import CropImage
from .vendor.MiniFASNet import MiniFASNetV2

PROJECT_ROOT = Path(__file__).resolve().parents[2]

WEIGHTS_PATH = (
    PROJECT_ROOT
    / "models"
    / "antispoof"
    / "2.7_80x80_MiniFASNetV2.pth"
)


def load_model(weights_path: Optional[Union[str, Path]] = None) -> MiniFASNetV2:
    """Loads the pretrained MiniFASNetV2 checkpoint on CPU."""
    target_path = Path(weights_path) if weights_path else WEIGHTS_PATH
    if not target_path.is_file():
        raise FileNotFoundError(f"Missing anti-spoof checkpoint: {target_path}")

    model = MiniFASNetV2(conv6_kernel=(5, 5), num_classes=3)

    weights = torch.load(
        target_path,
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


class PassiveLivenessEngine:
    """
    Callable passive liveness inference engine expected by app_scanner.py.
    
    Extracts an 80x80 patch around the face bounding box (scale=2.7), feeds it into
    MiniFASNetV2, and determines whether the face is genuine (Class 1) or a spoof attempt.
    """

    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        threshold: Optional[float] = None,
    ) -> None:
        if threshold is None:
            self.threshold = float(os.getenv("DCCP_LIVENESS_THRESHOLD", "0.50"))
        else:
            self.threshold = float(threshold)

        self.model = load_model(weights_path)
        self.cropper = CropImage()
        print(f"[INFO] PassiveLivenessEngine loaded MiniFASNetV2 (threshold={self.threshold:.2f}).")

    def check_passive_liveness(
        self, frame_bgr: np.ndarray, bbox: Optional[List[int]] = None
    ) -> Tuple[bool, float]:
        """
        Evaluates whether a detected face is live or a spoof.

        Args:
            frame_bgr: Full frame from camera in BGR format.
            bbox: Bounding box from InsightFace in [x1, y1, x2, y2] format.

        Returns:
            Tuple of (is_real: bool, liveness_score: float)
            - is_real is True if the model predicts Class 1 (LIVE) and meets the threshold.
            - liveness_score is the softmax probability for the LIVE class (0.0 to 1.0).
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return False, 0.0

        h, w = frame_bgr.shape[:2]

        # Convert [x1, y1, x2, y2] from InsightFace to [x, y, box_w, box_h] for MiniFASNet
        if bbox is not None and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            box_x = max(0, int(x1))
            box_y = max(0, int(y1))
            box_w = max(1, int(x2 - x1))
            box_h = max(1, int(y2 - y1))
        else:
            # Fall back to center crop if bbox is omitted
            box_x, box_y, box_w, box_h = 0, 0, w, h

        # Crop face patch with MiniFASNet standard scale 2.7 to 80x80
        try:
            crop = self.cropper.crop(
                org_img=frame_bgr,
                bbox=[box_x, box_y, box_w, box_h],
                scale=2.7,
                out_w=80,
                out_h=80,
                crop=True,
            )
        except Exception as e:
            print(f"[WARN] Anti-spoof crop failed: {e}")
            return False, 0.0

        # Prepare tensor: preserve OpenCV BGR order and 0-255 float range (as trained)
        tensor = (
            torch.from_numpy(crop.transpose(2, 0, 1).copy())
            .float()
            .unsqueeze(0)
        )

        with torch.inference_mode():
            logits = self.model(tensor)
            scores = torch.softmax(logits, dim=1)[0]

        predicted_class = int(scores.argmax().item())
        live_score = float(scores[1].item())

        # Class 1 corresponds to live person in MiniFASNet
        is_real = (predicted_class == 1) and (live_score >= self.threshold)
        return is_real, live_score


if __name__ == "__main__":
    engine = PassiveLivenessEngine()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_bbox = [100, 100, 250, 250]
    is_real, score = engine.check_passive_liveness(dummy_frame, dummy_bbox)
    print(f"Smoke test passed: is_real={is_real}, live_score={score:.4f}")
