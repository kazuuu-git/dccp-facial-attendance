"""
Milestone 2: Passive Liveness & Anti-Spoofing Detection
Owner: Corliss (Implementation, Callable Engine Interface, and Threshold Policy)

Uses MiniFASNet models to detect presentation attacks (printed photos, replay screens).
Integrates directly with app_scanner.py and app_dashboard.py via PassiveLivenessEngine.
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch

from .threshold import evaluate_liveness
from .vendor.generate_patches import CropImage
from .vendor.MiniFASNet import MiniFASNetV1SE, MiniFASNetV2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS_DIR = PROJECT_ROOT / "models" / "antispoof"
DEFAULT_WEIGHTS_PATH = WEIGHTS_DIR / "2.7_80x80_MiniFASNetV2.pth"

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


def _load_checkpoint(checkpoint_path: Path):
    if checkpoint_path.name not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported checkpoint: {checkpoint_path.name}")

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


def load_model(weights_path: Optional[Union[str, Path]] = None) -> MiniFASNetV2:
    """Backward-compatible loader for the original MiniFASNetV2 model."""
    target_path = Path(weights_path) if weights_path else DEFAULT_WEIGHTS_PATH
    if not target_path.is_file():
        raise FileNotFoundError(f"Missing anti-spoof checkpoint: {target_path}")

    return _load_checkpoint(target_path)["model"]


def load_models(weights_dir: Optional[Union[str, Path]] = None) -> list:
    """Load every supported anti-spoofing checkpoint found in the weights folder."""
    target_dir = Path(weights_dir) if weights_dir else WEIGHTS_DIR
    loaded_models = []

    for checkpoint_path in sorted(target_dir.glob("*.pth")):
        if checkpoint_path.name not in MODEL_CONFIGS:
            continue
        loaded_models.append(_load_checkpoint(checkpoint_path))

    if not loaded_models:
        raise FileNotFoundError(
            f"No supported anti-spoofing checkpoints found in: {target_dir}"
        )

    return loaded_models


class PassiveLivenessEngine:
    """
    Authoritative passive liveness inference engine for kiosk and dashboard.

    Extracts multi-scale patches around the face bounding box, feeds them into
    the loaded MiniFASNet checkpoint(s), and applies the strict threshold policy:
      1. LIVE (Class 1) must be the unique highest-scoring class (ties rejected).
      2. Average live score must meet or exceed the configured threshold.
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

        if not (0.0 <= self.threshold <= 1.0):
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {self.threshold}")

        if weights_path is not None:
            path = Path(weights_path)
            if not path.is_file():
                raise FileNotFoundError(f"Missing anti-spoof checkpoint: {path}")
            self.model_entries = [_load_checkpoint(path)]
        else:
            self.model_entries = load_models(WEIGHTS_DIR)

        self.cropper = CropImage()
        model_names = [m["name"] for m in self.model_entries]
        print(
            f"[INFO] PassiveLivenessEngine initialized with {len(self.model_entries)} "
            f"model(s): {model_names} (threshold={self.threshold:.2f})."
        )

    def _invalid(self, reason: str) -> dict:
        """Decision for a check that could not be completed (no score is reported)."""
        return {
            "status": "INVALID",
            "passed": False,
            "live_score": None,
            "average_scores": None,
            "threshold": self.threshold,
            "reason": reason,
        }

    def check_passive_liveness(
        self, frame_bgr: np.ndarray, bbox: Optional[List[int]] = None
    ) -> Tuple[bool, float]:
        """
        Evaluates whether a detected face is live or a spoof attack.

        Backward-compatible wrapper around evaluate(); runs inference once.

        Args:
            frame_bgr: Full frame from camera in OpenCV BGR format.
            bbox: Bounding box in [x1, y1, x2, y2] format.

        Returns:
            Tuple of (is_real: bool, liveness_score: float)
            - is_real is True ONLY IF LIVE uniquely wins and live_score >= threshold.
            - liveness_score is the average softmax score for the LIVE class
              (0.0 when the check could not be completed).
        """
        return decision_to_result(self.evaluate(frame_bgr, bbox))

    def evaluate(
        self, frame_bgr: np.ndarray, bbox: Optional[List[int]] = None
    ) -> dict:
        """
        Runs passive liveness inference once and returns the full decision.

        Returns a dict with keys: status ("PASS" | "REJECT" | "INVALID"),
        passed, live_score (None when INVALID), average_scores, threshold, reason.
        Scores are model outputs, not calibrated probabilities.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return self._invalid("No usable frame.")

        h, w = frame_bgr.shape[:2]

        if bbox is None or len(bbox) != 4:
            return self._invalid("No usable face bounding box.")

        x1, y1, x2, y2 = bbox
        box_x = max(0, int(x1))
        box_y = max(0, int(y1))
        box_w = int(x2 - x1)
        box_h = int(y2 - y1)

        if box_w <= 0 or box_h <= 0 or box_x >= w or box_y >= h:
            return self._invalid("Face bounding box is empty or outside the frame.")

        # Collect softmax score vectors from each loaded model
        model_scores = []

        for entry in self.model_entries:
            in_h, in_w = entry["input_size"]
            try:
                crop = self.cropper.crop(
                    org_img=frame_bgr,
                    bbox=[box_x, box_y, box_w, box_h],
                    scale=entry["crop_scale"],
                    out_w=in_w,
                    out_h=in_h,
                    crop=True,
                )
            except Exception as e:
                print(f"[WARN] Anti-spoof crop failed for {entry['name']}: {e}")
                return self._invalid("Face crop failed.")

            if crop is None or crop.size == 0 or crop.shape[:2] != (in_h, in_w):
                return self._invalid("Face crop is empty or has the wrong size.")

            # Preserve OpenCV BGR order and float32 in range [0, 255]
            tensor = (
                torch.from_numpy(crop.transpose(2, 0, 1).copy())
                .float()
                .unsqueeze(0)
            )

            try:
                with torch.inference_mode():
                    logits = entry["model"](tensor)
                    scores = torch.softmax(logits, dim=1)[0].tolist()
                model_scores.append(scores)
            except Exception as e:
                print(f"[WARN] Anti-spoof inference failed for {entry['name']}: {e}")
                return self._invalid("Model inference failed.")

        if not model_scores:
            return self._invalid("No model scores were produced.")

        return evaluate_liveness(model_scores, threshold=self.threshold)


def decision_to_result(decision: dict) -> Tuple[bool, float]:
    """Converts an evaluate() decision to the legacy (is_real, live_score) tuple."""
    passed = bool(decision["passed"])
    live_score = float(decision["live_score"]) if decision["live_score"] is not None else 0.0
    return passed, live_score


if __name__ == "__main__":
    engine = PassiveLivenessEngine()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_bbox = [100, 100, 250, 250]
    is_real, score = engine.check_passive_liveness(dummy_frame, dummy_bbox)
    print(f"Smoke test passed: is_real={is_real}, live_score={score:.4f}")
