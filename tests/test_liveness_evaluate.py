"""
P5 tests for PassiveLivenessEngine.evaluate() and the check_passive_liveness()
wrapper: one shared inference path, identical legacy results, and INVALID
decisions that never carry an invented score.

The real checkpoints are loaded (as in test_liveness_policy.py), then each
model is replaced with a fake that returns fixed logits and counts calls, so
results are deterministic without a camera or real faces.
"""

import math
import unittest
from unittest import mock

import numpy as np
import torch

import app_scanner
from src.liveness.anti_spoof import PassiveLivenessEngine, decision_to_result
from src.liveness.threshold import evaluate_liveness

FRAME = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
BBOX = [220, 150, 360, 320]


class FakeModel:
    def __init__(self, probs=None, logits=None, error=None):
        self.logits = logits if logits is not None else [math.log(p) for p in probs]
        self.error = error
        self.calls = 0

    def __call__(self, tensor):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return torch.tensor([self.logits], dtype=torch.float32)


class EvaluateTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = PassiveLivenessEngine(threshold=0.50)
        cls.original_models = [entry["model"] for entry in cls.engine.model_entries]

    def tearDown(self):
        for entry, model in zip(self.engine.model_entries, self.original_models):
            entry["model"] = model

    def install(self, *fakes):
        self.assertEqual(len(fakes), len(self.engine.model_entries))
        for entry, fake in zip(self.engine.model_entries, fakes):
            entry["model"] = fake
        return fakes


class TestSharedInferencePath(EvaluateTestCase):
    CASES = {
        "pass": ([0.05, 0.90, 0.05], [0.10, 0.85, 0.05]),
        "below_threshold": ([0.30, 0.45, 0.25], [0.35, 0.40, 0.25]),
        "spoof_wins": ([0.80, 0.10, 0.10], [0.70, 0.20, 0.10]),
        "tie": ([0.45, 0.45, 0.10], [0.45, 0.45, 0.10]),
    }

    def test_wrapper_matches_evaluate_and_policy(self):
        for name, probs in self.CASES.items():
            with self.subTest(case=name):
                self.install(*(FakeModel(p) for p in probs))
                decision = self.engine.evaluate(FRAME, BBOX)
                legacy = self.engine.check_passive_liveness(FRAME, BBOX)
                self.assertEqual(legacy, decision_to_result(decision))

                # Same result as applying the threshold policy to the model scores.
                scores = [torch.softmax(torch.tensor([[math.log(v) for v in p]]), dim=1)[0].tolist() for p in probs]
                expected = evaluate_liveness(scores, threshold=0.50)
                self.assertEqual(decision["status"], expected["status"])
                self.assertAlmostEqual(decision["live_score"], expected["live_score"], places=6)
                self.assertEqual(decision["threshold"], 0.50)

        self.install(*(FakeModel(p) for p in self.CASES["tie"]))
        self.assertEqual(self.engine.evaluate(FRAME, BBOX)["status"], "REJECT")

    def test_each_model_runs_once_per_check(self):
        fakes = self.install(*(FakeModel(p) for p in self.CASES["pass"]))
        self.engine.check_passive_liveness(FRAME, BBOX)
        self.assertEqual([f.calls for f in fakes], [1] * len(fakes))
        self.engine.evaluate(FRAME, BBOX)
        self.assertEqual([f.calls for f in fakes], [2] * len(fakes))


class TestInvalidDecisions(EvaluateTestCase):
    def assert_invalid(self, decision):
        self.assertEqual(decision["status"], "INVALID")
        self.assertFalse(decision["passed"])
        self.assertIsNone(decision["live_score"], "INVALID must not report a score")
        self.assertEqual(decision["threshold"], 0.50)
        self.assertTrue(decision["reason"])
        self.assertEqual(decision_to_result(decision), (False, 0.0))

    def test_unusable_inputs(self):
        self.install(*(FakeModel([0.05, 0.9, 0.05]) for _ in self.engine.model_entries))
        for frame, bbox in [
            (None, BBOX),
            (np.zeros((0, 0, 3), dtype=np.uint8), BBOX),
            (FRAME, None),
            (FRAME, [1, 2, 3]),
            (FRAME, [50, 50, 50, 50]),
            (FRAME, [700, 700, 800, 800]),
        ]:
            with self.subTest(bbox=bbox, frame_none=frame is None):
                self.assert_invalid(self.engine.evaluate(frame, bbox))
                self.assertEqual(self.engine.check_passive_liveness(frame, bbox), (False, 0.0))

    def test_crop_failure(self):
        self.install(*(FakeModel([0.05, 0.9, 0.05]) for _ in self.engine.model_entries))
        with mock.patch.object(self.engine.cropper, "crop", side_effect=RuntimeError("synthetic")):
            self.assert_invalid(self.engine.evaluate(FRAME, BBOX))

    def test_inference_failure(self):
        fakes = [FakeModel(logits=[0.0, 0.0, 0.0], error=RuntimeError("synthetic"))] + [
            FakeModel([0.05, 0.9, 0.05]) for _ in self.engine.model_entries[1:]
        ]
        self.install(*fakes)
        self.assert_invalid(self.engine.evaluate(FRAME, BBOX))

    def test_non_finite_scores(self):
        self.install(*(FakeModel(logits=[float("nan"), 0.0, 0.0]) for _ in self.engine.model_entries))
        self.assert_invalid(self.engine.evaluate(FRAME, BBOX))


class TestLivenessOverlayText(unittest.TestCase):
    def check_wording(self, text):
        self.assertNotIn("%", text)
        self.assertNotIn("probab", text.lower())
        self.assertNotIn("spoof", text.lower())

    def test_pass_reject_invalid(self):
        text, _ = app_scanner.format_liveness_line({"status": "PASS", "live_score": 0.8734, "threshold": 0.5})
        self.assertEqual(text, "Liveness: PASS | model live score 0.87 | threshold 0.50")
        self.check_wording(text)

        text, _ = app_scanner.format_liveness_line({"status": "REJECT", "live_score": 0.31, "threshold": 0.5})
        self.assertEqual(text, "Liveness: REJECT | model live score 0.31 | threshold 0.50")
        self.check_wording(text)

        text, _ = app_scanner.format_liveness_line({"status": "INVALID", "live_score": None, "threshold": 0.5})
        self.assertEqual(text, "Liveness: INVALID - check could not be completed | threshold 0.50")
        self.assertNotIn("score", text)
        self.check_wording(text)

    def test_blocked_wording(self):
        self.assertEqual(app_scanner.BLOCKED_BANNER, "BLOCKED: liveness check failed")


if __name__ == "__main__":
    unittest.main()
