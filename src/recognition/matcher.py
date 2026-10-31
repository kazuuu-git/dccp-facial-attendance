"""
Milestone 1: Core Facial Recognition & Enrollment Engine
Owner: Karl Stephen V. Zubiri

Can be imported by the main pipeline OR run directly (`python -m src.recognition.matcher`)
to test face enrollment and cosine similarity matching independently before database/liveness modules are merged.

Performance notes:
  - Only the detection + recognition modules are loaded (no landmarks / gender-age).
  - ONNX Runtime threads are capped and busy-waiting (spinning) is disabled, which is the
    main cause of "100% CPU" on many-core machines. Tune with the env var DCCP_ORT_THREADS.
"""

import os
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import onnxruntime as ort
    from insightface.app import FaceAnalysis
except ImportError:
    ort = None
    FaceAnalysis = None


def _build_session_options(num_threads: int):
    """ONNX Runtime options: few threads, no busy-wait spinning."""
    so = ort.SessionOptions()
    so.intra_op_num_threads = num_threads
    so.inter_op_num_threads = 1
    so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return so


class FaceRecognitionEngine:
    """Handles 512-D facial embedding extraction and cosine similarity matching via InsightFace."""

    def __init__(
        self,
        det_size: Tuple[int, int] = (320, 320),
        sim_threshold: float = 0.50,
        num_threads: Optional[int] = None,
    ):
        self.sim_threshold = sim_threshold
        self.app = None

        if FaceAnalysis is None:
            return

        if num_threads is None:
            num_threads = int(os.getenv("DCCP_ORT_THREADS", "2"))

        kwargs = dict(
            name="buffalo_s",
            providers=["CPUExecutionProvider"],
            allowed_modules=["detection", "recognition"],  # skip landmarks & gender/age
        )
        try:
            self.app = FaceAnalysis(sess_options=_build_session_options(num_threads), **kwargs)
        except TypeError:
            # Older insightface versions may not forward sess_options; fall back safely.
            print("[WARN] This insightface version ignores sess_options; thread limit not applied.")
            self.app = FaceAnalysis(**kwargs)

        self.app.prepare(ctx_id=-1, det_size=det_size)  # -1 = CPU

    def extract_embedding(self, frame_bgr: np.ndarray) -> Optional[Tuple[np.ndarray, list]]:
        """
        Detects the primary face in a BGR frame and extracts its normalized 512-D embedding.

        Returns:
            Tuple of (512-D numpy array, bbox [x1, y1, x2, y2]) or None if no face is detected.
        """
        if self.app is None:
            raise RuntimeError("InsightFace is not installed. Run: pip install insightface onnxruntime")

        faces = self.app.get(frame_bgr)
        if not faces:
            return None

        # Select the largest detected face in the frame
        primary_face = max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        )
        embedding = primary_face.normed_embedding.astype(np.float32)
        bbox = [int(coord) for coord in primary_face.bbox]
        return embedding, bbox

    def extract_averaged_embedding(self, frames_bgr: List[np.ndarray]) -> Optional[np.ndarray]:
        """
        Extracts 512-D embeddings across multiple frames, averages them, and L2-normalizes
        the result for stronger lighting/angle robustness during employee enrollment.
        """
        embeddings = []
        for frame in frames_bgr:
            result = self.extract_embedding(frame)
            if result is not None:
                emb, _ = result
                embeddings.append(emb)

        if not embeddings:
            return None

        mean_emb = np.mean(embeddings, axis=0).astype(np.float32)
        norm = np.linalg.norm(mean_emb)
        return mean_emb / norm if norm > 0 else mean_emb

    @staticmethod
    def compute_cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Computes cosine similarity between two L2-normalized 512-D embeddings."""
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        return float(np.dot(emb1, emb2) / (norm1 * norm2))

    def recognize_employee(
        self, frame_bgr: np.ndarray, enrolled_db: Dict[str, np.ndarray]
    ) -> Tuple[Optional[str], float, Optional[list]]:
        """
        Matches the largest detected face in the frame against enrolled employee embeddings.

        Args:
            frame_bgr: OpenCV BGR image array.
            enrolled_db: Dict mapping employee_id -> 512-D numpy embedding array.

        Returns:
            Tuple of (employee_id or None, best_similarity_score, bbox or None).
        """
        extraction = self.extract_embedding(frame_bgr)
        if extraction is None:
            return None, 0.0, None

        live_embedding, bbox = extraction
        best_emp_id = None
        best_score = -1.0

        for emp_id, ref_embedding in enrolled_db.items():
            score = self.compute_cosine_similarity(live_embedding, ref_embedding)
            if score > best_score:
                best_score = score
                best_emp_id = emp_id

        if best_score >= self.sim_threshold:
            return best_emp_id, best_score, bbox

        return None, max(best_score, 0.0), bbox


if __name__ == "__main__":
    # Standalone Milestone 1 test runner (works before Earl's DB or Corliss's Liveness exist)
    print("[INFO] Initializing InsightFace Recognition Engine (Milestone 1 Standalone Test)...")
    engine = FaceRecognitionEngine(sim_threshold=0.50)
    in_memory_db: Dict[str, np.ndarray] = {}

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("[CONTROLS] Press 'e' to enroll your face as DCCP-KARL-001 | Press 'q' to quit.")

    PROCESS_INTERVAL = 0.25  # seconds between recognitions
    last_proc = 0.0
    emp_id, score, bbox = None, 0.0, None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()

        if time.time() - last_proc >= PROCESS_INTERVAL:
            last_proc = time.time()
            emp_id, score, bbox = engine.recognize_employee(frame, in_memory_db)

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            color = (0, 200, 0) if emp_id else (0, 165, 255)
            label = f"{emp_id} ({score:.2f})" if emp_id else f"Unenrolled ({score:.2f})"
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display, label, (x1, max(25, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.putText(
            display,
            "Press 'e' to Enroll | Press 'q' to Quit",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )
        cv2.imshow("Milestone 1: Standalone Recognition Test", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("e"):
            res = engine.extract_embedding(frame)
            if res is not None:
                emb, _ = res
                in_memory_db["DCCP-KARL-001"] = emb
                print("[SUCCESS] Enrolled DCCP-KARL-001 into test memory!")
            else:
                print("[WARNING] No face detected to enroll.")
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()