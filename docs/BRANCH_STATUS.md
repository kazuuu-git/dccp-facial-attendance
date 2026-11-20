# Branch Cleanup Status & Recommendations

> **Date:** November 17, 2026  
> **Status:** Post-Liveness & Anti-Spoofing Web Verification Integration

The anti-spoofing web verification fix, threshold policy, dual-model ensemble, and regression test suite have been fully validated and merged into `develop` (commit `fc70dbd`).

---

## Feature Branch Status

| Branch | Status | Notes |
| :--- | :---: | :--- |
| `feat/corliss-minifasnet-loader` | **Okay for deletion** | Fully merged and superseded by `feat/corliss-spoof-threshold` and `develop`. |
| `feat/corliss-spoof-threshold` | **Okay for deletion** | All logic (`threshold.py`, multi-model configuration, test harness) is merged into `develop`. |
| `feat/karl-face-recognition` | **Okay for deletion** | All commits merged into `develop` in PR #1 (`4774dc2`). |
| `feat/earl-streamlit-dashboard` | **DO NOT DELETE** | Contains Earl's active UI styling enhancements and metrics (commits `03e5c02` and `0de5008`). |
| `Corlis1-test-1` | **Deleted** | Stale remote reference pruned (already removed on GitHub). |
| `backup/*` (5 branches) | **Retain** | Local safety rollback checkpoints. Keep until final capstone submission. |
| `main` | **Retain** | Protected baseline branch. Merge `develop` into `main` for final release. |
