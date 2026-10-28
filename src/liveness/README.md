## Implementation status

- MiniFASNetV2 loads and runs on CPU.
- Synthetic-input smoke test passes.
- Manual face selection and single-image inference work.
- Sample T1 returned LIVE; sample F2 returned SPOOF.

Pending:
- Second-model integration.
- Callable interface for the scanner.
- Webcam evaluation and threshold calibration.

Upstream code and checkpoints:
https://github.com/minivision-ai/Silent-Face-Anti-Spoofing

Reused architecture and cropping code are stored under vendor/.
The upstream license is preserved there.