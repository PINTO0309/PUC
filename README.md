# PUC

Phone Usage Classifier (PUC) is a four-class image classification pipeline for understanding how people
interact with smartphones. The `sc` package has been renamed to `puc`, and the core model (`PUC`) now predicts
the following behaviors from cropped images:

- `classid=0` (`no_action`): No interaction with a phone.
- `classid=1` (`call`): Holding the phone to the ear to make or take a call.
- `classid=2` (`point`): Pointing or tapping directly on the phone.
- `classid=3` (`point_somewhere`): Pointing/tapping near the phone or gesturing elsewhere.

Prepare a dataset that contains these four classes (either via the `class_id` column or the `label` text field),
then run the training pipeline:

```bash
python -m puc.pipeline train --data_root /path/to/dataset --output_dir ./outputs
```

Use `python -m puc.pipeline predict ...`, `webcam`, or `webcam_onnx` to run inference with trained checkpoints.
