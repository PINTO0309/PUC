# PUC

Phone Usage Classifier (PUC) is a four-class image classification pipeline for understanding how people
interact with smartphones.

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

## Dataset Preparation

```bash
python 01_data_prep_realdata.py
python 02_make_parquet.py --embed-images
```
```
Split counts:
  train: 24994
    val: 6185
Label counts:
         no_action: 7224
              call: 7695
             point: 8713
   point_somewhere: 7547
```

## Training Pipeline

- Use the images located under `dataset/output/002_xxxx_front_yyyyyy` together with their annotations in `dataset/output/002_xxxx_front.csv`.
- Every augmented image that originates from the same `still_image` stays in the same split to prevent leakage.
- The training loop relies on `BCEWithLogitsLoss` plus class-balanced `pos_weight` to stabilise optimisation under class imbalance; inference produces sigmoid probabilities. Use `--train_resampling weighted` to switch on the previous `WeightedRandomSampler` behaviour, or `--train_resampling balanced` to physically duplicate minority classes before shuffling.
- Training history, validation metrics, optional test predictions, checkpoints, configuration JSON, and ONNX exports are produced automatically.
- Per-epoch checkpoints named like `puc_epoch_0001.pt` are retained (latest 10), as well as the best checkpoints named `puc_best_epoch0004_f1_0.9321.pt` (also latest 10).
- The backbone can be switched with `--arch_variant`. Supported combinations with `--head_variant` are:

  | `--arch_variant` | Default (`--head_variant auto`) | Explicitly selectable heads | Remarks |
  |------------------|-----------------------------|---------------------------|------|
  | `baseline`       | `avg`                       | `avg`, `avgmax_mlp`       | When using `transformer`/`mlp_mixer`, you need to adjust the height and width of the feature map so that they are divisible by `--token_mixer_grid` (if left as is, an exception will occur during ONNX conversion or inference). |
  | `inverted_se`    | `avgmax_mlp`                | `avg`, `avgmax_mlp`       | When using `transformer`/`mlp_mixer`, it is necessary to adjust `--token_mixer_grid` as above. |
  | `convnext`       | `transformer`               | `avg`, `avgmax_mlp`, `transformer`, `mlp_mixer` | For both heads, the grid must be divisible by the feature map (default `3x2` fits with 30x48 input). |
- The classification head is selected with `--head_variant` (`avg`, `avgmax_mlp`, `transformer`, `mlp_mixer`, or `auto` which derives a sensible default from the backbone).
- Pass `--rgb_to_yuv_to_y` to convert RGB crops to YUV, keep only the Y (luma) channel inside the network, and train a single-channel stem without modifying the dataloader.
- Alternatively, use `--rgb_to_lab` or `--rgb_to_luv` to convert inputs to CIE Lab/Luv (3-channel) before the stem; these options are mutually exclusive with each other and with `--rgb_to_yuv_to_y`.
- Mixed precision can be enabled with `--use_amp` when CUDA is available.
- Resume training with `--resume path/to/puc_epoch_XXXX.pt`; all optimiser/scheduler/AMP states and history are restored.
- Loss/accuracy/F1 metrics are logged to TensorBoard under `output_dir`, and `tqdm` progress bars expose per-epoch progress for train/val/test loops.

Baseline depthwise-separable CNN:

```bash
SIZE=32x24
uv run python -m puc train \
--data_root data/dataset.parquet \
--output_dir runs/puc_${SIZE} \
--epochs 100 \
--batch_size 256 \
--image_size ${SIZE} \
--base_channels 32 \
--num_blocks 4 \
--arch_variant baseline \
--seed 42 \
--device auto \
--use_amp
```

Inverted residual + SE variant (recommended for higher capacity):

```bash
SIZE=32x24
uv run python -m puc train \
--data_root data/dataset.parquet \
--output_dir runs/puc_is_s_${SIZE} \
--epochs 100 \
--batch_size 256 \
--image_size ${SIZE} \
--base_channels 32 \
--num_blocks 4 \
--arch_variant inverted_se \
--head_variant avgmax_mlp \
--seed 42 \
--device auto \
--use_amp
```

ConvNeXt-style backbone with transformer head over pooled tokens:

```bash
SIZE=32x24
uv run python -m puc train \
--data_root data/dataset.parquet \
--output_dir runs/puc_convnext_${SIZE} \
--epochs 100 \
--batch_size 256 \
--image_size ${SIZE} \
--base_channels 32 \
--num_blocks 4 \
--arch_variant convnext \
--head_variant transformer \
--token_mixer_grid 2x2 \
--seed 42 \
--device auto \
--use_amp
```

- Outputs include the latest 10 `puc_epoch_*.pt`, the latest 10 `puc_best_epochXXXX_f1_YYYY.pt` (highest validation F1, or training F1 when no validation split), `history.json`, `summary.json`, optional `test_predictions.csv`, and `train.log`.
- After every epoch a confusion matrix and ROC curve are saved under `runs/puc/diagnostics/<split>/confusion_<split>_epochXXXX.png` and `roc_<split>_epochXXXX.png`.
- `--image_size` accepts either a single integer for square crops (e.g. `--image_size 48`) or `HEIGHTxWIDTH` to resize non-square frames (e.g. `--image_size 64x48`).
- Add `--resume <checkpoint>` to continue from an earlier epoch. Remember that `--epochs` indicates the desired total epoch count (e.g. resuming `--epochs 40` after training to epoch 30 will run 10 additional epochs).
- Launch TensorBoard with:
  ```bash
  tensorboard --logdir runs/puc
  ```

### ONNX Export

```bash
uv run python -m puc exportonnx \
--checkpoint runs/puc_is_s_32x24/puc_best_epoch0049_f1_0.9939.pt \
--output puc_s.onnx \
--opset 17
```

- The saved graph exposes `images` as input and `prob_pointing` as output (batch dimension is dynamic); probabilities can be consumed directly.
- After exporting, the tool runs `onnxsim` for simplification and rewrites any remaining BatchNormalization nodes into affine `Mul`/`Add` primitives. If simplification fails, a warning is emitted and the unsimplified model is preserved.
