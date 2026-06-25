# PUC
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17666420.svg)](https://doi.org/10.5281/zenodo.17666420) ![GitHub License](https://img.shields.io/github/license/pinto0309/PUC)
 [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/PINTO0309/puc)

Phone Usage Classifier (PUC) is a three-class image classification pipeline for understanding how people interact with smartphones.

- `classid=0` (`no_action`): No interaction with a smartphone.
- `classid=1` (`point_somewhere`): Pointing the smartphone somewhere other than the camera.
- `classid=2` (`point`): Pointing the smartphone towards the camera.

https://github.com/user-attachments/assets/56cb5147-78e7-4a9e-85ac-59db79b442c3

|Variant|Size|F1|CPU<br>inference<br>latency|ONNX|
|:-:|:-:|:-:|:-:|:-:|
|P|115 KB|0.9160| 0.24 ms|[Download](https://github.com/PINTO0309/PUC/releases/download/onnx/puc_p_48x48.onnx)|
|N|176 KB|0.9337| 0.39 ms|[Download](https://github.com/PINTO0309/PUC/releases/download/onnx/puc_n_48x48.onnx)|
|T|280 KB|0.9468| 0.51 ms|[Download](https://github.com/PINTO0309/PUC/releases/download/onnx/puc_t_48x48.onnx)|
|S|495 KB|0.9672| 0.66 ms|[Download](https://github.com/PINTO0309/PUC/releases/download/onnx/puc_s_48x48.onnx)|
|C|876 KB|0.9722| 0.73 ms|[Download](https://github.com/PINTO0309/PUC/releases/download/onnx/puc_c_48x48.onnx)|
|M|1.7 MB|0.9774| 0.86 ms|[Download](https://github.com/PINTO0309/PUC/releases/download/onnx/puc_m_48x48.onnx)|
|L|6.4 MB|0.9944| 1.07 ms|[Download](https://github.com/PINTO0309/PUC/releases/download/onnx/puc_l_48x48.onnx)|

## Data sample

|no<br>action|no<br>action|point<br>somewhere|point<br>somewhere|point|point|
|:-:|:-:|:-:|:-:|:-:|:-:|
<img width="48" height="48" alt="no_action_008364" src="https://github.com/user-attachments/assets/327c71a0-c636-4ea9-8700-ed5a28a0050e" />|<img width="48" height="48" alt="no_action_008001" src="https://github.com/user-attachments/assets/9d080aa1-fc47-4c83-85a6-9e1f1fa087b8" />|<img width="48" height="48" alt="point_somewhere_002145" src="https://github.com/user-attachments/assets/2d816974-3ddd-4d2c-ae61-0df6cbe5c14f" />|<img width="48" height="48" alt="point_somewhere_002068" src="https://github.com/user-attachments/assets/108d9652-0c07-47f0-8b34-428ee5f23dfa" />|<img width="48" height="48" alt="point_003496" src="https://github.com/user-attachments/assets/0bc4d7bd-e85e-43f9-a893-dbbb070f46da" />|<img width="48" height="48" alt="point_003008" src="https://github.com/user-attachments/assets/cbf23eed-5ded-4709-8e7a-0cc8eab920eb" />|

## Setup

```bash
git clone https://github.com/PINTO0309/PUC.git && cd PUC
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
source .venv/bin/activate
```

## Inference
```bash
uv run python demo_puc.py \
-v 0 \
-pm puc_l_48x48.onnx \
-dlr -dnm -dgm -dhm \
-ep cuda

uv run python demo_puc.py \
-v 0 \
-pm puc_l_48x48.onnx \
-dlr -dnm -dgm -dhm \
-ep tensorrt
```

## Dataset Preparation

```bash
uv run python 01_data_prep_realdata.py

uv run python 01_data_prep_realdata.py \
--input-image-dir real_images \
--start-folder 1001 \
--allow-multi-body

uv run python 02_make_parquet.py --overwrite
```
```
Split counts:
  train: 23988
    val: 2667
Label counts:
         no_action: 8885
   point_somewhere: 8885
             point: 8885
```

<img width="600" alt="class_distribution" src="https://github.com/user-attachments/assets/35622164-f354-4491-9847-d7d54f1a5e42" />

## Training Pipeline

- Use the labeled image folders under `data/no_action`, `data/point_somewhere`, and `data/point`.
- `02_make_parquet.py` writes pre-defined train/val splits into `data/dataset.parquet` using an image-level 9:1 split per class.
- The training loop relies on `BCEWithLogitsLoss` plus class-balanced `pos_weight` to stabilise optimisation under class imbalance; inference produces sigmoid probabilities. Use `--train_resampling weighted` to switch on the previous `WeightedRandomSampler` behaviour, or `--train_resampling balanced` to physically duplicate minority classes before shuffling.
- Training history, validation metrics, optional test predictions, checkpoints, configuration JSON, and ONNX exports are produced automatically.
- Per-epoch checkpoints named like `puc_epoch_0001.pt` are retained (latest 10), as well as the best checkpoints named `puc_best_epoch0004_f1_0.9321.pt` (also latest 10).
- The backbone can be switched with `--arch_variant`. Supported combinations with `--head_variant` are:

  | `--arch_variant` | Default (`--head_variant auto`) | Explicitly selectable heads | Remarks |
  |------------------|-----------------------------|---------------------------|------|
  | `baseline`       | `avg`                       | `avg`, `avgmax_mlp`       | When using `transformer`/`mlp_mixer`, you need to adjust the height and width of the feature map so that they are divisible by `--token_mixer_grid` (if left as is, an exception will occur during ONNX conversion or inference). |
  | `inverted_se`    | `avgmax_mlp`                | `avg`, `avgmax_mlp`       | When using `transformer`/`mlp_mixer`, it is necessary to adjust `--token_mixer_grid` as above. |
  | `convnext`       | `transformer`               | `avg`, `avgmax_mlp`, `transformer`, `mlp_mixer` | For token mixer heads, the feature map dimensions must be divisible by `--token_mixer_grid` (default `2x3`). |
- The classification head is selected with `--head_variant` (`avg`, `avgmax_mlp`, `transformer`, `mlp_mixer`, or `auto` which derives a sensible default from the backbone).
- Pass `--rgb_to_yuv_to_y` to convert RGB crops to YUV, keep only the Y (luma) channel inside the network, and train a single-channel stem without modifying the dataloader.
- Alternatively, use `--rgb_to_lab` or `--rgb_to_luv` to convert inputs to CIE Lab/Luv (3-channel) before the stem; these options are mutually exclusive with each other and with `--rgb_to_yuv_to_y`.
- Mixed precision can be enabled with `--use_amp` when CUDA is available.
- Resume training with `--resume path/to/puc_epoch_XXXX.pt`; all optimiser/scheduler/AMP states and history are restored.
- Loss/accuracy/F1 metrics are logged to TensorBoard under `output_dir`, and `tqdm` progress bars expose per-epoch progress for train/val/test loops.

Baseline depthwise-separable CNN:

```bash
SIZE=48x48
uv run python -m puc train \
--data_root data/dataset.parquet \
--output_dir runs/puc_${SIZE} \
--epochs 100 \
--batch_size 256 \
--train_resampling balanced \
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
SIZE=48x48
VAR=s
uv run python -m puc train \
--data_root data/dataset.parquet \
--output_dir runs/puc_is_${VAR}_${SIZE} \
--epochs 100 \
--batch_size 256 \
--train_resampling balanced \
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
SIZE=48x48
uv run python -m puc train \
--data_root data/dataset.parquet \
--output_dir runs/puc_convnext_${SIZE} \
--epochs 100 \
--batch_size 256 \
--train_resampling balanced \
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
--checkpoint runs/puc_is_s_48x48/puc_best_epoch0049_f1_0.9939.pt \
--output puc_s_48x48.onnx \
--opset 17
```

- The saved graph exposes `images` as input and `prob_pointing` as output (batch dimension is dynamic); probabilities can be consumed directly.
- After exporting, the tool runs `onnxsim` for simplification and rewrites any remaining BatchNormalization nodes into affine `Mul`/`Add` primitives. If simplification fails, a warning is emitted and the unsimplified model is preserved.

## Arch

<img width="350" alt="puc_p_48x48" src="https://github.com/user-attachments/assets/4f4bacfd-5ac1-4af6-a68b-379377f3dc49" />

## Ultra-lightweight classification model series
1. [VSDLM: Visual-only speech detection driven by lip movements](https://github.com/PINTO0309/VSDLM) - MIT License
2. [OCEC: Open closed eyes classification. Ultra-fast wink and blink estimation model](https://github.com/PINTO0309/OCEC) - MIT License
3. [PGC: Ultrafast pointing gesture classification](https://github.com/PINTO0309/PGC) - MIT License
4. [SC: Ultrafast sitting classification](https://github.com/PINTO0309/SC) - MIT License
5. [PUC: Phone Usage Classifier is a three-class image classification pipeline for understanding how people
interact with smartphones](https://github.com/PINTO0309/PUC) - MIT License
6. [HSC: Happy smile classifier](https://github.com/PINTO0309/HSC) - MIT License
7. [WHC: Waving Hand Classification](https://github.com/PINTO0309/WHC) - MIT License
8. [UHD: Ultra-lightweight human detection](https://github.com/PINTO0309/UHD) - MIT License
9. [MWC: Mask wearing classifier](https://github.com/PINTO0309/MWC) - MIT License
10. [SGC: Classification of wearing vs. not wearing sunglasses. 48x48.](https://github.com/PINTO0309/SGC) - MIT License

## Citation

If you find this project useful, please consider citing:

```bibtex
@software{hyodo2025puc,
  author    = {Katsuya Hyodo},
  title     = {PINTO0309/PUC},
  month     = {11},
  year      = {2025},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.17666420},
  url       = {https://github.com/PINTO0309/puc},
  abstract  = {Phone Usage Classifier (PUC) is a three-class image classification pipeline for understanding how people
interact with smartphones.},
}
```

## Acknowledgements
- https://github.com/PINTO0309/PINTO_model_zoo/tree/main/472_DEIMv2-Wholebody34: Apache 2.0 License
  ```bibtex
  @software{DEIMv2-Wholebody34,
    author={Katsuya Hyodo},
    title={Lightweight human detection models generated on high-quality human data sets. It can detect objects with high accuracy and speed in a total of 28 classes: body, adult, child, male, female, body_with_wheelchair, body_with_crutches, head, front, right-front, right-side, right-back, back, left-back, left-side, left-front, face, eye, nose, mouth, ear, collarbone, shoulder, solar_plexus, elbow, wrist, hand, hand_left, hand_right, abdomen, hip_joint, knee, ankle, foot.},
    url={https://github.com/PINTO0309/PINTO_model_zoo/tree/main/472_DEIMv2-Wholebody34},
    year={2025},
    month={10},
    doi={10.5281/zenodo.17625710}
  }
  ```
- https://github.com/PINTO0309/bbalg: MIT License
