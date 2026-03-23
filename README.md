# FIAS

This repository provides a PyTorch implementation scaffold for the ISBI (Oral) paper:

**FIAS: Feature Imbalance-Aware Medical Image Segmentation with Dynamic Fusion and Mixing Attention**




## 🛠 Environment Setup

### 1. Create a Python environment

Recommended Python version:

- `Python 3.10` or `Python 3.11`

Example:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Or install in editable mode:

```bash
pip install -e .
```

## 🗂 Dataset Preparation

This part is important. To reproduce the paper cleanly, keep the raw data and processed slice data separated.

### Recommended directory layout

```text
ISBI/
├── 3Dsegmentation/
│   └── data/
│       ├── DATASET_Synapse/
│       │   └── unetr_pp_raw/
│       │       └── unetr_pp_raw_data/
│       │           └── Task02_Synapse/
│       │               ├── dataset.json
│       │               ├── imagesTr/
│       │               ├── labelsTr/
│       │               └── imagesTs/
│       └── DATASET_Acdc/
│           └── unetr_pp_raw/
│               └── unetr_pp_raw_data/
│                   └── Task001_ACDC/
│                       ├── dataset.json
│                       ├── imagesTr/
│                       ├── labelsTr/
│                       └── imagesTs/
├── data/
│   ├── processed/
│   │   ├── synapse/
│   │   └── acdc/
│   └── splits/
│       ├── synapse_train.txt
│       ├── synapse_val.txt
│       ├── synapse_test.txt
│       ├── acdc_train.txt
│       ├── acdc_val.txt
│       └── acdc_test.txt
└── outputs/
    ├── checkpoints/
    ├── predictions/
    └── logs/
```

### Official dataset notes

#### Synapse

- Official task name used in this repository: `Task02_Synapse`
- Official modality: `CT`
- Official labels in `dataset.json`: 14 semantic IDs including background
- The paper reports results on **8 target organs**
- In this codebase, `num_classes=9` means:
  - `0`: background
  - `1-8`: the 8 paper-reported organs

#### ACDC

- Official task name used in this repository: `Task001_ACDC`
- Official modality: `MRI`
- Official labels in `dataset.json`:
  - `0`: background
  - `1`: RV
  - `2`: MYO
  - `3`: LV
- In this codebase, `num_classes=4`

## 📥 How to Place the Raw Data

### Synapse

Place the official files under:

```text
3Dsegmentation/data/DATASET_Synapse/unetr_pp_raw/unetr_pp_raw_data/Task02_Synapse/
```

Expected subfolders:

- `imagesTr/`
- `labelsTr/`
- `imagesTs/`
- `dataset.json`

### ACDC

Place the official files under:

```text
3Dsegmentation/data/DATASET_Acdc/unetr_pp_raw/unetr_pp_raw_data/Task001_ACDC/
```

Expected subfolders:

- `imagesTr/`
- `labelsTr/`
- `imagesTs/`
- `dataset.json`

## ⚙️ Preprocessing

The training code expects 2D `.pt` slices after preprocessing.

### Preprocess Synapse

```bash
python scripts/preprocess_synapse.py \
  --images-dir 3Dsegmentation/data/DATASET_Synapse/unetr_pp_raw/unetr_pp_raw_data/Task02_Synapse/imagesTr \
  --labels-dir 3Dsegmentation/data/DATASET_Synapse/unetr_pp_raw/unetr_pp_raw_data/Task02_Synapse/labelsTr \
  --output-dir data/processed/synapse
```

### Preprocess ACDC

```bash
python scripts/preprocess_acdc.py \
  --root 3Dsegmentation/data/DATASET_Acdc/unetr_pp_raw/unetr_pp_raw_data/Task001_ACDC \
  --output-dir data/processed/acdc
```

After preprocessing, the directory will contain files like:

```text
data/processed/synapse/
├── img0001_000_image.pt
├── img0001_000_mask.pt
├── img0001_001_image.pt
└── ...
```

## 🧾 Split Files

Create split files under `data/splits/`.

Each line should contain one sample ID matching the processed file prefix.

Example:

```text
img0001_000
img0001_001
img0002_003
```

Recommended split files:

- `synapse_train.txt`
- `synapse_val.txt`
- `synapse_test.txt`
- `acdc_train.txt`
- `acdc_val.txt`
- `acdc_test.txt`

## 🚀 Training

The current `train.py` uses command-line arguments directly.
The YAML files in `configs/` are provided as experiment references and should be aligned with your command arguments.

### Train on Synapse

```bash
python scripts/train.py \
  --dataset synapse \
  --data-root data/processed/synapse \
  --split-file data/splits/synapse_train.txt \
  --epochs 200 \
  --batch-size 8 \
  --image-size 256 256 \
  --num-classes 9 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --output-dir outputs/checkpoints/synapse
```

### Train on ACDC

```bash
python scripts/train.py \
  --dataset acdc \
  --data-root data/processed/acdc \
  --split-file data/splits/acdc_train.txt \
  --epochs 200 \
  --batch-size 8 \
  --image-size 256 256 \
  --num-classes 4 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --output-dir outputs/checkpoints/acdc
```

## 📊 Evaluation

### Evaluate on Synapse

```bash
python scripts/evaluate.py \
  --dataset synapse \
  --data-root data/processed/synapse \
  --split-file data/splits/synapse_val.txt \
  --checkpoint outputs/checkpoints/synapse/last.pt \
  --batch-size 8 \
  --image-size 256 256 \
  --num-classes 9
```

### Evaluate on ACDC

```bash
python scripts/evaluate.py \
  --dataset acdc \
  --data-root data/processed/acdc \
  --split-file data/splits/acdc_val.txt \
  --checkpoint outputs/checkpoints/acdc/last.pt \
  --batch-size 8 \
  --image-size 256 256 \
  --num-classes 4
```

## 🔎 Inference

Run inference on one `.pt` image tensor:

```bash
python scripts/infer.py \
  --input data/processed/synapse/img0001_000_image.pt \
  --checkpoint outputs/checkpoints/synapse/last.pt \
  --output outputs/predictions/synapse/img0001_000_pred.pt \
  --image-size 256 256 \
  --num-classes 9
```

## 📤 Export Predictions

Export predictions for one full split:

```bash
python scripts/export_predictions.py \
  --dataset synapse \
  --data-root data/processed/synapse \
  --split-file data/splits/synapse_test.txt \
  --checkpoint outputs/checkpoints/synapse/last.pt \
  --output-dir outputs/predictions/synapse \
  --batch-size 8 \
  --image-size 256 256 \
  --num-classes 9
```

## 🧪 Testing

The repository includes unit tests for:

- datasets
- DMK encoder
- CAF
- MixAtt decoder
- losses
- full model forward

Run tests with:

```bash
pytest tests -q
```

## 🔁 Reproducibility Guideline

To reproduce the paper in a controlled way, use this order:

1. Create a fresh Python environment.
2. Install dependencies from `requirements.txt`.
3. Place Synapse or ACDC raw data exactly under the directory structure shown above.
4. Run preprocessing to generate `.pt` slice files.
5. Create split files in `data/splits/`.
6. Use the dataset-specific config in `configs/synapse/` or `configs/acdc/` as your experiment reference.
7. Train with:
   - `epochs=200`
   - `batch_size=8`
   - `image_size=256x256`
   - `optimizer=AdamW`
   - `lr=1e-4`
   - `weight_decay=1e-4`
   - `loss = 0.4 * Dice + 0.6 * CrossEntropy`
8. Evaluate with Dice and HD95.
9. Export predictions for qualitative comparison.


## 📚 Citation

If you use this repository, please cite the original paper.

```bibtex
@inproceedings{liu2025fias,
  title={FIAS: Feature Imbalance-Aware Medical Image Segmentation with Dynamic Fusion and Mixing Attention},
  author={Liu, Xiwei and Xu, Min and Ho, Qirong},
  booktitle={2025 IEEE 22nd International Symposium on Biomedical Imaging (ISBI)},
  pages={1--5},
  year={2025},
  organization={IEEE}
}
```

