# L-VARC: Language-Guided Abstraction for Visual Reasoning

This repository contains the implementation of **L-VARC**, a language-guided extension of VARC for ARC reasoning.

- L-VARC introduces a privileged language branch during training (SCM + CAP).
- The language branch is removed at inference time, keeping the main model lightweight.
- Current reproduced results are provided in `outputs/` for both ARC-1 and ARC-2.

## 1. Environment Setup

Installation is basically the same as the original VARC baseline:

```bash
conda create -n visarc python==3.10
conda activate visarc
pip install -r requirements.txt
hf auth login
wandb login
```

## 2. Data Preparation

Please make sure the following datasets are available under `raw_data/`:

- `raw_data/ARC-AGI`
- `raw_data/ARC-AGI-2`
- `raw_data/re_arc`
- `raw_data/text_description`

If your workspace already matches this structure, you can run directly.

## 3. Quick Reproduce

### 3.1 Build augmented TTT data

```bash
python augment_data.py
bash script/sanity_ARC1.sh
bash script/sanity_ARC2.sh
```

### 3.2 Offline training

```bash
bash script/offline_train_VARC_ViT.sh
```

### 3.3 Test-time training (TTT)

ARC-1:

```bash
bash script/test_time_training_VARC_ViT_ARC1.sh
```

ARC-2:

```bash
bash script/test_time_training_VARC_ViT_ARC2.sh
```

## 4. Results in This Repo

Main analysis outputs are already in:

- `outputs/arc_agi_1.html` (ARC-1)
- `outputs/arc_agi_2.html` (ARC-2)

You can regenerate analysis pages with:

```bash
bash script/analysis/arc_1_vit.sh
bash script/analysis/arc_2_vit.sh
```

## 5. Project Structure (Brief)

- `src/`: model definitions (`ARC_ViT.py`, loader)
- `utils/`: training / evaluation / visualization utilities
- `script/`: runnable shell scripts for train, TTT, analysis
- `raw_data/`: ARC-related datasets and text descriptions
- `outputs/`: predictions and HTML analysis results

