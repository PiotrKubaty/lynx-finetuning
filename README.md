# lynx-finetuning

Finetuning feature matching modules for better performance on lynx-reidentification.

## Instalation

Clone RDD and change commit
```bash
git clone --recursive https://github.com/xtcpete/rdd
cd rdd
git checkout 539508b270095969f9934c574cf7026bf37c434c
cd .. # root directory
```
Download `RDD-v2.pth` and `RDD_lg-v2.pth` checkpoints to rdd/weights

Install packages
```bash
conda create -n lynx-finetuning python=3.12
conda activate lynx-finetuning

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
pip install -r rdd/requirements.txt
pip install -r contrastive_finetuning/requirements.txt
```

## Utilities

### `contrastive_finetuning/inspect_sequence_sampling.py`

Inspect how positive examples are chosen for training and verify that
sequence-aware sampling prefers:

- different source/video first
- then different sequence
- then same-sequence fallback

Useful when checking whether the dataset hierarchy is being interpreted
correctly or when comparing sequence-aware vs identity-only sampling.

Example:

```bash
python -m contrastive_finetuning.inspect_sequence_sampling \
  --data_root /shared/sets/datasets/confidential/lynx/processed_frames/segmented/dfk-June-2026-merged/lynx/train \
  --num_examples 8 \
  --seed 0 \
  --sequence_aware_sampling true \
  --compare_modes
```

### `contrastive_finetuning/backfill_eval_visuals.py`

Generate validation positive/negative match visualizations for existing
fine-tuning checkpoints after training has already finished. This is useful
when a run was trained without `--save_eval_visuals` and you want to create
the same style of images later for each saved epoch.

It loads each `epoch_XX/model.safetensors` checkpoint from a run directory and
writes images into:

```text
<run_dir>/val_visuals/epoch_XX/
```

Example:

```bash
python -m contrastive_finetuning.backfill_eval_visuals \
  --run_dir /shared/sets/datasets/confidential/lynx/checkpoints/contrastive-finetuning/balanced_topk_v1/20260622_202516_job429464 \
  --val_data /shared/sets/datasets/confidential/lynx/processed_frames/segmented/dfk-June-2026-merged/lynx/test \
  --num_eval_visuals 16 \
  --resize 128 \
  --top_k 64
```

### `rdd/scripts/lynx_eval_visuals.py`

Generate baseline positive/negative match visualizations using the stock
`RDD-v2` descriptor and frozen `RDD_lg-v2` LightGlue weights, without any
fine-tuning checkpoint.

This script is useful for:

- comparing baseline RDD against a fine-tuned run
- checking where baseline matches land on positive and negative pairs
- inspecting whether background or mask edges are producing matches

It saves one positive panel and one negative panel per sampled anchor, for
example:

```text
pair_000_pos.jpg
pair_000_neg.jpg
...
```

The script now supports `--sequence_aware_sampling` and defaults to `true`, so
its positive sampling matches the training-side sequence-aware policy.

Example:

```bash
python rdd/scripts/lynx_eval_visuals.py \
  --data_root /shared/sets/datasets/confidential/lynx/processed_frames/segmented/dfk-June-2026-merged/lynx/test \
  --output_dir /shared/results/common/kargin/lynx/results/rdd/baseline_eval_visuals \
  --num_examples 32 \
  --resize 448 \
  --top_k 2048 \
  --device cuda \
  --sequence_aware_sampling true
```

