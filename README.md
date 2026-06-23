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

### `contrastive_finetuning/build_pair_quality_cache.py`

Build an offline pair-quality cache using frozen baseline `RDD-v2 + RDD_lg-v2`.
The cache scores sampled same-identity positives and cross-identity negatives per
anchor image, then assigns quality bands (`high`, `medium`, `low`) used by the
training loader when `--use_pair_quality_mining` is enabled.

Output layout:

```text
<pair_quality_cache_dir>/
  metadata.json
  pairs.pt
```

Example:

```bash
python -m contrastive_finetuning.build_pair_quality_cache \
  --train_data /shared/sets/datasets/confidential/lynx/processed_frames/segmented/dfk-June-2026-merged/lynx/train \
  --output_dir /shared/sets/datasets/confidential/lynx/checkpoints/pair_quality_cache/baseline_v1 \
  --rdd_weights rdd/weights/RDD-v2.pth \
  --lg_weights rdd/weights/RDD_lg-v2.pth \
  --resize 256 \
  --top_k 256 \
  --device cuda \
  --max_positive_candidates_per_anchor 32 \
  --max_negative_candidates_per_anchor 32 \
  --sequence_aware_sampling true
```

Slurm launcher script:

```bash
sbatch build_pair_quality_cache.sh
```

Useful overrides:

```bash
# Smoke test on a subset of anchors
MAX_ANCHORS=100 CACHE_TAG=baseline_v1_smoke sbatch build_pair_quality_cache.sh

# Full run with explicit output directory
OUTPUT_DIR=/shared/sets/datasets/confidential/lynx/checkpoints/pair_quality_cache/baseline_v1 \
  sbatch build_pair_quality_cache.sh
```

Training with pair-quality mining (opt-in; fails fast if cache metadata mismatches
`--train_data`, `--resize`, `--top_k`, or weight paths):

```bash
accelerate launch --num_processes 1 -m contrastive_finetuning.train \
  --train_data /shared/sets/datasets/confidential/lynx/processed_frames/segmented/dfk-June-2026-merged/lynx/train \
  --val_data /shared/sets/datasets/confidential/lynx/processed_frames/segmented/dfk-June-2026-merged/lynx/test \
  --rdd_weights rdd/weights/RDD-v2.pth \
  --lg_weights rdd/weights/RDD_lg-v2.pth \
  --output_dir /tmp/lynx-mining-run \
  --batch_mode balanced \
  --loss_type batch_hard_topk \
  --use_pair_quality_mining \
  --pair_quality_cache_dir /shared/sets/datasets/confidential/lynx/checkpoints/pair_quality_cache/baseline_v1 \
  --sequence_aware_sampling true \
  --resize 256 \
  --top_k 256
```

Optional benchmark-faithful retrieval probe during training (same scoring style as
`rdd/scripts/lynx_benchmark.py`, on a fixed small subset):

```bash
accelerate launch --num_processes 1 -m contrastive_finetuning.train \
  ... \
  --use_retrieval_probe \
  --retrieval_probe_num_queries 8 \
  --retrieval_probe_gallery_per_id 2 \
  --retrieval_probe_frames_per_seq 2 \
  --retrieval_probe_every_n_epochs 1
```

Logs: `retrieval_probe/top1_acc`, `retrieval_probe/top5_acc`, `retrieval_probe/mAP`,
`retrieval_probe/balanced_top1_acc`.

Fallback behavior: if mining is enabled but a given anchor has no usable cached
candidates, the loader falls back to the existing sequence-aware sampler.

### `contrastive_finetuning/visualize_pair_quality_cache.py`

Visualize pair-quality cache selections as side-by-side match panels, similar to
`backfill_eval_visuals.py` and `lynx_eval_visuals.py`.

For each sampled anchor, the script loads the cached **best high-quality positive**
and **hardest negative** candidate (configurable), re-runs baseline RDD + LightGlue
matching, and saves:

```text
pair_000_pos.jpg
pair_000_neg.jpg
...
```

Panel titles include cache metadata: quality band, composite score, cached match
count, live match count, and anchor participation.

Example:

```bash
python -m contrastive_finetuning.visualize_pair_quality_cache \
  --cache_dir /shared/sets/datasets/confidential/lynx/checkpoints/pair_quality_cache/baseline_v1 \
  --output_dir /shared/results/common/kargin/lynx/results/pair_quality_cache_visuals \
  --num_examples 32 \
  --seed 0 \
  --device cuda \
  --positive_selection best_high \
  --negative_selection best_hard
```

Use `--positive_selection worst_low` to inspect low-quality same-identity pairs the
miner would downweight or exclude.
