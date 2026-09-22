# CzechLynx time-closed workflow

Paths come from each repository's `env.sh` (see WILDLIFE.md, "Site configuration").
Commands starting with `sbatch slurm_scripts/...` are run from the root of the repository
that owns the script: `lynx-finetuning` for `build_*`/`train_*`, `rdd-parallel-benchmark`
for `prepare_*`, `spawn_*` and `*_evaluate.sh`.

The integration uses masked real CzechLynx images and the metadata
`split-time_closed` protocol. The converter creates a symlink view with
`train`, `val`, and `test` directories. The strict protocol uses a
deterministic 20% encounter holdout for validation, while the legacy protocol
uses the metadata test split for training-time validation to match the
previous Lynx workflow.

## Run order

From the benchmark repository:

```bash
sbatch slurm_scripts/prepare_czechlynx.sh
```

Then build the RDD cache and mine the shared train/validation indices:

```bash
sbatch slurm_scripts/build_czechlynx_rdd_cache.sh
bash slurm_scripts/spawn_czechlynx_mining.sh
```

After aggregation, run RDD first and LoMa second:

```bash
sbatch slurm_scripts/train_czechlynx_rdd.sh
```

The RDD SLURM entry point requests two GPUs and uses synchronized
shape-bucketed global batches. `--batch_size 8` remains per GPU, so the
effective batch size is 16. Triplets are grouped by their cached `(H, W)`
signature before Accelerate splits each global batch identically across ranks;
this handles mixed dimensions such as `480x512` and `512x512` safely.

```bash
sbatch slurm_scripts/build_czechlynx_loma_cache.sh
sbatch slurm_scripts/train_czechlynx_loma.sh
```

The benchmark evaluator is launched with `CZECHLYNX_BACKEND=rdd` or
`CZECHLYNX_BACKEND=loma`, and writes separate full-gallery or top-15 JSON
reports. Existing original-Lynx scripts and artifact directories are not
changed.

## Split protocols

CzechLynx supports two fine-tuning protocols through
`CZECHLYNX_SPLIT_PROTOCOL`:

```text
legacy (default): train → fine-tuning; test → validation and final reporting
strict:           train → fine-tuning; val → validation; test → final reporting
```

The legacy mode is intentionally compatible with the previous Lynx workflow.
Because the test-derived index is used for checkpoint selection during
training, its final metrics are not an untouched test estimate. The strict
mode preserves the encounter-level 20% validation holdout for clean model
selection.

The mining command creates all three indices and keeps separate logs for each
query split:

```bash
bash slurm_scripts/spawn_czechlynx_mining.sh
```

The default mining report directory is
`outputs/czechlynx-time-closed/legacy/strong-matches`, and the training entry
points default to the corresponding legacy index root. The default protocol is
also `legacy`, so no environment variables are required for the standard
legacy-compatible run.

It produces:

```text
outputs/czechlynx-time-closed/strong-matches_train_combined.json
outputs/czechlynx-time-closed/strong-matches_val_combined.json
outputs/czechlynx-time-closed/strong-matches_test_combined.json
```

Mining uses 20 frames per collection, `top_k_frames=5`, `top_m=10`, and the
pretrained RDD/LightGlue weights. Test queries are scored against the train
gallery, matching the previous Lynx protocol.

### Optional combined train+validation index

For a larger training set, the existing train and validation indices can be
combined into a separate file. This does not modify the original indices:

```bash
cd "${RDD_BENCHMARK_ROOT}"

python - <<PY
import json
from pathlib import Path

root = Path("outputs/czechlynx-time-closed/legacy")
train = json.loads((root / "strong-matches_train_combined.json").read_text())
val = json.loads((root / "strong-matches_val_combined.json").read_text())

train_queries = {item["query_frame"] for item in train}
val_queries = {item["query_frame"] for item in val}
assert not train_queries & val_queries, "Train/validation query overlap detected"

output = root / "strong-matches_trainval_combined.json"
output.write_text(json.dumps(train + val, indent=2))
print(f"Wrote {output} with {len(train) + len(val)} entries")
PY
```

The resulting index contains approximately 4,826 entries (3,009 train and
1,817 validation entries). Use it explicitly for RDD or LoMa fine-tuning:

```bash
CZECHLYNX_TRAIN_INDEX=${RDD_BENCHMARK_ROOT}/outputs/czechlynx-time-closed/legacy/strong-matches_trainval_combined.json \
sbatch slurm_scripts/train_czechlynx_rdd.sh
```

For LoMa, use the same `CZECHLYNX_TRAIN_INDEX` override with
`train_czechlynx_loma.sh`. The legacy validation index remains
`strong-matches_test_combined.json`; the combined file is used only as the
training index.

This is a direct concatenation of already mined entries. The validation
entries were originally mined against the train gallery, so this does not
create a newly mined train+validation gallery. A true combined-gallery
experiment requires re-mining with both the train and validation collections
available as gallery candidates.

### Legacy RDD training

Legacy mode is the default, but it can be set explicitly:

```bash
CZECHLYNX_SPLIT_PROTOCOL=legacy \
sbatch slurm_scripts/train_czechlynx_rdd.sh
```

This uses `strong-matches_test_combined.json` as `val_index` and writes to the
legacy-specific checkpoint directory:

```text
/shared/sets/datasets/vision/czechlynx/checkpoints/czechlynx-time-closed/rdd-finetuned-legacy
```

### Strict RDD training

```bash
CZECHLYNX_SPLIT_PROTOCOL=strict \
sbatch slurm_scripts/train_czechlynx_rdd.sh
```

This uses `strong-matches_val_combined.json` as `val_index` and writes to
`rdd-finetuned-strict`.

### Legacy and strict LoMa training

Use the same setting with the LoMa entry point:

```bash
CZECHLYNX_SPLIT_PROTOCOL=legacy \
sbatch slurm_scripts/train_czechlynx_loma.sh

CZECHLYNX_SPLIT_PROTOCOL=strict \
sbatch slurm_scripts/train_czechlynx_loma.sh
```

RDD and LoMa use the same aggregated train/test/validation indices by default.
Each run prints its selected protocol and index paths and writes
`czechlynx_protocol.json` in its output directory.

## RDD test-set evaluation

The following commands evaluate the CzechLynx metadata test split using the
RDD cache. Run them from `rdd-parallel-benchmark`.

The pretrained LightGlue checkpoint is:

```bash
${LG_WEIGHTS}
```

The completed CzechLynx run stores fine-tuned checkpoints under:

```bash
/shared/sets/datasets/vision/czechlynx/checkpoints/czechlynx-time-closed/rdd-finetuned/
```

For example, `epoch_299/model.safetensors` is the final checkpoint. Select a
different epoch if validation identifies a better checkpoint.

### Pretrained full-gallery evaluation

```bash
CZECHLYNX_BACKEND=rdd \
CZECHLYNX_CACHE=/shared/sets/datasets/vision/czechlynx/checkpoints/czechlynx-time-closed/rdd-cache \
CZECHLYNX_WEIGHTS=${LG_WEIGHTS} \
CZECHLYNX_MODE=full \
CZECHLYNX_OUTPUT=${RDD_BENCHMARK_ROOT}/outputs/czechlynx-time-closed/pretrained-full.json \
sbatch slurm_scripts/czechlynx_evaluate.sh
```

### Fine-tuned full-gallery evaluation

```bash
CZECHLYNX_BACKEND=rdd \
CZECHLYNX_CACHE=/shared/sets/datasets/vision/czechlynx/checkpoints/czechlynx-time-closed/rdd-cache \
CZECHLYNX_WEIGHTS=/shared/sets/datasets/vision/czechlynx/checkpoints/czechlynx-time-closed/rdd-finetuned/epoch_299/model.safetensors \
CZECHLYNX_MODE=full \
CZECHLYNX_OUTPUT=${RDD_BENCHMARK_ROOT}/outputs/czechlynx-time-closed/epoch299-full.json \
sbatch slurm_scripts/czechlynx_evaluate.sh
```

### Pretrained top-15 evaluation

```bash
CZECHLYNX_BACKEND=rdd \
CZECHLYNX_CACHE=/shared/sets/datasets/vision/czechlynx/checkpoints/czechlynx-time-closed/rdd-cache \
CZECHLYNX_WEIGHTS=${LG_WEIGHTS} \
CZECHLYNX_PRESELECT_WEIGHTS=${LG_WEIGHTS} \
CZECHLYNX_MODE=top15 \
CZECHLYNX_OUTPUT=${RDD_BENCHMARK_ROOT}/outputs/czechlynx-time-closed/pretrained-top15.json \
sbatch slurm_scripts/czechlynx_evaluate.sh
```

### Fine-tuned top-15 evaluation

```bash
CZECHLYNX_BACKEND=rdd \
CZECHLYNX_CACHE=/shared/sets/datasets/vision/czechlynx/checkpoints/czechlynx-time-closed/rdd-cache \
CZECHLYNX_WEIGHTS=/shared/sets/datasets/vision/czechlynx/checkpoints/czechlynx-time-closed/rdd-finetuned/epoch_299/model.safetensors \
CZECHLYNX_PRESELECT_WEIGHTS=${LG_WEIGHTS} \
CZECHLYNX_MODE=top15 \
CZECHLYNX_OUTPUT=${RDD_BENCHMARK_ROOT}/outputs/czechlynx-time-closed/epoch299-top15.json \
sbatch slurm_scripts/czechlynx_evaluate.sh
```

In top-15 mode, the pretrained checkpoint selects the strongest query frame
and 15 diverse gallery frames. The target checkpoint then reranks those 15
frames. This makes the pretrained and fine-tuned top-15 results directly
comparable.

## Few-shot experiments

The same few-shot protocol as for WildlifeReID (WILDLIFE.md, "Few-shot experiments") applied
to the time-closed view: `rdd-parallel-benchmark/scripts/czechlynx_fewshot.py` keeps a
fraction of the training frames per identity (exact budget, never below two frames per
identity, seeded and nested; `val` and `test` unchanged; canonical frame names preserved so
the existing RDD cache is reused) and writes the probe metadata copy
`CzechLynx_v2/metadata_fewshot/CzechLynxDataset-Metadata-Real.csv` with one column
`split_frac<f>_seed<s>` per view (`train` = kept training frames, `unused` = every other
row whose `split-time_closed` is `train`, including the view's validation holdout, `test`
unchanged). All four fractions are feasible (median 19 frames per identity):

```text
1/8: 2,797 frames   1/4: 5,594   1/2: 11,188   full: 22,375   (315 identities, 6 singletons)
```

Everything is queued by one command from the benchmark root:

```bash
cd "${RDD_BENCHMARK_ROOT}"
bash slurm_scripts/fewshot/pipeline_czechlynx.sh            # seed 0, fractions 1/8 1/4 1/2 1
```

Per fraction it submits: mining as two arrays of `FEWSHOT_CZ_MINE_WORKERS` (16) strided
workers (train and test queries, `czechlynx_mine.py --all_queries`; rtx4090 by default)
→ aggregation (CPU, `afterok`) → fine-tuning (`czechlynx_train_job.sh`: dgxh100,
`--qos=normal`, 2 GPUs × batch 8 = the reference global batch 16, cache staged to the
node-local disk, private copy of `train_czechlynx_rdd.sh`) — each step skipped when its
output exists — and finally the probe chain (reduced + full gallery, `probe-fewshot-wildlife.sh
CzechLynx`) and the collector (`reports/fewshot/CzechLynx/`). Full-gallery runs use the
original `split-time_closed` column, i.e. the paper's 27,836-image database, while the
reduced gallery of the `1.0` view is the view's training split (22,375 frames: the
validation holdout is not fine-tuning data). `czechlynx_evaluate.sh` is not part of the
chain (≈50 M pairs per checkpoint).

### LoMa backend

`FEWSHOT_BACKEND=loma bash slurm_scripts/fewshot/pipeline_czechlynx.sh` runs the same chain
with the LoMa-B matcher: the pairs are mined with LoMa (`czechlynx_mine.py --backend loma`,
kargin's `loma-b-cache` of the full view, `loma_B.pt`), so the indices live under
`indices/CzechLynx/legacy/<view>/loma/` — an index always belongs to the matcher that mined
it and is never shared between RDD and LoMa — and the fine-tuning uses
`train_czechlynx_loma.sh` (reference run 4 × batch 8 = global 32, so 4 GPUs by default)
writing `checkpoints/CzechLynx/legacy/<view>/loma-finetuned/`. LoMa mining defaults to
`dgxh100` (`--qos=quick`, one GPU per worker) and LoMa training to one `rtx4090_batch` node
(`--qos=batch`, 4 GPUs, 40 CPUs, 125G, 12 h — the reference run finished 300 epochs in
~2–3 h on that setup; `FEWSHOT_CZ_MINE_SBATCH` / `FEWSHOT_CZ_TRAIN_SBATCH` override both);
the probe chain then adds the `LoMa (fine-tuned)` variant to both galleries automatically.
The view preparation reuses a complete view as is (`records.jsonl` identical to the
selection; `--force` rewrites the symlinks), so re-running the pipeline for the second
backend does not re-verify the ~110k links over NFS.

Cost estimate at fraction 1.0: mining ≈ 103 M LightGlue pairs (≈ 45 GPU-hours, so ~3 h with
16 workers), training ≈ 94 steps/epoch × 300 epochs (≈ 10–20 h on 2 H100s at the measured
~3 s/step of the single-process loop); smaller fractions scale down with the gallery.

### Descriptor fine-tuning (RDD / LoMa keypoint descriptors instead of the matcher)

Both trainers can unfreeze the keypoint-descriptor network instead of, or together with,
the matcher; the training signal stays the same margin loss on the matcher's own match
confidences, the gradient just reaches further back:

| wrapper variable | value | what is trained | trainer flags |
|---|---|---|---|
| `CZECHLYNX_RDD_TRAIN_COMPONENT` | `lg` (default) | LightGlue on cached RDD features | `--trained_model lg --keypoint_cache …` |
| | `descriptor` | RDD's descriptor network (backbone + deformable transformer), LightGlue frozen | `--trained_model rdd --rdd_train_component descriptor` |
| | `lg+descriptor` | LightGlue and RDD's descriptor jointly | `--trained_model lg+rdd --rdd_train_component descriptor` |
| | `rdd`, `lg+rdd` | as above with the RDD detector unfrozen too (`--rdd_train_component all`) | |
| `CZECHLYNX_LOMA_TRAIN_COMPONENT` | `matcher` (default) | the LoMa transformer on cached DaD keypoints + DeDoDe descriptors | `--loma_train_component matcher --loma_cache …` |
| | `descriptor` | DeDoDe's descriptor through the frozen matcher | `--loma_train_component descriptor --loma_keypoint_cache …` |
| | `descriptor+matcher` | DeDoDe's descriptor and the matcher jointly | `--loma_train_component descriptor+matcher --loma_keypoint_cache …` |

Images, not cached features: the descriptor modes decode every frame and run the detector +
descriptor on it, and these datasets are per-animal crops whose sizes all differ (a CzechLynx
triplet mixes e.g. 1712x1713 with 1563x1564). The dataset therefore resizes each frame to its
canonical grid in the DataLoader worker — long side `--resize`, both sides divisible by 32,
the same rule `resize_long_side` applies on the GPU, so nothing changes geometrically — which
collapses CzechLynx to four shapes; `collate_variable_images` keeps a batch that still mixes
shapes as a list instead of failing in `torch.stack`, and `features_from_batch` groups such a
batch by shape, runs RDD once per group and returns per-frame image sizes (the layout the
cached path already produced). Without that, `--trained_model rdd`/`lg+rdd` crash in the
pre-training evaluation, which is what the `lg` runs never hit because their cache stores
padded features rather than images.

The detectors (RDD's detection head, LoMa's DaD) stay frozen: keypoint positions come out of
a non-differentiable NMS/top-k, so unfreezing them adds nothing but BatchNorm drift
(`rdd`/`lg+rdd` exist for completeness). Consequently the descriptor modes cannot use the
feature caches — RDD re-detects and re-describes every image, LoMa keeps only the DaD
keypoints (`loma-b-keypoint-cache`, built by `contrastive_finetuning.build_loma_keypoint_cache`
when missing; a complete shared cache is reused as is) and recomputes the DeDoDe descriptors
with gradient (`LoMaDescriptorTrainingModel`, `--descriptor_microbatch_size`). Measured on
Turhan's runs (4 × rtx4090): RDD descriptor ≈ 8 min/epoch, LoMa descriptor ≈ 6 min/epoch,
i.e. 30–40 h for 300 epochs, against 2–3 h for the matcher runs. Since the `batch` QoS of
`rtx4090_batch` allows 24 h, both trainers resume: `--resume <output>/epoch_NN` (RDD: prepared
models, optimizer, `scheduler.pt` and every rank's RNG file restored by
`accelerator.load_state`; LoMa: its `accelerate_state`), exposed as `CZECHLYNX_RESUME_FROM`
in the wrappers.

For the few-shot views the whole chain is one call per backend/component (fraction 1.0 by
default, RDD-mined pairs for RDD and LoMa-mined pairs for LoMa — the same indices as the
matcher runs; LoMa trainings wait for the queued `fs-CzechLynx-1.0-loma-aggregate` job when
the LoMa pipeline is still mining):

```bash
cd "${RDD_BENCHMARK_ROOT}"
bash slurm_scripts/fewshot/pipeline_czechlynx_components.sh rdd descriptor          # one experiment
FEWSHOT_EPOCHS=30 bash slurm_scripts/fewshot/spawn_czechlynx_components.sh          # all six (2 models x {matcher, descriptor, both}), 30 epochs
FEWSHOT_DRY_RUN=1 bash slurm_scripts/fewshot/spawn_czechlynx_components.sh          # print the sbatch calls
```

`FEWSHOT_EPOCHS` (300) is the training length (the cosine schedule spans it); any other
length is a separate experiment tagged `ep<N>` in the checkpoint directory
(`rdd-descriptor-finetuned-ep30/epoch_29`, `loma-finetuned-ep30/epoch_029`), the job and W&B run
names, the probe labels (`custom-descriptor-ep30`) and the report series
(`RDD-LightGlue (fine-tuned descriptor, ep30)`), so it sits next to the 300-epoch runs instead
of overwriting them; runs of at most 100 epochs get one 12 h job and evaluation every 10
epochs. The spawner's `FEWSHOT_RDD_COMPONENTS` (`lg descriptor lg+descriptor`) and
`FEWSHOT_LOMA_COMPONENTS` (`matcher descriptor descriptor+matcher`) select the variants.

Before queueing a long run, `sbatch slurm_scripts/fewshot/smoke_train_component.sh <backend>
<component>` trains the same wrapper on a couple of dozen index entries for one epoch on one
GPU, with W&B off and the checkpoints in node-local scratch: it exercises the pre-training
evaluation, a training step and the checkpoint write in a few minutes.

`pipeline_czechlynx_components.sh` submits `FEWSHOT_TRAIN_CHAIN` (3) copies of
`czechlynx_train_job.sh` chained with `--dependency=afterany` (one `rtx4090_batch` node each:
4 GPUs, 40 CPUs, 125G, 24 h; `FEWSHOT_CZ_COMPONENT_SBATCH` overrides) — each copy resumes from
the newest complete epoch of `checkpoints/CzechLynx/legacy/<view>/<backend>-<component>-finetuned/`
(`rdd-descriptor-finetuned`, `rdd-lg-descriptor-finetuned`, `loma-descriptor-finetuned`,
`loma-descriptor-matcher-finetuned`) and exits at once when `epoch_299` exists — followed by
the probe chain (`fewshot_probe_chain.sh CzechLynx 1.0`: reduced + full gallery, then the
collector). `probe-fewshot-wildlife.sh` finds those checkpoint directories itself and submits
them as the variants `custom-descriptor`, `custom-lg-descriptor` (the RDD + LightGlue files of
the epoch directory) and `custom-descriptor-matcher`; the probe loads the components each file
holds (`checkpoint_components=auto`) and the report shows them as
`RDD-LightGlue (fine-tuned descriptor)`, `RDD-LightGlue (fine-tuned LG+descriptor)`,
`LoMa (fine-tuned descriptor)`, `LoMa (fine-tuned descriptor+matcher)` next to the matcher
runs. W&B: project `czechlynx-fewshot-<backend>`, run `CzechLynx-<backend>-<component>-legacy-<view>`
(a resumed job starts a new W&B run under the same name).
