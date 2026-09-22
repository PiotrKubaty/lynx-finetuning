# Generic WildlifeReID-10k workflow

## Site configuration

Every SLURM script reads its paths from the `env.sh` at the root of its repository
(`lynx-finetuning/env.sh`, `rdd-parallel-benchmark/env.sh`,
`explainable_individual_reidentification/env.sh`): conda (`CONDA_SH`, `CONDA_ENV_*`),
the pretrained checkpoints (`RDD_WEIGHTS`, `LG_WEIGHTS`, `LOMA_WEIGHTS`), the data roots
(`WILDLIFE_PROCESSED_ROOT`, `CHECKPOINTS_ROOT`, `FEWSHOT_ROOT`) and the location of the
other repositories (`RDD_BENCHMARK_ROOT`, `LYNX_FINETUNING_ROOT`, `EXREID_ROOT`). Scripts
are always run from the root of their own repository; nothing assumes that the other
repositories are checked out next to it. Every value in `env.sh` is a default that can be
overridden from the calling shell, e.g. `WILDLIFE_EPOCHS=1 sbatch slurm_scripts/train_wildlife_rdd.sh`.

The views, caches and checkpoints under `/shared/sets/datasets/vision/czechlynx/` were
produced by the original runs and are read-only; the caches are reused as they are, and
everything new is written under `FEWSHOT_ROOT` (see "Few-shot experiments").

The generic pipeline trains and evaluates one animal dataset at a time. The
supported configurations are in
`rdd-parallel-benchmark/configs/wildlife/`:

```text
BelugaID+
NyalaData+
WhaleSharkID+
ZindiTurtleRecall+ (labeled rows with a resolved masked image)
AmvrakikosTurtles
ATRW+
CowDataset+
Giraffes+
GiraffeZebraID+
HyenaID2022+
LeopardID2022+
ReunionTurtles
SeaStarReID2023+
StripeSpotter+
ZakynthosTurtles
```

NDD20 is intentionally not configured: its masked paths and identity labels
are not usable yet.

The 11 newly added datasets use metadata from
`metadata_mdsplit_no_background/metadata_<dataset>.csv`, the shared
`masked_images` tree, and the `identity`, `path`, and `split` columns. The
preparation step never falls back to unmasked images. Rows with an empty or
`unknown` identity, invalid split, or missing masked target are excluded and
recorded in the generated manifest; metadata/schema errors or a dataset with
no usable records stop preparation.

No encounter or session field is currently available in these metadata files,
so identity is used as the collection fallback. This limitation is recorded
by each configuration's `collection_rule: identity` setting and in the
generated experiment metadata. It should be considered when interpreting
collection-level metrics.

## 1. Select a dataset and prepare its canonical view

Run these commands from `rdd-parallel-benchmark`:

```bash
export WILDLIFE_CONFIG=configs/wildlife/BelugaID.json
export WILDLIFE_PROTOCOL=legacy
sbatch slurm_scripts/prepare_wildlife.sh
```

The view is symlink-based and is written to:

```text
/shared/sets/datasets/vision/czechlynx/wildlife_processed/<dataset>/<protocol>/
```

`strict` uses official train images with a deterministic 20% validation holdout
per identity, while official test remains untouched. `legacy` uses official
train for fine-tuning and official test for both validation and final reporting.

## 2. Build caches and mine shared indices

After preparation, build the backend caches from `lynx-finetuning`:

```bash
export WILDLIFE_CONFIG=configs/wildlife/BelugaID.json
export WILDLIFE_PROTOCOL=legacy
sbatch slurm_scripts/build_wildlife_rdd_cache.sh
sbatch slurm_scripts/build_wildlife_loma_cache.sh
```

Mine the training pairs with the backend you want to fine-tune. RDD remains
the default if `WILDLIFE_MINING_BACKEND` is unset:

```bash
cd "${RDD_BENCHMARK_ROOT}"   # rdd-parallel-benchmark root
export WILDLIFE_CONFIG=configs/wildlife/BelugaID.json
export WILDLIFE_PROTOCOL=legacy
export WILDLIFE_MINING_BACKEND=rdd
bash slurm_scripts/spawn_wildlife_mining.sh
```

For LoMa, use its compatible cache and pretrained checkpoint:

```bash
export WILDLIFE_MINING_BACKEND=loma
# WILDLIFE_LOMA_CACHE and LOMA_WEIGHTS default to the values in env.sh
bash slurm_scripts/spawn_wildlife_mining.sh
```

The defaults are 20 frames per collection, `top_k_frames=5`, `top_m=10`, and
array concurrency 30. Explicit backend selection stores indices under
`outputs/wildlife-reid-10k/<dataset>/indices/rdd/` or `loma/`; the historical
RDD path remains available when no selector is supplied. Each combined index
keeps the trainer-compatible JSON list and has a `.metadata.json` sidecar with
the backend, checkpoint, cache, protocol, and mining settings. RDD and LoMa
indices are intentionally separate. In legacy mode, the validation combined
file is an explicit alias of the test combined file; no second test mining pass
is performed.

## 3. Fine-tune

```bash
cd "${LYNX_FINETUNING_ROOT}"   # lynx-finetuning root
export WILDLIFE_CONFIG=configs/wildlife/BelugaID.json
export WILDLIFE_PROTOCOL=legacy
sbatch slurm_scripts/train_wildlife_rdd.sh
sbatch slurm_scripts/train_wildlife_loma.sh
```

The training wrappers select the backend-specific combined index when it is
present (`indices/rdd/` for RDD and `indices/loma/` for LoMa), while retaining
the historical shared RDD path as a fallback. Validation is
`strong-matches_val_combined.json` in strict mode and
`strong-matches_test_combined.json` in legacy mode. Checkpoints and W&B runs
are isolated by dataset, backend, and protocol. Unless overridden with
`WILDLIFE_WANDB_PROJECT`, the default projects are
`wildlife-reid-rdd-<dataset>-<protocol>` and
`wildlife-reid-loma-<dataset>-<protocol>`.

To run another dataset, replace `BelugaID.json` in the commands above with
one of the configuration files listed at the beginning of this document. The
cache, index, checkpoint, and evaluation paths are derived from that dataset
identifier.

## 4. Evaluate

Set `WILDLIFE_BACKEND`, `WILDLIFE_CACHE`, and `WILDLIFE_WEIGHTS` and submit the
generic evaluator. The test split is used for final reporting:

```bash
cd "${RDD_BENCHMARK_ROOT}"   # rdd-parallel-benchmark root
export WILDLIFE_CONFIG=configs/wildlife/BelugaID.json
export WILDLIFE_PROTOCOL=legacy
export WILDLIFE_BACKEND=rdd
export WILDLIFE_CACHE=${CHECKPOINTS_ROOT}/wildlife-reid-10k/BelugaID/rdd-cache
export WILDLIFE_WEIGHTS=${LG_WEIGHTS}
export WILDLIFE_MODE=full
sbatch slurm_scripts/wildlife_evaluate.sh
```

For the existing top-15 protocol, set `WILDLIFE_MODE=top15` and provide
`WILDLIFE_PRESELECT_WEIGHTS`. Change `WILDLIFE_BACKEND` and the cache/checkpoint
paths to run LoMa.

## Few-shot experiments (fraction of the training images)

`rdd-parallel-benchmark/scripts/wildlife_fewshot.py` derives, from the full canonical
view, a view in which the training split keeps a fraction `f ∈ {1/8, 1/4, 1/2, 1}` of
its frames and `test` is untouched:

- the budget `round(f · N_train)` is exact and spread proportionally over identities;
- no identity is reduced below 2 frames, so every training frame keeps at least one
  positive (identities that already have a single frame stay single: they are gallery
  entries and never training queries, as in the full data). Identities whose share is
  below 2 are pinned at 2 and the richer ones absorb the difference; only when the
  minimums alone exceed the budget is the effective fraction larger than `f`, which
  `fewshot.json` reports (`budget_feasible`, `effective_fraction`);
- selections are seeded and nested (1/8 ⊂ 1/4 ⊂ 1/2 ⊂ full);
- canonical frame names are unchanged, so the existing RDD/LoMa caches are reused, but
  the gallery changes, so **mining, fine-tuning and evaluation are repeated per
  fraction** into separate directories:

```text
$FEWSHOT_ROOT/views/<dataset>/<protocol>/frac<f>-seed<s>/
$FEWSHOT_ROOT/indices/<dataset>/<protocol>/frac<f>-seed<s>/<backend>/strong-matches_*_combined.json
$FEWSHOT_ROOT/checkpoints/<dataset>/<protocol>/frac<f>-seed<s>/<backend>-finetuned/epoch_*/
$FEWSHOT_ROOT/eval/<dataset>/<protocol>/frac<f>-seed<s>/<backend>-{pretrained,epoch299}-full.json
$WILDLIFE_SOURCE_ROOT/metadata_fewshot/metadata_<dataset>.csv   (probe metadata, one split column per view)
```

One fraction, end to end (view → mining → fine-tuning → evaluation, chained with SLURM
dependencies; the fine-tuning job is submitted from `LYNX_FINETUNING_ROOT`):

```bash
cd "${RDD_BENCHMARK_ROOT}"
bash slurm_scripts/fewshot/run_wildlife_fewshot.sh configs/wildlife/StripeSpotter.json 0.125
bash slurm_scripts/fewshot/run_wildlife_fewshot.sh configs/wildlife/StripeSpotter.json 0.25
bash slurm_scripts/fewshot/run_wildlife_fewshot.sh configs/wildlife/StripeSpotter.json 0.5
bash slurm_scripts/fewshot/run_wildlife_fewshot.sh configs/wildlife/StripeSpotter.json 1.0
# or all four at once:
bash slurm_scripts/fewshot/run_wildlife_fewshot_all_sizes.sh configs/wildlife/StripeSpotter.json
```

`FEWSHOT_BACKEND=loma` runs the LoMa variant, `FEWSHOT_DRY_RUN=1` builds the view,
validates the cache and prints the `sbatch` calls without submitting, `FEWSHOT_FORCE=1`
re-mines/re-trains existing outputs. The fraction `1.0` view is the full dataset again,
rebuilt through the same path so that its numbers can be compared with the original runs.

### Single-job variant (dgxh100)

When array jobs are impractical (per-user limits on `dgxh100`: 6 GPUs / 60 CPUs / 384 GB),
`fewshot_job.sh` runs the whole chain of one fraction inside a single allocation — mining
processes every query collection of a split in one process per GPU (`wildlife_mine.py
--all_queries`, byte-identical reports), then aggregation, fine-tuning
(`accelerate --num_processes <gpus>`, global batch 32 split over the job's GPUs so that it
matches the 4×8 reference runs) and both evaluations. One job per fraction:

```bash
cd "${RDD_BENCHMARK_ROOT}"
bash slurm_scripts/fewshot/submit_fewshot_dgxh100.sh configs/wildlife/CowDataset.json
# defaults: --partition=dgxh100 --qos=quick --gres=gpu:1 --cpus-per-task=10 --mem=64G --time=24:00:00
#           (6 GPUs / 60 CPUs / 384 GB per user on dgxh100 -> six such jobs run concurrently)
# FEWSHOT_FRACTIONS, FEWSHOT_QOS, FEWSHOT_GPUS, FEWSHOT_CPUS, FEWSHOT_MEM, FEWSHOT_TIME override them
```

Logs: `logs/fewshot/fs-<dataset>-<fraction>-<backend>-<job>.out` (per-step timings at the
end) and `logs/fewshot/<dataset>-<protocol>-<view>-<backend>/mine-*.log`; the run record
`$FEWSHOT_ROOT/runs/<dataset>/<protocol>/<view>-<backend>.json` keeps the step timings.
`FEWSHOT_STOP_AFTER=view|mining|aggregate|training` stops a job early.

### Whole experiment per dataset (default workflow)

`pipeline_dataset.sh` queues everything for one dataset and `spawn_datasets.sh` does it for
a list of datasets (default order: datasets with cosine top-1 < 90 % in the reference
results first, cheapest first; then the saturated ones):

```bash
cd "${RDD_BENCHMARK_ROOT}"
bash slurm_scripts/fewshot/pipeline_dataset.sh configs/wildlife/NyalaData.json   # one dataset
bash slurm_scripts/fewshot/spawn_datasets.sh                                     # the default list
bash slurm_scripts/fewshot/spawn_datasets.sh SeaStarReID2023 HyenaID2022         # a subset
```

Per dataset: (1) the views of all fractions are built inline and fractions whose view is
identical to the previous one (minimum-of-two-images rule) are dropped; (2) one
`fewshot_job.sh` per remaining fraction is submitted (skipped when the checkpoint exists or
a job of that name is already queued); (3) a CPU job with `--dependency` on them submits the
probe arrays for both retrieval settings (`explainable_individual_reidentification/probe-fewshot-wildlife.sh`,
reduced gallery and `--full-gallery`; variants with a completed run are skipped); (4) a CPU
collector job with `--dependency` on the arrays writes
`$EXREID_ROOT/reports/fewshot/<dataset>/` (`fewshot_summary.md`, `fewshot_results.csv`,
`fewshot_<metric>.png` for the reduced gallery, `fewshot_fullgallery_<metric>.png` for the
full gallery). `fewshot_job.sh` stages the feature cache to the node-local disk first
(`FEWSHOT_STAGE_CACHE=0` disables it): training reads ~100 cache files per step and was
I/O bound on NFS (3.4 s/step on dgxh100 with 8 loader workers).

The probe benchmark uses the same selection through `metadata_fewshot/metadata_<dataset>.csv`
with `dataset.split_col=split_frac<f>_seed<s>` (`train` = kept training rows = gallery,
`unused` = dropped training rows, `test` unchanged). The full-gallery setting uses the
original `split` column of that file (whole training split as the database) with the
few-shot fine-tuned checkpoints; its default-checkpoint numbers are fraction independent
and equal the reference tables.
