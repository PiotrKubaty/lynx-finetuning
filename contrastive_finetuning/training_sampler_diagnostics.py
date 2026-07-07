from __future__ import annotations

import random
import re
import shlex
import sys
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from contrastive_finetuning.loading import (
    BalancedBatchSampler,
    LabeledImageFolder,
    TripletImageFolder,
)
from contrastive_finetuning.models import build_masked_lg, build_rdd
from contrastive_finetuning.pair_quality import PairMiningConfig, PairQualityCache
from contrastive_finetuning.train import (
    batch_features,
    build_pair_mining_config,
    build_train_transform,
    cv2,
    draw_matches,
    extract_train,
    load_pair_quality_cache_for_training,
    parse_args,
    resize_long_side,
    tensor_to_bgr,
    unpack_matches,
)


_SIMPLE_VAR_PATTERN = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")


@dataclass
class ShellLaunchConfig:
    shell_path: Path
    repo_root: Path
    variables: dict[str, str]
    cli_args: list[str]
    use_pair_quality_mining: bool
    use_retrieval_probe: bool


@dataclass
class DiagnosticContext:
    args: Any
    shell: ShellLaunchConfig
    train_transform: Any
    pair_mining_config: PairMiningConfig
    pair_quality_cache: PairQualityCache | None
    triplet_dataset: TripletImageFolder
    labeled_dataset: LabeledImageFolder | None
    match_backend: dict[str, Any] | None = None


@dataclass
class TripletExample:
    mode: str
    batch_index: int | None
    group_index: int | None
    anchor_idx: int
    positive_idx: int
    negative_idx: int
    positive_origin: str
    positive_relation: str
    negative_origin: str
    positive_cache_record: dict[str, Any] | None
    negative_cache_record: dict[str, Any] | None


def _to_bool(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _strip_inline_comment(line: str) -> str:
    in_single = False
    in_double = False
    escaped = False
    out: list[str] = []
    for ch in line:
        if escaped:
            out.append(ch)
            escaped = False
            continue
        if ch == "\\":
            out.append(ch)
            escaped = True
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
            out.append(ch)
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            out.append(ch)
            continue
        if ch == "#" and not in_single and not in_double:
            break
        out.append(ch)
    return "".join(out).strip()


def _expand_shell_value(value: str, variables: dict[str, str]) -> str:
    text = value.strip()
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1]
    for _ in range(8):
        replaced = _expand_braced_variables(text, variables)
        replaced = _SIMPLE_VAR_PATTERN.sub(lambda m: variables.get(m.group(1), ""), replaced)
        if replaced == text:
            return replaced
        text = replaced
    return text


def _expand_braced_variables(text: str, variables: dict[str, str]) -> str:
    result: list[str] = []
    i = 0
    while i < len(text):
        if text.startswith("${", i):
            end = _find_matching_brace(text, i + 2)
            if end == -1:
                result.append(text[i:])
                break
            expr = text[i + 2 : end]
            result.append(_resolve_braced_expression(expr, variables))
            i = end + 1
            continue
        result.append(text[i])
        i += 1
    return "".join(result)


def _find_matching_brace(text: str, start: int) -> int:
    depth = 1
    i = start
    while i < len(text):
        if text.startswith("${", i):
            depth += 1
            i += 2
            continue
        if text[i] == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _resolve_braced_expression(expr: str, variables: dict[str, str]) -> str:
    name, default = _split_default_expr(expr)
    if default is None:
        return variables.get(name, "")
    value = variables.get(name)
    if value not in (None, ""):
        return value
    return _expand_shell_value(default, variables)


def _split_default_expr(expr: str) -> tuple[str, str | None]:
    depth = 0
    i = 0
    while i < len(expr) - 1:
        if expr.startswith("${", i):
            depth += 1
            i += 2
            continue
        if expr[i] == "}" and depth > 0:
            depth -= 1
            i += 1
            continue
        if depth == 0 and expr[i : i + 2] == ":-":
            return expr[:i], expr[i + 2 :]
        i += 1
    return expr, None


def _parse_shell_variables(shell_text: str) -> dict[str, str]:
    variables: dict[str, str] = {}
    for raw_line in shell_text.splitlines():
        line = _strip_inline_comment(raw_line)
        if not line or line.startswith("source ") or line.startswith("export ") or line.startswith("if "):
            continue
        match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$", line)
        if not match:
            continue
        key, raw_value = match.groups()
        if raw_value.startswith("$("):
            continue
        variables[key] = _expand_shell_value(raw_value, variables)
    return variables


def _extract_accelerate_cli_args(shell_text: str, variables: dict[str, str]) -> list[str]:
    lines = shell_text.splitlines()
    start_idx = next((idx for idx, line in enumerate(lines) if "contrastive_finetuning.train" in line), None)
    if start_idx is None:
        raise ValueError("Could not find contrastive_finetuning.train launcher in finetune shell file.")

    command_parts: list[str] = []
    for idx in range(start_idx, len(lines)):
        piece = _strip_inline_comment(lines[idx])
        if not piece:
            continue
        trailing_backslash = piece.endswith("\\")
        if trailing_backslash:
            piece = piece[:-1].strip()
        command_parts.append(piece)
        if not trailing_backslash:
            break

    joined = " ".join(command_parts)
    tokens = shlex.split(joined)
    if "-m" not in tokens:
        raise ValueError("Could not parse accelerate command from finetune shell file.")
    mod_idx = tokens.index("-m")
    if mod_idx + 1 >= len(tokens) or tokens[mod_idx + 1] != "contrastive_finetuning.train":
        raise ValueError("Accelerate command does not target contrastive_finetuning.train.")

    cli_args: list[str] = []
    for token in tokens[mod_idx + 2 :]:
        if token == "${EXTRA_TRAIN_ARGS[@]}":
            continue
        cli_args.append(_expand_shell_value(token, variables))
    return cli_args


def load_train_args_from_finetune_sh(shell_path: str | Path = "finetune.sh") -> tuple[Any, ShellLaunchConfig]:
    shell_path = Path(shell_path)
    shell_text = shell_path.read_text()
    variables = _parse_shell_variables(shell_text)
    cli_args = _extract_accelerate_cli_args(shell_text, variables)
    use_pair_quality_mining = _to_bool(variables.get("USE_PAIR_QUALITY_MINING", "false"))
    use_retrieval_probe = _to_bool(variables.get("USE_RETRIEVAL_PROBE", "false"))

    if use_pair_quality_mining:
        cli_args.extend(
            [
                "--use_pair_quality_mining",
                "--pair_quality_cache_dir",
                variables.get("PAIR_QUALITY_CACHE_DIR", ""),
                "--positive_quality_mode",
                variables.get("POSITIVE_QUALITY_MODE", "bucketed"),
                "--hard_negative_ratio",
                variables.get("HARD_NEGATIVE_RATIO", "0.5"),
            ]
        )
    if use_retrieval_probe:
        cli_args.extend(
            [
                "--use_retrieval_probe",
                "--retrieval_probe_num_queries",
                variables.get("RETRIEVAL_PROBE_NUM_QUERIES", "16"),
                "--retrieval_probe_gallery_per_id",
                variables.get("RETRIEVAL_PROBE_GALLERY_PER_ID", "2"),
                "--retrieval_probe_frames_per_seq",
                variables.get("RETRIEVAL_PROBE_FRAMES_PER_SEQ", "2"),
                "--retrieval_probe_every_n_epochs",
                variables.get("RETRIEVAL_PROBE_EVERY_N_EPOCHS", "1"),
            ]
        )

    shell_cfg = ShellLaunchConfig(
        shell_path=shell_path,
        repo_root=shell_path.resolve().parent,
        variables=variables,
        cli_args=cli_args,
        use_pair_quality_mining=use_pair_quality_mining,
        use_retrieval_probe=use_retrieval_probe,
    )
    with _temporary_argv(["diagnostic", *cli_args]):
        args = parse_args()
    return args, shell_cfg


@contextmanager
def _temporary_argv(argv: list[str]):
    previous = sys.argv[:]
    sys.argv = argv
    try:
        yield
    finally:
        sys.argv = previous


@contextmanager
def _temporary_random_seed(seed: int):
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)


def _resolve_repo_relative_path(path_value: Any, repo_root: Path) -> Path | Any:
    if path_value is None:
        return None
    path_obj = Path(path_value)
    if path_obj.is_absolute():
        return path_obj
    return (repo_root / path_obj).resolve()


def _normalize_repo_relative_args(args: Any, repo_root: Path) -> Any:
    for attr in ("train_data", "val_data", "rdd_weights", "lg_weights", "pair_quality_cache_dir", "output_dir"):
        if hasattr(args, attr):
            value = getattr(args, attr)
            if value is not None:
                setattr(args, attr, _resolve_repo_relative_path(value, repo_root))
    return args


def build_diagnostic_context(shell_path: str | Path = "finetune.sh") -> DiagnosticContext:
    args, shell_cfg = load_train_args_from_finetune_sh(shell_path)
    args = _normalize_repo_relative_args(args, shell_cfg.repo_root)
    train_transform = build_train_transform(args)
    pair_mining_config = build_pair_mining_config(args)
    try:
        pair_quality_cache = load_pair_quality_cache_for_training(args)
    except FileNotFoundError as exc:
        if args.use_pair_quality_mining:
            warnings.warn(
                f"{exc}. Falling back to sequence-aware sampling without pair-quality cache for diagnostics."
            )
            args.use_pair_quality_mining = False
            args.pair_quality_cache_dir = None
            pair_mining_config = build_pair_mining_config(args)
            pair_quality_cache = None
        else:
            raise

    triplet_dataset = TripletImageFolder(
        args.train_data,
        transform=train_transform,
        anchor_transform=train_transform,
        sequence_aware_sampling=args.sequence_aware_sampling,
        pair_quality_cache=pair_quality_cache,
        pair_mining_config=pair_mining_config,
    )
    labeled_dataset = None
    if args.batch_mode == "balanced":
        labeled_dataset = LabeledImageFolder(
            args.train_data,
            transform=train_transform,
            sequence_aware_sampling=args.sequence_aware_sampling,
            pair_quality_cache=pair_quality_cache,
            pair_mining_config=pair_mining_config,
        )

    return DiagnosticContext(
        args=args,
        shell=shell_cfg,
        train_transform=train_transform,
        pair_mining_config=pair_mining_config,
        pair_quality_cache=pair_quality_cache,
        triplet_dataset=triplet_dataset,
        labeled_dataset=labeled_dataset,
    )


def summarize_context(context: DiagnosticContext) -> dict[str, Any]:
    args = context.args
    return {
        "train_data": str(args.train_data),
        "batch_mode": args.batch_mode,
        "resize": args.resize,
        "top_k": args.top_k,
        "sequence_aware_sampling": bool(args.sequence_aware_sampling),
        "pair_quality_mining": bool(args.use_pair_quality_mining),
        "positive_quality_mode": args.positive_quality_mode,
        "hard_negative_ratio": args.hard_negative_ratio,
        "n_classes": getattr(args, "n_classes", None),
        "n_samples_per_class": getattr(args, "n_samples_per_class", None),
        "batch_size": getattr(args, "batch_size", None),
        "aug_profile": args.aug_profile,
        "num_workers": args.num_workers,
    }


def _candidate_record(
    cache: PairQualityCache | None,
    mapping_name: str,
    anchor_idx: int,
    candidate_idx: int,
) -> dict[str, Any] | None:
    if cache is None:
        return None
    mapping = getattr(cache, mapping_name)
    for record in mapping.get(anchor_idx, []):
        if int(record.get("candidate_index", -1)) == candidate_idx:
            return record
    return None


def _positive_relation(dataset: Any, anchor_idx: int, positive_idx: int) -> str:
    anchor_meta = dataset._sample_meta[anchor_idx]
    positive_meta = dataset._sample_meta[positive_idx]
    if positive_meta.source_id != anchor_meta.source_id:
        return "different_source"
    if positive_meta.sequence_id != anchor_meta.sequence_id:
        return "different_sequence"
    if positive_idx != anchor_idx:
        return "same_sequence"
    return "self_fallback"


def _sample_negative_with_reason(
    dataset: Any,
    label: int,
    anchor_idx: int,
    rng: random.Random,
) -> tuple[int, str, dict[str, Any] | None]:
    cache = dataset.pair_quality_cache
    config = dataset.pair_mining_config
    if (
        cache is not None
        and config is not None
        and config.use_pair_quality_mining
        and config.use_hard_negative_cache
        and rng.random() < config.hard_negative_ratio
    ):
        hard_candidates = cache.negative_candidates_by_anchor.get(anchor_idx, [])
        hard_candidates = [c for c in hard_candidates if not c.get("same_identity", True)]
        if hard_candidates:
            if config.positive_quality_mode == "ranked":
                picked = hard_candidates[0]
            else:
                picked = rng.choice(hard_candidates[: max(1, min(8, len(hard_candidates)))])
            return int(picked["candidate_index"]), "mined_hard_negative", picked

    neg_label = label
    class_ids = list(dataset._class_to_indices.keys())
    while neg_label == label:
        neg_label = rng.choice(class_ids)
    neg_idx = rng.choice(dataset._class_to_indices[neg_label])
    return neg_idx, "random_negative", None


def sample_triplet_examples(
    context: DiagnosticContext,
    num_examples: int = 8,
    seed: int = 0,
) -> list[TripletExample]:
    examples: list[TripletExample] = []
    triplet_ds = context.triplet_dataset
    rng = random.Random(seed)

    if context.args.batch_mode == "balanced":
        if context.labeled_dataset is None:
            raise ValueError("Balanced diagnostic requested without a labeled dataset.")
        sampler = BalancedBatchSampler(
            context.labeled_dataset,
            n_classes=context.args.n_classes,
            n_samples=context.args.n_samples_per_class,
        )
        with _temporary_random_seed(seed):
            for batch_index, batch_indices in enumerate(sampler):
                for group_index in range(context.args.n_classes):
                    start = group_index * context.args.n_samples_per_class
                    group = batch_indices[start : start + context.args.n_samples_per_class]
                    if len(group) < 2:
                        continue
                    anchor_idx = int(group[0])
                    positive_idx = int(group[1])
                    label = context.labeled_dataset.targets[anchor_idx]
                    negative_idx, negative_origin, negative_record = _sample_negative_with_reason(
                        triplet_ds, label, anchor_idx, rng
                    )
                    examples.append(
                        TripletExample(
                            mode="balanced",
                            batch_index=batch_index,
                            group_index=group_index,
                            anchor_idx=anchor_idx,
                            positive_idx=positive_idx,
                            negative_idx=negative_idx,
                            positive_origin="balanced_group",
                            positive_relation=_positive_relation(context.labeled_dataset, anchor_idx, positive_idx),
                            negative_origin=negative_origin,
                            positive_cache_record=_candidate_record(
                                context.pair_quality_cache,
                                "positive_candidates_by_anchor",
                                anchor_idx,
                                positive_idx,
                            ),
                            negative_cache_record=negative_record
                            or _candidate_record(
                                context.pair_quality_cache,
                                "negative_candidates_by_anchor",
                                anchor_idx,
                                negative_idx,
                            ),
                        )
                    )
                    if len(examples) >= num_examples:
                        return examples
    else:
        anchor_indices = list(range(len(triplet_ds)))
        rng.shuffle(anchor_indices)
        for anchor_idx in anchor_indices[: min(num_examples, len(anchor_indices))]:
            label = triplet_ds.targets[anchor_idx]
            positive_idx = triplet_ds._sample_positive_index(anchor_idx, rng=rng)
            negative_idx, negative_origin, negative_record = _sample_negative_with_reason(
                triplet_ds, label, anchor_idx, rng
            )
            examples.append(
                TripletExample(
                    mode="triplet",
                    batch_index=None,
                    group_index=None,
                    anchor_idx=anchor_idx,
                    positive_idx=positive_idx,
                    negative_idx=negative_idx,
                    positive_origin="triplet_sampler",
                    positive_relation=_positive_relation(triplet_ds, anchor_idx, positive_idx),
                    negative_origin=negative_origin,
                    positive_cache_record=_candidate_record(
                        context.pair_quality_cache,
                        "positive_candidates_by_anchor",
                        anchor_idx,
                        positive_idx,
                    ),
                    negative_cache_record=negative_record
                    or _candidate_record(
                        context.pair_quality_cache,
                        "negative_candidates_by_anchor",
                        anchor_idx,
                        negative_idx,
                    ),
                )
            )
    return examples


def sample_balanced_batches(
    context: DiagnosticContext,
    num_batches: int = 1,
    seed: int = 0,
) -> list[list[int]]:
    if context.args.batch_mode != "balanced" or context.labeled_dataset is None:
        raise ValueError("Balanced batch sampling is only available when finetune.sh is in balanced mode.")
    sampler = BalancedBatchSampler(
        context.labeled_dataset,
        n_classes=context.args.n_classes,
        n_samples=context.args.n_samples_per_class,
    )
    batches: list[list[int]] = []
    with _temporary_random_seed(seed):
        for batch in sampler:
            batches.append([int(idx) for idx in batch])
            if len(batches) >= num_batches:
                break
    return batches


def _format_meta(dataset: Any, idx: int, role: str) -> str:
    meta = dataset._sample_meta[idx]
    return (
        f"{role}\n"
        f"id={meta.identity}\n"
        f"source={meta.source_id}\n"
        f"seq={meta.sequence_id}\n"
        f"path={meta.path.name}"
    )


def _format_record(record: dict[str, Any] | None, fallback: str) -> str:
    if record is None:
        return fallback
    return (
        f"{fallback} | band={record.get('quality_band', 'n/a')} | "
        f"score={float(record.get('composite_score', 0.0)):.3f} | "
        f"matches={int(record.get('match_count', 0))} | "
        f"mean_conf={float(record.get('mean_confidence', 0.0)):.3f}"
    )


def _load_raw_image(dataset: Any, idx: int) -> Image.Image:
    if hasattr(dataset, "_load"):
        return dataset._load(idx)
    sample_path, _ = dataset.samples[idx]
    return dataset.loader(sample_path)


def _apply_transform(transform: Any, image: Image.Image, seed: int) -> torch.Tensor:
    with _temporary_random_seed(seed):
        result = transform(image)
    return result


def _tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    data = tensor.detach().cpu().float().clamp(0, 1)
    if data.ndim == 3:
        data = data.permute(1, 2, 0)
    return data.numpy()


def render_triplet_panel(
    context: DiagnosticContext,
    example: TripletExample,
    seed: int = 0,
    figsize: tuple[float, float] = (16, 9),
) -> plt.Figure:
    ds = context.triplet_dataset
    raw_images = {
        "anchor": _load_raw_image(ds, example.anchor_idx),
        "positive": _load_raw_image(ds, example.positive_idx),
        "negative": _load_raw_image(ds, example.negative_idx),
    }
    aug_images = {
        "anchor": _apply_transform(ds.anchor_transform, raw_images["anchor"], seed + 11),
        "positive": _apply_transform(ds.transform, raw_images["positive"], seed + 17),
        "negative": _apply_transform(ds.transform, raw_images["negative"], seed + 23),
    }

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    roles = ["anchor", "positive", "negative"]
    for col, role in enumerate(roles):
        axes[0, col].imshow(np.asarray(raw_images[role]))
        axes[0, col].set_title(_format_meta(ds, getattr(example, f"{role}_idx"), role), fontsize=9)
        axes[0, col].axis("off")

        axes[1, col].imshow(_tensor_to_image(aug_images[role]))
        axes[1, col].set_title(f"{role} augmented", fontsize=9)
        axes[1, col].axis("off")

    header_bits = [f"mode={example.mode}"]
    if example.batch_index is not None:
        header_bits.append(f"batch={example.batch_index}")
    if example.group_index is not None:
        header_bits.append(f"group={example.group_index}")
    header_bits.append(f"positive_relation={example.positive_relation}")
    fig.suptitle(" | ".join(header_bits), fontsize=12)

    footer_lines = [
        _format_record(
            example.positive_cache_record,
            f"positive_origin={example.positive_origin}",
        ),
        _format_record(
            example.negative_cache_record,
            f"negative_origin={example.negative_origin}",
        ),
        f"anchor_path={ds._sample_meta[example.anchor_idx].path}",
        f"positive_path={ds._sample_meta[example.positive_idx].path}",
        f"negative_path={ds._sample_meta[example.negative_idx].path}",
    ]
    fig.text(0.01, 0.01, "\n".join(footer_lines), fontsize=8, va="bottom", family="monospace")
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    return fig


def display_triplet_examples(
    context: DiagnosticContext,
    num_examples: int = 8,
    seed: int = 0,
    figsize: tuple[float, float] = (16, 9),
) -> list[TripletExample]:
    examples = sample_triplet_examples(context, num_examples=num_examples, seed=seed)
    for idx, example in enumerate(examples):
        fig = render_triplet_panel(context, example, seed=seed + idx * 101, figsize=figsize)
        plt.show()
        plt.close(fig)
    return examples


def _resolve_match_device(device: str | torch.device = "auto") -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def ensure_match_backend(context: DiagnosticContext, device: str | torch.device = "auto") -> dict[str, Any]:
    resolved_device = _resolve_match_device(device)
    backend = context.match_backend
    if backend is not None and str(backend.get("device")) == str(resolved_device):
        return backend

    rdd = build_rdd(context.args.rdd_weights, resolved_device, context.args.top_k)
    lg = build_masked_lg(resolved_device, weights=context.args.lg_weights)
    lg.eval()
    backend = {"device": resolved_device, "rdd": rdd, "lg": lg}
    context.match_backend = backend
    return backend


def _bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    if cv2 is not None:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image[..., ::-1]


def _draw_keypoints_pair(
    img0: np.ndarray,
    img1: np.ndarray,
    key0: np.ndarray,
    key1: np.ndarray,
    title: str,
    max_keypoints: int = 256,
) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("OpenCV is required for keypoint diagnostics.")
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    canvas = np.zeros((max(h0, h1) + 28, w0 + w1, 3), dtype=np.uint8)
    canvas[28 : 28 + h0, :w0] = img0
    canvas[28 : 28 + h1, w0:] = img1
    palette = [(80, 200, 255), (0, 220, 120)]
    for x, y in key0[:max_keypoints]:
        cv2.circle(canvas, (int(x), int(y) + 28), 2, palette[0], -1)
    for x, y in key1[:max_keypoints]:
        cv2.circle(canvas, (int(x) + w0, int(y) + 28), 2, palette[1], -1)
    cv2.putText(canvas, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
    return canvas


def compute_triplet_match_artifacts(
    context: DiagnosticContext,
    example: TripletExample,
    seed: int = 0,
    device: str | torch.device = "auto",
) -> dict[str, Any]:
    if cv2 is None:
        raise RuntimeError("OpenCV is required for match diagnostics.")
    backend = ensure_match_backend(context, device=device)
    ds = context.triplet_dataset

    raw_images = {
        "anchor": _load_raw_image(ds, example.anchor_idx),
        "positive": _load_raw_image(ds, example.positive_idx),
        "negative": _load_raw_image(ds, example.negative_idx),
    }
    aug_images = {
        "anchor": _apply_transform(ds.anchor_transform, raw_images["anchor"], seed + 11),
        "positive": _apply_transform(ds.transform, raw_images["positive"], seed + 17),
        "negative": _apply_transform(ds.transform, raw_images["negative"], seed + 23),
    }

    anchors = aug_images["anchor"].unsqueeze(0)
    positives = aug_images["positive"].unsqueeze(0)
    negatives = aug_images["negative"].unsqueeze(0)
    anchors_r = resize_long_side(anchors, context.args.resize).to(backend["device"])
    positives_r = resize_long_side(positives, context.args.resize).to(backend["device"])
    negatives_r = resize_long_side(negatives, context.args.resize).to(backend["device"])

    h_r, w_r = anchors_r.shape[-2:]
    h_orig, w_orig = anchors.shape[-2:]
    scale = np.array([w_orig / w_r, h_orig / h_r], dtype=np.float32)

    with torch.no_grad():
        feats_a = extract_train(backend["rdd"], anchors_r)
        feats_p = extract_train(backend["rdd"], positives_r)
        feats_n = extract_train(backend["rdd"], negatives_r)
        data_a = batch_features(feats_a, h_r, w_r)
        data_p = batch_features(feats_p, h_r, w_r)
        data_n = batch_features(feats_n, h_r, w_r)
        pred_pos = backend["lg"]({"image0": data_a, "image1": data_p})
        pred_neg = backend["lg"]({"image0": data_a, "image1": data_n})

    anchor_bgr = tensor_to_bgr(anchors[0])
    pos_bgr = tensor_to_bgr(positives[0])
    neg_bgr = tensor_to_bgr(negatives[0])
    key_a = feats_a[0]["keypoints"].detach().cpu().numpy() * scale
    key_p = feats_p[0]["keypoints"].detach().cpu().numpy() * scale
    key_n = feats_n[0]["keypoints"].detach().cpu().numpy() * scale
    pts0_p, pts1_p, scores_p = unpack_matches(pred_pos, 0, key_a, key_p)
    pts0_n, pts1_n, scores_n = unpack_matches(pred_neg, 0, key_a, key_n)

    return {
        "anchor_bgr": anchor_bgr,
        "positive_bgr": pos_bgr,
        "negative_bgr": neg_bgr,
        "key_a": key_a,
        "key_p": key_p,
        "key_n": key_n,
        "pts0_p": pts0_p,
        "pts1_p": pts1_p,
        "scores_p": scores_p,
        "pts0_n": pts0_n,
        "pts1_n": pts1_n,
        "scores_n": scores_n,
    }


def render_triplet_match_panel(
    context: DiagnosticContext,
    example: TripletExample,
    seed: int = 0,
    device: str | torch.device = "auto",
    figsize: tuple[float, float] = (16, 12),
    max_matches: int = 40,
    max_keypoints: int = 256,
) -> plt.Figure:
    artifacts = compute_triplet_match_artifacts(context, example, seed=seed, device=device)

    pos_key_canvas = _draw_keypoints_pair(
        artifacts["anchor_bgr"],
        artifacts["positive_bgr"],
        artifacts["key_a"],
        artifacts["key_p"],
        title=f"positive pair keypoints | anchor={len(artifacts['key_a'])} | positive={len(artifacts['key_p'])}",
        max_keypoints=max_keypoints,
    )
    neg_key_canvas = _draw_keypoints_pair(
        artifacts["anchor_bgr"],
        artifacts["negative_bgr"],
        artifacts["key_a"],
        artifacts["key_n"],
        title=f"negative pair keypoints | anchor={len(artifacts['key_a'])} | negative={len(artifacts['key_n'])}",
        max_keypoints=max_keypoints,
    )
    pos_match_canvas = draw_matches(
        artifacts["pts0_p"],
        artifacts["pts1_p"],
        artifacts["scores_p"],
        artifacts["anchor_bgr"],
        artifacts["positive_bgr"],
        title=(
            f"positive matches | matches={len(artifacts['scores_p'])} | "
            f"conf={float(artifacts['scores_p'].mean()):.3f}" if len(artifacts['scores_p']) else "positive matches | matches=0 | conf=0.000"
        ),
        max_matches=max_matches,
    )
    neg_match_canvas = draw_matches(
        artifacts["pts0_n"],
        artifacts["pts1_n"],
        artifacts["scores_n"],
        artifacts["anchor_bgr"],
        artifacts["negative_bgr"],
        title=(
            f"negative matches | matches={len(artifacts['scores_n'])} | "
            f"conf={float(artifacts['scores_n'].mean()):.3f}" if len(artifacts['scores_n']) else "negative matches | matches=0 | conf=0.000"
        ),
        max_matches=max_matches,
    )

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    canvases = [pos_key_canvas, pos_match_canvas, neg_key_canvas, neg_match_canvas]
    titles = [
        "positive pair detected keypoints",
        "positive pair LightGlue matches",
        "negative pair detected keypoints",
        "negative pair LightGlue matches",
    ]
    for ax, canvas, title in zip(axes.flat, canvases, titles):
        ax.imshow(_bgr_to_rgb(canvas))
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    header_bits = [f"mode={example.mode}", f"positive_relation={example.positive_relation}"]
    if example.batch_index is not None:
        header_bits.append(f"batch={example.batch_index}")
    if example.group_index is not None:
        header_bits.append(f"group={example.group_index}")
    fig.suptitle(" | ".join(header_bits), fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def display_triplet_match_examples(
    context: DiagnosticContext,
    num_examples: int = 4,
    seed: int = 0,
    device: str | torch.device = "auto",
    figsize: tuple[float, float] = (16, 12),
    max_matches: int = 40,
    max_keypoints: int = 256,
) -> list[TripletExample]:
    examples = sample_triplet_examples(context, num_examples=num_examples, seed=seed)
    for idx, example in enumerate(examples):
        fig = render_triplet_match_panel(
            context,
            example,
            seed=seed + idx * 101,
            device=device,
            figsize=figsize,
            max_matches=max_matches,
            max_keypoints=max_keypoints,
        )
        plt.show()
        plt.close(fig)
    return examples


def render_balanced_batch_overview(
    context: DiagnosticContext,
    batch_indices: list[int],
    seed: int = 0,
    show_augmented: bool = True,
    figsize_per_cell: tuple[float, float] = (3.2, 3.0),
) -> plt.Figure:
    if context.labeled_dataset is None:
        raise ValueError("Balanced batch overview requires a labeled dataset.")
    ds = context.labeled_dataset
    rows = context.args.n_classes
    cols = context.args.n_samples_per_class
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(figsize_per_cell[0] * cols, figsize_per_cell[1] * rows),
        squeeze=False,
    )
    for row in range(rows):
        for col in range(cols):
            idx = batch_indices[row * cols + col]
            image = _load_raw_image(ds, idx)
            display_img = (
                _tensor_to_image(_apply_transform(ds.transform, image, seed + row * 100 + col))
                if show_augmented
                else np.asarray(image)
            )
            axes[row, col].imshow(display_img)
            axes[row, col].set_title(_format_meta(ds, idx, f"r{row}c{col}"), fontsize=8)
            axes[row, col].axis("off")
    mode_label = "augmented" if show_augmented else "raw"
    fig.suptitle(f"balanced batch overview ({mode_label})", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def display_balanced_batch_overviews(
    context: DiagnosticContext,
    num_batches: int = 1,
    seed: int = 0,
    show_augmented: bool = True,
) -> list[list[int]]:
    batches = sample_balanced_batches(context, num_batches=num_batches, seed=seed)
    for batch_idx, batch in enumerate(batches):
        fig = render_balanced_batch_overview(
            context,
            batch,
            seed=seed + batch_idx * 103,
            show_augmented=show_augmented,
        )
        plt.show()
        plt.close(fig)
    return batches
