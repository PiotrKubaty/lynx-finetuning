from pathlib import Path

import torch

from rdd_patch.lightglue_masked import LightGlueMasked
from rdd.RDD.RDD import build
from rdd.RDD.utils import read_config

def build_rdd(weights: Path, device: torch.device, top_k: int):
    rdd_conf = read_config("rdd/configs/default.yaml")
    model = build(rdd_conf, weights=str(weights))
    model.top_k = top_k
    model.set_softdetect(top_k=top_k)
    model.to(device)
    model.eval()
    return model

def load_lg_state_dict(path: Path | str) -> dict:
    """LightGlue weights from a .pth or from an accelerate .safetensors dump."""
    if Path(path).suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path))
    return torch.load(str(path), map_location="cpu")


def build_masked_lg(
    device: torch.device, weights="rdd/weights/RDD_lg-v2.pth", init_threshold=0.01,
    detach_descriptors=True,
):
    """LightGlueMasked with `weights` loaded.

    `weights` may be a pretrained .pth or an accelerate `save_state` dump
    (`model.safetensors`) from a finetuning run.
    """
    n_layers = 9
    # LightGlueMasked's own loader is torch.load-only, so .safetensors goes
    # through load_lg_state_dict below instead and the module is built weightless.
    from_safetensors = Path(weights).suffix == ".safetensors"
    lg_conf = {
        "name": "lightglue",
        "input_dim": 256,
        "descriptor_dim": 256,
        "add_scale_ori": False,
        "n_layers": n_layers,
        "num_heads": 4,
        "flash": True,
        "mp": False,
        "filter_threshold": init_threshold,
        "depth_confidence": -1,
        "width_confidence": -1,
        "weights": None if from_safetensors else weights,
        "detach_descriptors": detach_descriptors,
    }
    lg = LightGlueMasked("rdd", **lg_conf)

    if from_safetensors:
        state_dict = load_lg_state_dict(weights)
        # Same legacy-key rewrite LightGlueMasked applies when it loads weights itself.
        for i in range(n_layers):
            for pattern in ((f"self_attn.{i}", f"transformers.{i}.self_attn"),
                            (f"cross_attn.{i}", f"transformers.{i}.cross_attn")):
                state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
        missing, unexpected = lg.load_state_dict(state_dict, strict=False)
        # `confidence_thresholds` is a buffer recomputed at init, so a checkpoint
        # legitimately may lack it. Anything else missing means the checkpoint did
        # not actually land in the model — which would silently evaluate
        # pretrained weights instead of the finetuned ones (LightGlueMasked loads
        # with strict=False and would say nothing).
        missing = [k for k in missing if k != "confidence_thresholds"]
        if missing or unexpected:
            raise RuntimeError(
                f"{weights}: state_dict mismatch — {len(missing)} missing, {len(unexpected)} "
                f"unexpected. missing[:5]={missing[:5]} unexpected[:5]={list(unexpected)[:5]}"
            )

    return lg.to(device).eval()
