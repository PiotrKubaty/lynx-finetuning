import importlib.util
from pathlib import Path

import torch

from rdd_patch.lightglue_masked import LightGlueMasked
from rdd.RDD.RDD import build
from rdd.RDD.utils import read_config


def _load_lightglue_class():
    lg_path = Path(__file__).resolve().parents[1] / "rdd" / "RDD" / "matchers" / "lightglue.py"
    spec = importlib.util.spec_from_file_location("rdd_lightglue_module", lg_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load LightGlue from {lg_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LightGlue

def build_rdd(weights: Path, device: torch.device, top_k: int):
    rdd_conf = read_config("rdd/configs/default.yaml")
    rdd_conf["device"] = str(device)
    model = build(rdd_conf, weights=str(weights))
    model.top_k = top_k
    model.set_softdetect(top_k=top_k)
    model.to(device)
    model.eval()
    return model

def _lightglue_config(weights: str, init_threshold: float = 0.01) -> dict:
    return {
        "name": "lightglue",
        "input_dim": 256,
        "descriptor_dim": 256,
        "add_scale_ori": False,
        "n_layers": 9,
        "num_heads": 4,
        "flash": True,
        "mp": False,
        "filter_threshold": init_threshold,
        "depth_confidence": -1,
        "width_confidence": -1,
        "weights": weights,
    }


def build_lightglue(device: torch.device, weights="rdd/weights/RDD_lg-v2.pth", init_threshold=0.01):
    """Unmasked LightGlue for single-pair scoring (benchmark / cache builder)."""
    LightGlue = _load_lightglue_class()
    lg = LightGlue("rdd", **_lightglue_config(weights, init_threshold)).to(device).eval()
    return lg


def build_masked_lg(device: torch.device, weights="rdd/weights/RDD_lg-v2.pth", init_threshold=0.01):
    lg = LightGlueMasked("rdd", **_lightglue_config(weights, init_threshold)).to(device).eval()
    return lg
