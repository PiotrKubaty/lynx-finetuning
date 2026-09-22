import torch
from torch import nn

from contrastive_finetuning.train_by_lg_matches import (
    configure_rdd_trainable_component,
    set_rdd_training_mode,
)


class FakeRDD(nn.Module):
    def __init__(self):
        super().__init__()
        self.detector = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3))
        self.descriptor = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3))


def test_rdd_descriptor_mode_freezes_detector_and_trains_descriptor():
    model = FakeRDD()
    configure_rdd_trainable_component(model, True, "descriptor")
    set_rdd_training_mode(model, True, "descriptor")

    assert all(not parameter.requires_grad for parameter in model.detector.parameters())
    assert all(parameter.requires_grad for parameter in model.descriptor.parameters())
    assert not model.detector.training
    assert model.descriptor.training


def test_legacy_rdd_mode_still_trains_all_rdd_weights():
    model = FakeRDD()
    configure_rdd_trainable_component(model, True, "all")
    set_rdd_training_mode(model, True, "all")

    assert all(parameter.requires_grad for parameter in model.parameters())
    assert model.detector.training
    assert model.descriptor.training


def test_frozen_rdd_mode_keeps_every_parameter_frozen():
    model = FakeRDD()
    configure_rdd_trainable_component(model, False, "all")
    set_rdd_training_mode(model, False, "all")

    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert not model.training


def test_resume_argument_requires_a_saved_epoch_directory_and_stateless_flags(tmp_path, monkeypatch):
    import pytest
    from contrastive_finetuning.train_by_lg_matches import parse_args, resume_epoch

    assert resume_epoch(tmp_path / "epoch_07") == 7
    assert resume_epoch(tmp_path / "latest") is None
    base = [
        "prog", "--train_index", "train.json", "--val_index", "val.json", "--data_root", str(tmp_path),
        "--rdd_weights", "rdd.pth", "--lg_weights", "lg.pth", "--trained_model", "rdd",
        "--rdd_train_component", "descriptor",
    ]
    checkpoint = tmp_path / "epoch_12"
    checkpoint.mkdir()
    monkeypatch.setattr("sys.argv", base + ["--resume", str(checkpoint)])
    with pytest.raises(SystemExit):  # no accelerator state inside
        parse_args()
    (checkpoint / "random_states_0.pkl").write_bytes(b"x")
    monkeypatch.setattr("sys.argv", base + ["--resume", str(checkpoint)])
    assert parse_args().resume == checkpoint
    monkeypatch.setattr("sys.argv", base + ["--resume", str(checkpoint), "--ema_decay", "0.99"])
    with pytest.raises(SystemExit):  # per-run state that is not checkpointed
        parse_args()
