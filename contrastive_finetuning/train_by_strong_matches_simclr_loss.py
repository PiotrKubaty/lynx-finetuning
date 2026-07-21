from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from contrastive_finetuning.train_common import (
    add_common_args, retrieval_boost_loss, run_training, scale_grad,
)


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Contrastive fine-tuning of RDD descriptor (SimCLR-style InfoNCE)")
    add_common_args(p)
    p.add_argument("--temperature", type=float, default=0.1, help="Softmax temperature for InfoNCE loss")
    return p.parse_args()


# ── loss ──────────────────────────────────────────────────────────────────────
def simclr_descriptor_loss(
    feats_a: list[dict],
    feats_p: list[dict],
    feats_n: list[dict],
    matches_pos: list[torch.Tensor],
    matches_neg: list[torch.Tensor],
    temperature: float,
    neg_grad_scale: float = 1.0,
    match_boost_weight: float = 0.0,
    match_boost_temperature: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """
    SimCLR-style loss using LightGlue matches as correspondence oracle.

    `matches_pos`/`matches_neg` are correspondence indices computed upstream
    (train_common.run_lg_matching) — possibly from an EMA teacher's
    descriptors rather than `feats_*` themselves, see --ema_decay.

    For each image in the batch, mirroring matched_descriptor_loss's use of
    pos_sims.mean() / neg_sims.mean() as single pair-level scores:
      - pos_score = mean dot product of LG-matched (anchor, positive) pairs
      - neg_score = mean dot product of LG-matched (anchor, negative) pairs
      - logits = [pos_score, neg_score] / temperature
      - loss   = cross_entropy(logits, label=0)

    This is a 2-class softmax over the same two scalars the margin loss
    compares, i.e. a smooth substitute for relu(margin - pos_score + neg_score):
      loss = softplus(-(pos_score - neg_score) / temperature) / const
    No other batch samples or unmatched descriptors are pulled in as negatives.

    Images with matched positives but zero matched negatives are skipped, since
    there is no neg_score to contrast against (the triplet loss instead falls
    back to relu(margin - pos_sims).mean() in that case).

    `neg_grad_scale` < 1 weakens the gradient of the neg_score term without
    changing the forward loss value — see --neg_grad_scale.

    `match_boost_weight` > 0 adds a full-candidate-set retrieval cross-entropy
    (see retrieval_boost_loss) that directly targets match uniqueness rather
    than raw similarity magnitude — meant to be activated dynamically when
    matches/mean_pos trends down, see --match_boost_weight.
    """
    device = feats_a[0]["descriptors"].device

    losses = []
    n_skipped = 0
    na_list, np_list, nn_list, pos_match_list, neg_match_list = [], [], [], [], []
    pos_sim_list, neg_sim_list, retrieval_acc_list = [], [], []

    for fa, fp, fn, m_pos, m_neg in zip(
        feats_a, feats_p, feats_n, matches_pos, matches_neg,
    ):
        da = fa["descriptors"]  # (Na, D) — gradient flows here
        dp = fp["descriptors"]  # (Np, D)
        dn = fn["descriptors"]  # (Nn, D)

        if m_pos.shape[0] == 0 or da.shape[0] == 0:
            n_skipped += 1
            continue

        na_list.append(da.shape[0])
        np_list.append(dp.shape[0])
        nn_list.append(dn.shape[0])
        pos_match_list.append(m_pos.shape[0])
        neg_match_list.append(m_neg.shape[0])

        pos_sims = (da[m_pos[:, 0]] * dp[m_pos[:, 1]]).sum(dim=-1)  # (Mp,)
        pos_score = pos_sims.mean()
        pos_sim_list.append(pos_score.item())

        if m_neg.shape[0] == 0:
            # No neg_score to contrast against: skip, like InfoNCE with a
            # single class (the triplet loss instead falls back to
            # relu(margin - pos_sims).mean() here).
            n_skipped += 1
            continue

        neg_sims = (da[m_neg[:, 0]] * dn[m_neg[:, 1]]).sum(dim=-1)  # (Mn,)
        neg_sim_list.append(neg_sims.mean().item())
        neg_score = scale_grad(neg_sims.mean(), neg_grad_scale)

        logits = torch.stack([pos_score, neg_score]) / temperature  # (2,)
        label = torch.zeros(1, dtype=torch.long, device=device)
        sample_loss = F.cross_entropy(logits.unsqueeze(0), label)

        if match_boost_weight > 0 and dp.shape[0] > 1:
            sample_loss = sample_loss + match_boost_weight * retrieval_boost_loss(
                da, dp, m_pos, match_boost_temperature
            )

        losses.append(sample_loss)

        with torch.no_grad():
            retrieval_acc_list.append(float((pos_score > neg_score).item()))

    def _mean(lst):
        return sum(lst) / len(lst) if lst else 0.0

    stats = {
        "n_skipped":        n_skipped,
        "mean_na":          _mean(na_list),
        "mean_np":          _mean(np_list),
        "mean_nn":          _mean(nn_list),
        "mean_pos_matches": _mean(pos_match_list),
        "mean_neg_matches": _mean(neg_match_list),
        "mean_pos_sim":     _mean(pos_sim_list),
        "mean_neg_sim":     _mean(neg_sim_list),
        "retrieval_acc":    _mean(retrieval_acc_list),
    }

    if not losses:
        stats["n_skipped"] += len(feats_a)
        return torch.zeros(1, device=device, requires_grad=True).squeeze(), stats
    return torch.stack(losses).mean(), stats


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    run_training(args, simclr_descriptor_loss, args.temperature)


if __name__ == "__main__":
    main()
