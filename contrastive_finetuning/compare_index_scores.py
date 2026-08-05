import argparse
import json
from pathlib import Path

"""
Joins the two per-pair score dumps — contrastive_finetuning/dump_index_scores.py
(--ours) and rdd-benchmark/scripts/lynx_dump_index_scores.py (--theirs), both
scoring the SAME persisted index file with the SAME finetuned checkpoint — by
(query_frame, candidate) path, and reports:
  - each side's overall video-level accuracy on that shared pool
  - the largest per-pair score deltas between the two scoring pipelines
  - per-video agreement/disagreement on the predicted lynx_id
  - for disagreements, each pipeline's winning pair AND what the OTHER
    pipeline scored that same pair (and vice versa) — a genuine cross-repo
    disagreement on which pair is strongest looks different from one side
    just nudging a near-tie the other side also saw as close.

Pure stdlib — no torch/GPU needed, runs directly on the login node.
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ours", type=Path, required=True,
                    help="contrastive_finetuning/dump_index_scores.py output")
    p.add_argument("--theirs", type=Path, required=True,
                    help="rdd-benchmark/scripts/lynx_dump_index_scores.py output")
    p.add_argument("--top_deltas", type=int, default=20)
    return p.parse_args()


def index_by_pair(dump: dict) -> dict:
    out = {}
    for pq in dump["per_query"]:
        q = pq["query_frame"]
        for c in pq["candidates"]:
            out[(q, c["path"])] = c
    return out


def acc_str(dump: dict) -> str:
    videos = dump["videos"]
    n = sum(1 for v in videos.values() if v["best_lynx"] == v["true_lynx"])
    return f"{dump['video_accuracy']:.4f} ({n}/{len(videos)})"


def main() -> None:
    args = parse_args()
    ours = json.load(open(args.ours))
    theirs = json.load(open(args.theirs))

    print(f"ours   (chunk_size={ours.get('chunk_size')}, lg_weights={ours.get('lg_weights')}): "
          f"video_accuracy = {acc_str(ours)}")
    print(f"theirs (batch_size={theirs.get('batch_size')}, score_weights={theirs.get('score_weights')}): "
          f"video_accuracy = {acc_str(theirs)}")

    ours_pairs = index_by_pair(ours)
    theirs_pairs = index_by_pair(theirs)
    common = set(ours_pairs) & set(theirs_pairs)
    only_ours = set(ours_pairs) - set(theirs_pairs)
    only_theirs = set(theirs_pairs) - set(ours_pairs)
    print(f"\n{len(common)} common (query,candidate) pairs, "
          f"{len(only_ours)} only in ours, {len(only_theirs)} only in theirs")
    if only_ours or only_theirs:
        print("  (non-zero here means the two sides didn't score the exact same "
              "pool — e.g. a cached feature is missing on one side)")
        for q, c in list(only_ours)[:5]:
            print(f"  only ours:   {q} / {c}")
        for q, c in list(only_theirs)[:5]:
            print(f"  only theirs: {q} / {c}")

    deltas = [
        (q, c, theirs_pairs[(q, c)]["score"] - ours_pairs[(q, c)]["score"])
        for q, c in common
    ]
    deltas.sort(key=lambda x: abs(x[2]), reverse=True)
    print(f"\nlargest |score| deltas (theirs - ours), top {args.top_deltas}:")
    for q, c, d in deltas[:args.top_deltas]:
        oc, tc = ours_pairs[(q, c)], theirs_pairs[(q, c)]
        kp_note = ""
        if oc.get("n_keypoints") is not None and tc.get("n_keypoints") is not None:
            kp_note = f"  n_kp ours={oc['n_keypoints']} theirs={tc['n_keypoints']}"
        print(f"  {d:+.4f}  ours={oc['score']:.4f} theirs={tc['score']:.4f}  "
              f"query={Path(q).name} cand={Path(c).name}{kp_note}")

    videos = sorted(set(ours["videos"]) | set(theirs["videos"]))
    print(f"\n{'video':50s} {'true':8s} {'ours':8s} {'theirs':8s}")
    disagreements = []
    for v in videos:
        ov, tv = ours["videos"].get(v), theirs["videos"].get(v)
        if ov is None or tv is None:
            print(f"  {v:50s}  MISSING in {'ours' if ov is None else 'theirs'}")
            continue
        agree = ov["best_lynx"] == tv["best_lynx"]
        marker = "" if agree else "  <-- DISAGREE"
        print(f"{v:50s} {ov['true_lynx']:8s} {ov['best_lynx']:8s} {tv['best_lynx']:8s}{marker}")
        if not agree:
            disagreements.append(v)

    print(f"\n{len(disagreements)} video(s) where the two pipelines disagree on predicted lynx_id:")
    for v in disagreements:
        ov, tv = ours["videos"][v], theirs["videos"][v]
        print(f"\n  video={v} true_lynx={ov['true_lynx']}")
        print(f"    ours   winner: query={Path(ov['best_query_frame']).name} "
              f"-> {Path(ov['best_candidate']).name} ({ov['best_lynx']})  score={ov['best_score']:.4f}")
        print(f"    theirs winner: query={Path(tv['best_query_frame']).name} "
              f"-> {Path(tv['best_candidate']).name} ({tv['best_lynx']})  score={tv['best_score']:.4f}")
        cross_1 = ours_pairs.get((tv["best_query_frame"], tv["best_candidate"]))
        cross_2 = theirs_pairs.get((ov["best_query_frame"], ov["best_candidate"]))
        if cross_1 is not None:
            print(f"    ours's own score for theirs' winning pair:   {cross_1['score']:.4f}")
        if cross_2 is not None:
            print(f"    theirs's own score for ours' winning pair:   {cross_2['score']:.4f}")


if __name__ == "__main__":
    main()
