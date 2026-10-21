"""Plug-in pi_mode for SC vs CDV using the project grader (not string equality)."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.answers import answers_equivalent, majority_vote  # noqa: E402

BUNDLE = ROOT / "results_bundle"


def load_item_answers(run_dir: Path):
    out = {}
    with (run_dir / "graded.jsonl").open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            eid = rec.get("example_id")
            answers = rec.get("sample_answers")
            if answers is None and rec.get("extra"):
                answers = rec["extra"].get("sample_answers")
            if not answers:
                answers = [rec.get("final_answer")]
            out[eid] = {
                "answers": list(answers),
                "gold": str(rec.get("gold_answer")),
                "is_correct": bool(rec.get("is_correct")),
                "answer_type": rec.get("answer_type") or "math",
                "choices": (rec.get("meta") or {}).get("choices"),
            }
    return out


def find_runs(model, dataset, elic, seed=0, n_samples=24, load_4bit=False):
    hits = []
    for mpath in BUNDLE.rglob("manifest.json"):
        st = json.loads(mpath.read_text(encoding="utf-8"))
        cfg = st["config"]
        if cfg["strategy"]["name"] != "configured_sampling":
            continue
        if cfg["model"]["name_or_path"].split("/")[-1] != model:
            continue
        if cfg["data"]["name"] != dataset:
            continue
        if (cfg.get("elicitation") or {}).get("id") != elic:
            continue
        if cfg.get("runtime", {}).get("seed") != seed:
            continue
        if bool(cfg["model"].get("load_in_4bit")) != load_4bit:
            continue
        ns = (cfg["strategy"].get("params") or {}).get("n_samples")
        if n_samples is not None and ns != n_samples:
            continue
        hits.append(mpath.parent)
    return hits


def pi_mode_from_is_correct(item_map):
    n = len(item_map)
    hit = sum(1 for row in item_map.values() if row["is_correct"])
    return hit / n if n else float("nan"), n


def cdv_pi_mode(maps_by_cfg, n_per_cfg: int):
    items = set.intersection(*[set(m) for m in maps_by_cfg.values()])
    hit = 0
    for eid in items:
        pool = []
        gold = None
        answer_type = "math"
        choices = None
        for m in maps_by_cfg.values():
            row = m[eid]
            pool.extend(row["answers"][:n_per_cfg])
            gold = row["gold"]
            answer_type = row["answer_type"]
            choices = row.get("choices")
        vote, _info = majority_vote(pool, answer_type=answer_type, choices=choices)
        if vote is not None and answers_equivalent(vote, gold, answer_type):
            hit += 1
    return hit / len(items) if items else float("nan"), len(items)


def main():
    model = "Qwen2.5-3B-Instruct"
    dataset = "math500"
    print(f"=== {model} / {dataset} graded plug-in pi_mode ===")

    maps24 = {}
    for c in ["c0", "c1", "c2"]:
        runs = find_runs(model, dataset, c, seed=0, n_samples=24)
        maps24[c] = load_item_answers(runs[0])
        pm, n = pi_mode_from_is_correct(maps24[c])
        print(f"  SC {c} N=24: {100*pm:.1f}%  n={n}")

    cdv, n = cdv_pi_mode(maps24, n_per_cfg=8)
    print(f"  CDV(c0-c2) 8*3=24 tokensamples: {100*cdv:.1f}%  n={n}")
    print(f"  Delta vs best SC: {100*(cdv - max(pi_mode_from_is_correct(m)[0] for m in maps24.values())):+.1f} pp")
    print(f"  Delta vs SC(c0): {100*(cdv - pi_mode_from_is_correct(maps24['c0'])[0]):+.1f} pp")

    maps64 = {}
    for c in ["c0", "c1", "c2", "c3"]:
        runs = find_runs(model, dataset, c, seed=0, n_samples=64)
        maps64[c] = load_item_answers(runs[0])
        # Deduped load already; pi_mode from is_correct on majority of 64
        pm, n = pi_mode_from_is_correct(maps64[c])
        print(f"  SC {c} N=64: {100*pm:.1f}%  n={n}")

    use = {k: maps64[k] for k in ["c0", "c1", "c2", "c3"]}
    cdv64, n = cdv_pi_mode(use, n_per_cfg=16)
    sc0 = pi_mode_from_is_correct(maps64["c0"])[0]
    print(f"  CDV(c0-c3) 16*4=64: {100*cdv64:.1f}%  n={n}")
    print(f"  Delta CDV - SC(c0): {100*(cdv64 - sc0):+.1f} pp")
    best = max(pi_mode_from_is_correct(maps64[c])[0] for c in maps64)
    print(f"  Delta CDV - best single cfg: {100*(cdv64 - best):+.1f} pp  (B12 check)")


if __name__ == "__main__":
    main()
