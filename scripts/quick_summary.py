"""Fast per-run accuracy / pi_mode summary without full paper bootstrap.

Reads graded.jsonl only; prints a table suitable for paper drafting.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "results_bundle"


def majority(answers):
    answers = [a for a in answers if a not in (None, "", "⊥")]
    if not answers:
        return None, 0.0
    c = Counter(answers)
    top, n = c.most_common(1)[0]
    return top, n / len(answers)


def summarise_run(run_dir: Path):
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    cfg = manifest["config"]
    model = cfg["model"]["name_or_path"].split("/")[-1]
    ds = cfg["data"]["name"]
    subset = cfg["data"].get("subset")
    elic = (cfg.get("elicitation") or {}).get("id")
    seed = cfg.get("runtime", {}).get("seed")
    n_samples = (cfg.get("strategy", {}).get("params") or {}).get("n_samples")
    load_4bit = bool(cfg["model"].get("load_in_4bit"))
    strategy = cfg["strategy"]["name"]

    graded = run_dir / "graded.jsonl"
    if not graded.exists():
        return None
    n = correct = 0
    modal_hits = 0
    extraction_fail = 0
    with graded.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            n += 1
            if rec.get("is_correct"):
                correct += 1
            samples = rec.get("sample_answers") or []
            if not samples and rec.get("extra"):
                samples = rec["extra"].get("sample_answers") or []
            # Fallback: single final answer
            if not samples and rec.get("final_answer") is not None:
                samples = [rec.get("final_answer")]
            mode, _ = majority(samples)
            gold = str(rec.get("gold_answer", ""))
            # crude: compare string equality of mode to graded correctness of vote
            if rec.get("is_correct") and mode is not None:
                modal_hits += 1
            elif mode is not None and rec.get("vote_correct") is True:
                modal_hits += 1
            # Prefer explicit fields if present
            if "modal_correct" in rec:
                modal_hits = modal_hits  # keep
            fails = rec.get("n_extraction_failures")
            if fails is None and rec.get("extra"):
                fails = rec["extra"].get("n_extraction_failures")
            if fails:
                extraction_fail += int(fails)
    # Better pi_mode: fraction of items where majority vote is correct
    # Re-read with vote field
    modal_hits = 0
    with graded.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("is_correct"):
                modal_hits += 1
    acc = correct / n if n else float("nan")
    return {
        "model": model,
        "dataset": ds,
        "subset": subset or "",
        "elic": elic or "",
        "seed": seed,
        "N": n_samples,
        "4bit": load_4bit,
        "strategy": strategy,
        "n": n,
        "acc": acc,
        "gpu_h": float(manifest.get("gpu_hours") or 0),
        "status": manifest.get("status"),
        "dir": run_dir.name,
    }


def main():
    rows = []
    for m in sorted(ROOT.rglob("manifest.json")):
        row = summarise_run(m.parent)
        if row:
            rows.append(row)

    # Main grid seed0 N24 configured_sampling fp16
    print("=== MAIN GRID (configured_sampling, seed0, fp16, N=24) accuracy = maj@24 ===")
    print(f"{'model':28s} {'dataset':12s} {'c0':>7s} {'c1':>7s} {'c2':>7s}  range_pp")
    models = sorted({r["model"] for r in rows if r["strategy"] == "configured_sampling"})
    datasets = ["gsm8k", "math500"]
    for model in models:
        for ds in datasets:
            accs = {}
            for c in ["c0", "c1", "c2"]:
                hits = [
                    r
                    for r in rows
                    if r["model"] == model
                    and r["dataset"] == ds
                    and r["elic"] == c
                    and r["seed"] == 0
                    and not r["4bit"]
                    and r["strategy"] == "configured_sampling"
                    and r["N"] in (24, None)
                ]
                # Prefer exact N=24; if multiple hashes take first
                if not hits and ds == "math500":
                    hits = [
                        r
                        for r in rows
                        if r["model"] == model
                        and r["dataset"] == ds
                        and r["elic"] == c
                        and r["seed"] == 0
                        and not r["4bit"]
                        and r["strategy"] == "configured_sampling"
                        and r["N"] != 64
                    ]
                accs[c] = hits[0]["acc"] if hits else None
            vals = [v for v in accs.values() if v is not None]
            rng = (max(vals) - min(vals)) * 100 if len(vals) >= 2 else float("nan")
            def fmt(x):
                return f"{100*x:6.1f}%" if x is not None else "   n/a"
            print(
                f"{model:28s} {ds:12s} {fmt(accs.get('c0'))} {fmt(accs.get('c1'))} {fmt(accs.get('c2'))}  "
                f"{rng:5.1f}" if vals else f"{model:28s} {ds:12s} missing"
            )

    print("\n=== GSM-Symbolic (seed0) ===")
    print(f"{'model':28s} {'subset':6s} {'c0':>7s} {'c1':>7s} {'c2':>7s}  range_pp")
    for model in ["Qwen2.5-1.5B-Instruct", "Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct"]:
        for sub in ["main", "p2"]:
            accs = {}
            for c in ["c0", "c1", "c2"]:
                hits = [
                    r
                    for r in rows
                    if r["model"] == model
                    and r["dataset"] == "gsm_symbolic"
                    and r["subset"] == sub
                    and r["elic"] == c
                ]
                accs[c] = hits[0]["acc"] if hits else None
            vals = [v for v in accs.values() if v is not None]
            rng = (max(vals) - min(vals)) * 100 if len(vals) >= 2 else float("nan")
            def fmt(x):
                return f"{100*x:6.1f}%" if x is not None else "   n/a"
            print(
                f"{model:28s} {sub:6s} {fmt(accs.get('c0'))} {fmt(accs.get('c1'))} {fmt(accs.get('c2'))}  {rng:5.1f}"
            )

    print("\n=== MISSING 7B gsm8k c0 seed0 ===")
    hits = [
        r
        for r in rows
        if r["model"] == "Qwen2.5-7B-Instruct"
        and r["dataset"] == "gsm8k"
        and r["elic"] == "c0"
        and r["seed"] == 0
        and r["strategy"] == "configured_sampling"
    ]
    print("present" if hits else "MISSING — must re-run before GSM8K 7B crossed analysis")

    print("\n=== O2 N=64 3B MATH-500 ===")
    for c in ["c0", "c1", "c2", "c3"]:
        hits = [
            r
            for r in rows
            if r["model"] == "Qwen2.5-3B-Instruct"
            and r["dataset"] == "math500"
            and r["elic"] == c
            and r["N"] == 64
        ]
        if hits:
            print(f"  {c}: {100*hits[0]['acc']:.1f}%  n={hits[0]['n']}")
        else:
            print(f"  {c}: MISSING")

    print("\n=== PC 4bit vs fp16 MATH-500 seed0 ===")
    for model in ["Qwen2.5-1.5B-Instruct", "Qwen2.5-3B-Instruct"]:
        for c in ["c0", "c1", "c2"]:
            fp = [
                r
                for r in rows
                if r["model"] == model
                and r["dataset"] == "math500"
                and r["elic"] == c
                and r["seed"] == 0
                and not r["4bit"]
                and r["N"] in (24, None)
            ]
            b4 = [
                r
                for r in rows
                if r["model"] == model
                and r["dataset"] == "math500"
                and r["elic"] == c
                and r["4bit"]
            ]
            a = 100 * fp[0]["acc"] if fp else float("nan")
            b = 100 * b4[0]["acc"] if b4 else float("nan")
            print(f"  {model:28s} {c}: fp16={a:5.1f}%  4bit={b:5.1f}%  delta={b-a:+5.1f}pp")

    out = ROOT.parent / "paper_artefacts" / "quick_summary.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
