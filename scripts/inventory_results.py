"""Inventory results_bundle completeness against the Tier A/B checklist."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "results_bundle"


def load_rows():
    rows = []
    for m in sorted(ROOT.rglob("manifest.json")):
        st = json.loads(m.read_text(encoding="utf-8"))
        cfg = st.get("config", {})
        model = cfg.get("model", {}).get("name_or_path", "")
        data = cfg.get("data", {})
        strat = cfg.get("strategy", {})
        elic = (cfg.get("elicitation") or {}).get("id")
        gen = cfg.get("generation", {})
        results_path = m.parent / "results.jsonl"
        n_lines = 0
        if results_path.exists():
            with results_path.open(encoding="utf-8", errors="ignore") as f:
                n_lines = sum(1 for _ in f)
        rows.append(
            {
                "dir": m.parent.name,
                "status": st.get("status"),
                "n_done": st.get("n_completed"),
                "n_total": st.get("n_total"),
                "n_err": st.get("n_errors"),
                "gpu_h": round(float(st.get("gpu_hours") or 0), 3),
                "model": model.split("/")[-1] if model else "?",
                "dataset": data.get("name"),
                "subset": data.get("subset"),
                "strategy": strat.get("name"),
                "n_samples": (strat.get("params") or {}).get("n_samples"),
                "elicitation": elic,
                "seed": cfg.get("runtime", {}).get("seed"),
                "T": gen.get("temperature"),
                "max_tok": gen.get("max_new_tokens"),
                "load_4bit": bool(cfg.get("model", {}).get("load_in_4bit")),
                "graded": (m.parent / "graded.jsonl").exists(),
                "results_lines": n_lines,
            }
        )
    return rows


def ok(hits):
    if not hits:
        return "MISSING"
    r = hits[0]
    if r["status"] != "complete" or r["n_done"] != r["n_total"]:
        return f"PARTIAL {r['n_done']}/{r['n_total']}"
    return "OK"


def find(rows, **kw):
    out = []
    for r in rows:
        if any(r.get(k) != v for k, v in kw.items()):
            continue
        out.append(r)
    return out


def math_subset_ok(r):
    return r["subset"] in (None, "null", "")


def main():
    rows = load_rows()
    print(f"TOTAL RUNS: {len(rows)}")
    print(f"COMPLETE: {sum(1 for r in rows if r['status'] == 'complete')}")
    print(f"INCOMPLETE: {sum(1 for r in rows if r['status'] != 'complete')}")
    print(f"TOTAL GPU-H: {sum(r['gpu_h'] for r in rows):.2f}")
    print(f"UNGRADED: {sum(1 for r in rows if not r['graded'])}")
    print()

    for r in rows:
        if r["status"] != "complete" or r["n_done"] != r["n_total"]:
            print(
                "INCOMPLETE:",
                r["dir"],
                r["status"],
                f"{r['n_done']}/{r['n_total']}",
                "err",
                r["n_err"],
            )

    models_qwen = [
        "Qwen2.5-1.5B-Instruct",
        "Qwen2.5-3B-Instruct",
        "Qwen2.5-7B-Instruct",
    ]
    models_all = models_qwen + ["Llama-3.2-3B-Instruct"]

    print("\n=== PHASE 1 main grid (seed0, N=24 or unset, c0-c2) ===")
    missing = []
    for model in models_all:
        for ds in ["gsm8k", "math500"]:
            for c in ["c0", "c1", "c2"]:
                hits = []
                for r in rows:
                    if r["strategy"] != "configured_sampling":
                        continue
                    if r["model"] != model or r["dataset"] != ds:
                        continue
                    if r["elicitation"] != c or r["seed"] != 0:
                        continue
                    if r["load_4bit"]:
                        continue
                    if r["n_samples"] not in (24, 64, None):
                        continue
                    if ds == "math500" and not math_subset_ok(r):
                        continue
                    if ds == "gsm8k" and r["subset"] not in ("main", None, "null", ""):
                        continue
                    # Prefer N=24 over N=64 for phase1 listing
                    hits.append(r)
                # Prefer complete N=24
                pref = [h for h in hits if h["n_samples"] in (24, None) and h["status"] == "complete"]
                use = pref or hits
                mark = ok(use)
                n = use[0]["n_samples"] if use else "?"
                line = f"  {mark:16s} {model:28s} {ds:12s} {c} seed0 N={n}"
                print(line)
                if not mark.startswith("OK"):
                    missing.append(line)

    print("\n=== PHASE 2 seeds at c0 (seed 1,2) ===")
    for model in models_qwen:
        for ds in ["gsm8k", "math500"]:
            for seed in [1, 2]:
                hits = [
                    r
                    for r in rows
                    if r["strategy"] == "configured_sampling"
                    and r["model"] == model
                    and r["dataset"] == ds
                    and r["elicitation"] == "c0"
                    and r["seed"] == seed
                    and not r["load_4bit"]
                ]
                mark = ok(hits)
                line = f"  {mark:16s} {model:28s} {ds:12s} c0 seed{seed}"
                print(line)
                if not mark.startswith("OK"):
                    missing.append(line)

    print("\n=== 7B gsm8k c0 seed0 (critical) ===")
    hits = [
        r
        for r in rows
        if r["model"] == "Qwen2.5-7B-Instruct"
        and r["dataset"] == "gsm8k"
        and r["elicitation"] == "c0"
        and r["seed"] == 0
        and r["strategy"] == "configured_sampling"
    ]
    print(" ", ok(hits), hits[0]["dir"] if hits else "MISSING")
    if not hits:
        missing.append("7B gsm8k c0 seed0 configured_sampling")

    print("\n=== GSM-Symbolic E4 ===")
    for model in models_qwen:
        for sub in ["main", "p2"]:
            for c in ["c0", "c1", "c2"]:
                hits = [
                    r
                    for r in rows
                    if r["dataset"] == "gsm_symbolic"
                    and r["model"] == model
                    and r["subset"] == sub
                    and r["elicitation"] == c
                ]
                mark = ok(hits)
                line = f"  {mark:16s} {model:28s} symbolic/{sub} {c}"
                print(line)
                if not mark.startswith("OK"):
                    missing.append(line)

    print("\n=== Llama E3a ===")
    for ds in ["gsm8k", "math500"]:
        for c in ["c0", "c1", "c2"]:
            hits = [
                r
                for r in rows
                if r["model"] == "Llama-3.2-3B-Instruct"
                and r["dataset"] == ds
                and r["elicitation"] == c
                and r["seed"] == 0
            ]
            mark = ok(hits)
            line = f"  {mark:16s} Llama-3.2-3B {ds} {c}"
            print(line)
            if not mark.startswith("OK"):
                missing.append(line)

    print("\n=== O2 deep N=64 (3B MATH-500 c0-c3) ===")
    for c in ["c0", "c1", "c2", "c3"]:
        hits = [
            r
            for r in rows
            if r["model"] == "Qwen2.5-3B-Instruct"
            and r["dataset"] == "math500"
            and r["elicitation"] == c
            and r["n_samples"] == 64
        ]
        mark = ok(hits)
        extra = f" done={hits[0]['n_done']}/{hits[0]['n_total']}" if hits else ""
        line = f"  {mark:16s} 3B math500 {c} N=64{extra}"
        print(line)
        if not mark.startswith("OK"):
            missing.append(line)

    print("\n=== O2b c1 seeds on 3B MATH-500 ===")
    for seed in [1, 2]:
        hits = [
            r
            for r in rows
            if r["model"] == "Qwen2.5-3B-Instruct"
            and r["dataset"] == "math500"
            and r["elicitation"] == "c1"
            and r["seed"] == seed
        ]
        mark = ok(hits)
        line = f"  {mark:16s} 3B math500 c1 seed{seed} N={hits[0]['n_samples'] if hits else '?'}"
        print(line)
        if not mark.startswith("OK"):
            missing.append(line)

    print("\n=== PC 4-bit ===")
    for model in ["Qwen2.5-1.5B-Instruct", "Qwen2.5-3B-Instruct"]:
        for c in ["c0", "c1", "c2"]:
            hits = [
                r
                for r in rows
                if r["model"] == model
                and r["dataset"] == "math500"
                and r["elicitation"] == c
                and r["load_4bit"]
            ]
            mark = ok(hits)
            line = f"  {mark:16s} {model} math500 {c} 4bit"
            print(line)
            if not mark.startswith("OK"):
                missing.append(line)

    print("\n=== Temperature check (paper needs T=0.8) ===")
    bad_T = [
        r
        for r in rows
        if r["strategy"] == "configured_sampling" and r["T"] not in (0.8, None)
    ]
    print(f"  runs with T!=0.8: {len(bad_T)}")
    for r in bad_T[:10]:
        print(f"    T={r['T']} {r['dir']}")

    print("\n=== MISSING SUMMARY ===")
    print(f"count={len(missing)}")
    for line in missing:
        print(line)


if __name__ == "__main__":
    main()
