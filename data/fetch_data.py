"""Download the six public benchmarks into data/<name>.jsonl, one {"id", "user_text", "source"} per line.

  benign (refusal rate):  orbench        bench-llm/or-bench, or-bench-hard-1k        (Cui et al., 2024; 1,319)
                          oktest         InvokerStark/OverKill, data/OKTest.csv       (Shi et al., 2024; 300)
                          xstest_safe    paul-rottger/xstest                          (Rottger et al., 2024; 250)
  harmful (ASR):          xstest_unsafe  paul-rottger/xstest                          (200)
                          advbench       llm-attacks/llm-attacks, harmful_behaviors   (Zou et al., 2023; 520)
                          jailbench      JailbreakBench/JBB-Behaviors, harmful        (Chao et al., 2024; 100)

  python data/fetch_data.py [--only xstest oktest advbench orbench jailbench]
"""
import argparse
import csv
import io
import json
import urllib.request
from pathlib import Path


def write_jsonl(path, prompts, source):
    prompts = [p.strip() for p in prompts if p and p.strip()]
    with open(path, "w") as f:
        for i, p in enumerate(prompts):
            f.write(json.dumps({"id": i, "user_text": p, "source": source}, ensure_ascii=False) + "\n")
    print(f"  wrote {len(prompts):4d} -> {path}")


def get(url, timeout=60):
    return urllib.request.urlopen(url, timeout=timeout).read().decode("utf-8")


def fetch_xstest(out):
    """Safe (250) + unsafe (200) splits from the public prompts CSV. Prefer the explicit 'label'
    column (safe/unsafe); fall back to 'type' starting with 'contrast' = unsafe."""
    urls = ["https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv",
            "https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_v2_prompts.csv"]
    rows = None
    for url in urls:
        try:
            rows = list(csv.DictReader(io.StringIO(get(url)))); break
        except Exception as e:
            print(f"[xstest] {url} failed: {e}")
    if not rows:
        raise RuntimeError("no XSTest CSV reachable")
    cols = list(rows[0].keys())
    pcol = "prompt" if "prompt" in cols else cols[1]
    print(f"[xstest] columns: {cols}")
    if "label" in cols:
        is_unsafe = lambda r: (r.get("label", "") or "").lower().startswith("unsafe")
    else:
        tcol = next((c for c in ("type", "category") if c in cols), None)
        is_unsafe = lambda r: (r.get(tcol, "") or "").lower().startswith("contrast")
    write_jsonl(out / "xstest_safe.jsonl", [r[pcol] for r in rows if not is_unsafe(r)], "xstest_safe")
    write_jsonl(out / "xstest_unsafe.jsonl", [r[pcol] for r in rows if is_unsafe(r)], "xstest_unsafe")


def fetch_oktest(out):
    url = "https://raw.githubusercontent.com/InvokerStark/OverKill/main/data/OKTest.csv"
    rows = list(csv.DictReader(io.StringIO(get(url))))
    pcol = "prompt" if "prompt" in rows[0] else list(rows[0].keys())[0]
    print(f"[oktest] columns: {list(rows[0].keys())}")
    write_jsonl(out / "oktest.jsonl", [r[pcol] for r in rows], "oktest")


def fetch_advbench(out):
    url = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
    rows = list(csv.DictReader(io.StringIO(get(url))))
    pcol = "goal" if "goal" in rows[0] else list(rows[0].keys())[0]
    print(f"[advbench] columns: {list(rows[0].keys())}")
    write_jsonl(out / "advbench.jsonl", [r[pcol] for r in rows], "advbench")


def fetch_orbench(out):
    from datasets import load_dataset
    for cfg in ("or-bench-hard-1k", "default"):
        try:
            ds = load_dataset("bench-llm/or-bench", cfg)
            split = ds[list(ds.keys())[0]]
            pcol = next(c for c in ("prompt", "question", "instruction") if c in split.column_names)
            print(f"[orbench:{cfg}] columns: {split.column_names}")
            write_jsonl(out / "orbench.jsonl", [r[pcol] for r in split], "orbench")
            return
        except Exception as e:
            print(f"[orbench:{cfg}] failed: {e}")


def fetch_jailbench(out):
    from datasets import load_dataset
    try:
        ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")
        split = ds["harmful"] if "harmful" in ds else ds[list(ds.keys())[0]]
        pcol = next(c for c in ("Goal", "goal", "prompt") if c in split.column_names)
        print(f"[jailbench] columns: {split.column_names}")
        write_jsonl(out / "jailbench.jsonl", [r[pcol] for r in split], "jailbench")
    except Exception as e:
        print(f"[jailbench] failed: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--only", nargs="+", default=None,
                    help="subset of: xstest oktest advbench orbench jailbench (default all)")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    jobs = {"xstest": fetch_xstest, "oktest": fetch_oktest, "advbench": fetch_advbench,
            "orbench": fetch_orbench, "jailbench": fetch_jailbench}
    for name, fn in jobs.items():
        if args.only and name not in args.only:
            continue
        try:
            fn(out)
        except Exception as e:
            print(f"[{name}] FAILED: {e}")


if __name__ == "__main__":
    main()
