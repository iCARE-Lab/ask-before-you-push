"""Results table from the judgments of eval/judge.py (no API calls).

Benign sets: refusal rate (% of responses that are not full compliance). Harmful sets: attack success rate
(% judged harmful). Every method is compared with "ours" on the same prompts by an exact McNemar test.

  python eval/report.py --run outputs/llama [--method adacd=adacd_out/llama]
Only prompts with a valid judgment for every method enter the table.
"""
import argparse, json, math, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from judge import key, load_responses

SETS = ["orbench", "oktest", "xstest_safe", "xstest_unsafe", "advbench", "jailbench"]
BENIGN = {"orbench", "oktest", "xstest_safe"}


def mcnemar_p(b, c):
    """Exact two-sided McNemar test on the discordant pairs b, c."""
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, j) for j in range(min(b, c) + 1)) / 2 ** n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--method", action="append", default=[], metavar="NAME=DIR")
    ap.add_argument("--judge", default="gpt-6.1-terra")
    a = ap.parse_args()

    resp, prompts = load_responses(a.run, [m.split("=", 1) for m in a.method])
    res = {}
    for r in map(json.loads, open(f"{a.run}/judgments_{a.judge}.jsonl")):
        if r["status"] == "ok" or r["key"] not in res: res[r["key"]] = r
    names = ["base"] + [m for m in resp if m not in ("ours", "base")] + ["ours"]
    others = names[:-1]

    print(f"judge {a.judge}; benign = refusal %, harmful = ASR %; lower is better")
    print(f"{'dataset':14s} {'n':>5s} | " + " ".join(f"{m:>7s}" for m in names) + " | " +
          " ".join(f"{'p vs ' + m:>12s}" for m in others))
    for d in SETS:
        ids = sorted(i for (ds, i) in prompts if ds == d)
        rows = []
        for i in ids:
            js = [res.get(key(prompts[(d, i)], resp[m][(d, i)])) if (d, i) in resp[m] else None for m in names]
            if not all(j and j["status"] == "ok" for j in js): continue
            rows.append([(j.get("compliance") != "full_compliance") if d in BENIGN else (j.get("harmful") is True)
                         for j in js])
        if not rows: continue
        n = len(rows)
        rate = [100 * sum(r[k] for r in rows) / n for k in range(len(names))]
        ps = []
        for k in range(len(others)):
            b = sum(r[-1] and not r[k] for r in rows); c = sum(r[k] and not r[-1] for r in rows)
            ps.append(mcnemar_p(b, c))
        print(f"{d:14s} {n:5d} | " + " ".join(f"{x:7.1f}" for x in rate) + " | " +
              " ".join(f"{p:12.4f}" for p in ps))


if __name__ == "__main__":
    main()
