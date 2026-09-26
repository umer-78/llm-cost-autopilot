"""python -m autopilot bench   measure every strategy, print the tables, write results/bench.json
python -m autopilot gate    rerun and fail if the router lost quality or saving against the baseline"""
import argparse
import json
import sys

from . import bench


def gate(now, base):
    problems = []
    for name, g in now["registries"].items():
        f, r, was = g["frontier"], g["router"], base["registries"][name]["router"]
        if r["accuracy"] < g["single"][f]["accuracy"] - 0.01:
            problems.append(f"{name}: router accuracy {r['accuracy']:.1%} is more than a point under {f}'s")
        if r["saving"] < was["saving"] - 0.05:
            problems.append(f"{name}: saving {r['saving']:.1%}, more than 5 points under the baseline {was['saving']:.1%}")
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser(prog="autopilot")
    ap.add_argument("command", choices=["bench", "gate"])
    args = ap.parse_args(argv)
    result = bench.run()
    print(bench.report(result))
    if args.command == "bench":
        bench.save(result)
        return 0
    problems = gate(result, json.loads((bench.RESULTS / "baseline.json").read_text()))
    for p in problems:
        print(f"GATE: {p}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
