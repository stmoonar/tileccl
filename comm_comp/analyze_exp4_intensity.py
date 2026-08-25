#!/usr/bin/env python3

"""Report how the CE/TMA granularity result moves with compute density.

The H sweep runs at one compute intensity, where B/A ~ 1.  That ratio is what
makes granularity matter at all: a ready group leaves A/(4*cps) of compute
with nothing to overlap, and the tail is hidden only when

    B <= A - A/(4*cps)          (cps = 1  =>  B <= 0.75*A)

flux's AG uses SPLIT=1, so a ready group is a whole peer shard (cps = 1, the
worst point in the sweep), yet it pays nothing for it because a real GEMM puts
B/A near 0.36.  This script reads the --intensity cases and prints, per
density: the measured A and B, the resulting no-stall bound, and where CE
crosses below TMA -- i.e. how far the fine-granularity conclusion travels.
"""

import argparse
import csv
import collections
from pathlib import Path

PANELS_PER_SHARD = 16384  # rps * P at M=65536, K=8192, world=4


def worst(rows, variant, mode, field="t_us_mean"):
    out = collections.defaultdict(float)
    for r in rows:
        if r["variant"] == variant and r["mode"] == mode and int(r["rank"]) >= 0:
            h = int(r["panel_h"])
            out[h] = max(out[h], float(r[field]))
    return out


def analyse(path: Path) -> None:
    rows = list(csv.DictReader(open(path)))
    if not rows:
        raise SystemExit(f"{path}: empty")
    bad = [r for r in rows if r["verify"] != "ok" or int(r["err_count"]) != 0]
    if bad:
        raise SystemExit(f"{path}: {len(bad)} invalid rows")
    intensity = sorted({int(r["intensity"]) for r in rows})
    if len(intensity) != 1:
        raise SystemExit(f"{path}: mixed intensity {intensity}")

    a_ce = worst(rows, "ce-aggregate", "compute-only")
    a_tm = worst(rows, "tma", "compute-only")
    f_ce = worst(rows, "ce-aggregate", "fused")
    f_tm = worst(rows, "tma", "fused")
    b_ce = worst(rows, "ce-aggregate", "bystander")
    b_tm = worst(rows, "tma", "bystander")

    hs = sorted(f_ce)
    # B is not measured directly here; recover it from the tail model at the
    # coarsest H, where the tail term dominates and is exactly A/4.
    coarse = max(hs)
    cps_coarse = PANELS_PER_SHARD // coarse
    b_ce_est = f_ce[coarse] - a_ce[coarse] / (4 * cps_coarse)
    b_tm_est = f_tm[coarse] - a_tm[coarse] / (4 * cps_coarse)

    ref = a_ce[coarse]
    print(f"\n=== intensity {intensity[0]}  ({path.name}) ===")
    print(f"  A (CE compute-only, 132 SM) : {ref:8.0f} us")
    print(f"  A (TMA compute-only, 116 SM): {a_tm[coarse]:8.0f} us"
          f"   SM tax {a_tm[coarse]/ref:.3f}")
    print(f"  B (CE,  from tail model)    : {b_ce_est:8.0f} us   B/A {b_ce_est/ref:.3f}")
    print(f"  B (TMA, from tail model)    : {b_tm_est:8.0f} us   B/A {b_tm_est/a_tm[coarse]:.3f}")
    print(f"  no-stall bound at cps=1 (flux's granularity): B <= {0.75*ref:.0f} us"
          f"  -> CE {'hidden' if b_ce_est <= 0.75*ref else 'EXPOSED'}")

    print(f"\n  {'H':>6} {'chunk':>7} {'cps':>5} | {'CE':>7} {'TMA':>7} | "
          f"{'CE byst':>8} {'TMA byst':>8} | winner")
    cross = None
    for h in hs:
        cps = PANELS_PER_SHARD // h
        rc, rt = f_ce[h] / ref, f_tm[h] / ref
        bc = b_ce[h] / ref if h in b_ce else float("nan")
        bt = b_tm[h] / ref if h in b_tm else float("nan")
        win = "CE" if rc < rt else "TMA"
        if win == "CE" and cross is None:
            cross = h
        lbl = f"{h*16//1024}M" if h >= 64 else f"{h*16}K"
        print(f"  {h:>6} {lbl:>7} {cps:>5} | {rc:>7.3f} {rt:>7.3f} | "
              f"{bc:>8.3f} {bt:>8.3f} | {win}")
    print(f"  crossover: {'H=' + str(cross) if cross else 'none in this range'}"
          "   (bystander = comm running, gates pre-armed: interference only)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle", type=Path, help="extracted result directory")
    args = ap.parse_args()
    files = sorted(args.bundle.glob("exp4_intensity_*.csv"),
                   key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    if not files:
        raise SystemExit(f"no exp4_intensity_*.csv under {args.bundle}")
    for f in files:
        analyse(f)


if __name__ == "__main__":
    main()
