#!/usr/bin/env python3
"""Analyse a rerun_m32768.sh results directory and rule on the m=32768 upturn.

  python3 analyze_rerun.py rerun_20260812_101500

Reads step1_stats.csv / step2_decouple.csv / step3_order.csv plus the
iters_*.rank<N>.csv per-iteration dumps, and prints the four verdicts the
rerun was designed to produce:

  1. shift or tail        -- does comm_sd rise on p50s, or only on means?
  2. tiles or duration    -- which one does the absolute comm cost track?
  3. run-order bias       -- is struct_sd < 1 real or an artefact of timing
                             base first?
  4. tail shape           -- isolated outliers, or a drift over the run?

Missing files are skipped (QUICK=1 only produces step 1), so this is safe to
run against a partial directory.
"""
import glob
import os
import statistics
import sys

# ---------------------------------------------------------------------------
# CSV reading
# ---------------------------------------------------------------------------
# exp3_gemm_rs_fused appends AND re-emits its header on every invocation, so a
# file accumulating several shapes has header lines interleaved with data. Track
# the most recent header instead of assuming line 0 is the only one.


def read_rows(path):
    if not os.path.exists(path):
        return []
    rows, hdr = [], None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if parts[0] == "m":
                hdr = parts
                continue
            if hdr is None:
                continue
            row = dict(zip(hdr, parts))
            for key, val in list(row.items()):
                try:
                    row[key] = float(val)
                except ValueError:
                    pass
            rows.append(row)
    return rows


def need(row, col):
    """Fail loudly on a pre-patch CSV rather than silently reporting zeros."""
    if col not in row:
        sys.exit(
            "column '%s' missing -- this CSV predates the p50/dump patch;\n"
            "rebuild exp3_gemm_rs_fused and rerun." % col
        )
    return row[col]


def by_shape(rows):
    """Group rank rows into one record per (m,n,k,order)."""
    groups = {}
    for r in rows:
        key = (int(r["m"]), int(r["n"]), int(r["k"]), r.get("order", "?"))
        groups.setdefault(key, []).append(r)
    out = []
    for (m, n, k, order), rs in sorted(groups.items()):
        ctrl_mean = [need(r, "ctrl_us") for r in rs]
        ctrl_p50 = [need(r, "ctrl_p50") for r in rs]
        fused_mean = [need(r, "fused_us") for r in rs]
        fused_p50 = [need(r, "fused_p50") for r in rs]
        base_mean = [need(r, "base_us") for r in rs]
        ctrl_max = max(ctrl_mean)
        out.append(
            dict(
                m=m, n=n, k=k, order=order, ranks=len(rs),
                # tiles: the fused kernel is a persistent 132-CTA grid, so this
                # is the static scheduler's work count, not a hardware wave count
                tiles=(m // 128) * (n // 256),
                pull_mib=0.75 * m * n * 2 / 2**20,
                base=avg(base_mean),
                ctrl=avg(ctrl_mean), ctrl_p50=avg(ctrl_p50),
                fused=avg(fused_mean), fused_p50=avg(fused_p50),
                comm_mean=avg(fused_mean) / avg(ctrl_mean),
                comm_p50=avg(fused_p50) / avg(ctrl_p50),
                comm_skew=avg(fused_mean) / ctrl_max,
                struct=avg(ctrl_mean) / avg(base_mean),
                total=avg(fused_mean) / avg(base_mean),
                c_abs=avg(fused_mean) - avg(ctrl_mean),
                c_p50=avg(fused_p50) - avg(ctrl_p50),
            )
        )
    return out


def avg(xs):
    return sum(xs) / len(xs)


def find(shapes, m, k, order=None):
    for s in shapes:
        if s["m"] == m and s["k"] == k and (order is None or s["order"] == order):
            return s
    return None


# ---------------------------------------------------------------------------
# step 1: shift or tail
# ---------------------------------------------------------------------------


def step1(outdir):
    shapes = by_shape(read_rows(os.path.join(outdir, "step1_stats.csv")))
    if not shapes:
        return None
    print("=" * 78)
    print("STEP 1  shift or tail?   (K=8192, comm_sd on means vs on medians)")
    print("=" * 78)
    print(
        "%8s %9s %9s | %8s %8s %8s | %9s %9s"
        % ("M", "ctrl us", "fusd us", "comm avg", "comm p50", "comm skw",
           "C avg us", "C p50 us")
    )
    for s in sorted(shapes, key=lambda s: s["m"]):
        print(
            "%8d %9.1f %9.1f | %8.3f %8.3f %8.3f | %9.1f %9.1f"
            % (s["m"], s["ctrl"], s["fused"], s["comm_mean"], s["comm_p50"],
               s["comm_skew"], s["c_abs"], s["c_p50"])
        )

    lo, hi = find(shapes, 16384, 8192), find(shapes, 32768, 8192)
    if not (lo and hi):
        print("\n(need both M=16384 and M=32768 to rule)")
        return shapes, None
    d_mean = hi["comm_mean"] - lo["comm_mean"]
    d_p50 = hi["comm_p50"] - lo["comm_p50"]
    print(
        "\n  16384 -> 32768:  comm_sd(mean) %+.3f,  comm_sd(p50) %+.3f"
        % (d_mean, d_p50)
    )
    if d_p50 < 0.02 and d_mean > 0.04:
        verdict = "TAIL"
        print(
            "  VERDICT: TAIL. The median tax is flat -- the large-M plateau\n"
            "           holds, and the 20260811 rise was outlier iterations\n"
            "           leaking into the mean. Rewrite ANALYSIS sec 1 as\n"
            "           'plateau continues; fused grows a fat tail at M=32768'."
        )
    elif d_p50 >= 0.04:
        verdict = "SHIFT"
        print(
            "  VERDICT: SHIFT. The whole distribution moved, so there is a real\n"
            "           communication cost that grows with size, with a tail on\n"
            "           top. The ANALYSIS sec 1 warning becomes a finding -- go\n"
            "           to step 2 to locate it."
        )
    else:
        verdict = "MIXED"
        print("  VERDICT: MIXED (p50 rise %+.3f) -- both effects present." % d_p50)
    return shapes, verdict


# ---------------------------------------------------------------------------
# step 2: tiles or duration
# ---------------------------------------------------------------------------


def step2(outdir, s1, s1_verdict):
    shapes = by_shape(read_rows(os.path.join(outdir, "step2_decouple.csv")))
    if not shapes:
        return
    if s1:
        shapes = shapes + [s for s in s1 if s["k"] == 8192]
    print()
    print("=" * 78)
    print("STEP 2  does the cost track TILES or DURATION?")
    print("=" * 78)
    print("        comm_sd is NOT comparable across these rows (K moves the")
    print("        denominator). Read C = fused-ctrl, in absolute us.")
    if s1_verdict == "TAIL":
        print()
        print("        NOTE: step 1 ruled TAIL, i.e. there is no median-level")
        print("        excess to localise. The rows below are still worth")
        print("        reading for the C-per-tile trend, but the verdict is")
        print("        moot -- chase the tail (step 4) instead.")
    print()
    print(
        "%7s %6s %6s %7s %8s | %9s %9s | %8s %8s"
        % ("M", "K", "tiles", "pullMi", "dur ms", "C avg us", "C p50 us",
           "us/ktile", "us/ms")
    )
    for s in sorted(shapes, key=lambda s: (-s["m"], s["k"])):
        dur = s["fused"] / 1000.0
        print(
            "%7d %6d %6d %7.0f %8.2f | %9.1f %9.1f | %8.1f %8.1f"
            % (s["m"], s["k"], s["tiles"], s["pull_mib"], dur, s["c_abs"],
               s["c_p50"], 1000.0 * s["c_p50"] / s["tiles"],
               s["c_p50"] / dur)
        )

    anom = find(shapes, 32768, 8192)
    same_tiles = find(shapes, 32768, 4096)   # tiles held, duration halved
    same_dur = find(shapes, 16384, 16384)    # duration held, tiles halved
    base = find(shapes, 16384, 8192)
    if not all([anom, same_tiles, same_dur, base]):
        print("\n(need all four shapes to rule)")
        return
    # Which control keeps the anomaly's excess cost?
    r_tiles = same_tiles["c_p50"] / anom["c_p50"]
    r_dur = same_dur["c_p50"] / anom["c_p50"]
    print(
        "\n  holding tiles+volume, halving duration: C keeps %.0f%% of the anomaly"
        % (100 * r_tiles)
    )
    print(
        "  holding duration, halving tiles+volume: C keeps %.0f%% of the anomaly"
        % (100 * r_dur)
    )
    if r_tiles > 0.7 and r_dur < 0.5:
        print(
            "  VERDICT: TILE/VOLUME BOUND. The cost follows the flag+fetch path,\n"
            "           not elapsed time -- a per-tile protocol cost that grows\n"
            "           superlinearly once the flag array gets long. Next: Nsight\n"
            "           on the fetch warps' system-scope polling."
        )
    elif r_dur > 0.7 and r_tiles < 0.5:
        print(
            "  VERDICT: DURATION BOUND. The cost follows elapsed time, not work\n"
            "           -- inter-rank skew accumulating within an iteration, or\n"
            "           clock/thermal drift. Next: check step 4's drift line and\n"
            "           per-rank spread; this is a harness property, not a flux\n"
            "           design property, and must be reported as such."
        )
    else:
        print(
            "  VERDICT: BOTH (tiles %.0f%%, duration %.0f%%). Neither control\n"
            "           removes the excess; the two contributions are comparable\n"
            "           and separating them needs profiling, not more shapes."
            % (100 * r_tiles, 100 * r_dur)
        )


# ---------------------------------------------------------------------------
# step 3: run-order bias
# ---------------------------------------------------------------------------


def step3(outdir, s1):
    rev = by_shape(read_rows(os.path.join(outdir, "step3_order.csv")))
    if not rev or not s1:
        return
    print()
    print("=" * 78)
    print("STEP 3  is struct_sd < 1 real, or does `base` just run first?")
    print("=" * 78)
    print(
        "%8s | %10s %10s | %10s %10s"
        % ("M", "struct fwd", "struct rev", "comm fwd", "comm rev")
    )
    verdicts = []
    for r in sorted(rev, key=lambda s: s["m"]):
        f = find(s1, r["m"], r["k"])
        if not f:
            continue
        print(
            "%8d | %10.3f %10.3f | %10.3f %10.3f"
            % (r["m"], f["struct"], r["struct"], f["comm_p50"], r["comm_p50"])
        )
        verdicts.append((f["struct"], r["struct"]))
    if not verdicts:
        return
    fwd = avg([v[0] for v in verdicts])
    rvs = avg([v[1] for v in verdicts])
    print("\n  mean struct_sd: forward %.3f, reversed %.3f (%+.3f)" % (fwd, rvs, rvs - fwd))
    if fwd < 0.99 and rvs - fwd > 0.02:
        print(
            "  VERDICT: RUN-ORDER BIAS. struct_sd < 1 is an artefact of timing\n"
            "           `base` first. Retract 'the restructuring is slightly\n"
            "           faster at large M' from BENCHMARK_REPORT sec 1.4 -- the\n"
            "           honest claim is 'free, within run-order noise'."
        )
    else:
        print(
            "  VERDICT: REAL. struct_sd survives order reversal, so the\n"
            "           restructured kernel genuinely is not slower (the smem\n"
            "           carveout / stage-count explanation stands)."
        )


# ---------------------------------------------------------------------------
# step 4: what the tail actually looks like
# ---------------------------------------------------------------------------


def step4(outdir):
    dumps = sorted(glob.glob(os.path.join(outdir, "iters_*.rank*.csv")))
    if not dumps:
        return
    print()
    print("=" * 78)
    print("STEP 4  tail shape from the raw per-iteration series (fused only)")
    print("=" * 78)
    print(
        "%-20s %4s %7s %8s %8s %6s %7s %8s"
        % ("tag", "rank", "p50", "mean", "max", ">5%", "1st 3rd", "last 3rd")
    )
    drifts = []
    for path in dumps:
        name = os.path.basename(path)
        tag = name[len("iters_"):].split(".rank")[0]
        rank = name.split(".rank")[1].split(".")[0]
        series = [
            float(p[7])
            for p in (l.strip().split(",") for l in open(path))
            if len(p) == 8 and p[5] == "fused"
        ]
        if len(series) < 9:
            continue
        p50 = statistics.median(series)
        third = len(series) // 3
        first_m = statistics.median(series[:third])
        last_m = statistics.median(series[-third:])
        slow = sum(1 for x in series if x > 1.05 * p50)
        drifts.append((last_m - first_m) / first_m)
        print(
            "%-20s %4s %7.1f %8.1f %8.1f %5d%% %7.1f %8.1f"
            % (tag, rank, p50, avg(series), max(series),
               round(100 * slow / len(series)), first_m, last_m)
        )
    if not drifts:
        return
    worst = max(drifts, key=abs)
    print(
        "\n  worst first-third -> last-third drift: %+.1f%%" % (100 * worst)
    )
    if abs(worst) > 0.03:
        print(
            "  VERDICT: DRIFT. The kernel gets monotonically slower over the run\n"
            "           -- thermal/power, or state accumulating across iterations\n"
            "           (flags not fully cleared). This is a harness bug to fix\n"
            "           before any m=32768 number is quotable."
        )
    else:
        print(
            "  VERDICT: NO DRIFT. Slow iterations are isolated events, not a\n"
            "           downward slope -- consistent with a scheduling/skew\n"
            "           outlier rather than heating."
        )


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    outdir = sys.argv[1]
    got = step1(outdir)
    s1, s1_verdict = got if got else (None, None)
    step2(outdir, s1, s1_verdict)
    step3(outdir, s1)
    step4(outdir)
    print()


if __name__ == "__main__":
    main()
