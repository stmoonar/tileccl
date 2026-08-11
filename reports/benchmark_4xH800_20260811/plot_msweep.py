#!/usr/bin/env python3
"""Generate fig7 (the fixed-K M-sweep) for ANALYSIS_MSWEEP_4xH800.md.

Reads:
  comm_comp/results_20260811_065005/exp3_epilogue_msweep.csv   (v1, M sweep)
  comm_comp/results_20260811_065005/exp3_fused_msweep.csv      (v2, M sweep)
  comm_comp/results_20260811_065005/exp3_fused.csv             (v2, m=8192 point)

Writes figs/fig7_msweep.png. Usage:  python plot_msweep.py
Style matches ../benchmark_4xH800_20260810/plot_report.py.
"""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "comm_comp", "results_20260811_065005")
OUT = os.path.join(HERE, "figs")

TEAL, COPPER, SLATE, RED, GREY = "#0E6B67", "#B4560F", "#5B5E8D", "#A83232", "#8A938F"

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#DDDDDD",
    "grid.linewidth": 0.6,
    "figure.facecolor": "white",
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})


def rows(path):
    with open(path, newline="") as f:
        out = list(csv.DictReader(f))
    # multi-run CSVs re-append their header line; drop those pseudo-rows
    return [r for r in out if str(r.get("m", "0")).isdigit()]


def fig_msweep():
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(10.5, 3.9))

    # ---- left: v1 epilogue remote-write tax vs M (tma epilogue) ----
    data = rows(os.path.join(RES, "exp3_epilogue_msweep.csv"))
    ms = sorted({int(r["m"]) for r in data})

    def sd(mode):
        return [float(r["slowdown"]) for m in ms for r in data
                if int(r["m"]) == m and r["epi"] == "tma" and r["mode"] == mode]

    series = [("tma remote", sd("remote"), TEAL, "-"),
              ("tma scatter", sd("scatter"), TEAL, "--"),
              ("tma scatter-local (segmentation only)", sd("scatter-local"),
               SLATE, "-")]
    for label, vals, color, ls in series:
        axl.plot(ms, vals, "o", ls=ls, color=color, label=label, ms=4, lw=1.8,
                 zorder=3)
    axl.text(70, 2.75, "4x ~80 µs kernel floor\n(segmentation, not remoteness)",
             fontsize=7.5, color=SLATE)
    axl.annotate("1 wave (128 CTAs):\nlast-wave drain exposed", (512, 1.447),
                 textcoords="offset points", xytext=(-12, -30), fontsize=7.5,
                 color=TEAL)
    axl.annotate("pinned by K: 1.27...1.23", (8192, 1.24),
                 textcoords="offset points", xytext=(0, 10), ha="center",
                 fontsize=7.5, color=TEAL)
    axl.axhline(1.0, color=GREY, lw=1, ls="--", zorder=2)
    axl.set_yscale("log")
    axl.set_ylim(0.92, 4.6)
    axl.yaxis.set_major_locator(FixedLocator([1, 1.5, 2, 3, 4]))
    axl.yaxis.set_major_formatter("{x:.1f}x")
    axl.yaxis.set_minor_formatter(NullFormatter())
    axl.set_ylabel("slowdown vs same-epilogue local (log)")
    axl.set_title("exp3 v1: remote-write tax vs M (Mx8192x8192, K fixed)")

    # ---- right: v2 fused struct/comm/total vs M ----
    fdata = rows(os.path.join(RES, "exp3_fused_msweep.csv")) + \
        [r for r in rows(os.path.join(RES, "exp3_fused.csv"))
         if int(r["k"]) == 8192]
    fms = sorted({int(r["m"]) for r in fdata})

    def col(key, agg):
        out = []
        for m in fms:
            vs = [float(r[key]) for r in fdata if int(r["m"]) == m]
            out.append(agg(vs))
        return out

    mean = lambda vs: sum(vs) / len(vs)
    for label, vals, color, ls in [
            ("comm = fused/ctrl (mean)", col("comm_sd", mean), TEAL, "-"),
            ("total = fused/base (worst rank)", col("total_sd", max), COPPER,
             "-"),
            ("struct = ctrl/base (mean)", col("struct_sd", mean), SLATE, "-")]:
        axr.plot(fms, vals, "o", ls=ls, color=color, label=label, ms=4, lw=1.8,
                 zorder=3)
    axr.annotate("latency-bound:\npull only 26 GB/s", (512, 2.19),
                 textcoords="offset points", xytext=(10, -18), fontsize=7.5,
                 color=TEAL)
    axr.text(3300, 1.42, "comm plateau 1.09-1.15\n(production-limited, ~54 GB/s)",
             fontsize=7.5, color=TEAL)
    axr.annotate("uptick + fat tail\n(8192 tiles)", (32768, 1.17),
                 textcoords="offset points", xytext=(-10, 14), ha="right",
                 fontsize=7.5, color=RED)
    axr.text(2600, 0.905, "struct: cost at 1-4 waves, gain at >=4096",
             fontsize=7.5, color=SLATE)
    axr.axhline(1.0, color=GREY, lw=1, ls="--", zorder=2)
    axr.set_yscale("log")
    axr.set_ylim(0.86, 2.55)
    axr.yaxis.set_major_locator(FixedLocator([1, 1.5, 2]))
    axr.yaxis.set_major_formatter("{x:.1f}x")
    axr.yaxis.set_minor_formatter(NullFormatter())
    axr.set_title("exp3 v2: fused GEMM+RS vs M (K=8192, 4-rank)")

    for ax, ticks in ((axl, ms), (axr, fms)):
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(
            lambda x, _: f"{int(x/1024)}K" if x >= 1024 else f"{int(x)}")
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("M")
        ax.legend(frameon=False, fontsize=8)

    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, "fig7_msweep.png"))
    plt.close(fig)
    print("wrote figs/fig7_msweep.png")


if __name__ == "__main__":
    fig_msweep()
