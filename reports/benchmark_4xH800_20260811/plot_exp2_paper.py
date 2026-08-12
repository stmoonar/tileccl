#!/usr/bin/env python3
"""Paper figure: exp2 AG+GEMM end-to-end time vs copy message size.

Single curve (pull / 3 streams, M=N=K=8192), x-axis is the per-copy chunk
size in MiB (= 32 MiB shard / S) instead of the split count S. Exports a
vector PDF sized for a single column.

Reads comm_comp/results_20260811_065005/exp2_pull.csv.
Writes figs/fig_exp2_msgsize.pdf (+ .png preview). Usage: python plot_exp2_paper.py
"""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "comm_comp", "results_20260811_065005", "exp2_pull.csv")
OUT = os.path.join(HERE, "figs")

TEAL, GREY = "#0E6B67", "#8A938F"

plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#DDDDDD",
    "grid.linewidth": 0.5,
    "figure.facecolor": "white",
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,  # embed TrueType, no Type-3 (camera-ready requirement)
    "ps.fonttype": 42,
})


def main():
    with open(CSV, newline="") as f:
        raw = [r for r in csv.DictReader(f) if r["m"].isdigit()]
    # keep the M=8192 shape; the CSV may repeat rows across appended runs
    by_s = {}
    for r in raw:
        if int(r["m"]) == 8192 and r["mode"] == "pull":
            by_s[int(r["S"])] = r
    ss = sorted(by_s)                                 # S = 1 .. 32
    mib = [float(by_s[s]["chunk_mib"]) for s in ss]   # 32 .. 1 MiB
    tovl = [float(by_s[s]["t_ovl_us"]) / 1e3 for s in ss]  # ms
    tfull = float(by_s[ss[0]]["t_full_us"]) / 1e3

    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.plot(mib, tovl, "o-", color=TEAL, ms=3.5, lw=1.6, zorder=3,
            clip_on=False)

    # perfect-overlap floor: the unsegmented GEMM alone
    ax.axhline(tfull, color=GREY, lw=1, ls="--", zorder=2)
    ax.text(1.0, tfull * 0.965, f"unsegmented GEMM alone ({tfull:.2f} ms)",
            fontsize=7, color=GREY, va="top")

    # direct labels on the two ends and the optimum
    best = min(ss, key=lambda s: float(by_s[s]["t_ovl_us"]))
    bx, by = float(by_s[best]["chunk_mib"]), float(by_s[best]["t_ovl_us"]) / 1e3
    ax.annotate(f"best: {bx:.0f} MiB ({by:.2f} ms)", (bx, by),
                textcoords="offset points", xytext=(-2, -13), ha="center",
                fontsize=7, color=TEAL)
    ax.annotate(f"{tovl[-1]:.1f} ms", (mib[-1], tovl[-1]),
                textcoords="offset points", xytext=(8, -2), fontsize=7,
                color=TEAL)
    ax.set_ylim(1.35, 12)

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(mib))
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:g}")
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("copy message size (MiB)")

    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator([1.5, 2, 3, 5, 10]))
    ax.yaxis.set_major_formatter("{x:g}")
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylabel("AG+GEMM end-to-end (ms)")

    # same axis, second reading: the split count S per 32 MiB shard
    top = ax.secondary_xaxis("top")
    top.set_xscale("log", base=2)
    top.xaxis.set_major_locator(FixedLocator(mib))
    top.xaxis.set_major_formatter(
        lambda x, _: f"S={int(round(32 / x))}" if x > 0 else "")
    top.xaxis.set_minor_formatter(NullFormatter())
    top.tick_params(labelsize=7, colors=GREY, length=0)
    top.spines["top"].set_visible(False)

    # shape annotation (requested: keep N, K visible)
    ax.text(0.985, 0.955,
            "GEMM M=N=K=8192, fp16, TP world=4\n"
            "all-gathered shard 2048$\\times$8192 = 32 MiB/rank",
            transform=ax.transAxes, ha="right", va="top", fontsize=7,
            color="#444444")

    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig_exp2_msgsize.{ext}"),
                    dpi=300 if ext == "png" else None)
        print(f"wrote figs/fig_exp2_msgsize.{ext}")
    plt.close(fig)


if __name__ == "__main__":
    main()
