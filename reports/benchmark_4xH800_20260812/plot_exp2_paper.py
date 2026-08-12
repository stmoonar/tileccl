#!/usr/bin/env python3
"""Paper figure: exp2 AG+GEMM effective throughput vs copy message size.

Standalone version of the exp2 panel from fig1_interference_granularity.png:
only the M=8192 e2e curve, x-axis converted from the split count S to the
per-copy chunk size in MiB (= 32 MiB shard / S). Held-clock (1305 MHz) data.

Reads comm_comp/results_20260812_083024/exp2_pull.csv.
Writes figs/fig_exp2_msgsize.pdf (+ .png preview). Usage: python plot_exp2_paper.py
"""
import csv
import io
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "comm_comp", "results_20260812_083024", "exp2_pull.csv")
OUT = os.path.join(HERE, "figs")

ROOFLINE = 651  # measured GEMM roofline at the held 1305 MHz lock

TEAL, GREY = "#0E6B67", "#8A938F"

plt.rcParams.update({
    "font.size": 8,
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
    with io.open(CSV, newline="", encoding="utf-8") as f:
        raw = [r for r in csv.DictReader(f) if r["m"].isdigit()]
    by_s = {int(r["S"]): r for r in raw
            if int(r["m"]) == 8192 and r["mode"] == "pull"}
    ss = sorted(by_s)                                  # S = 1 .. 32
    mib = [float(by_s[s]["chunk_mib"]) for s in ss]    # 32 .. 1 MiB
    tflops = [float(by_s[s]["e2e_tflops"]) for s in ss]

    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.plot(mib, tflops, "o-", color=TEAL, ms=3.5, lw=1.6, zorder=3,
            clip_on=False)

    ax.axhline(ROOFLINE, color=GREY, lw=1, ls="--", zorder=2)
    ax.text(1.0, ROOFLINE - 14, f"GEMM roofline {ROOFLINE} TFLOP/s",
            fontsize=7, color=GREY, va="top")

    best = max(ss, key=lambda s: float(by_s[s]["e2e_tflops"]))
    bx = float(by_s[best]["chunk_mib"])
    by = float(by_s[best]["e2e_tflops"])
    ax.annotate(f"best: {bx:.0f} MiB, {by:.0f} TFLOP/s\n"
                f"({100 * by / ROOFLINE:.0f}% of roofline)", (bx, by),
                textcoords="offset points", xytext=(0, -24), ha="center",
                fontsize=7, color=TEAL)
    ax.annotate(f"{tflops[-1]:.0f}", (mib[-1], tflops[-1]),
                textcoords="offset points", xytext=(0, -11), ha="center",
                fontsize=7, color=TEAL)

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(mib))
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:g}")
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("copy message size (MiB)")

    ax.set_ylim(0, 700)
    ax.set_ylabel("AG+GEMM effective TFLOP/s")

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
    ax.text(0.985, 0.05,
            "GEMM M=N=K=8192, fp16, TP world=4\n"
            "all-gathered shard 2048$\\times$8192 = 32 MiB/rank",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
            color="#444444")

    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig_exp2_msgsize.{ext}"),
                    dpi=300 if ext == "png" else None)
        print(f"wrote figs/fig_exp2_msgsize.{ext}")
    plt.close(fig)


if __name__ == "__main__":
    main()
