#!/usr/bin/env python3
"""Plot the pk_bw_sweep results in the style of the ParallelKittens paper
(arXiv:2511.13940, Figures 2 and 3), plus the reg thread-count sweep.

Data below was measured on 2x RTX PRO 5000 Blackwell (sm_120, 110 SMs)
over PCIe Gen5 x16, payload 1 GiB per measurement, push direction,
destination-side verification passing. Re-run pk_bw_sweep and paste new
numbers here to refresh.

Usage:  python plot_pk_bw.py [--out figs] [--dpi 200]
"""

import argparse
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

# ---------------------------------------------------------------------------
# measured data
# ---------------------------------------------------------------------------

# Fig.2 analog: bandwidth (GB/s) vs message size, 1 GiB total, 110 blocks.
SIZES = [128, 512, 2**11, 2**13, 2**15, 2**16, 2**17, 2**19,
         2**21, 2**23, 2**25, 2**27, 2**29, 2**30]
SIZE_LABELS = ["128B", "512B", "2K", "8K", "32K", "64K", "128K", "512K",
               "2M", "8M", "32M", "128M", "512M", "1G"]
CE_BW = [0.08, 0.30, 1.23, 4.87, 19.48, 30.73, 38.86, 50.68,
         55.06, 56.21, 56.49, 56.56, 56.56, 56.56]
TMA_BW = [44.60, 50.83, 51.19, 51.19, 51.18, 51.17]  # measured up to 64 KiB
REG_BW = [51.18, 51.19, 51.19, 51.18, 51.18, 51.18, 51.18, 51.19,
          51.19, 51.19, 51.19, 51.19, 51.19, 51.19]

# Fig.3 analog: bandwidth vs SM count (msg: tma=25216 B, reg=16 KiB @1024 thr).
SMS = [1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 110]
TMA_SMS = [43.92] + [51.19] * 16
REG_SMS = [50.92, 51.14, 51.15, 51.17, 51.16, 51.18, 51.18, 51.19, 51.18,
           51.19, 51.19, 51.19, 51.19, 51.19, 51.19, 51.19, 51.18]
CE_REF = 56.56  # single 1 GiB cudaMemcpyPeerAsync, zero SMs

# Thread sweep: reg bandwidth vs threads/CTA, msg 16 KiB, at 1/2/4 SMs.
THREADS = [32, 64, 128, 256, 512, 1024]
REG_THR = {  # SMs -> bandwidth per thread count
    1: [5.06, 10.07, 19.86, 38.12, 51.13, 50.89],
    2: [10.09, 19.97, 38.70, 51.17, 51.09, 51.14],
    4: [19.90, 39.43, 51.17, 51.18, 51.17, 51.17],
}
TMA_1SM = 43.92  # TMA, 1 SM, a single issuing thread

PCIE_PEAK = 63.0        # PCIe Gen5 x16 effective payload limit
DEVICE_CEIL = 51.19     # shared TMA/reg ceiling observed

# Fixed series colors (identity follows the mechanism everywhere).
C_CE, C_TMA, C_REG = "#2a78d6", "#eb6834", "#1baf7a"
REG_RAMP = {1: "#4dbd90", 2: "#189567", 4: "#0e6b49"}  # ordered: SMs
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def style_axis(ax, xlabel):
    ax.set_ylim(0, 66)
    ax.set_yticks(range(0, 61, 10))
    ax.set_ylabel("Bandwidth (GB/s)", color=INK)
    ax.set_xlabel(xlabel, color=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    for lab in ax.get_yticklabels() + ax.get_xticklabels():
        lab.set_color(INK)


def log2_axis(ax, xs, labels):
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(xs))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    if labels is not None:
        ax.set_xticklabels(labels)
    ax.set_xlim(xs[0] / 1.35, xs[-1] * 1.35)


def ref_line_data(ax, y, text, color, above=True):
    ax.axhline(y, color=color, linewidth=1.2, linestyle=(0, (5, 4)), alpha=0.9)
    ax.annotate(text, xy=(0.99, y), xycoords=("axes fraction", "data"),
                xytext=(0, 3 if above else -4), textcoords="offset points",
                ha="right", va="bottom" if above else "top",
                fontsize=8, color=color)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def fig_msgsize(path, dpi):
    fig, ax = plt.subplots(figsize=(7.0, 4.1), layout="constrained")
    log2_axis(ax, SIZES, SIZE_LABELS)

    ref_line_data(ax, PCIE_PEAK, "PCIe Gen5 x16 effective ~63 GB/s", MUTED)

    ax.plot(SIZES, CE_BW, color=C_CE, linewidth=2, marker="o", markersize=4.5,
            markeredgecolor="white", markeredgewidth=1, label="Copy Engine", zorder=3)
    n = len(TMA_BW)
    ax.plot(SIZES[:n], TMA_BW, color=C_TMA, linewidth=2, marker="o",
            markersize=4.5, markeredgecolor="white", markeredgewidth=1,
            label="TMA Op", zorder=4)
    ax.plot(SIZES[n - 1:], [TMA_BW[-1]] * (len(SIZES) - n + 1), color=C_TMA,
            linewidth=1.8, linestyle=(0, (4, 4)), zorder=4,
            label="TMA (> smem, held constant)")
    ax.plot(SIZES, REG_BW, color=C_REG, linewidth=2, marker="o", markersize=4.5,
            markeredgecolor="white", markeredgewidth=1, label="Register Op", zorder=3)

    # direct labels at meaningful spots
    ax.annotate("ce 56.6", xy=(SIZES[-1], CE_BW[-1]), xytext=(2, 7),
                textcoords="offset points", ha="right", fontsize=9,
                fontweight="bold", color=C_CE)
    ax.annotate("reg 51.2", xy=(SIZES[-1], REG_BW[-1]), xytext=(2, -14),
                textcoords="offset points", ha="right", fontsize=9,
                fontweight="bold", color=C_REG)
    ax.annotate("tma 44.6 @128B", xy=(SIZES[0], TMA_BW[0]), xytext=(6, -14),
                textcoords="offset points", fontsize=9, fontweight="bold",
                color=C_TMA)
    ax.annotate("~2 us per cudaMemcpyPeerAsync call", xy=(SIZES[2], CE_BW[2]),
                xytext=(8, 2), textcoords="offset points", fontsize=8,
                color=C_CE, alpha=0.9)

    style_axis(ax, "Message size")
    for lab in ax.get_xticklabels():
        lab.set_rotation(35)
        lab.set_horizontalalignment("right")
    ax.set_title("P2P bandwidth vs. message granularity  (1 GiB payload, cf. PK Fig. 2)",
                 fontsize=10.5, color=INK, loc="left")
    ax.legend(loc="center right", fontsize=8, frameon=False)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def fig_smcount(path, dpi):
    fig, ax = plt.subplots(figsize=(7.0, 4.1), layout="constrained")
    ticks = [1, 2, 4, 8, 16, 32, 64, 110]
    log2_axis(ax, SMS, None)
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.set_xticklabels([str(t) for t in ticks])

    ref_line_data(ax, CE_REF, "copy engine (0 SMs, 1 call) 56.6", C_CE)

    ax.plot(SMS, TMA_SMS, color=C_TMA, linewidth=2, marker="o", markersize=4.5,
            markeredgecolor="white", markeredgewidth=1,
            label="TMA (1 issuing thread / SM)", zorder=4)
    ax.plot(SMS, REG_SMS, color=C_REG, linewidth=2, marker="o", markersize=4.5,
            markeredgecolor="white", markeredgewidth=1,
            label="Register Op (1024 threads / SM)", zorder=3)

    ax.annotate("43.9 @1 SM", xy=(1, TMA_SMS[0]), xytext=(6, -16),
                textcoords="offset points", fontsize=9, fontweight="bold",
                color=C_TMA)
    ax.annotate("saturated from 2 SMs", xy=(2, 51.19), xytext=(8, 6),
                textcoords="offset points", fontsize=8, color=INK)

    style_axis(ax, "Number of SMs driving the transfer")
    ax.set_title("P2P bandwidth vs. SM count  (msg: tma 25 KB / reg 16 KiB, cf. PK Fig. 3)",
                 fontsize=10.5, color=INK, loc="left")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def fig_threads(path, dpi):
    fig, ax = plt.subplots(figsize=(7.0, 4.1), layout="constrained")
    log2_axis(ax, THREADS, [str(t) for t in THREADS])

    ref_line_data(ax, TMA_1SM, "TMA: 1 SM, a single thread, 43.9", C_TMA,
                  above=False)

    for sms, bw in REG_THR.items():
        ax.plot(THREADS, bw, color=REG_RAMP[sms], linewidth=2, marker="o",
                markersize=4.5, markeredgecolor="white", markeredgewidth=1,
                label=f"reg, {sms} SM{'s' if sms > 1 else ''}", zorder=3)
        ax.annotate(f"{sms} SM", xy=(THREADS[-1], bw[-1]),
                    xytext=(6, {1: -10, 2: 0, 4: 9}[sms]),
                    textcoords="offset points", fontsize=9, fontweight="bold",
                    color=REG_RAMP[sms])

    ax.annotate("pre-saturation: ~0.157 GB/s x SMs x threads",
                xy=(64, 10.07), xytext=(10, -6), textcoords="offset points",
                fontsize=8, color=MUTED)

    style_axis(ax, "Threads per SM (reg kernel, msg 16 KiB)")
    ax.set_title("Register-op bandwidth vs. thread count  (in-flight requests are the limit)",
                 fontsize=10.5, color=INK, loc="left")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figs", help="output directory")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.titleweight": "bold",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })

    fig_msgsize(os.path.join(args.out, "pk_fig2_msgsize.png"), args.dpi)
    fig_smcount(os.path.join(args.out, "pk_fig3_smcount.png"), args.dpi)
    fig_threads(os.path.join(args.out, "pk_fig4_regthreads.png"), args.dpi)
    print(f"wrote 3 figures to {args.out}/")


if __name__ == "__main__":
    main()
