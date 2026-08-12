#!/usr/bin/env python3
"""Plot the pk_bw_sweep results in the style of the ParallelKittens paper
(arXiv:2511.13940, Figures 2 and 3), plus the reg thread-count sweep.

Two measured machines, data embedded below:
  rtx5000  2x RTX PRO 5000 Blackwell (sm_120, 110 SMs), PCIe Gen5 x16
  h20      2x H20 (sm_90, 78 SMs), NVLink (450 GB/s per direction)

All runs: payload 1 GiB per measurement, push direction, destination-side
verification passing. Re-run pk_bw_sweep and paste new numbers to refresh.

Usage:  python plot_pk_bw.py [--out figs] [--dpi 200] [--machine rtx5000|h20|all]
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

# ---------------------------------------------------------------------------
# shared axes / style
# ---------------------------------------------------------------------------

SIZES = [128, 512, 2**11, 2**13, 2**15, 2**16, 2**17, 2**19,
         2**21, 2**23, 2**25, 2**27, 2**29, 2**30]
SIZE_LABELS = ["128B", "512B", "2K", "8K", "32K", "64K", "128K", "512K",
               "2M", "8M", "32M", "128M", "512M", "1G"]

# Fixed series colors (identity follows the mechanism everywhere).
C_CE, C_TMA, C_REG = "#2a78d6", "#eb6834", "#1baf7a"
REG_RAMP = {1: "#4dbd90", 2: "#189567", 4: "#0e6b49"}  # ordered: SMs (rtx fig)
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"

# ---------------------------------------------------------------------------
# measured data
# ---------------------------------------------------------------------------

RTX5000 = dict(
    tag="rtx5000",
    name="2x RTX PRO 5000 (sm_120), PCIe Gen5 x16",
    ymax=66, ystep=10,
    link_ref=(63.0, "PCIe Gen5 x16 effective ~63 GB/s"),
    # Fig.2: 1 GiB payload, blocks=110
    ce=[0.08, 0.30, 1.23, 4.87, 19.48, 30.73, 38.86, 50.68,
        55.06, 56.21, 56.49, 56.56, 56.56, 56.56],
    tma=[44.60, 50.83, 51.19, 51.19, 51.18, 51.17],   # measured up to 64 KiB
    reg=[51.18, 51.19, 51.19, 51.18, 51.18, 51.18, 51.18, 51.19,
         51.19, 51.19, 51.19, 51.19, 51.19, 51.19],
    # Fig.3: msg tma=25216 B, reg=16 KiB @ 1024 threads
    sms=[1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 110],
    sms_ticks=[1, 2, 4, 8, 16, 32, 64, 110],
    tma_sms=[43.92] + [51.19] * 16,
    reg_sms=[50.92, 51.14, 51.15, 51.17, 51.16, 51.18, 51.18, 51.19, 51.18,
             51.19, 51.19, 51.19, 51.19, 51.19, 51.19, 51.19, 51.18],
    ce_ref=56.56,
    msg_note="msg: tma 25 KB / reg 16 KiB",
)

H20 = dict(
    tag="h20",
    name="2x H20 (sm_90), NVLink",
    ymax=470, ystep=50,
    link_ref=(450.0, "NVLink 450 GB/s per direction"),
    # Fig.2: 1 GiB payload, blocks=78 (128 KiB point is single-buffer)
    ce=[0.06, 0.21, 0.82, 3.09, 12.04, 23.87, 43.89, 135.37,
        267.42, 348.20, 379.64, 393.43, 396.54, 397.09],
    tma=[46.11, 181.95, 352.95, 369.90, 372.95, 373.02, 371.57],  # to 128 KiB
    reg=[353.61, 368.75, 370.11, 369.58, 369.05, 370.03, 367.76, 361.14,
         362.65, 369.70, 370.66, 369.78, 368.90, 368.59],
    # Fig.3: msg tma=57984 B, reg=16 KiB @ 1024 threads
    sms=[1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 32, 40, 48, 64, 78],
    sms_ticks=[1, 2, 4, 8, 16, 32, 64, 78],
    tma_sms=[42.81, 85.61, 128.39, 170.93, 256.32, 341.68, 364.11, 365.67,
             366.32, 372.38, 373.12, 372.19, 372.53, 372.81, 371.35],
    reg_sms=[37.23, 74.54, 111.70, 148.75, 222.17, 297.22, 354.57, 351.55,
             354.89, 365.23, 365.50, 370.56, 370.04, 370.47, 369.89],
    ce_ref=396.97,
    msg_note="msg: tma 57 KB / reg 16 KiB",
)

# Thread sweep, rtx5000: reg bandwidth vs threads/CTA, msg 16 KiB, 1/2/4 SMs.
RTX_THREADS = [32, 64, 128, 256, 512, 1024]
RTX_REG_THR = {
    1: [5.06, 10.07, 19.86, 38.12, 51.13, 50.89],
    2: [10.09, 19.97, 38.70, 51.17, 51.09, 51.14],
    4: [19.90, 39.43, 51.17, 51.18, 51.17, 51.17],
}
RTX_TMA_1SM = 43.92          # TMA, 1 SM, a single issuing thread
RTX_PER_THREAD = 0.157       # GB/s per thread before saturation

# Thread sweep, h20: reg bandwidth vs threads/CTA, msg 16 KiB, 1..64 SMs.
H20_THREADS = [32, 64, 128, 256, 512, 1024]
H20_REG_THR = {
    1:  [1.25, 2.49, 4.97, 9.85, 19.38, 37.23],
    4:  [4.97, 9.90, 19.72, 39.16, 77.20, 148.75],
    8:  [9.86, 19.68, 39.21, 77.97, 154.16, 297.35],
    16: [19.72, 39.37, 78.50, 156.45, 309.41, 351.56],
    32: [39.38, 78.66, 157.27, 313.82, 328.50, 364.55],
    64: [78.35, 156.96, 313.96, 342.64, 358.58, 370.61],
}
H20_TMA_1SM = 42.81
H20_PER_THREAD = 0.039       # GB/s per thread before saturation
H20_REG_CEIL = 370.0

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def style_axis(ax, xlabel, ymax, ystep):
    ax.set_ylim(0, ymax)
    ax.set_yticks(range(0, int(ymax - ymax * 0.05), ystep))
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


def line(ax, xs, ys, color, label, zorder=3, dash=False):
    ax.plot(xs, ys, color=color, linewidth=2 if not dash else 1.8,
            linestyle=(0, (4, 4)) if dash else "-",
            marker=None if dash else "o", markersize=4.5,
            markeredgecolor="white", markeredgewidth=1,
            label=label, zorder=zorder)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def fig_msgsize(d, path, dpi):
    fig, ax = plt.subplots(figsize=(7.0, 4.1), layout="constrained")
    log2_axis(ax, SIZES, SIZE_LABELS)

    ref_line_data(ax, *d["link_ref"], MUTED)

    line(ax, SIZES, d["ce"], C_CE, "Copy Engine")
    n = len(d["tma"])
    line(ax, SIZES[:n], d["tma"], C_TMA, "TMA Op", zorder=4)
    line(ax, SIZES[n - 1:], [d["tma"][-1]] * (len(SIZES) - n + 1), C_TMA,
         "TMA (> smem, held constant)", zorder=4, dash=True)
    line(ax, SIZES, d["reg"], C_REG, "Register Op")

    ax.annotate(f"ce {d['ce'][-1]:.0f}", xy=(SIZES[-1], d["ce"][-1]),
                xytext=(2, 7), textcoords="offset points", ha="right",
                fontsize=9, fontweight="bold", color=C_CE)
    ax.annotate(f"reg {d['reg'][-1]:.0f}", xy=(SIZES[-1], d["reg"][-1]),
                xytext=(2, -15), textcoords="offset points", ha="right",
                fontsize=9, fontweight="bold", color=C_REG)
    ax.annotate(f"tma {d['tma'][0]:.0f} @128B", xy=(SIZES[0], d["tma"][0]),
                xytext=(6, -14), textcoords="offset points", fontsize=9,
                fontweight="bold", color=C_TMA)

    style_axis(ax, "Message size", d["ymax"], d["ystep"])
    for lab in ax.get_xticklabels():
        lab.set_rotation(35)
        lab.set_horizontalalignment("right")
    ax.set_title(f"P2P bandwidth vs. message granularity  ({d['name']})",
                 fontsize=10.5, color=INK, loc="left")
    ax.legend(loc="center right", fontsize=8, frameon=False)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def fig_smcount(d, path, dpi):
    fig, ax = plt.subplots(figsize=(7.0, 4.1), layout="constrained")
    log2_axis(ax, d["sms"], None)
    ax.xaxis.set_major_locator(FixedLocator(d["sms_ticks"]))
    ax.set_xticklabels([str(t) for t in d["sms_ticks"]])

    ref_line_data(ax, d["ce_ref"], f"copy engine (0 SMs, 1 call) {d['ce_ref']:.0f}",
                  C_CE)

    line(ax, d["sms"], d["tma_sms"], C_TMA, "TMA (1 issuing thread / SM)",
         zorder=4)
    line(ax, d["sms"], d["reg_sms"], C_REG, "Register Op (1024 threads / SM)")

    ax.annotate(f"{d['tma_sms'][0]:.0f} @1 SM", xy=(1, d["tma_sms"][0]),
                xytext=(6, -16), textcoords="offset points", fontsize=9,
                fontweight="bold", color=C_TMA)
    if d["tag"] == "h20":
        ax.annotate("linear: ~43 GB/s x SMs", xy=(4, 170.93), xytext=(-4, 8),
                    textcoords="offset points", ha="right", fontsize=8,
                    color=INK)
        ax.annotate("tma saturates ~12 SMs\nreg ~24 SMs", xy=(12, 364),
                    xytext=(8, -34), textcoords="offset points", fontsize=8,
                    color=INK)
    else:
        ax.annotate("saturated from 2 SMs", xy=(2, 51.19), xytext=(8, 6),
                    textcoords="offset points", fontsize=8, color=INK)

    style_axis(ax, "Number of SMs driving the transfer", d["ymax"], d["ystep"])
    ax.annotate(d["msg_note"], xy=(0.01, 0.99), xycoords="axes fraction",
                ha="left", va="top", fontsize=8, color=MUTED)
    ax.set_title(f"P2P bandwidth vs. SM count  ({d['name']})",
                 fontsize=10.5, color=INK, loc="left")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def fig_threads_rtx(path, dpi):
    fig, ax = plt.subplots(figsize=(7.0, 4.1), layout="constrained")
    log2_axis(ax, RTX_THREADS, [str(t) for t in RTX_THREADS])

    ref_line_data(ax, RTX_TMA_1SM, "TMA: 1 SM, a single thread, 43.9", C_TMA,
                  above=False)

    for sms, bw in RTX_REG_THR.items():
        ax.plot(RTX_THREADS, bw, color=REG_RAMP[sms], linewidth=2, marker="o",
                markersize=4.5, markeredgecolor="white", markeredgewidth=1,
                label=f"reg, {sms} SM{'s' if sms > 1 else ''}", zorder=3)
        ax.annotate(f"{sms} SM", xy=(RTX_THREADS[-1], bw[-1]),
                    xytext=(6, {1: -10, 2: 0, 4: 9}[sms]),
                    textcoords="offset points", fontsize=9, fontweight="bold",
                    color=REG_RAMP[sms])

    ax.annotate(f"pre-saturation: ~{RTX_PER_THREAD} GB/s x SMs x threads",
                xy=(64, 10.07), xytext=(10, -6), textcoords="offset points",
                fontsize=8, color=MUTED)

    style_axis(ax, "Threads per SM (reg kernel, msg 16 KiB)", 66, 10)
    ax.set_title("Register-op bandwidth vs. thread count  (2x RTX PRO 5000, PCIe)",
                 fontsize=10.5, color=INK, loc="left")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def fig_threads_h20(path, dpi):
    """All (SMs, threads) runs collapse onto one curve in total threads."""
    fig, ax = plt.subplots(figsize=(7.0, 4.1), layout="constrained")
    totals_ticks = [32, 128, 512, 2048, 8192, 32768]
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(totals_ticks))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.set_xticklabels(["32", "128", "512", "2K", "8K", "32K"])
    ax.set_xlim(32 / 1.5, 65536 * 1.5)

    # Ideal model: linear in total threads until the ~370 GB/s ceiling.
    # (y = k*x is a curve on a log-x axis, so sample it densely.)
    guide_x = [32 * 2 ** (i / 8) for i in range(0, 8 * 12 + 1)]
    guide_y = [min(x * H20_PER_THREAD, H20_REG_CEIL) for x in guide_x]
    ax.plot(guide_x, guide_y, color=MUTED, linewidth=1.2,
            linestyle=(0, (5, 4)), zorder=2)
    ax.annotate(f"model: min(~{H20_PER_THREAD} GB/s x total threads, 370)",
                xy=(2048, 2048 * H20_PER_THREAD), xytext=(-8, 10),
                textcoords="offset points", ha="right", fontsize=8, color=MUTED)

    ref_line_data(ax, H20_TMA_1SM, "TMA: 1 SM, a single thread, 43", C_TMA,
                  above=False)

    # One color (identity = reg), marker shape encodes the SM count.
    markers = {1: "o", 4: "s", 8: "^", 16: "D", 32: "v", 64: "P"}
    for sms, bw in H20_REG_THR.items():
        xs = [sms * t for t in H20_THREADS]
        ax.plot(xs, bw, color=C_REG, linewidth=1.2, alpha=0.55, zorder=3)
        ax.scatter(xs, bw, color=C_REG, marker=markers[sms], s=26, zorder=4,
                   edgecolors="white", linewidths=0.8,
                   label=f"{sms} SM{'s' if sms > 1 else ''}")

    style_axis(ax, "Total threads (SMs x threads/SM, reg kernel, msg 16 KiB)",
               470, 50)
    ax.set_title("Register-op bandwidth collapses onto total thread count  (2x H20, NVLink)",
                 fontsize=10.5, color=INK, loc="left")
    ax.legend(loc="upper left", fontsize=8, frameon=False, ncols=2,
              title="SMs used", title_fontsize=8)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figs", help="output directory")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--machine", default="all", choices=["rtx5000", "h20", "all"])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.titleweight": "bold",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })

    count = 0
    for d in (RTX5000, H20):
        if args.machine not in ("all", d["tag"]):
            continue
        fig_msgsize(d, os.path.join(args.out, f"pk_{d['tag']}_fig2_msgsize.png"),
                    args.dpi)
        fig_smcount(d, os.path.join(args.out, f"pk_{d['tag']}_fig3_smcount.png"),
                    args.dpi)
        if d["tag"] == "rtx5000":
            fig_threads_rtx(os.path.join(args.out,
                                         "pk_rtx5000_fig4_regthreads.png"),
                            args.dpi)
        else:
            fig_threads_h20(os.path.join(args.out,
                                         "pk_h20_fig4_regthreads.png"),
                            args.dpi)
        count += 3
    print(f"wrote {count} figures to {args.out}/")


if __name__ == "__main__":
    main()
