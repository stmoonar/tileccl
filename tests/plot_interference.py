#!/usr/bin/env python3
"""Plot the interference / sync-cost study results.

Data embedded below, measured on:
  rtx5000  2x RTX PRO 5000 Blackwell (sm_120, 110 SMs), PCIe Gen5 x16
  h20      2x H20 (sm_90, 78 SMs), NVLink

Sources: interference_matrix (--no-alone-cache runs), the --comm-sms sweep,
intra_sm_matrix v2, sync_cost, and the starvation mode. Re-run the tools and
paste new numbers to refresh.

Figures written to --out (default figs/):
  if_sc_heatmap.png   S_c: how much communication slows compute (per probe x mover)
  if_sm_heatmap.png   S_m: how much compute slows communication (source side)
  if_sm_sweep.png     S_m + achieved BW vs comm SMs (hbm probe under load)
  if_intra.png        intra-SM S_intra / S_warp (warp-specialized fusion)
  if_sync.png         sync-primitive costs (producer sequence, signals)
  if_starve.png       starvation under a full-occupancy compute kernel

Usage:  python plot_interference.py [--out figs] [--dpi 200]
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize

# ---------------------------------------------------------------------------
# shared style (same identity colors as plot_pk_bw.py: color follows mechanism)
# ---------------------------------------------------------------------------

C_CE, C_TMA, C_REG = "#2a78d6", "#eb6834", "#1baf7a"
TMA_RAMP = {1: "#f2a37f", 2: "#eb6834", 4: "#b04618"}   # comm warps 1/2/4
REG_RAMP = {1: "#6cc9a4", 2: "#1baf7a", 4: "#0e7a52"}
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
SURF = "#fcfcfb"

HEAT = LinearSegmentedColormap.from_list(
    "interf", ["#fcfcfb", "#f7d7c6", "#ee8a5c", "#c2571f", "#7d3410"])

plt.rcParams.update({
    "font.size": 9,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.grid": False,
    "figure.facecolor": "white",
})

PROBES = ["mma", "ffma", "hbm", "l2", "smem"]
MOVERS = ["ce", "tma", "reg"]

# ---------------------------------------------------------------------------
# measured data
# ---------------------------------------------------------------------------

# interference_matrix, S_c rows = probes, cols = movers (ce, tma, reg).
SC = {
    ("rtx5000", "src"): [[1.00, 1.00, 1.00],
                         [1.00, 1.00, 1.00],
                         [1.03, 1.03, 1.03],
                         [1.00, 1.01, 1.00],
                         [1.00, 1.00, 1.00]],
    ("rtx5000", "dst"): [[1.00, 1.00, 1.00],
                         [1.00, 1.00, 1.00],
                         [1.07, 1.07, 1.07],
                         [1.00, 1.00, 1.00],
                         [1.00, 1.00, 1.00]],
    ("h20", "src"):     [[1.00, 1.00, 1.00],
                         [1.00, 1.00, 1.00],
                         [1.03, 1.02, 1.01],
                         [1.10, 1.62, 1.91],
                         [1.00, 1.00, 1.00]],
    ("h20", "dst"):     [[1.00, 1.00, 1.00],
                         [1.00, 1.00, 1.00],
                         [1.07, 1.07, 1.04],
                         [1.67, 1.66, 1.64],
                         [1.00, 1.00, 1.00]],
}

# S_m, source side (dst side is 1.00 everywhere on both machines).
SM = {
    "rtx5000": [[1.00, 1.00, 1.00],
                [1.00, 1.00, 1.00],
                [1.41, 1.07, 1.01],
                [1.00, 1.00, 1.00],
                [1.00, 1.00, 1.00]],
    "h20":     [[1.01, 1.07, 1.02],
                [1.01, 1.00, 1.00],
                [1.06, 1.16, 1.28],
                [1.00, 1.00, 1.00],
                [1.00, 1.00, 1.00]],
}

# --comm-sms sweep, hbm probe on src. (sms, S_m, alone GB/s)
SWEEP_SMS = [2, 8, 16, 32, 64]
SWEEP = {
    "rtx5000": {
        "tma": dict(sm=[4.07, 1.07, 1.08, 1.01, 1.00],
                    bw=[50.77, 51.07, 51.06, 51.07, 51.07]),
        "reg": dict(sm=[1.28, 1.01, 1.02, 1.00, 1.00],
                    bw=[51.08, 51.07, 51.07, 51.09, 51.09]),
    },
    "h20": {
        "tma": dict(sm=[1.16, 1.16, 1.15, 1.00, 1.00],
                    bw=[65.94, 257.55, 337.49, 356.22, 361.06]),
        "reg": dict(sm=[1.30, 1.28, 1.29, 0.99, 0.99],
                    bw=[38.74, 151.83, 299.98, 315.43, 350.57]),
    },
}

# intra_sm_matrix v2, RTX5000 (H20 pending). S_intra rows follow IN_PROBES,
# one list per comm warp count 1/2/4.
IN_PROBES = ["mma", "ffma", "smem", "hbm"]
INTRA = {
    "all-CTA (110 comm CTAs)": {
        "ldst": {1: [1.000, 1.014, 1.003, 1.033],
                 2: [1.019, 1.110, 1.016, 1.427],
                 4: [1.062, 1.236, 1.036, 3.493]},
        "tma":  {1: [1.129, 1.478, 1.062, 5.656],
                 2: [1.244, 1.763, 1.109, 11.710],
                 4: [1.344, 1.785, 1.117, 12.304]},
    },
    "concentrated (8 of 110 CTAs)": {
        "ldst": {1: [1.001, 1.003, 1.005, 1.038],
                 2: [1.001, 1.004, 1.010, 1.079],
                 4: [1.002, 1.011, 1.017, 1.166]},
        "tma":  {1: [1.010, 1.022, 1.014, 1.246],
                 2: [1.012, 1.057, 1.022, 1.562],
                 4: [1.017, 1.071, 1.028, 2.177]},
    },
}
S_WARP = {  # static cost of ceding warps, per probe, cw 1/2/4
    "mma": [1.000, 1.000, 1.003],
    "ffma": [1.066, 1.143, 1.333],
    "smem": [1.055, 1.117, 1.260],
    "hbm": [0.997, 0.996, 0.959],
}

# sync_cost, RTX5000 (PCIe Gen5). Per-message producer sequence, us.
SYNC_PAYLOAD = ["0B", "512B", "2K", "8K", "32K", "128K"]
SYNC_STORES = [0.076, 0.118, 0.197, 0.344, 0.830, 14.504]
SYNC_REL = [0.526, 0.618, 0.726, 0.805, 1.292, 14.951]
SYNC_FENCE = [0.520, 0.780, 0.901, 1.509, 4.203, 14.962]
SYNC_ONEWAY = 1.307   # flag ping-pong RTT/2, us
SYNC_ATOMIC = 1.145   # dependent atomicAdd_system on peer, us/op
SYNC_MBARRIER_NS = 26  # 63 cycles @ 2.4 GHz

# starvation: (alone us, under-squat us, starved?)
STARVE = {
    "rtx5000": {"ce": (1222.2, 1225.3, False),
                "tma": (1725.8, 1001887.3, True),
                "reg": (1319.7, 1001798.7, True)},
    "h20": {"ce": (288.7, 439.6, False),
            "tma": (264.9, 1001367.4, True),
            "reg": (450.9, 1000958.1, True)},
}

MACHINE_NAME = {"rtx5000": "RTX PRO 5000 x2, PCIe Gen5",
                "h20": "H20 x2, NVLink"}

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=3, color=MUTED)


def heat_panel(ax, data, title, vmax):
    data = np.array(data)
    norm = Normalize(vmin=1.0, vmax=vmax)
    ax.imshow(data, cmap=HEAT, norm=norm, aspect="auto")
    ax.set_xticks(range(len(MOVERS)), MOVERS)
    ax.set_yticks(range(len(PROBES)), PROBES)
    ax.set_title(title, fontsize=9, pad=6)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    # white 2px grid between cells
    for i in range(1, len(PROBES)):
        ax.axhline(i - 0.5, color="white", lw=2)
    for j in range(1, len(MOVERS)):
        ax.axvline(j - 0.5, color="white", lw=2)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            ax.text(j, i, f"{v:.2f}",
                    ha="center", va="center", fontsize=8.5,
                    color="white" if norm(v) > 0.55 else INK,
                    fontweight="bold" if v >= 1.10 else "normal")


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def fig_sc_heatmap(out, dpi):
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.4))
    vmax = 2.0
    for ax, key in zip(axes.flat, [("rtx5000", "src"), ("rtx5000", "dst"),
                                   ("h20", "src"), ("h20", "dst")]):
        mach, side = key
        side_txt = "sender side" if side == "src" else "receiver side"
        heat_panel(ax, SC[key], f"{MACHINE_NAME[mach]} - {side_txt}", vmax)
    fig.suptitle("How much does communication slow compute?\n"
                 "S_c = compute alone / compute overlapped", fontsize=10.5,
                 y=0.99)
    fig.text(0.47, 0.895,
             "1.00 = no interference.  Compute-, L2(PCIe)- and smem-bound rows"
             " are untouched everywhere;\nthe only large channel is L2"
             " pollution on NVLink (up to 1.9x), plus a fixed ~7%"
             " receiver-side HBM tax.",
             ha="center", fontsize=8, color=MUTED)
    sm = plt.cm.ScalarMappable(cmap=HEAT, norm=Normalize(1.0, vmax))
    cb = fig.colorbar(sm, ax=axes, fraction=0.03, pad=0.02)
    cb.set_label("S_c (slowdown factor)", fontsize=8)
    cb.outline.set_visible(False)
    fig.subplots_adjust(top=0.83, hspace=0.35, wspace=0.25, left=0.09,
                        right=0.86)
    fig.savefig(os.path.join(out, "if_sc_heatmap.png"), dpi=dpi)
    plt.close(fig)


def fig_sm_heatmap(out, dpi):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.6))
    vmax = 1.5
    for ax, mach in zip(axes, ["rtx5000", "h20"]):
        heat_panel(ax, SM[mach], MACHINE_NAME[mach], vmax)
    fig.suptitle("How much does compute slow communication?  "
                 "S_m = comm BW alone / comm BW overlapped  (sender side)",
                 fontsize=11, y=1.00)
    fig.text(0.5, 0.90,
             "Receiver-side S_m is 1.00 everywhere on both machines: incoming"
             " writes cannot be throttled by local compute.\nOn PCIe the copy"
             " engine loses 41% under an HBM-saturated kernel; on NVLink the"
             " under-provisioned SM movers lose instead.",
             ha="center", fontsize=8.5, color=MUTED)
    sm = plt.cm.ScalarMappable(cmap=HEAT, norm=Normalize(1.0, vmax))
    cb = fig.colorbar(sm, ax=axes, fraction=0.03, pad=0.02)
    cb.set_label("S_m (slowdown factor)", fontsize=8)
    cb.outline.set_visible(False)
    fig.subplots_adjust(top=0.72, wspace=0.25, left=0.09, right=0.88,
                        bottom=0.12)
    fig.savefig(os.path.join(out, "if_sm_heatmap.png"), dpi=dpi)
    plt.close(fig)


def fig_sm_sweep(out, dpi):
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.6), sharex=True)
    x = np.arange(len(SWEEP_SMS))
    for col, mach in enumerate(["rtx5000", "h20"]):
        ax_bw, ax_sm = axes[0][col], axes[1][col]
        for mover, color in [("tma", C_TMA), ("reg", C_REG)]:
            d = SWEEP[mach][mover]
            ax_bw.plot(x, d["bw"], "-o", color=color, lw=2, ms=5, label=mover)
            ax_sm.plot(x, d["sm"], "-o", color=color, lw=2, ms=5, label=mover)
        ax_bw.set_title(f"{MACHINE_NAME[mach]}  (hbm-bound compute on sender)",
                        fontsize=9)
        ax_bw.set_ylim(0, 60 if mach == "rtx5000" else 400)
        ax_bw.set_ylabel("comm BW alone (GB/s)" if col == 0 else "")
        ax_sm.set_ylabel("S_m under compute" if col == 0 else "")
        ax_sm.set_xlabel("SMs given to the mover")
        ax_sm.set_xticks(x, [str(s) for s in SWEEP_SMS])
        ax_sm.axhline(1.0, color=MUTED, lw=1, ls=":")
        for ax in (ax_bw, ax_sm):
            style_axes(ax)
            ax.grid(axis="y", color=GRID, lw=0.7)
    axes[1][0].annotate("tma@2 SMs collapses 4.1x:\ntoo few bytes in flight"
                        " to ride out\nDRAM-contention latency",
                        xy=(0, 4.07), xytext=(0.9, 3.3), fontsize=8,
                        color=INK,
                        arrowprops=dict(arrowstyle="->", color=MUTED))
    axes[1][1].annotate("vulnerable until provisioned\nto saturation"
                        " (~32 SMs)", xy=(2, 1.29), xytext=(1.1, 1.14),
                        fontsize=8, color=INK,
                        arrowprops=dict(arrowstyle="->", color=MUTED))
    axes[0][0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Communication robustness scales with bytes-in-flight, not"
                 " mechanism", fontsize=11)
    fig.subplots_adjust(top=0.90, hspace=0.18, wspace=0.24, left=0.09,
                        right=0.97, bottom=0.10)
    fig.savefig(os.path.join(out, "if_sm_sweep.png"), dpi=dpi)
    plt.close(fig)


def fig_intra(out, dpi):
    fig = plt.figure(figsize=(8.6, 4.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.25, 0.9], wspace=0.32)
    modes = list(INTRA.keys())
    for p, mode in enumerate(modes):
        ax = fig.add_subplot(gs[0, p])
        base = np.arange(len(IN_PROBES)) * 1.0
        width = 0.13
        for gi, (comm, ramp) in enumerate([("ldst", REG_RAMP),
                                           ("tma", TMA_RAMP)]):
            for ci, cw in enumerate([1, 2, 4]):
                offs = (gi * 3 + ci - 2.5) * width
                vals = INTRA[mode][comm][cw]
                ax.bar(base + offs, vals, width=width * 0.92,
                       color=ramp[cw],
                       label=f"{comm} cw={cw}" if p == 0 else None)
        ax.set_yscale("log")
        ax.set_ylim(0.95, 16)
        ax.set_yticks([1, 2, 4, 8, 16], ["1", "2", "4", "8", "16"])
        ax.axhline(1.0, color=MUTED, lw=1, ls=":")
        ax.set_xticks(base, IN_PROBES)
        ax.set_title(mode, fontsize=9)
        if p == 0:
            ax.set_ylabel("S_intra (dynamic slowdown, log)")
        style_axes(ax)
        ax.grid(axis="y", color=GRID, lw=0.7)
    ax = fig.add_subplot(gs[0, 2])
    cws = [1, 2, 4]
    colors = {"mma": C_CE, "ffma": C_TMA, "smem": C_REG, "hbm": "#eda100"}
    for probe in IN_PROBES:
        ax.plot(cws, S_WARP[probe], "-o", lw=2, ms=5, color=colors[probe],
                label=probe)
    ax.axhline(1.0, color=MUTED, lw=1, ls=":")
    ax.set_xticks(cws, [str(c) for c in cws])
    ax.set_xlabel("warps ceded per CTA")
    ax.set_title("S_warp: static cost of\nceding warps (no comm)", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    style_axes(ax)
    ax.grid(axis="y", color=GRID, lw=0.7)
    fig.suptitle("Intra-SM interference (warp-specialized fusion, RTX PRO"
                 " 5000, PCIe)  -  link-saturating backpressure only hurts"
                 " memory-touching compute, and only on comm-carrying SMs"
                 " (S_pure = 1.00)", fontsize=10)
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=6, frameon=False, fontsize=8,
               loc="lower center", bbox_to_anchor=(0.42, -0.01))
    fig.subplots_adjust(top=0.82, bottom=0.20, left=0.07, right=0.98)
    fig.savefig(os.path.join(out, "if_intra.png"), dpi=dpi,
                bbox_inches="tight")
    plt.close(fig)


def fig_sync(out, dpi):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.6))
    x = np.arange(len(SYNC_PAYLOAD))
    ax1.plot(x, SYNC_STORES, "-o", color=C_CE, lw=2, ms=5,
             label="payload stores only")
    ax1.plot(x, SYNC_REL, "-o", color=C_REG, lw=2, ms=5,
             label="+ st.release.sys flag")
    ax1.plot(x, SYNC_FENCE, "-o", color=C_TMA, lw=2, ms=5,
             label="+ __threadfence_system + flag")
    ax1.set_yscale("log")
    ax1.set_xticks(x, SYNC_PAYLOAD)
    ax1.set_xlabel("payload per message")
    ax1.set_ylabel("us per message (log)")
    ax1.set_title("producer sequence cost, one warp", fontsize=9)
    ax1.legend(frameon=False, fontsize=8, loc="upper left")
    style_axes(ax1)
    ax1.grid(axis="y", color=GRID, lw=0.7)

    drel = [r - s for r, s in zip(SYNC_REL, SYNC_STORES)]
    dfence = [f - s for f, s in zip(SYNC_FENCE, SYNC_STORES)]
    w = 0.36
    ax2.bar(x - w / 2, drel, width=w * 0.92, color=C_REG,
            label="release flag (marginal)")
    ax2.bar(x + w / 2, dfence, width=w * 0.92, color=C_TMA,
            label="fence + flag (marginal)")
    ax2.axhline(SYNC_ONEWAY, color=C_CE, lw=1.4, ls="--")
    ax2.text(0.02, SYNC_ONEWAY + 0.08, "one-way flag latency 1.31 us",
             fontsize=8, color=C_CE)
    ax2.axhline(SYNC_ATOMIC, color=MUTED, lw=1.2, ls=":")
    ax2.text(-0.45, 0.93, "remote atomic 1.15 us/op",
             fontsize=8, color=MUTED, ha="left")
    ax2.set_xticks(x, SYNC_PAYLOAD)
    ax2.set_xlabel("payload per message")
    ax2.set_ylabel("marginal signal cost (us)")
    ax2.set_title("what the signal adds on top of the payload", fontsize=9)
    ax2.legend(frameon=False, fontsize=8, loc="upper left")
    style_axes(ax2)
    ax2.grid(axis="y", color=GRID, lw=0.7)
    fig.text(0.5, 0.015,
             "local smem mbarrier: 63 cycles (~%d ns) - negligible.  Release"
             " flag is flat ~0.45 us (PCIe posted-write ordering);\nthe fence"
             " must drain in-flight P2P writes, so its cost grows with the"
             " payload it follows." % SYNC_MBARRIER_NS,
             ha="center", fontsize=7.5, color=MUTED)
    fig.suptitle("Sync-primitive costs (RTX PRO 5000 pair, PCIe Gen5)",
                 fontsize=11)
    fig.subplots_adjust(top=0.85, bottom=0.26, wspace=0.28, left=0.08,
                        right=0.98)
    fig.savefig(os.path.join(out, "if_sync.png"), dpi=dpi)
    plt.close(fig)


def fig_starve(out, dpi):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), sharex=True)
    movers = ["ce", "tma", "reg"]
    colors = {"ce": C_CE, "tma": C_TMA, "reg": C_REG}
    for ax, mach in zip(axes, ["rtx5000", "h20"]):
        y = np.arange(len(movers))[::-1] * 1.0
        h = 0.32
        for i, m in enumerate(movers):
            alone, squat, starved = STARVE[mach][m]
            ax.barh(y[i] + h / 2 + 0.02, alone, height=h, color=colors[m],
                    alpha=0.45)
            ax.barh(y[i] - h / 2 - 0.02, squat, height=h, color=colors[m])
            if starved:
                ax.text(squat * 0.7, y[i] - h / 2 - 0.02, "STARVED",
                        va="center", ha="right", fontsize=8.5, color="white",
                        fontweight="bold")
            else:
                ax.text(squat * 1.3, y[i] - h / 2 - 0.02,
                        f"{squat / alone:.2f}x", va="center", fontsize=8,
                        color=MUTED)
        ax.set_xscale("log")
        ax.set_xlim(100, 4e6)
        ax.set_yticks(y, movers)
        ax.set_xlabel("time to move one 64 MiB payload (us, log)")
        ax.set_title(MACHINE_NAME[mach], fontsize=9)
        style_axes(ax)
        ax.grid(axis="x", color=GRID, lw=0.7)
    fig.suptitle("Progress model: only the copy engine advances under a"
                 " fully occupied GPU", fontsize=11, y=0.99)
    fig.text(0.5, 0.885, "light = alone,   solid = launched under a"
             " full-occupancy compute kernel (1s deadline)",
             ha="center", fontsize=8, color=MUTED)
    fig.subplots_adjust(top=0.76, bottom=0.19, wspace=0.16, left=0.07,
                        right=0.98)
    fig.savefig(os.path.join(out, "if_starve.png"), dpi=dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figs")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    fig_sc_heatmap(args.out, args.dpi)
    fig_sm_heatmap(args.out, args.dpi)
    fig_sm_sweep(args.out, args.dpi)
    fig_intra(args.out, args.dpi)
    fig_sync(args.out, args.dpi)
    fig_starve(args.out, args.dpi)
    print("wrote 6 figures to", args.out)


if __name__ == "__main__":
    main()
