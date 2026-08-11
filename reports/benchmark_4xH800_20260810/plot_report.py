#!/usr/bin/env python3
"""Generate the figures for ANALYSIS_4xH800.md from the benchmark CSVs.

Reads:
  comm_comp/results_20260810_121354/*.csv        (this bundle, 4xH800)
  signalling/signalling_4xH800.csv
  ../comm_comp_signalling_4xH20/comm_comp_results/*.csv   (H20 comparison)

Writes figs/*.png. Usage:  python plot_report.py
"""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
H800 = os.path.join(HERE, "comm_comp", "results_20260810_121354")
H800_SIG = os.path.join(HERE, "signalling", "signalling_4xH800.csv")
H20 = os.path.join(HERE, "..", "comm_comp_signalling_4xH20", "comm_comp_results")
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
    return [r for r in out if str(r.get("m", r.get("mode", "0"))).replace(".", "").isdigit()
            or "mode" in r and r["mode"] not in (None, "mode")]


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)
    print("wrote figs/" + name)


# ---------------------------------------------------------------- fig 1: exp1
def fig_exp1():
    data = rows(os.path.join(H800, "exp1_default.csv"))
    patterns = ["pull", "push", "allgather", "bystander", "engine-only", "local"]
    sizes = [2048, 4096, 8192]
    sc = {(r["pattern"], int(r["m"])): float(r["S_c"]) for r in data}

    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    w = 0.26
    colors = [TEAL, COPPER, SLATE]
    ymax = 1.40
    for si, (m, c) in enumerate(zip(sizes, colors)):
        xs = [i + (si - 1) * w for i in range(len(patterns))]
        vals = [sc[(p, m)] for p in patterns]
        clipped = [v > ymax for v in vals]
        ax.bar(xs, [min(v, ymax) for v in vals], width=w, color=c,
               alpha=[0.45 if cl else 0.95 for cl in clipped][0] if False else 0.95,
               label=f"{m}$^3$", zorder=3)
        for x, v, cl in zip(xs, vals, clipped):
            ax.text(x, min(v, ymax) + 0.008,
                    (f"↑{v:.2f}" if cl else f"{v:.3f}"),
                    ha="center", va="bottom", fontsize=7.2, rotation=0)
    ax.axhline(1.0, color=GREY, lw=1, ls="--", zorder=2)
    ax.set_xticks(range(len(patterns)))
    ax.set_xticklabels(patterns)
    ax.set_ylim(0.95, 1.47)
    ax.set_ylabel("S_c = t_overlap / t_alone")
    ax.set_title("exp1: CE traffic vs independent GEMM (msg = 64 MiB)")
    ax.legend(frameon=False, ncol=3, loc="upper right")
    save(fig, "fig1_exp1_sc.png")


# ---------------------------------------------------------------- fig 2: exp2
def fig_exp2():
    # push/pull overlap almost exactly for S>=4: differentiate by marker,
    # dash and draw order so all three stay visible
    runs = [("exp2_push.csv", "push / 3 streams", SLATE, "--", "^", 3),
            ("exp2_serial.csv", "serial / 1 stream", COPPER, "-", "s", 4),
            ("exp2_pull.csv", "pull / 3 streams", TEAL, "-", "o", 5)]
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    fulls = []
    for fname, label, color, ls, mk, z in runs:
        data = [r for r in rows(os.path.join(H800, fname)) if int(r["m"]) == 8192]
        S = [int(r["S"]) for r in data]
        tovl = [float(r["t_ovl_us"]) for r in data]
        fulls.append(float(data[0]["t_full_us"]))
        ax.plot(S, tovl, marker=mk, ls=ls, color=color, label=label,
                ms=4.5, lw=1.8, zorder=z, markerfacecolor="none" if mk == "^" else color)
    ax.annotate("serial S=1: 1643", (1, 1643.4), textcoords="offset points",
                xytext=(6, -14), fontsize=8, color=COPPER)
    ax.annotate("pull S=2: 1707", (2, 1706.6), textcoords="offset points",
                xytext=(0, 8), ha="center", fontsize=8, color=TEAL)
    tf = sum(fulls) / len(fulls)
    ax.axhline(tf, color=GREY, lw=1.2, ls="--", zorder=2)
    ax.text(1.02, tf * 1.04, f"t_full ≈ {tf:.0f} µs", fontsize=8.5, color=GREY)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(FixedLocator([1, 2, 4, 8, 16, 32]))
    ax.xaxis.set_major_formatter("S={x:.0f}")
    ax.yaxis.set_major_locator(FixedLocator([1500, 2000, 3000, 5000, 10000]))
    ax.yaxis.set_major_formatter("{x:.0f}")
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylabel("t_ovl (µs, log)")
    ax.set_title("exp2: AG+GEMM end-to-end time vs copy granularity (8192$^3$, shard M=2048)")
    ax.legend(frameon=False)
    save(fig, "fig2_exp2_granularity.png")


# ------------------------------------------------------------- fig 3: exp3 v1
def fig_exp3v1():
    data = rows(os.path.join(H800, "exp3_epilogue.csv"))
    ks = [512, 2048, 8192]

    def sd(epi, mode):
        return [float(r["slowdown"]) for k in ks for r in data
                if int(r["k"]) == k and r["epi"] == epi and r["mode"] == mode]

    series = [("tma remote", sd("tma", "remote"), TEAL, "-"),
              ("tma scatter (1/4 local + 3/4 remote)", sd("tma", "scatter"), TEAL, "--"),
              ("nosmem remote (scalar stores)", sd("nosmem", "remote"), RED, "-"),
              ("tma scatter-local (segmentation only)", sd("tma", "scatter-local"), SLATE, "-")]
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    for label, vals, color, ls in series:
        ax.plot(ks, vals, "o", ls=ls, color=color, label=label, ms=4.5, lw=1.8, zorder=3)
        for k, v in zip(ks, vals):
            if ls == "-":
                ax.annotate(f"{v:.2f}", (k, v), textcoords="offset points",
                            xytext=(0, 6), ha="center", fontsize=7.5)
    ax.axhline(1.0, color=GREY, lw=1, ls="--", zorder=2)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(FixedLocator(ks))
    ax.xaxis.set_major_formatter("K={x:.0f}")
    ax.yaxis.set_major_locator(FixedLocator([1, 2, 4, 8, 16]))
    ax.yaxis.set_major_formatter("{x:.0f}×")
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylabel("slowdown vs same-epilogue local (log)")
    ax.set_title("exp3 v1: epilogue remote-write tax vs arithmetic intensity (8192×8192×K)")
    ax.legend(frameon=False, fontsize=8.5)
    save(fig, "fig3_exp3v1_remote_tax.png")


# ------------------------------------------------------------- fig 4: exp3 v2
def fused_by_k(path):
    """{k: (mean struct, mean comm, worst total, mean pull_gbps)}"""
    out = {}
    data = rows(path)
    for k in sorted({int(r["k"]) for r in data}, reverse=True):
        rs = [r for r in data if int(r["k"]) == k]
        out[k] = (sum(float(r["struct_sd"]) for r in rs) / len(rs),
                  sum(float(r["comm_sd"]) for r in rs) / len(rs),
                  max(float(r["total_sd"]) for r in rs),
                  sum(float(r["pull_gbps"]) for r in rs) / len(rs))
    return out


def fig_fused():
    f = fused_by_k(os.path.join(H800, "exp3_fused.csv"))
    ks = list(f)  # [8192, 2048, 512]
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    w = 0.24
    for i, (label, idx, color) in enumerate(
            [("struct = ctrl/base", 0, SLATE),
             ("comm = fused/ctrl", 1, TEAL),
             ("total (worst rank)", 2, COPPER)]):
        xs = [j + (i - 1) * w for j in range(len(ks))]
        vals = [f[k][idx] for k in ks]
        ax.bar(xs, vals, width=w, bottom=0, color=color, label=label, zorder=3)
        for x, v in zip(xs, vals):
            ax.text(x, v * 1.03, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)
    ax.axhline(1.0, color=GREY, lw=1, ls="--", zorder=4)
    ax.set_yscale("log")
    ax.set_ylim(0.8, 12)
    ax.yaxis.set_major_locator(FixedLocator([1, 2, 4, 8]))
    ax.yaxis.set_major_formatter("{x:.0f}×")
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([f"K={k}\n(pull {f[k][3]:.0f} GB/s)" for k in ks])
    ax.set_ylabel("slowdown (log)")
    ax.set_title("exp3 v2: fused GEMM+RS — structure tax vs communication tax (RS-off control)")
    ax.legend(frameon=False)
    save(fig, "fig4_exp3v2_fused.png")


# ---------------------------------------------------------- fig 5: H800 vs H20
def fig_cross():
    ks = [8192, 2048, 512]

    def remote_tax(dirpath):
        data = rows(os.path.join(dirpath, "exp3_epilogue.csv"))
        return [float(r["slowdown"]) for k in ks for r in data
                if int(r["k"]) == k and r["epi"] == "tma" and r["mode"] == "remote"]

    h800_rw = remote_tax(H800)
    h20_rw = remote_tax(H20)
    h800_f = fused_by_k(os.path.join(H800, "exp3_fused.csv"))
    h20_f = fused_by_k(os.path.join(H20, "exp3_fused.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6), sharey=True)
    panels = [("epilogue TMA remote-write tax", h800_rw, h20_rw),
              ("fused RS communication tax (comm)", [h800_f[k][1] for k in ks],
               [h20_f[k][1] for k in ks])]
    w = 0.32
    for ax, (title, a, b) in zip(axes, panels):
        xs = range(len(ks))
        ax.bar([x - w / 2 for x in xs], a, width=w, color=TEAL, label="H800", zorder=3)
        ax.bar([x + w / 2 for x in xs], b, width=w, color=COPPER, label="H20", zorder=3)
        for x, v in zip(xs, a):
            ax.text(x - w / 2, v * 1.04, f"{v:.2f}", ha="center", fontsize=7.5)
        for x, v in zip(xs, b):
            ax.text(x + w / 2, v * 1.04, f"{v:.2f}", ha="center", fontsize=7.5)
        ax.axhline(1.0, color=GREY, lw=1, ls="--", zorder=4)
        ax.set_yscale("log")
        ax.set_ylim(0.9, 12)
        ax.yaxis.set_major_locator(FixedLocator([1, 2, 4, 8]))
        ax.yaxis.set_major_formatter("{x:.0f}×")
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xticks(list(xs))
        ax.set_xticklabels([f"K={k}" for k in ks])
        ax.set_title(title, fontsize=9.5)
        ax.legend(frameon=False, fontsize=8.5)
    fig.suptitle("Same design choice, opposite verdict: link headroom / compute ratio decides", y=1.02)
    save(fig, "fig5_cross_h800_h20.png")


# ------------------------------------------------------------ fig 6: signalling
def fig_signalling():
    data = rows(H800_SIG)

    def pick(mode, fanin):
        rs = [r for r in data if r["mode"] == mode and int(r["fanin"]) == fanin]
        rs.sort(key=lambda r: int(r["bytes"]))
        return ([int(r["bytes"]) for r in rs], [float(r["p50_us"]) for r in rs])

    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    for mode, fanin, color, alpha in [("pull", 1, TEAL, 1.0), ("pull", 32, TEAL, 0.45),
                                      ("push", 1, COPPER, 1.0), ("push", 32, COPPER, 0.45)]:
        bx, py = pick(mode, fanin)
        ax.plot(bx, py, "o-", color=color, alpha=alpha, ms=4, lw=1.8,
                label=f"{mode} F={fanin}", zorder=3)
    sig = [float(r["p50_us"]) for r in data if r["mode"] == "signal" and int(r["fanin"]) == 1]
    if sig:
        ax.axhline(sig[0], color=GREY, lw=1, ls="--", zorder=2)
        ax.text(4500, sig[0] * 1.1, f"signal floor (go handshake) ≈ {sig[0]:.1f} µs",
                fontsize=8, color=GREY)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ticks = [4096, 8192, 32768, 131072, 1048576, 4194304]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(lambda x, _: f"{int(x/1024)}K" if x < 1 << 20 else f"{int(x/1048576)}M")
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(FixedLocator([2, 5, 10, 30, 100, 300, 1000]))
    ax.yaxis.set_major_formatter("{x:.0f}")
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("payload size")
    ax.set_ylabel("p50 (µs, log)")
    ax.set_title("signalling: push vs pull completion latency (target-side clock64)")
    ax.legend(frameon=False, fontsize=8.5)
    save(fig, "fig6_signalling.png")


if __name__ == "__main__":
    fig_exp1()
    fig_exp2()
    fig_exp3v1()
    fig_fused()
    fig_cross()
    fig_signalling()
