#!/usr/bin/env python3
"""Figures for BENCHMARK_REPORT_4xH800.md (the held-clock suite).

Reads comm_comp/results_20260812_083024/ and writes figs/fig1..fig3.
Usage: python plot_final.py
"""
import csv
import io
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "comm_comp", "results_20260812_083024")
OUT = os.path.join(HERE, "figs")

TEAL, COPPER, SLATE, RED, GREY = "#0E6B67", "#B4560F", "#5B5E8D", "#A83232", "#8A938F"

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#DDDDDD", "grid.linewidth": 0.6,
    "figure.facecolor": "white", "savefig.dpi": 150, "savefig.bbox": "tight",
})


def rows(name):
    """The CSVs re-emit their header per invocation; drop those pseudo-rows."""
    with io.open(os.path.join(RES, name), newline="", encoding="utf-8") as f:
        out = list(csv.DictReader(f))
    key = "pattern" if name.startswith("exp1") else "m"
    return [r for r in out if r.get(key) and r[key] not in ("m", "pattern")]


def mean(v):
    return sum(v) / len(v)


def logx(ax, ticks, label):
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(
        lambda x, _: f"{int(x/1024)}K" if x >= 1024 else f"{int(x)}")
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(label)


# ---------------------------------------------------------------------------
# fig1: exp1 interference + exp2 granularity
# ---------------------------------------------------------------------------
def fig1():
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(10.5, 3.9))

    d = rows("exp1_default.csv")
    pats = ["engine-only", "bystander", "pull", "push", "allgather", "local"]
    ms = sorted({int(r["m"]) for r in d})
    for p, col in zip(pats, [GREY, SLATE, TEAL, TEAL, COPPER, RED]):
        ls = "--" if p == "push" else "-"
        v = [float(r["S_c"]) for m in ms for r in d
             if int(r["m"]) == m and r["pattern"] == p]
        axl.plot(ms, v, "o", ls=ls, color=col, ms=4, lw=1.8, label=p, zorder=3)
    axl.axhline(1.0, color=GREY, lw=1, ls="--", zorder=2)
    axl.annotate("only local D2D really bites:\nit shares this GPU's HBM",
                 (2048, 1.975), textcoords="offset points", xytext=(14, -6),
                 fontsize=7.5, color=RED)
    axl.set_yscale("log")
    axl.yaxis.set_major_locator(FixedLocator([1, 1.25, 1.5, 2]))
    axl.yaxis.set_major_formatter("{x:.2f}x")
    axl.yaxis.set_minor_formatter(NullFormatter())
    axl.set_ylabel("S_c = GEMM overlapped / alone (paired)")
    axl.set_title("exp1: CE traffic vs an independent GEMM")
    logx(axl, ms, "GEMM M=N=K")
    axl.legend(frameon=False, fontsize=7.5, loc="upper right")

    d = rows("exp2_pull.csv")
    for m, col in ((8192, TEAL), (16384, COPPER)):
        rs = sorted((r for r in d if int(r["m"]) == m), key=lambda r: int(r["S"]))
        S = [int(r["S"]) for r in rs]
        axr.plot(S, [float(r["e2e_tflops"]) for r in rs], "o-", color=col,
                 ms=4, lw=1.8, label=f"M={m} e2e", zorder=3)
        axr.plot(S, [float(r["seg_eff"]) * 650 for r in rs], "s:", color=col,
                 ms=3, lw=1.3, alpha=0.65, label=f"M={m} seg_eff (x650)")
    axr.axhline(651, color=GREY, lw=1, ls="--", zorder=2)
    axr.text(1.05, 664, "GEMM roofline 651 TFLOP/s @ 1305 MHz",
             fontsize=7.5, color=GREY)
    axr.annotate("S=2 optimum: comm fully hidden", (2, 626.7),
                 textcoords="offset points", xytext=(10, -26), fontsize=7.5,
                 color=TEAL)
    axr.set_ylabel("effective TFLOP/s")
    axr.set_title("exp2: AG+GEMM vs chunks per remote shard")
    logx(axr, [1, 2, 4, 8, 16, 32], "S (chunks per shard)")
    axr.legend(frameon=False, fontsize=7.5, loc="lower left")

    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, "fig1_interference_granularity.png"))
    plt.close(fig)
    print("wrote figs/fig1_interference_granularity.png")


# ---------------------------------------------------------------------------
# fig2: exp3 v1 remote-write tax + v2 fused, both vs M
# ---------------------------------------------------------------------------
def fig2():
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(10.5, 3.9))

    d = [r for r in rows("exp3_epilogue_msweep.csv") if r["epi"] == "tma"]
    ms = sorted({int(r["m"]) for r in d})
    sd = lambda mode: [float(r["slowdown"]) for m in ms for r in d
                       if int(r["m"]) == m and r["mode"] == mode]
    for lbl, v, col, ls in (("remote (all of D remote)", sd("remote"), TEAL, "-"),
                            ("scatter (RS write pattern)", sd("scatter"), COPPER, "-"),
                            ("scatter-local (segmentation only)",
                             sd("scatter-local"), SLATE, "--")):
        axl.plot(ms, v, "o", ls=ls, color=col, ms=4, lw=1.8, label=lbl, zorder=3)
    axl.axhline(1.0, color=GREY, lw=1, ls="--", zorder=2)
    axl.annotate("4 sub-GEMMs x the 113 us single-wave floor;\n"
                 "the cost is segmentation, not remoteness",
                 (128, 3.93), textcoords="offset points", xytext=(8, -34),
                 fontsize=7.5, color=SLATE)
    axl.annotate("pinned by K, flat in M", (16384, 1.32),
                 textcoords="offset points", xytext=(-4, 16), ha="right",
                 fontsize=7.5, color=TEAL)
    axl.set_yscale("log")
    axl.set_ylim(0.92, 4.8)
    axl.yaxis.set_major_locator(FixedLocator([1, 1.5, 2, 3, 4]))
    axl.yaxis.set_major_formatter("{x:.1f}x")
    axl.yaxis.set_minor_formatter(NullFormatter())
    axl.set_ylabel("slowdown vs local epilogue (log)")
    axl.set_title("exp3 v1: epilogue remote write vs M (K=8192, tma)")
    logx(axl, ms, "M")
    axl.legend(frameon=False, fontsize=7.5, loc="upper right")

    d = rows("exp3_fused_msweep.csv") + \
        [r for r in rows("exp3_fused.csv") if int(r["k"]) == 8192]
    fms = sorted({int(r["m"]) for r in d})
    col_ = lambda k: [mean([float(r[k]) for r in d if int(r["m"]) == m])
                      for m in fms]
    for lbl, v, c in (("comm = fused/ctrl (p50)", col_("comm_sd_p50"), TEAL),
                      ("total = fused/base", col_("total_sd"), COPPER),
                      ("struct = ctrl/base", col_("struct_sd"), SLATE)):
        axr.plot(fms, v, "o-", color=c, ms=4, lw=1.8, label=lbl, zorder=3)
    axr.axhline(1.0, color=GREY, lw=1, ls="--", zorder=2)
    axr.annotate("latency-bound: pull only 23 GB/s", (512, 2.337),
                 textcoords="offset points", xytext=(12, -8), fontsize=7.5,
                 color=TEAL)
    axr.annotate("plateau 1.10, no upper bound in M\n"
                 "(per-tile residual flat at 82-92 ns)",
                 (32768, 1.099), textcoords="offset points", xytext=(-6, 30),
                 ha="right", fontsize=7.5, color=TEAL,
                 arrowprops=dict(arrowstyle="->", color=TEAL, lw=0.9,
                                 shrinkA=0, shrinkB=3))
    axr.text(600, 0.93, "struct ~ 1.00 everywhere: the restructuring is free",
             fontsize=7.5, color=SLATE)
    axr.set_yscale("log")
    axr.set_ylim(0.90, 2.7)
    axr.yaxis.set_major_locator(FixedLocator([1, 1.5, 2, 2.5]))
    axr.yaxis.set_major_formatter("{x:.1f}x")
    axr.yaxis.set_minor_formatter(NullFormatter())
    axr.set_title("exp3 v2: fused GEMM+RS vs M (K=8192, 4 ranks)")
    logx(axr, fms, "M")
    axr.legend(frameon=False, fontsize=7.5, loc="upper right")

    fig.savefig(os.path.join(OUT, "fig2_epilogue_fused.png"))
    plt.close(fig)
    print("wrote figs/fig2_epilogue_fused.png")


# ---------------------------------------------------------------------------
# fig3: the clock, i.e. why this bundle exists
# ---------------------------------------------------------------------------
def fig3():
    import re
    fig, ax = plt.subplots(figsize=(10.5, 3.4))
    for tag, path, lock, col in (
            ("nominal lock 1830 MHz (20260812_080039)",
             os.path.join(HERE, "comm_comp", "..", "..", "..", "tmp",
                          "results_20260812_080039"), 1830, RED),
            ("held lock 1305 MHz (this bundle)", RES, 1305, TEAL)):
        p = os.path.join(path, "gpu_trace.csv")
        if not os.path.exists(p):
            continue
        ep = {}
        for line in io.open(os.path.join(path, "manifest.txt"), encoding="utf-8"):
            m = re.search(r"^\s+(\S+) epoch_start=([0-9.]+)", line)
            if m:
                ep[m.group(1)] = float(m.group(2))
        S = []
        for line in io.open(p, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            ts, _, rest = line.partition(",")
            try:
                t = float(ts)
            except ValueError:
                continue
            for c in rest.split(";"):
                f = [x.strip() for x in c.split(",")]
                if len(f) >= 8 and f[0] == "0":
                    try:
                        S.append((t, int(f[1]), float(f[3])))
                    except ValueError:
                        pass
        names = sorted(ep, key=lambda k: ep[k])
        xs, ys = [], []
        for i, n in enumerate(names):
            if not n.startswith("exp3_fused_m"):
                continue
            t0 = ep[n]
            t1 = ep[names[i + 1]] if i + 1 < len(names) else S[-1][0]
            sm = sorted(s for t, s, pw in S if t0 <= t < t1 and pw > 300)
            if len(sm) < 3:
                continue
            xs.append(int(n.replace("exp3_fused_m", "")))
            ys.append(sm[len(sm) // 2])
        if xs:
            ax.plot(xs, ys, "o-", color=col, ms=5, lw=1.9, label=tag, zorder=3)
            ax.axhline(lock, color=col, lw=1, ls=":", alpha=0.6)
    ax.annotate("longer shape -> longer under load -> capped harder.\n"
                "Part of what the old M-sweep measured was the clock.",
                (2048, 1455), textcoords="offset points", xytext=(24, 8),
                fontsize=8, color=RED)
    ax.set_ylabel("median SM clock under load (MHz)")
    ax.set_title("Why this bundle exists: `-lgc 1830` is a request, not a lock")
    logx(ax, [512, 1024, 2048, 4096, 8192, 16384, 32768], "M (fused GEMM+RS sweep)")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.savefig(os.path.join(OUT, "fig3_clock.png"))
    plt.close(fig)
    print("wrote figs/fig3_clock.png")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
