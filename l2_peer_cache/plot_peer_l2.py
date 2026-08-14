#!/usr/bin/env python3
"""Plot a peer_l2_probe result bundle.

Reads the CSVs that run_all.sh drops into results_<stamp>/ and draws the three
panels the conclusion is read from:

  (a) read bandwidth vs working set   -- the local curve must step at L2,
                                         the peer curve must not
  (b) chase latency  vs working set   -- same claim on the latency axis
  (c) NVLink bytes / bytes asked for  -- ~1.0 on peer, ~0 on local

Solid = default build (L1 on), dashed = peer_l2_probe_nol1 (-dlcm=cg).

Usage:
  python plot_peer_l2.py results_20260814_101500 [--out figs] [--dpi 200]
"""

import argparse
import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# same identity convention as the other plot scripts in this repo: color
# follows the thing being measured, linestyle follows the build variant
C_LOCAL, C_PEER = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
STYLE = {"ca": "-", "cg": "--"}
LABEL = {"ca": "L1 on", "cg": "L1 bypassed"}

plt.rcParams.update({
    "font.size": 9,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "figure.facecolor": "white",
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,  # embed TrueType, no Type-3
    "ps.fonttype": 42,
})


def read_csv(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return []
    with open(path, newline="") as f:
        return [r for r in csv.DictReader(f) if r.get("build")]


def load(bundle):
    """-> (bw rows, lat rows, wire rows, L2 bytes or None)"""
    bw, lat, wire = [], [], []
    for build in ("ca", "cg"):
        bw += read_csv(os.path.join(bundle, f"{build}_bw.csv"))
        lat += read_csv(os.path.join(bundle, f"{build}_lat.csv"))
        wire += read_csv(os.path.join(bundle, f"{build}_wire.csv"))
    wire += read_csv(os.path.join(bundle, "bigbuf_wire.csv"))
    l2 = None
    for r in bw:
        frac = float(r["frac_l2"])
        if frac > 0:
            l2 = float(r["bytes"]) / frac
            break
    return bw, lat, wire, l2


def series(rows, xkey, ykey):
    """(build, target) -> sorted [(x, y)]"""
    out = defaultdict(list)
    for r in rows:
        out[(r["build"], r["target"])].append(
            (float(r[xkey]), float(r[ykey])))
    return {k: sorted(v) for k, v in out.items()}


def curve_panel(ax, data, l2, ylabel, title, logy):
    for (build, target), pts in sorted(data.items()):
        if not pts:
            continue
        xs = [p[0] / 2**20 for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, STYLE.get(build, "-"),
                color=C_PEER if target == "peer" else C_LOCAL,
                marker="o", markersize=3, linewidth=1.4,
                label=f"{target}, {LABEL.get(build, build)}")
    if l2:
        ax.axvline(l2 / 2**20, color=MUTED, linewidth=1, linestyle=":")
        ax.annotate("L2 capacity", xy=(l2 / 2**20, 0.99),
                    xycoords=("data", "axes fraction"), color=MUTED,
                    fontsize=8, ha="right", va="top", xytext=(-3, 0),
                    textcoords="offset points")
    ax.set_xscale("log", base=2)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("working set (MiB)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9, loc="left")
    ax.grid(True, which="major", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def wire_panel(ax, rows):
    bars = []
    for r in rows:
        if r.get("counters_ok") != "1":
            continue
        bars.append((r["target"], float(r["bytes"]) / 2**20,
                     float(r["rx_over_asked"]), r["build"]))
    if not bars:
        ax.text(0.5, 0.5, "no NVLink counters in this bundle",
                ha="center", va="center", color=MUTED, transform=ax.transAxes)
        ax.set_axis_off()
        return
    # color = target (as in the other panels), hatch = L1-bypassed build
    bars.sort(key=lambda b: (b[0], b[3], b[1]))
    vals = [v for _, _, v, _ in bars]
    ax.bar(range(len(bars)), vals, width=0.7,
           color=[C_PEER if t == "peer" else C_LOCAL for t, _, _, _ in bars],
           hatch=["//" if b == "cg" else "" for _, _, _, b in bars],
           edgecolor="white", linewidth=0)
    ax.axhline(1.0, color=MUTED, linewidth=1, linestyle=":")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.03, f"{v:.2f}", ha="center", fontsize=7, color=INK)
    ax.set_xticks(range(len(bars)))
    ax.set_xticklabels([f"{s:.0f}M" for _, s, _, _ in bars], fontsize=7)
    ax.set_ylabel("NVLink RX bytes / bytes asked for")
    ax.set_ylim(0, max(1.25, max(vals) * 1.3))
    ax.set_title("(c) what actually crossed the link\n"
                 "    hatched = L1 bypassed; 1.0 = nothing cached locally",
                 fontsize=8, loc="left")
    ax.set_xlabel("working set")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c, label=t)
                       for t, c in (("local", C_LOCAL), ("peer", C_PEER))],
              frameon=False, fontsize=8, loc="upper left")
    ax.grid(True, axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle", help="results_<stamp>/ directory")
    ap.add_argument("--out", default=None, help="output dir (default <bundle>/figs)")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    bw, lat, wire, l2 = load(args.bundle)
    if not bw and not lat and not wire:
        raise SystemExit(f"no probe CSVs found in {args.bundle}")
    out = args.out or os.path.join(args.bundle, "figs")
    os.makedirs(out, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    curve_panel(axes[0], series(bw, "bytes", "gbps"), l2,
                "achieved read GB/s",
                "(a) bandwidth: the local step is the control", logy=True)
    curve_panel(axes[1], series(lat, "bytes", "ns_per_hop"), l2,
                "dependent-load latency (ns)",
                "(b) latency: same question, other axis", logy=True)
    wire_panel(axes[2], wire)
    axes[0].legend(frameon=False, fontsize=8, loc="best")

    name = os.path.basename(os.path.normpath(args.bundle)) or "peer_l2"
    fig.suptitle(f"Is peer memory cached in the requester's L2?   ({name})",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext in ("png", "pdf"):
        path = os.path.join(out, f"peer_l2.{ext}")
        fig.savefig(path, dpi=args.dpi)
        print("wrote", path)


if __name__ == "__main__":
    main()
