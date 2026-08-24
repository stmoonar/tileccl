#!/usr/bin/env python3

"""Plot only the corrected Fig2-left dependent-compute slowdown."""

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


LABELS = {
    "ce-aggregate": "CE aggregate",
    "ce-panelized": "CE 16-KiB copies",
    "tma": "TMA",
}
COLORS = {
    "ce-aggregate": "#d55e00",
    "ce-panelized": "#cc79a7",
    "tma": "#0072b2",
}
MARKERS = {"ce-aggregate": "o", "ce-panelized": "s", "tma": "^"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input",
        type=Path,
        help="exp4_fig2_left.csv or the extracted result directory",
    )
    args = parser.parse_args()
    csv = args.input / "exp4_fig2_left.csv" if args.input.is_dir() else args.input
    out_dir = csv.parent / "figs"
    out_dir.mkdir(exist_ok=True)

    data = pd.read_csv(csv)
    bad = data[(data["verify"] != "ok") | (data["err_count"] != 0)]
    if not bad.empty:
        raise SystemExit(f"refusing to plot {len(bad)} invalid rows")
    if set(data["mode"]) != {"compute-only", "fused"}:
        raise SystemExit(f"unexpected modes: {sorted(set(data['mode']))}")
    if data["drift_pct"].abs().max() > 2.0:
        raise SystemExit("paired compute baseline drift exceeds 2%")
    fused = data[(data["mode"] == "fused") & (data["rank"] >= 0)]
    fused = (
        fused.groupby(["variant", "panel_h"], as_index=False)
        .agg(slowdown=("slowdown", "max"))
        .sort_values(["variant", "panel_h"])
    )
    expected = {
        (variant, h)
        for variant in LABELS
        for h in range(1, 8193)
        if h & (h - 1) == 0
    }
    observed = set(zip(fused["variant"], fused["panel_h"]))
    if observed != expected:
        raise SystemExit(f"missing/unexpected sweep points: {sorted(expected ^ observed)}")

    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 220,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    for variant in LABELS:
        rows = fused[fused["variant"] == variant]
        ax.plot(
            rows["panel_h"] * 16,
            rows["slowdown"],
            label=LABELS[variant],
            color=COLORS[variant],
            marker=MARKERS[variant],
            linewidth=2,
        )
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ticks = [16 * (1 << power) for power in range(14)]
    ax.set_xticks(ticks)
    ax.set_xticklabels(
        [str(x) if x < 1024 else f"{x // 1024}M" for x in ticks],
        rotation=35,
        ha="right",
    )
    ax.set_xlabel("Bytes per ready flag (KiB below 1 MiB, then MiB)")
    ax.set_ylabel("Fused compute completion / compute-only")
    ax.set_title("Dependent-compute slowdown (worst rank)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "fig2_left_compute_slowdown.png", bbox_inches="tight")
    fig.savefig(out_dir / "fig2_left_compute_slowdown.pdf", bbox_inches="tight")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
