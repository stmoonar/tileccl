#!/usr/bin/env python3

"""Plot corrected Fig2-left mean, p50, and p95 compute slowdowns."""

import argparse
from pathlib import Path

import matplotlib as mpl
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import pandas as pd


VARIANTS = ("ce-aggregate", "tma")
COLORS = {
    "ce-aggregate": "#d55e00",
    "tma": "#0072b2",
}
MARKERS = {"ce-aggregate": "o", "tma": "^"}
STAT_COLUMNS = {
    "mean": "t_us_mean",
    "p50": "t_us_p50",
    "p95": "t_us_p95",
}
EXPECTED_WORLD = 4
EXPECTED_M = 65536
EXPECTED_K = 8192


def variant_labels(data: pd.DataFrame) -> dict:
    """Derive labels from the CSV so config changes cannot silently mislabel
    the figure; each variant must be internally consistent instead of
    matching a hardcoded expectation."""
    tma_comm_sms = sorted(set(data.loc[data["variant"] == "tma", "n_comm"]))
    if len(tma_comm_sms) != 1:
        raise SystemExit(f"mixed TMA n_comm values in one bundle: {tma_comm_sms}")
    streams = sorted(set(data.loc[data["variant"] == "ce-aggregate",
                                  "comm_streams"]))
    ce_label = (
        f"CE aggregate ({streams[0]} stream{'s' if streams[0] > 1 else ''})"
        if len(streams) == 1
        else f"CE aggregate (streams {'/'.join(map(str, streams))} by chunk)"
    )
    return {
        "ce-aggregate": ce_label,
        "tma": f"TMA ({tma_comm_sms[0]} comm SMs)",
    }


def plot_one(paired: pd.DataFrame, stat: str, out_dir: Path,
             labels: dict) -> None:
    column = STAT_COLUMNS[stat]
    value = f"slowdown_{stat}"
    paired = paired.copy()
    paired[value] = paired[f"{column}_fused"] / paired[f"{column}_base"]
    curves = (
        paired.groupby(["variant", "panel_h"], as_index=False)
        .agg(**{value: (value, "max")})
        .sort_values(["variant", "panel_h"])
    )

    fig = Figure(figsize=(8.2, 4.2))
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    for variant in VARIANTS:
        rows = curves[curves["variant"] == variant]
        ax.plot(
            rows["panel_h"] * 16,  # KiB: one physical panel is 16 KiB
            rows[value],
            label=labels[variant],
            color=COLORS[variant],
            marker=MARKERS[variant],
            linewidth=2,
        )
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ticks = [16 * (1 << power) for power in range(15)]
    ax.set_xticks(ticks)
    ax.set_xticklabels(
        [f"{x}K" if x < 1024 else f"{x // 1024}M" for x in ticks],
        rotation=35,
        ha="right",
    )
    ax.set_xlabel("Data released per ready flag")
    ax.set_ylabel(f"Fused {stat} / compute-only {stat}")
    ax.set_title(f"Dependent-compute slowdown: {stat} (worst rank)")
    ax.legend(frameon=False)
    fig.tight_layout()
    stem = out_dir / f"fig2_left_compute_slowdown_{stat}"
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.clear()


def main() -> None:
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
    # Older bundles may contain the retired CE 16-KiB-copy diagnostic.  It is
    # deliberately ignored rather than allowed to dominate the main figure.
    data = data[data["variant"].isin(VARIANTS)].copy()
    bad = data[(data["verify"] != "ok") | (data["err_count"] != 0)]
    if not bad.empty:
        raise SystemExit(f"refusing to plot {len(bad)} invalid rows")
    if set(data["mode"]) != {"compute-only", "fused"}:
        raise SystemExit(f"unexpected modes: {sorted(set(data['mode']))}")
    if data["drift_pct"].abs().max() > 2.0:
        raise SystemExit("paired compute baseline drift exceeds 2%")
    for column, expected_value in {
        "world": EXPECTED_WORLD,
        "m": EXPECTED_M,
        "k": EXPECTED_K,
    }.items():
        observed_values = set(data[column])
        if observed_values != {expected_value}:
            raise SystemExit(
                f"expected {column}={expected_value}, got {sorted(observed_values)}"
            )
    labels = variant_labels(data)

    keys = ["variant", "panel_h", "rank"]
    columns = keys + list(STAT_COLUMNS.values())
    base = data[(data["mode"] == "compute-only") & (data["rank"] >= 0)][columns]
    fused = data[(data["mode"] == "fused") & (data["rank"] >= 0)][columns]
    paired = fused.merge(
        base,
        on=keys,
        suffixes=("_fused", "_base"),
        validate="one_to_one",
    )
    if any(
        (paired[f"{column}_base"] <= 0).any()
        for column in STAT_COLUMNS.values()
    ):
        raise SystemExit("non-positive compute-only timing")

    expected = {
        (variant, h)
        for variant in VARIANTS
        for h in range(1, 16385)
        if h & (h - 1) == 0
    }
    observed = set(zip(paired["variant"], paired["panel_h"]))
    if observed != expected:
        raise SystemExit(f"missing/unexpected sweep points: {sorted(expected ^ observed)}")
    expected_rows = len(expected) * 4
    if len(paired) != expected_rows:
        raise SystemExit(f"expected {expected_rows} paired rank rows, got {len(paired)}")

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
    for stat in STAT_COLUMNS:
        plot_one(paired, stat, out_dir, labels)
    print(f"wrote mean/p50/p95 figures to {out_dir}")


if __name__ == "__main__":
    main()
