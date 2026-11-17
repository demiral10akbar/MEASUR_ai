"""Regenerate the cross-system comparison figures (manuscript Figures 9 and 10) from the data.

Usage:  python scripts/make_figures.py
Writes: results/fig_savings_comparison.png, results/fig_cost_vs_payback.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rotaeq.analysis import scenario_kpis

OUT = Path(__file__).resolve().parents[1] / "results"
LABELS = {"A-AI": "A\nCompressor", "B-S1": "B\nSc.1 Leak", "B-S2": "B\nSc.2 Pressure",
          "B-S3": "B\nSc.3 Schedule", "C-AI": "C\nFan", "D-AI": "D\nPumps", "E-AI": "E\nMixer"}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    df = scenario_kpis()

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar([LABELS[s] for s in df["scenario"]], df["savings_pct"], color="#2E75B6")
    for b, v in zip(bars, df["savings_pct"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.1f}%", ha="center", fontsize=11)
    ax.axhline(df["savings_pct"].mean(), ls="--", color="grey", lw=1)
    ax.set_ylabel("Predicted energy saving (%) (S)", fontsize=12)
    ax.set_ylim(0, df["savings_pct"].max() * 1.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_savings_comparison.png", dpi=200)
    plt.close(fig)

    d = df[df["implementation_cost_usd"] > 0]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sizes = 2000 * d["cost_saving_usd"] / d["cost_saving_usd"].max() + 60
    ax.scatter(d["implementation_cost_usd"], d["payback_months"], s=sizes, alpha=0.55, color="#ED7D31",
               edgecolor="k")
    for _, r in d.iterrows():
        ax.annotate(r["scenario"], (r["implementation_cost_usd"], r["payback_months"]),
                    xytext=(8, 6), textcoords="offset points", fontsize=11)
    ax.axhline(12, ls="--", color="#C00000", lw=1)
    ax.text(ax.get_xlim()[0], 12.2, "12-month threshold", color="#C00000", fontsize=10)
    ax.set_xlabel("Implementation cost (USD) (E)", fontsize=12)
    ax.set_ylabel("Simple payback period (months) (S,E)", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_cost_vs_payback.png", dpi=200)
    plt.close(fig)
    print("Figures written to", OUT)


if __name__ == "__main__":
    main()
