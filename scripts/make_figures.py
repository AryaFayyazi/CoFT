#!/usr/bin/env python3
"""Figures 3 and 4 -- the validation Pareto sweeps over lambda and alpha.

Reads ``results/<model>/sweeps.json`` (written by ``scripts/run_sweep.py``) and
writes ``results/<model>/fig3_lambda.{pdf,png}`` and ``fig4_alpha.{pdf,png}``,
plus a coverage-diagnostic panel corresponding to Fig. 7(c-d).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, RED, GREY = "#2b6cb0", "#c53030", "#4a5568"


def _pareto_panel(ax, points, knob: str, chosen: float, title: str):
    xs = [p["bias_avg"] for p in points]
    ys = [p["utility_avg"] for p in points]
    ax.plot(xs, ys, "-o", color=BLUE, markersize=5, linewidth=1.4, label=f"{knob} sweep (val)")
    for p, x, y in zip(points, xs, ys):
        ax.annotate(f"{p[knob]:g}", (x, y), textcoords="offset points", xytext=(4, 5), fontsize=8)
    for p, x, y in zip(points, xs, ys):
        if abs(p[knob] - chosen) < 1e-9:
            ax.plot([x], [y], marker="*", color=RED, markersize=16, linestyle="none",
                    label="chosen knee", zorder=5)
            break
    ax.set_xlabel(r"BiasAvg $\downarrow$")
    ax.set_ylabel(r"UtilityAvg $\uparrow$")
    ax.set_title(title)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(fontsize=8, loc="best")


def _coverage_panel(ax, points, chosen: float):
    alphas = [p["alpha"] for p in points]
    miscov = [p["empirical_miscoverage"] for p in points]
    sizes = [p["normalized_set_size"] for p in points]
    ax.plot(alphas, miscov, "-o", color=BLUE, markersize=5, label="empirical miscoverage")
    ax.plot(alphas, alphas, "--", color=GREY, linewidth=1.2, label=r"ideal target $\alpha$")
    ax.plot(alphas, sizes, ":s", color=RED, markersize=4, label=r"normalized $|\mathcal{C}_t|$")
    ax.axvline(chosen, color=RED, alpha=0.25, linewidth=1.0)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"miscoverage / normalized $|\mathcal{C}_t|$")
    ax.set_title(r"Calibration diagnostics vs. $\alpha$")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(fontsize=8, loc="best")


def _save(fig, stem: Path):
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  wrote {stem}.pdf / .png")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results")
    args = ap.parse_args()

    root = Path(args.results)
    found = 0
    for sweep_file in sorted(root.glob("*/sweeps.json")):
        payload = json.loads(sweep_file.read_text())
        label = payload["model"]["label"]
        out_dir = sweep_file.parent
        found += 1

        if "lambda_sweep" in payload:
            s = payload["lambda_sweep"]
            fig, ax = plt.subplots(figsize=(4.2, 3.2))
            _pareto_panel(ax, s["points"], "lambda", s["chosen"],
                          rf"Ablation: $\lambda$ -- {label}")
            _save(fig, out_dir / "fig3_lambda")

        if "alpha_sweep" in payload:
            s = payload["alpha_sweep"]
            fig, ax = plt.subplots(figsize=(4.2, 3.2))
            _pareto_panel(ax, s["points"], "alpha", s["chosen"],
                          rf"Ablation: $\alpha$ -- {label}")
            _save(fig, out_dir / "fig4_alpha")

            fig, ax = plt.subplots(figsize=(4.2, 3.2))
            _coverage_panel(ax, s["points"], s["chosen"])
            _save(fig, out_dir / "fig4b_coverage")

    if not found:
        print(f"no sweeps.json under {root}; run scripts/run_sweep.py first")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
