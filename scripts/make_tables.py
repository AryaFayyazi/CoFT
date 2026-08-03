#!/usr/bin/env python3
"""Render the JSON produced by ``run_*.py`` into Markdown and LaTeX tables.

    python scripts/make_tables.py --results results --out results/TABLES.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coft.registry import METHOD_LABELS  # noqa: E402

BIAS_COLUMNS = [
    ("SS", "v", 2),
    ("CP Acc", "^", 1),
    ("BBQ Bias", "v", 3),
    ("BOLD Tox", "v", 3),
    ("Utrecht DP", "v", 3),
    ("COMPAS Gap", "v", 3),
    ("Avg. Rank", "v", 1),
]
UTILITY_COLUMNS = [
    ("GSM8K", "^", 1),
    ("StrategyQA", "^", 1),
    ("ARC-easy", "^", 1),
    ("PIQA", "^", 1),
    ("PPL", "v", 1),
    ("MAUVE", "^", 2),
]


def _fmt(v: Optional[float], nd: int) -> str:
    if v is None or v != v:
        return "n/a"
    return f"{v:.{nd}f}"


def _md_table(headers: List[str], rows: List[List[str]], bold_row: Optional[int] = None) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for i, r in enumerate(rows):
        cells = [f"**{c}**" for c in r] if i == bold_row else r
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def _latex_table(headers, rows, caption, label, bold_row=None) -> str:
    align = "l" + "c" * (len(headers) - 1)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{align}}}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]
    for i, r in enumerate(rows):
        cells = [f"\\textbf{{{c}}}" for c in r] if i == bold_row else r
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def render_bias(payload: Dict) -> Dict[str, str]:
    table = payload["table"]
    methods = list(table)
    present = [c for c in BIAS_COLUMNS if any(c[0] in table[m] for m in methods)]
    headers = ["Method"] + [f"{c} {a}" for c, a, _ in present]
    rows = [
        [METHOD_LABELS.get(m, m)] + [_fmt(table[m].get(c), nd) for c, _, nd in present]
        for m in methods
    ]
    bold = methods.index("coft") if "coft" in methods else None
    label = payload["model"]["label"]
    return {
        "md": f"### Table 1 -- Bias results ({label})\n\n" + _md_table(headers, rows, bold),
        "tex": _latex_table(
            headers, rows,
            f"Bias results on {label}. Lower is better except CP Acc.",
            f"tab:bias-{payload['model']['key']}", bold,
        ),
    }


def render_utility(payload: Dict) -> Dict[str, str]:
    table = payload["table"]
    methods = list(table)
    present = [c for c in UTILITY_COLUMNS if any(c[0] in table[m] for m in methods)]
    headers = ["Method"] + [f"{c} {a}" for c, a, _ in present]
    rows = [
        [METHOD_LABELS.get(m, m)] + [_fmt(table[m].get(c), nd) for c, _, nd in present]
        for m in methods
    ]
    bold = methods.index("coft") if "coft" in methods else None
    label = payload["model"]["label"]
    return {
        "md": f"### Table 2 -- Utility & quality ({label})\n\n" + _md_table(headers, rows, bold),
        "tex": _latex_table(
            headers, rows,
            f"Utility and language quality on {label}.",
            f"tab:utility-{payload['model']['key']}", bold,
        ),
    }


def render_efficiency(payload: Dict) -> Dict[str, str]:
    table = payload["table"]
    headers = ["Method", "tok/s ^", "Overhead", "Peak Mem (GB)"]
    rows = []
    for m, r in table.items():
        oh = "--" if r.get("overhead_pct") is None else f"{r['overhead_pct']:.1f}%"
        rows.append([METHOD_LABELS.get(m, m), _fmt(r["tokens_per_sec"], 1), oh, _fmt(r["peak_mem_gb"], 1)])
    bold = list(table).index("coft") if "coft" in table else None
    label = payload["model"]["label"]
    return {
        "md": f"### Table 3 -- Efficiency ({label})\n\n" + _md_table(headers, rows, bold),
        "tex": _latex_table(
            headers, rows, f"Efficiency on {label}.", f"tab:eff-{payload['model']['key']}", bold
        ),
    }


def render_ablation(payload: Dict) -> Dict[str, str]:
    table = payload["table"]
    headers = ["Variant", "BiasAvg v", "UtilityAvg ^"]
    rows = [
        [METHOD_LABELS.get(k, k), _fmt(v["BiasAvg"], 3), _fmt(v["UtilityAvg"], 1)]
        for k, v in table.items()
    ]
    bold = list(table).index("coft_full") if "coft_full" in table else None
    label = payload["model"]["label"]
    return {
        "md": f"### Table 4 -- Ablations ({label})\n\n" + _md_table(headers, rows, bold),
        "tex": _latex_table(
            headers, rows, f"Component ablation on {label}.",
            f"tab:ablation-{payload['model']['key']}", bold,
        ),
    }


RENDERERS = {
    "table1_bias.json": render_bias,
    "table2_utility.json": render_utility,
    "table3_efficiency.json": render_efficiency,
    "table4_ablation.json": render_ablation,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/TABLES.md")
    ap.add_argument("--tex-out", default="results/tables.tex")
    args = ap.parse_args()

    root = Path(args.results)
    md_parts: List[str] = ["# COFT -- reproduced tables\n"]
    tex_parts: List[str] = ["% Generated by scripts/make_tables.py\n"]

    model_dirs = sorted(d for d in root.iterdir() if d.is_dir() and d.name != "cache")
    if not model_dirs:
        print(f"no model result directories under {root}")
        return 1

    for mdir in model_dirs:
        md_parts.append(f"\n## {mdir.name}\n")
        for fname, render in RENDERERS.items():
            f = mdir / fname
            if not f.exists():
                continue
            payload = json.loads(f.read_text())
            try:
                out = render(payload)
            except Exception as exc:  # keep going on partial results
                print(f"  [warn] {f}: {exc}")
                continue
            md_parts.append(out["md"] + "\n")
            tex_parts.append(out["tex"] + "\n")
            print(f"  rendered {f.relative_to(root)}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(md_parts))
    Path(args.tex_out).write_text("\n".join(tex_parts))
    print(f"\nwrote {args.out} and {args.tex_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
