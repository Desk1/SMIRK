#!/usr/bin/env python3
"""
Generate a self-contained HTML evaluation report from SMIRK attack results.

Usage:
    python smirk/scripts/generate_report.py
    python smirk/scripts/generate_report.py --output-dir /path/to/output

Reads from ./output (or --output-dir) and writes output/attacks/report.html.
No model loading required — reads pre-computed .pt files and evaluation_results.csv.
"""

import argparse
import base64
import csv
import io
import sys
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parent.parent.parent


def _resolve_output_dir(cli_arg: str | None) -> Path:
    if cli_arg:
        return Path(cli_arg).resolve()
    return _REPO_ROOT / "output"


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _tensor_to_png_b64(img: torch.Tensor) -> str:
    """Convert a [C,H,W] or [1,C,H,W] float [0,1] tensor to a base64 PNG string."""
    if img.ndim == 4:
        img = img.squeeze(0)
    arr = (img.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_csv(csv_path: Path) -> dict[tuple, dict]:
    """Load evaluation_results.csv → {(target_label_int, filename): row_dict}."""
    if not csv_path.exists():
        return {}
    with open(csv_path) as f:
        return {
            (int(r["target_label"]), r["result_file"]): r
            for r in csv.DictReader(f)
        }


_STAGE_ORDER = ["original.pt", "0.pt", "final.pt"]
_STAGE_LABELS = {
    "original.pt": "White-box Elite (reference)",
    "0.pt":        "Black-box Initial",
    "final.pt":    "Black-box Final",
}


def _load_targets(output_dir: Path, csv_results: dict) -> list[dict]:
    blackbox_dir = output_dir / "attacks" / "blackbox"
    whitebox_dir = output_dir / "attacks" / "whitebox"

    if not blackbox_dir.exists():
        return []

    targets = []
    for target_dir in sorted(blackbox_dir.iterdir()):
        if not target_dir.is_dir() or not target_dir.name.startswith("target_"):
            continue

        label = int(target_dir.name.split("_")[1])

        # White-box elite from the whitebox output dir
        wb_path = whitebox_dir / f"target_{label}" / "elite.pt"
        wb = None
        if wb_path.exists():
            d = torch.load(wb_path, map_location="cpu", weights_only=False)
            wb = {
                "fitness_score": d.fitness_score,
                "image_b64":     _tensor_to_png_b64(d.generated_image),
            }

        # Black-box result stages
        stages = []
        for fname in _STAGE_ORDER:
            pt = target_dir / fname
            if not pt.exists():
                continue
            d = torch.load(pt, map_location="cpu", weights_only=False)
            row = csv_results.get((label, fname), {})
            stages.append({
                "name":          fname,
                "label":         _STAGE_LABELS[fname],
                "asr":           int(row.get("asr", 0)),
                "fitness_score": d.fitness_score,
                "image_b64":     _tensor_to_png_b64(d.generated_image),
            })

        targets.append({"label": label, "whitebox_elite": wb, "stages": stages})

    return targets


def _compute_summary(targets: list, csv_results: dict) -> dict:
    rows = list(csv_results.values())
    total = len(rows)
    successful = sum(1 for r in rows if int(r.get("asr", 0)) == 1)
    final_hits = sum(
        1 for t in targets
        if any(s["name"] == "final.pt" and s["asr"] == 1 for s in t["stages"])
    )
    fitness_vals = [
        float(r["fitness_score"]) for r in rows
        if r.get("fitness_score") not in (None, "N/A", "")
    ]
    return {
        "n_targets":          len(targets),
        "total_results":      total,
        "successful_attacks": successful,
        "asr_rate":           successful / total if total > 0 else 0.0,
        "final_asr_rate":     final_hits / len(targets) if targets else 0.0,
        "best_fitness":       max(fitness_vals) if fitness_vals else None,
        "worst_fitness":      min(fitness_vals) if fitness_vals else None,
        "generated_at":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

_CSS = """
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
  background: #f4f6fb;
  color: #1a1a2e;
  line-height: 1.5;
}
a { color: inherit; }

/* ---- Header ---- */
header {
  background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
  color: #fff;
  padding: 2.5rem 2rem;
  border-bottom: 3px solid #0f3460;
}
header h1 { font-size: 1.8rem; font-weight: 700; letter-spacing: -0.02em; }
header .subtitle { opacity: 0.6; font-size: 0.88rem; margin-top: 0.3rem; }

/* ---- Layout ---- */
.container { max-width: 1100px; margin: 2rem auto; padding: 0 1.5rem; }
h2.section-heading {
  font-size: 0.75rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: #666;
  margin-bottom: 1rem;
}

/* ---- Summary cards ---- */
.summary-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 1rem;
  margin-bottom: 2.5rem;
}
.stat-card {
  background: #fff;
  border-radius: 10px;
  padding: 1.25rem 1rem;
  box-shadow: 0 1px 6px rgba(0,0,0,.07);
  text-align: center;
}
.stat-card .value {
  font-size: 1.9rem;
  font-weight: 800;
  color: #0f3460;
  line-height: 1;
}
.stat-card .label {
  font-size: 0.72rem;
  color: #888;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  margin-top: 0.4rem;
}

/* ---- Target sections ---- */
.target-section {
  background: #fff;
  border-radius: 12px;
  padding: 1.75rem;
  margin-bottom: 1.5rem;
  box-shadow: 0 1px 6px rgba(0,0,0,.07);
}
.target-section h3 {
  font-size: 1.05rem;
  font-weight: 700;
  margin-bottom: 1.25rem;
  padding-bottom: 0.6rem;
  border-bottom: 2px solid #eef0f6;
  color: #0f3460;
}

/* ---- Image grid ---- */
.stages-grid { display: flex; gap: 1.5rem; flex-wrap: wrap; align-items: flex-start; }
.stage-card { flex: 1; min-width: 170px; max-width: 260px; }
.stage-card img {
  width: 100%;
  border-radius: 8px;
  border: 1px solid #e4e7f0;
  display: block;
  background: #f0f0f0;
}
.stage-label {
  font-size: 0.80rem;
  font-weight: 600;
  color: #333;
  margin-top: 0.6rem;
  text-align: center;
}
.stage-meta {
  margin-top: 0.35rem;
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 0.5rem;
  flex-wrap: wrap;
}
.fitness {
  font-size: 0.73rem;
  color: #777;
  font-family: 'Courier New', monospace;
}

/* ---- Badges ---- */
.badge {
  font-size: 0.68rem;
  font-weight: 700;
  padding: 0.15rem 0.55rem;
  border-radius: 20px;
  letter-spacing: 0.03em;
}
.badge-success { background: #d1fae5; color: #065f46; }
.badge-fail    { background: #fee2e2; color: #991b1b; }

/* ---- Divider between wb and bb ---- */
.divider {
  width: 1px;
  background: #e4e7f0;
  min-height: 180px;
  align-self: stretch;
  flex-shrink: 0;
  margin: 0 0.25rem;
}

/* ---- Results table ---- */
.results-section {
  background: #fff;
  border-radius: 12px;
  padding: 1.75rem;
  margin-bottom: 1.5rem;
  box-shadow: 0 1px 6px rgba(0,0,0,.07);
}
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.85rem;
}
th {
  text-align: left;
  padding: 0.6rem 0.9rem;
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: #666;
  border-bottom: 2px solid #eef0f6;
}
td {
  padding: 0.6rem 0.9rem;
  border-bottom: 1px solid #f0f2f8;
  font-family: 'Courier New', monospace;
  font-size: 0.82rem;
}
tr:last-child td { border-bottom: none; }
tr:hover td { background: #f9fafb; }
td.target-col { font-family: inherit; font-weight: 600; color: #0f3460; }
td.file-col   { color: #555; }

/* ---- Footer ---- */
footer {
  text-align: center;
  font-size: 0.75rem;
  color: #aaa;
  padding: 2rem;
}
"""


def _render_stage_card(s: dict, show_asr: bool = True) -> str:
    badge = ""
    if show_asr:
        if s["asr"]:
            badge = '<span class="badge badge-success">ASR ✓</span>'
        else:
            badge = '<span class="badge badge-fail">ASR ✗</span>'
    return f"""
        <div class="stage-card">
          <img src="data:image/png;base64,{s['image_b64']}" alt="{s['label']}">
          <div class="stage-label">{s['label']}</div>
          <div class="stage-meta">
            {badge}
            <span class="fitness">{s['fitness_score']:.4f}</span>
          </div>
        </div>"""


def _render_html(summary: dict, targets: list, csv_results: dict) -> str:
    # ---- Summary cards ----
    asr_pct = f"{summary['asr_rate'] * 100:.1f}%"
    final_pct = f"{summary['final_asr_rate'] * 100:.1f}%"
    best_fit = f"{summary['best_fitness']:.4f}" if summary["best_fitness"] is not None else "—"
    worst_fit = f"{summary['worst_fitness']:.4f}" if summary["worst_fitness"] is not None else "—"

    summary_html = f"""
      <div class="summary-grid">
        <div class="stat-card"><div class="value">{summary['n_targets']}</div><div class="label">Targets</div></div>
        <div class="stat-card"><div class="value">{summary['successful_attacks']}/{summary['total_results']}</div><div class="label">Successful&nbsp;Attacks</div></div>
        <div class="stat-card"><div class="value">{asr_pct}</div><div class="label">Overall ASR</div></div>
        <div class="stat-card"><div class="value">{final_pct}</div><div class="label">Final&nbsp;Result ASR</div></div>
        <div class="stat-card"><div class="value">{best_fit}</div><div class="label">Best Fitness</div></div>
        <div class="stat-card"><div class="value">{worst_fit}</div><div class="label">Worst Fitness</div></div>
      </div>"""

    # ---- Per-target sections ----
    target_sections = []
    for t in targets:
        cards = ""
        if t["whitebox_elite"]:
            wb = t["whitebox_elite"]
            cards += f"""
        <div class="stage-card">
          <img src="data:image/png;base64,{wb['image_b64']}" alt="White-box Elite">
          <div class="stage-label">White-box Elite</div>
          <div class="stage-meta">
            <span class="fitness">{wb['fitness_score']:.4f}</span>
          </div>
        </div>
        <div class="divider"></div>"""

        for s in t["stages"]:
            # Skip original.pt in the image grid — it duplicates the whitebox elite
            if s["name"] == "original.pt" and t["whitebox_elite"]:
                continue
            cards += _render_stage_card(s)

        target_sections.append(f"""
      <div class="target-section">
        <h3>Target Class {t['label']}</h3>
        <div class="stages-grid">{cards}
        </div>
      </div>""")

    # ---- Raw results table ----
    rows_html = ""
    for (label, fname), row in sorted(csv_results.items()):
        asr_val = int(row.get("asr", 0))
        fit_val = row.get("fitness_score", "N/A")
        try:
            fit_disp = f"{float(fit_val):.6f}"
        except (ValueError, TypeError):
            fit_disp = fit_val
        badge = (
            '<span class="badge badge-success">1</span>'
            if asr_val
            else '<span class="badge badge-fail">0</span>'
        )
        rows_html += f"""
          <tr>
            <td class="target-col">{label}</td>
            <td class="file-col">{fname}</td>
            <td>{badge}</td>
            <td>{fit_disp}</td>
          </tr>"""

    table_html = f"""
      <div class="results-section">
        <h2 class="section-heading">All Evaluation Results</h2>
        <table>
          <thead><tr><th>Target</th><th>File</th><th>ASR</th><th>Fitness Score</th></tr></thead>
          <tbody>{rows_html}
          </tbody>
        </table>
      </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SMIRK Evaluation Report</title>
  <style>{_CSS}</style>
</head>
<body>
<header>
  <h1>SMIRK Evaluation Report</h1>
  <div class="subtitle">Model Inversion Attack Results &mdash; Generated {summary['generated_at']}</div>
</header>
<div class="container">
  <h2 class="section-heading">Summary</h2>
  {summary_html}
  <h2 class="section-heading">Generated Images</h2>
  {"".join(target_sections)}
  {table_html}
</div>
<footer>SMIRK &mdash; Surrogate Model Inversion via Restricted Knowledge</footer>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def generate_report(output_dir: Path) -> Path:
    csv_path = output_dir / "attacks" / "evaluation_results.csv"
    report_path = output_dir / "attacks" / "report.html"

    print(f"[report] Loading CSV from {csv_path}")
    csv_results = _load_csv(csv_path)

    print(f"[report] Loading attack results from {output_dir / 'attacks'}")
    targets = _load_targets(output_dir, csv_results)

    if not targets:
        print("[report] No attack results found — nothing to render.", file=sys.stderr)
        sys.exit(1)

    summary = _compute_summary(targets, csv_results)
    html = _render_html(summary, targets, csv_results)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(html, encoding="utf-8")
    print(f"[report] Targets: {summary['n_targets']}  |  ASR: {summary['asr_rate']:.2%}")
    print(f"[report] Written → {report_path}")
    print(f"[report] Open in browser: file://{report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(description="Generate HTML evaluation report for SMIRK results.")
    parser.add_argument("--output-dir", default=None, help="Path to the output directory (default: ./output)")
    args = parser.parse_args()

    output_dir = _resolve_output_dir(args.output_dir)
    generate_report(output_dir)


if __name__ == "__main__":
    main()
