"""Recompute exact ST2 terms and descriptive results from saved scalar scores.

Run with Python 3.10+; only the standard library is required.
"""
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics

from scripts.st2 import exact_st2

ROOT = Path(__file__).resolve().parent


def read_table(name):
    with (ROOT / "data" / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def recompute():
    groups = defaultdict(dict)
    for row in read_table("coalition_scores.csv"):
        groups[row["photo"], row["layout"]][row["coalition"]] = float(row["target_score"])
    results = {key: exact_st2(values) for key, values in groups.items()}
    expected = read_table("expected_terms.csv")
    errors = []
    for row in expected:
        result = results[row["photo"], row["layout"]]
        terms = result["pairs"] if ":" in row["term"] else result["singletons"]
        errors.append(abs(terms[row["term"]] - float(row["value"])))
    photos = read_table("photo_scores.csv")
    summary = {"photos": len(photos), "coalition_scores": sum(map(len, groups.values())),
               "games": len(results), "terms_checked": len(expected),
               "max_term_error_vs_saved": max(errors),
               "max_efficiency_residual": max(abs(r["efficiency_error"]) for r in results.values()),
               "all_efficiency_checks_pass": all(r["efficiency_passed"] for r in results.values()),
               "historical_visual_qc": {state: sum(p["historical_visual_qc"] == state for p in photos)
                                        for state in ("failed", "pending")}, "layouts": {}}
    for layout in ("L1", "L2"):
        deltas = [float(row[layout]) - float(row["L0"]) for row in photos]
        summary["layouts"][layout] = {"decreased": sum(v < 0 for v in deltas),
                                       "increased": sum(v > 0 for v in deltas),
                                       "mean_delta": statistics.mean(deltas),
                                       "median_delta": statistics.median(deltas)}
    example = "photo_12"
    summary["example"] = {"photo": example, "head_wing_ST2": {
        layout: results[example, layout]["pairs"]["P1:P3"] for layout in ("L0", "L1", "L2")}}
    return summary, photos, results


def plot(photos, destination):
    """Draw a numerical plot without image assets or plotting dependencies."""
    left, top, width, row_height = 155, 150, 625, 24
    x = lambda value: left + (value + 36) / 44 * width
    elements = ['<svg xmlns="http://www.w3.org/2000/svg" width="860" height="740" viewBox="0 0 860 740">',
                '<rect width="860" height="740" fill="#fafafa"/>',
                '<g font-family="Arial, sans-serif" fill="#202a35">',
                '<text x="40" y="42" font-size="24" font-weight="bold">Moving parts changes the evidence</text>',
                '<text x="40" y="71" font-size="14">20 source photos · fixed target class · raw score change from L0</text>',
                '<text x="40" y="101" font-size="14" fill="#006f9b">● L1: head ↔ tail + separation</text>',
                '<text x="405" y="101" font-size="14" fill="#b35200">● L2: wing ↔ tail + separation</text>']
    for value in (-30, -20, -10, 0):
        elements += [f'<line x1="{x(value):.1f}" y1="125" x2="{x(value):.1f}" y2="635" stroke="{"#536273" if value == 0 else "#ddd"}"/>',
                     f'<text x="{x(value):.1f}" y="657" text-anchor="middle" font-size="13">{value}</text>']
    for index, row in enumerate(photos):
        y = top + index * row_height
        values = [float(row[layout]) - float(row["L0"]) for layout in ("L1", "L2")]
        elements += [f'<text x="130" y="{y+4}" text-anchor="end" font-size="12">{row["photo"]}</text>',
                     f'<line x1="{x(values[0]):.1f}" y1="{y}" x2="{x(values[1]):.1f}" y2="{y}" stroke="#c8cdd1"/>']
        for value, color in zip(values, ("#006f9b", "#b35200")):
            elements.append(f'<circle cx="{x(value):.1f}" cy="{y}" r="4" fill="{color}"><title>{row["photo"]}: {value:.6f}</title></circle>')
    elements += ['<text x="40" y="695" font-size="13">Exploratory results: historical visual QC was 13 failed / 7 pending.</text>',
                 '<text x="40" y="717" font-size="13">Joint position + separation intervention; scores are not probabilities.</text>', '</g></svg>']
    destination.write_text("\n".join(elements) + "\n")


if __name__ == "__main__":
    summary, photos, _ = recompute()
    (ROOT / "results_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot(photos, ROOT / "score_changes.svg")
    print(json.dumps(summary, indent=2))
    print("Wrote results_summary.json and score_changes.svg beside this script.")
