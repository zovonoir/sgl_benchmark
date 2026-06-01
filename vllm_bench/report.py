"""Aggregation and summary reporting for vLLM benchmark runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def aggregate_case(raw_path: Path, meta_path: Path, agg_path: Path) -> dict[str, Any]:
    """Create an aggregated per-GPU JSON file for one benchmark result."""

    with open(raw_path, encoding="utf-8") as f:
        raw = json.load(f)
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    tp = int(meta.get("tp", 1)) or 1
    total_tput = float(raw.get("total_token_throughput", 0.0))
    output_tput = float(raw.get("output_throughput", 0.0))
    input_tput = total_tput - output_tput

    agg = {
        "case_name": meta["case_name"],
        "model_path": meta["model_path"],
        "model_prefix": meta["model_prefix"],
        "precision": meta["precision"],
        "framework": meta["framework"],
        "runner_type": meta["runner_type"],
        "concurrency": meta["concurrency"],
        "isl": meta["isl"],
        "osl": meta["osl"],
        "num_prompts": meta["num_prompts"],
        "tp": tp,
        "duration": float(raw.get("duration", 0.0)),
        "request_throughput": float(raw.get("request_throughput", 0.0)),
        "total_token_throughput": total_tput,
        "output_throughput": output_tput,
        "input_throughput": input_tput,
        "total_tput_per_gpu": total_tput / tp,
        "output_tput_per_gpu": output_tput / tp,
        "input_tput_per_gpu": input_tput / tp,
        "mean_ttft_ms": float(raw.get("mean_ttft_ms", 0.0)),
        "p99_ttft_ms": float(raw.get("p99_ttft_ms", 0.0)),
        "mean_tpot_ms": float(raw.get("mean_tpot_ms", 0.0)),
        "p99_tpot_ms": float(raw.get("p99_tpot_ms", 0.0)),
        "mean_e2el_ms": float(raw.get("mean_e2el_ms", 0.0)),
        "p99_e2el_ms": float(raw.get("p99_e2el_ms", 0.0)),
    }

    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)
    return agg


def generate_summary(run_dir: Path) -> None:
    """Generate suite_summary_report.txt from benchmark case results."""

    agg_files = sorted(run_dir.glob("case_*/agg_*.json"))
    if not agg_files:
        print(f"[summary] No agg_*.json found under {run_dir}")
        return

    rows = []
    for path in agg_files:
        with open(path, encoding="utf-8") as f:
            rows.append(json.load(f))

    rows.sort(key=lambda row: (
        row["precision"],
        row["concurrency"],
        row["isl"],
        row["osl"],
        row["tp"],
        row["num_prompts"],
    ))

    headers = [
        "DTYPE",
        "CONC",
        "ISL",
        "OSL",
        "TP",
        "PROMPTS",
        "DUR(s)",
        "REQ/s",
        "TOTAL tok/s",
        "OUT tok/s",
        "TOTAL/GPU",
        "OUT/GPU",
        "TTFT(ms)",
        "P99 TTFT",
        "TPOT(ms)",
        "P99 TPOT",
        "E2EL(ms)",
    ]

    table_rows = []
    for row in rows:
        table_rows.append([
            str(row["precision"]),
            str(row["concurrency"]),
            str(row["isl"]),
            str(row["osl"]),
            str(row["tp"]),
            str(row["num_prompts"]),
            f'{row["duration"]:.2f}',
            f'{row["request_throughput"]:.2f}',
            f'{row["total_token_throughput"]:.2f}',
            f'{row["output_throughput"]:.2f}',
            f'{row["total_tput_per_gpu"]:.2f}',
            f'{row["output_tput_per_gpu"]:.2f}',
            f'{row["mean_ttft_ms"]:.2f}',
            f'{row["p99_ttft_ms"]:.2f}',
            f'{row["mean_tpot_ms"]:.2f}',
            f'{row["p99_tpot_ms"]:.2f}',
            f'{row["mean_e2el_ms"]:.2f}',
        ])

    col_widths = [len(h) for h in headers]
    for table_row in table_rows:
        for idx, cell in enumerate(table_row):
            col_widths[idx] = max(col_widths[idx], len(cell))

    def fmt_line(cells: list[str]) -> str:
        return "| " + " | ".join(
            cell.ljust(col_widths[idx]) for idx, cell in enumerate(cells)
        ) + " |"

    sep = "+-" + "-+-".join("-" * width for width in col_widths) + "-+"
    table = [sep, fmt_line(headers), sep]
    table.extend(fmt_line(row) for row in table_rows)
    table.append(sep)

    first = rows[0]
    lines = [
        f"Run directory : {run_dir}",
        f"Total cases   : {len(rows)}",
        f"Model path    : {first['model_path']}",
        f"Model prefix  : {first['model_prefix']}",
        f"Framework     : {first['framework']}",
        "",
        *table,
    ]

    report_path = run_dir / "suite_summary_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[summary] Report written to {report_path}")

