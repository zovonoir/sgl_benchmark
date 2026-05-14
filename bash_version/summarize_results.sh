#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:-${RUN_DIR:-}}"
if [[ -z "${RUN_DIR}" ]]; then
  echo "Usage: $0 <run_dir>" >&2
  exit 1
fi

if [[ ! -d "$RUN_DIR" ]]; then
  echo "Run directory not found: $RUN_DIR" >&2
  exit 1
fi

python3 - "$RUN_DIR" <<'PYEOF'
import glob
import json
import os
import sys

def parse_image_version(image: str) -> str:
    if not image:
        return "N/A"
    if "@" in image:
        return image.split("@", 1)[1]
    last_slash = image.rfind("/")
    last_colon = image.rfind(":")
    if last_colon > last_slash:
        return image[last_colon + 1:]
    return "latest"

run_dir = sys.argv[1]
meta_files = sorted(glob.glob(os.path.join(run_dir, "case_*", "meta_*.json")))
if not meta_files:
    print(f"[summary] No meta_*.json found under {run_dir}", file=sys.stderr)
    sys.exit(1)

rows = []
server_env = {}
for meta_file in meta_files:
    with open(meta_file) as fh:
        meta = json.load(fh)

    if not server_env:
        server_env = meta.get("server_env", {})

    result_filename = meta["result_filename"]
    case_dir = os.path.dirname(meta_file)
    raw_file = os.path.join(case_dir, f"{result_filename}.json")
    agg_file = os.path.join(case_dir, f"agg_{result_filename}.json")

    if not os.path.isfile(raw_file):
        raise FileNotFoundError(f"Missing raw result: {raw_file}")
    if not os.path.isfile(agg_file):
        raise FileNotFoundError(f"Missing aggregated result: {agg_file}")

    with open(raw_file) as fh:
      raw = json.load(fh)
    with open(agg_file) as fh:
      agg = json.load(fh)

    rows.append({
        "precision": meta["precision"],
        "concurrency": meta["concurrency"],
        "isl": meta["isl"],
        "osl": meta["osl"],
        "tp": meta["tp"],
        "ep": meta["ep"] if meta["ep"] is not None else 1,
        "num_prompts": meta["num_prompts"],
        "qps": raw.get("request_throughput", 0.0),
        "qps_per_tp": raw.get("request_throughput", 0.0) / meta["tp"] if meta["tp"] else 0.0,
        "mean_tpot_ms": raw.get("mean_tpot_ms", 0.0),
        "mean_ttft_ms": raw.get("mean_ttft_ms", 0.0),
        "input_tput_per_gpu": agg.get("input_tput_per_gpu", 0.0),
        "output_tput_per_gpu": agg.get("output_tput_per_gpu", 0.0),
        "total_tput_per_gpu": agg.get("tput_per_gpu", 0.0),
    })

rows.sort(key=lambda row: (row["precision"], row["concurrency"], row["isl"], row["osl"], row["tp"], row["ep"], row["num_prompts"]))

headers = [
    "DTYPE",
    "CONC",
    "ISL",
    "OSL",
    "TP",
    "EP",
    "PROMPTS",
    "QPS(req/s)",
    "QPS/TP",
    "TPOT(ms)",
    "TTFT(ms)",
    "Input Tput/GPU(tok/s)",
    "Output Tput/GPU(tok/s)",
    "Total Tput/GPU(tok/s)",
]

table_rows = []
for row in rows:
    table_rows.append([
        str(row["precision"]),
        str(row["concurrency"]),
        str(row["isl"]),
        str(row["osl"]),
        str(row["tp"]),
        str(row["ep"]),
        str(row["num_prompts"]),
        f'{row["qps"]:.4f}',
        f'{row["qps_per_tp"]:.4f}',
        f'{row["mean_tpot_ms"]:.1f}',
        f'{row["mean_ttft_ms"]:.1f}',
        f'{row["input_tput_per_gpu"]:.2f}',
        f'{row["output_tput_per_gpu"]:.2f}',
        f'{row["total_tput_per_gpu"]:.2f}',
    ])

col_widths = [len(header) for header in headers]
for row in table_rows:
    for idx, cell in enumerate(row):
        col_widths[idx] = max(col_widths[idx], len(cell))

def fmt_line(cells):
    return "| " + " | ".join(cell.ljust(col_widths[idx]) for idx, cell in enumerate(cells)) + " |"

sep = "+-" + "-+-".join("-" * width for width in col_widths) + "-+"
lines = [sep, fmt_line(headers), sep]
lines.extend(fmt_line(row) for row in table_rows)
lines.append(sep)

report_lines = [
    f"Run directory : {run_dir}",
    f"Total cases    : {len(rows)}",
    f"Model path     : {meta['model_path']}",
    f"Image          : {meta['image']}",
    f"Image version  : {parse_image_version(meta['image'])}",
    f"Request rate   : {meta.get('request_rate', 'inf')}",
    f"Burstiness     : {meta.get('burstiness', '1.0')}",
    "",
    "Server env:",
    *[f"  {k}={v}" for k, v in sorted(server_env.items())],
    "",
    *lines,
]

report_path = os.path.join(run_dir, "suite_summary_report.txt")
with open(report_path, "w") as fh:
    fh.write("\n".join(report_lines) + "\n")

print("\n".join(report_lines))
print(f"\n[summary] Report written to {report_path}")
PYEOF
