"""CLI helper to aggregate one vLLM benchmark result JSON."""

from __future__ import annotations

import argparse
from pathlib import Path

from vllm_bench.report import aggregate_case


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate a vLLM benchmark result")
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--meta", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    agg = aggregate_case(args.raw, args.meta, args.output)
    print(agg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

