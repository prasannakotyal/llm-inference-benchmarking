from __future__ import annotations

import argparse
import json
from pathlib import Path

METRICS = {
    "tokens_per_sec": "Generated token throughput.",
    "request_per_sec": "Completed request throughput.",
    "mean_ttft_ms": "Mean time to first token.",
    "p95_ttft_ms": "P95 time to first token.",
    "mean_itl_ms": "Mean inter-token latency.",
    "p95_itl_ms": "P95 inter-token latency.",
    "peak_memory_mb": "Peak allocated GPU memory.",
    "max_kv_cache_mb_per_request": "Maximum KV-cache memory per request.",
}


def load_rows(path: Path) -> list[dict[str, float | int | str]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    assert rows
    return rows


def labels(row: dict[str, float | int | str]) -> str:
    values = {
        "backend": row["backend"],
        "prompt_length": row["prompt_length"],
        "batch_size": row["batch_size"],
        "concurrency": row["concurrency"],
    }
    return ",".join(f'{key}="{value}"' for key, value in values.items())


def metric_name(name: str) -> str:
    return "llm_inference_" + name


def export_metrics(rows: list[dict[str, float | int | str]]) -> str:
    lines = []
    for name, help_text in METRICS.items():
        lines.append(f"# HELP {metric_name(name)} {help_text}")
        lines.append(f"# TYPE {metric_name(name)} gauge")
        for row in rows:
            lines.append(f"{metric_name(name)}{{{labels(row)}}} {row[name]}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export benchmark results as Prometheus metrics.")
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/runpod/prometheus_metrics.prom"),
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(export_metrics(load_rows(args.input)), encoding="utf-8")


if __name__ == "__main__":
    main()
