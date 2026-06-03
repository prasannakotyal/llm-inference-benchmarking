from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_rows(path: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    assert rows
    return pd.DataFrame(rows)


def fmt_float(value: float, digits: int = 1) -> str:
    return f"{value:,.{digits}f}"


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("|" + "|".join("---" for _ in columns) + "|")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize benchmark JSONL results.")
    parser.add_argument("path", type=Path)
    args = parser.parse_args()

    df = load_rows(args.path)
    display = df.copy()
    display["tokens/sec"] = display["tokens_per_sec"].map(lambda value: fmt_float(value, 1))
    display["mean TTFT"] = display["mean_ttft_ms"].map(lambda value: fmt_float(value, 1) + " ms")
    display["p95 TTFT"] = display["p95_ttft_ms"].map(lambda value: fmt_float(value, 1) + " ms")
    display["mean ITL"] = display["mean_itl_ms"].map(lambda value: fmt_float(value, 1) + " ms")
    display["peak memory"] = display["peak_memory_mb"].map(
        lambda value: fmt_float(value, 0) + " MB"
    )

    key_columns = [
        "backend",
        "prompt_length",
        "batch_size",
        "concurrency",
        "tokens/sec",
        "mean TTFT",
        "p95 TTFT",
        "mean ITL",
        "peak memory",
    ]
    print(markdown_table(display[key_columns], key_columns))


if __name__ == "__main__":
    main()
