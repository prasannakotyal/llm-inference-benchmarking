from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BACKEND_ORDER = ["hf-static", "continuous"]
BACKEND_COLORS = {
    "hf-static": "#1F2937",
    "continuous": "#0F9D58",
}
BACKEND_HATCHES = {
    "hf-static": "",
    "continuous": "///",
}


def load_rows(path: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    assert rows
    return pd.DataFrame(rows)


def save_bar(
    df: pd.DataFrame,
    title: str,
    value_column: str,
    ylabel: str,
    output_path: Path,
) -> None:
    pivot = df.pivot_table(
        index=["prompt_length", "batch_size", "concurrency"],
        columns="backend",
        values=value_column,
        aggfunc="mean",
    )[BACKEND_ORDER]
    colors = [BACKEND_COLORS[backend] for backend in pivot.columns]
    ax = pivot.plot(
        kind="bar",
        figsize=(16, 7),
        width=0.8,
        color=colors,
        edgecolor="#111827",
        linewidth=0.8,
    )
    for container, backend in zip(ax.containers, pivot.columns, strict=False):
        for patch in container.patches:
            patch.set_hatch(BACKEND_HATCHES[backend])

    ax.set_title(title, fontsize=16, weight="bold", pad=14)
    ax.set_xlabel("prompt tokens / max batch / concurrent requests")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, pivot.to_numpy().max() * 1.18)
    ax.grid(axis="y", alpha=0.22, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        title="Backend",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=2,
        frameon=True,
    )
    plt.xticks(rotation=35, ha="right", fontsize=9)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_line(
    df: pd.DataFrame,
    title: str,
    value_column: str,
    ylabel: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for backend in BACKEND_ORDER:
        group = df[df["backend"] == backend]
        grouped = group.groupby("prompt_length")[value_column].mean().sort_index()
        ax.plot(
            grouped.index,
            grouped.values,
            marker="o",
            linewidth=2.5,
            markersize=7,
            color=BACKEND_COLORS[backend],
            label=backend,
        )
    ax.set_title(title, fontsize=16, weight="bold", pad=14)
    ax.set_xlabel("Prompt length")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.22, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(title="Backend", loc="best", frameon=True)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate benchmark plots.")
    parser.add_argument("path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("assets/runpod"))
    args = parser.parse_args()

    df = load_rows(args.path)
    save_bar(
        df,
        "Decode Throughput",
        "tokens_per_sec",
        "Generated tokens/sec",
        args.output_dir / "throughput.png",
    )
    save_bar(
        df,
        "Time to First Token",
        "mean_ttft_ms",
        "Mean TTFT (ms)",
        args.output_dir / "ttft.png",
    )
    save_bar(
        df,
        "Inter-token Latency",
        "mean_itl_ms",
        "Mean ITL (ms)",
        args.output_dir / "itl.png",
    )
    save_line(
        df,
        "KV-cache Memory Scaling",
        "max_kv_cache_mb_per_request",
        "Max KV cache per request (MB)",
        args.output_dir / "kv_cache_scaling.png",
    )
    save_line(
        df,
        "Peak GPU Memory Scaling",
        "peak_memory_mb",
        "Peak allocated memory (MB)",
        args.output_dir / "peak_memory.png",
    )


if __name__ == "__main__":
    main()
