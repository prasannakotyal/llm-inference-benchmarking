from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import transformers.utils as transformers_utils
import transformers.utils.import_utils as transformers_import_utils

if TYPE_CHECKING:
    from transformers import AutoModelForCausalLM


def no_torchvision() -> bool:
    return False


transformers_import_utils.is_torchvision_available.cache_clear()
transformers_import_utils.is_torchvision_v2_available.cache_clear()
transformers_import_utils.is_torchvision_available = no_torchvision
transformers_import_utils.is_torchvision_v2_available = no_torchvision
transformers_import_utils.BACKENDS_MAPPING["torchvision"] = (
    no_torchvision,
    transformers_import_utils.TORCHVISION_IMPORT_ERROR,
)
transformers_utils.is_torchvision_available = no_torchvision
transformers_utils.is_torchvision_v2_available = no_torchvision


BACKENDS = ("hf-static", "continuous")


@dataclass(frozen=True)
class Workload:
    backend: str
    prompt_length: int
    batch_size: int
    concurrency: int
    requests: int
    output_pattern: tuple[int, ...]


@dataclass(eq=False)
class RequestState:
    request_id: int
    target_tokens: int
    start_time: float
    prompt_tokens: int
    generated_tokens: int = 0
    first_token_time: float | None = None
    token_times: list[float] = field(default_factory=list)
    last_token: torch.Tensor | None = None
    past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None
    kv_cache_mb: float = 0.0


def positive_int(raw: str) -> int:
    value = int(raw)
    assert value > 0
    return value


def int_tuple(raw: str) -> tuple[int, ...]:
    values = tuple(positive_int(part.strip()) for part in raw.split(",") if part.strip())
    assert values
    return values


def make_prompt_ids(
    request_id: int,
    length: int,
    vocab_size: int,
    device: torch.device,
) -> torch.Tensor:
    assert length > 0
    usable_vocab = max(vocab_size - 256, 256)
    start = 128 + (request_id * 17) % usable_vocab
    ids = (torch.arange(length, device=device) + start) % usable_vocab
    return ids.add(128).clamp(max=vocab_size - 1).unsqueeze(0)


def sample_next_token(logits: torch.Tensor) -> torch.Tensor:
    return torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)


def normalize_past(past: object) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    if hasattr(past, "to_legacy_cache"):
        past = past.to_legacy_cache()
    if hasattr(past, "layers"):
        return tuple((layer.keys, layer.values) for layer in past.layers)
    assert isinstance(past, tuple)
    return past


def past_batch_size(past: tuple[tuple[torch.Tensor, torch.Tensor], ...]) -> int:
    first_layer = past[0]
    return first_layer[0].shape[0]


def past_sequence_length(past: tuple[tuple[torch.Tensor, torch.Tensor], ...]) -> int:
    first_layer = past[0]
    return first_layer[0].shape[-2]


def past_cache_mb(past: tuple[tuple[torch.Tensor, torch.Tensor], ...]) -> float:
    total_bytes = 0
    for key, value in past:
        total_bytes += key.numel() * key.element_size()
        total_bytes += value.numel() * value.element_size()
    return total_bytes / 1024 / 1024


def slice_past(
    past: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    index: int,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    assert 0 <= index < past_batch_size(past)
    return tuple(
        (key[index : index + 1].contiguous(), value[index : index + 1].contiguous())
        for key, value in past
    )


def stack_past(
    states: list[RequestState],
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    assert states
    assert all(state.past_key_values is not None for state in states)
    layer_count = len(states[0].past_key_values)
    layers = []
    for layer_index in range(layer_count):
        keys = [state.past_key_values[layer_index][0] for state in states]
        values = [state.past_key_values[layer_index][1] for state in states]
        layers.append((torch.cat(keys, dim=0), torch.cat(values, dim=0)))
    return tuple(layers)


def to_model_cache(past: tuple[tuple[torch.Tensor, torch.Tensor], ...]) -> object:
    from transformers.cache_utils import DynamicCache

    return DynamicCache(past)


def record_token(
    state: RequestState,
    token: torch.Tensor,
    now: float,
    past: tuple[tuple[torch.Tensor, torch.Tensor], ...],
) -> None:
    state.generated_tokens += 1
    state.last_token = token
    state.past_key_values = past
    state.kv_cache_mb = past_cache_mb(past)
    state.token_times.append(now)
    if state.first_token_time is None:
        state.first_token_time = now


@torch.inference_mode()
def prefill_batch(
    model: AutoModelForCausalLM,
    states: list[RequestState],
    prompt_length: int,
    vocab_size: int,
    device: torch.device,
) -> None:
    assert states
    input_ids = torch.cat(
        [make_prompt_ids(state.request_id, prompt_length, vocab_size, device) for state in states],
        dim=0,
    )
    outputs = model(input_ids=input_ids, use_cache=True, return_dict=True)
    next_tokens = sample_next_token(outputs.logits)
    past = normalize_past(outputs.past_key_values)
    now = time.perf_counter()
    for index, state in enumerate(states):
        token = next_tokens[index : index + 1].contiguous()
        record_token(state, token, now, slice_past(past, index))


@torch.inference_mode()
def decode_batch(model: AutoModelForCausalLM, states: list[RequestState]) -> None:
    assert states
    assert all(state.last_token is not None for state in states)
    input_ids = torch.cat([state.last_token for state in states], dim=0)
    past = stack_past(states)
    outputs = model(
        input_ids=input_ids,
        past_key_values=to_model_cache(past),
        use_cache=True,
        return_dict=True,
    )
    next_tokens = sample_next_token(outputs.logits)
    next_past = normalize_past(outputs.past_key_values)
    now = time.perf_counter()
    for index, state in enumerate(states):
        token = next_tokens[index : index + 1].contiguous()
        record_token(state, token, now, slice_past(next_past, index))


def make_requests(workload: Workload, prompt_length: int) -> list[RequestState]:
    assert workload.requests >= workload.concurrency
    return [
        RequestState(
            request_id=index,
            target_tokens=workload.output_pattern[index % len(workload.output_pattern)],
            start_time=0.0,
            prompt_tokens=prompt_length,
        )
        for index in range(workload.requests)
    ]


def run_hf_static(
    model: AutoModelForCausalLM,
    workload: Workload,
    vocab_size: int,
    device: torch.device,
) -> list[RequestState]:
    completed: list[RequestState] = []
    waiting = make_requests(workload, workload.prompt_length)
    start_time = time.perf_counter()
    for state in waiting:
        state.start_time = start_time

    for offset in range(0, len(waiting), workload.batch_size):
        batch = waiting[offset : offset + workload.batch_size]
        prefill_batch(model, batch, workload.prompt_length, vocab_size, device)
        max_tokens = max(state.target_tokens for state in batch)
        for _ in range(1, max_tokens):
            decode_batch(model, batch)
        completed.extend(batch)
    return completed


def run_continuous(
    model: AutoModelForCausalLM,
    workload: Workload,
    vocab_size: int,
    device: torch.device,
) -> list[RequestState]:
    completed: list[RequestState] = []
    waiting = make_requests(workload, workload.prompt_length)
    start_time = time.perf_counter()
    for state in waiting:
        state.start_time = start_time

    active: list[RequestState] = []

    def admit() -> None:
        available_slots = workload.batch_size - len(active)
        if available_slots <= 0 or not waiting:
            return
        admitted = [waiting.pop(0) for _ in range(min(available_slots, len(waiting)))]
        prefill_batch(model, admitted, workload.prompt_length, vocab_size, device)
        for state in admitted:
            if state.generated_tokens >= state.target_tokens:
                completed.append(state)
            else:
                active.append(state)

    admit()
    while active:
        by_length: dict[int, list[RequestState]] = {}
        for state in active:
            assert state.past_key_values is not None
            by_length.setdefault(past_sequence_length(state.past_key_values), []).append(state)

        for sequence_length in sorted(by_length):
            group = [state for state in by_length[sequence_length] if state in active]
            if not group:
                continue
            decode_batch(model, group)
            for state in group:
                if state.generated_tokens >= state.target_tokens:
                    active.remove(state)
                    completed.append(state)
            admit()
    return completed


def summarize_run(
    workload: Workload,
    states: list[RequestState],
    started_at: float,
    ended_at: float,
    model_name: str,
    dtype: str,
    device_name: str,
    peak_memory_mb: float,
) -> dict[str, float | int | str]:
    assert states
    request_count = len(states)
    requested_output_tokens = sum(state.target_tokens for state in states)
    prompt_tokens = sum(state.prompt_tokens for state in states)
    total_seconds = ended_at - started_at
    ttft = [state.first_token_time - state.start_time for state in states]
    inter_token_latencies = []
    for state in states:
        token_times = state.token_times[: state.target_tokens]
        inter_token_latencies.extend(
            later - earlier
            for earlier, later in zip(token_times, token_times[1:], strict=False)
        )

    mean_itl = statistics.fmean(inter_token_latencies) if inter_token_latencies else 0.0
    p95_itl = percentile(inter_token_latencies, 0.95) if inter_token_latencies else 0.0
    return {
        "model": model_name,
        "backend": workload.backend,
        "dtype": dtype,
        "device": device_name,
        "prompt_length": workload.prompt_length,
        "batch_size": workload.batch_size,
        "concurrency": workload.concurrency,
        "requests": request_count,
        "output_pattern": "/".join(str(value) for value in workload.output_pattern),
        "prompt_tokens": prompt_tokens,
        "output_tokens": requested_output_tokens,
        "total_seconds": total_seconds,
        "tokens_per_sec": requested_output_tokens / total_seconds,
        "request_per_sec": request_count / total_seconds,
        "mean_ttft_ms": statistics.fmean(ttft) * 1000,
        "p95_ttft_ms": percentile(ttft, 0.95) * 1000,
        "mean_itl_ms": mean_itl * 1000,
        "p95_itl_ms": p95_itl * 1000,
        "peak_memory_mb": peak_memory_mb,
        "max_kv_cache_mb_per_request": max(state.kv_cache_mb for state in states),
    }


def percentile(values: list[float], q: float) -> float:
    assert values
    assert 0 <= q <= 1
    ordered = sorted(values)
    index = math.ceil(q * len(ordered)) - 1
    return ordered[max(index, 0)]


def load_model(
    model_name: str,
    dtype: str,
    device: torch.device,
) -> tuple[AutoModelForCausalLM, int]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    torch_dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[dtype]
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    model.eval().to(device)
    assert tokenizer.vocab_size > 512
    return model, tokenizer.vocab_size


def run_workload(
    model: AutoModelForCausalLM,
    workload: Workload,
    vocab_size: int,
    device: torch.device,
    model_name: str,
    dtype: str,
) -> dict[str, float | int | str]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started_at = time.perf_counter()
    if workload.backend == "hf-static":
        states = run_hf_static(model, workload, vocab_size, device)
    elif workload.backend == "continuous":
        states = run_continuous(model, workload, vocab_size, device)
    else:
        raise ValueError(f"unknown backend: {workload.backend}")
    torch.cuda.synchronize(device)
    ended_at = time.perf_counter()
    peak_memory_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024
    return summarize_run(
        workload=workload,
        states=states,
        started_at=started_at,
        ended_at=ended_at,
        model_name=model_name,
        dtype=dtype,
        device_name=torch.cuda.get_device_name(device),
        peak_memory_mb=peak_memory_mb,
    )


def build_workloads(args: argparse.Namespace) -> list[Workload]:
    workloads = []
    for backend in args.backends:
        assert backend in BACKENDS
        for prompt_length in args.prompt_lengths:
            for batch_size in args.batch_sizes:
                for concurrency in args.concurrencies:
                    workloads.append(
                        Workload(
                            backend=backend,
                            prompt_length=prompt_length,
                            batch_size=batch_size,
                            concurrency=concurrency,
                            requests=concurrency,
                            output_pattern=args.output_pattern,
                        )
                    )
    return workloads


def write_jsonl(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    assert rows
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark static and continuous LLM batching.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="fp16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backends", nargs="+", choices=BACKENDS, default=list(BACKENDS))
    parser.add_argument("--prompt-lengths", type=int_tuple, default=(64, 256, 512))
    parser.add_argument("--batch-sizes", type=int_tuple, default=(1, 4, 8))
    parser.add_argument("--concurrencies", type=int_tuple, default=(4, 8, 16))
    parser.add_argument("--output-pattern", type=int_tuple, default=(16, 32))
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/runpod/benchmark_results.jsonl"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    assert device.type == "cuda"
    model, vocab_size = load_model(args.model, args.dtype, device)

    if args.warmup:
        warmup = Workload("hf-static", 16, 1, 1, 1, (2,))
        run_workload(model, warmup, vocab_size, device, args.model, args.dtype)

    rows = []
    for workload in build_workloads(args):
        print(
            f"running backend={workload.backend} prompt={workload.prompt_length} "
            f"batch={workload.batch_size} concurrency={workload.concurrency}",
            flush=True,
        )
        row = run_workload(model, workload, vocab_size, device, args.model, args.dtype)
        rows.append(row)
        write_jsonl(args.output, rows)
        write_csv(args.output.with_suffix(".csv"), rows)


if __name__ == "__main__":
    main()
