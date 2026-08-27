"""Reproducible HTTP latency benchmark for the /predict endpoint.

IMPORTANT — environment dependence:
    Absolute latency numbers produced by this harness are entirely a
    function of the machine, network, and load it is run on (CPU, whether
    the API and client share a host, background load, container CPU
    limits, etc.). Numbers from a laptop are NOT representative of
    production. This module intentionally contains NO hardcoded benchmark
    results — every number in reports/latency/ is produced by actually
    executing requests against a live endpoint at run time. Always rerun
    this benchmark on the target deployment hardware / environment before
    trusting a pass/fail verdict for a release decision.

Usage:
    python -m payshap_ml.benchmarks.latency --params params.yaml
    make benchmark

This module is import-safe: importing it does not start a server, make
any network calls, or read files. All I/O happens inside main() / run_benchmark().
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import psutil

from payshap_ml.utils.config import ensure_parent_dir, get_params_arg_parser, load_params
from payshap_ml.utils.logging import get_logger

log = get_logger(__name__)


# --------------------------------------------------------------------------
# Synthetic payload generation
# --------------------------------------------------------------------------

def make_synthetic_payload(index: int) -> dict[str, Any]:
    """Build a realistic PayShap pacs.008-style transaction payload.

    Values vary per-request (amount, IDs, timestamp) so the benchmark
    exercises the same code paths (validation, feature construction,
    scoring) a real transaction stream would hit — not a single cached
    trivial case.
    """
    now = datetime.now(timezone.utc)
    proxy_types = ["MOBILE", "ID_NUMBER", "EMAIL", "ACCOUNT"]
    bics = ["FIRNZAJJ", "SBZAZAJJ", "ABSAZAJJ", "NEDSZAJJ", "CABLZAJJ"]
    purpose_codes = ["CASH", "SALA", "SUPP", "UTIL", None]

    return {
        "transaction_id": f"PS-BENCH-{uuid.uuid4().hex[:16]}",
        "debtor_shap_id": f"+2782{index % 10_000_000:07d}",
        "creditor_shap_id": f"+2783{(index * 7) % 10_000_000:07d}",
        "amount_zar": round(10 + (index * 37 % 49_990) + 0.55, 2),
        "clearing_system_property": "pacs.008",
        "timestamp": now.isoformat(),
        "proxy_type": proxy_types[index % len(proxy_types)],
        "debtor_agent_bic": bics[index % len(bics)],
        "creditor_agent_bic": bics[(index + 1) % len(bics)],
        "purpose_code": purpose_codes[index % len(purpose_codes)],
        "charge_bearer": "SLEV",
    }


# --------------------------------------------------------------------------
# Result containers
# --------------------------------------------------------------------------

@dataclass
class RequestResult:
    ok: bool
    status_code: int | None
    latency_ms: float | None
    error: str | None = None


@dataclass
class BenchmarkReport:
    timestamp_utc: str
    target_url: str
    concurrency: int
    target_rps: int
    warmup_runs: int
    measured_runs: int
    timeout_ms: int
    total_measured_requests: int
    successful_requests: int
    failed_requests: int
    error_rate: float
    throughput_rps: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    rss_mb_before: float
    rss_mb_after: float
    rss_mb_delta: float
    p95_budget_ms: float
    max_error_rate_budget: float
    passed_p95_budget: bool
    passed_error_rate_budget: bool
    passed: bool
    errors: list[str] = field(default_factory=list)


def _percentile_no_numpy(values: list[float], pct: float) -> float:
    """Nearest-rank percentile without a numpy dependency in this hot path."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return ordered[k]


# --------------------------------------------------------------------------
# Core benchmark execution
# --------------------------------------------------------------------------

async def _fire_request(
    client: httpx.AsyncClient, url: str, payload: dict[str, Any], timeout_s: float
) -> RequestResult:
    t0 = time.perf_counter()
    try:
        resp = await client.post(url, json=payload, timeout=timeout_s)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if resp.status_code == 200:
            return RequestResult(ok=True, status_code=resp.status_code, latency_ms=elapsed_ms)
        return RequestResult(
            ok=False,
            status_code=resp.status_code,
            latency_ms=elapsed_ms,
            error=f"HTTP {resp.status_code}",
        )
    except Exception as exc:  # noqa: BLE001 — network errors are expected outcomes here
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return RequestResult(ok=False, status_code=None, latency_ms=elapsed_ms, error=str(exc))


async def _run_batch(
    url: str, n_requests: int, concurrency: int, timeout_s: float, start_index: int
) -> list[RequestResult]:
    """Run n_requests total against url, holding at most `concurrency`
    requests in flight at once (a semaphore-bounded async gather)."""
    import asyncio

    results: list[RequestResult] = []
    semaphore = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient() as client:

        async def worker(i: int) -> RequestResult:
            async with semaphore:
                payload = make_synthetic_payload(start_index + i)
                return await _fire_request(client, url, payload, timeout_s)

        tasks = [worker(i) for i in range(n_requests)]
        results = await asyncio.gather(*tasks)

    return list(results)


async def run_benchmark_async(bench_cfg: dict) -> BenchmarkReport:
    import asyncio  # local import keeps module import-safe without asyncio side effects

    url = bench_cfg["target_url"]
    concurrency = bench_cfg["concurrency"]
    warmup_runs = bench_cfg["warmup_runs"]
    measured_runs = bench_cfg["measured_runs"]
    timeout_s = bench_cfg["timeout_ms"] / 1000.0
    target_rps = bench_cfg["target_rps"]

    process = psutil.Process()
    rss_before_mb = process.memory_info().rss / (1024 * 1024)

    log.info("Warmup phase: %d requests against %s (untimed for reporting)", warmup_runs, url)
    if warmup_runs > 0:
        await _run_batch(url, warmup_runs, concurrency, timeout_s, start_index=0)

    log.info(
        "Measured phase: %d requests, concurrency=%d, target_rps=%d",
        measured_runs,
        concurrency,
        target_rps,
    )

    # Fixed-rate pacing: split the measured run into 1-second ticks and
    # send target_rps requests per tick, so the harness exercises a
    # controlled arrival rate rather than an uncontrolled burst.
    results: list[RequestResult] = []
    remaining = measured_runs
    tick = 0
    wall_start = time.perf_counter()
    while remaining > 0:
        batch_size = min(target_rps, remaining)
        tick_start = time.perf_counter()
        batch_results = await _run_batch(
            url, batch_size, concurrency, timeout_s, start_index=warmup_runs + tick * target_rps
        )
        results.extend(batch_results)
        remaining -= batch_size
        tick += 1

        # Pace to ~1 tick/second if the batch completed faster than that,
        # so the achieved rate approximates target_rps rather than firing
        # as fast as possible.
        elapsed = time.perf_counter() - tick_start
        if elapsed < 1.0 and remaining > 0:
            await asyncio.sleep(1.0 - elapsed)

    wall_elapsed_s = time.perf_counter() - wall_start
    rss_after_mb = process.memory_info().rss / (1024 * 1024)

    latencies = [r.latency_ms for r in results if r.latency_ms is not None]
    successes = [r for r in results if r.ok]
    failures = [r for r in results if not r.ok]

    error_rate = len(failures) / len(results) if results else 1.0
    throughput = len(results) / wall_elapsed_s if wall_elapsed_s > 0 else 0.0

    p95 = _percentile_no_numpy(latencies, 95)
    p95_budget = bench_cfg["budgets"]["p95_latency_ms"]
    max_error_budget = bench_cfg["budgets"]["max_error_rate"]

    passed_p95 = p95 <= p95_budget
    passed_error_rate = error_rate <= max_error_budget

    report = BenchmarkReport(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        target_url=url,
        concurrency=concurrency,
        target_rps=target_rps,
        warmup_runs=warmup_runs,
        measured_runs=measured_runs,
        timeout_ms=bench_cfg["timeout_ms"],
        total_measured_requests=len(results),
        successful_requests=len(successes),
        failed_requests=len(failures),
        error_rate=round(error_rate, 6),
        throughput_rps=round(throughput, 3),
        avg_latency_ms=round(statistics.fmean(latencies), 3) if latencies else float("nan"),
        p50_latency_ms=round(_percentile_no_numpy(latencies, 50), 3),
        p95_latency_ms=round(p95, 3),
        p99_latency_ms=round(_percentile_no_numpy(latencies, 99), 3),
        min_latency_ms=round(min(latencies), 3) if latencies else float("nan"),
        max_latency_ms=round(max(latencies), 3) if latencies else float("nan"),
        rss_mb_before=round(rss_before_mb, 3),
        rss_mb_after=round(rss_after_mb, 3),
        rss_mb_delta=round(rss_after_mb - rss_before_mb, 3),
        p95_budget_ms=p95_budget,
        max_error_rate_budget=max_error_budget,
        passed_p95_budget=passed_p95,
        passed_error_rate_budget=passed_error_rate,
        passed=passed_p95 and passed_error_rate,
        errors=sorted({r.error for r in failures if r.error})[:20],
    )
    return report


def run_benchmark(bench_cfg: dict) -> BenchmarkReport:
    """Synchronous entrypoint wrapping the async implementation."""
    import asyncio

    return asyncio.run(run_benchmark_async(bench_cfg))


# --------------------------------------------------------------------------
# Output writers
# --------------------------------------------------------------------------

def write_json_report(report: BenchmarkReport, path: str) -> None:
    ensure_parent_dir(path)
    with open(path, "w") as f:
        json.dump(asdict(report), f, indent=2)


def write_csv_report(report: BenchmarkReport, path: str) -> None:
    ensure_parent_dir(path)
    row = asdict(report)
    row["errors"] = "; ".join(row["errors"])
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


# --------------------------------------------------------------------------
# CLI entrypoint
# --------------------------------------------------------------------------

def main() -> int:
    parser = get_params_arg_parser("PayShap /predict latency benchmark harness")
    parser.add_argument(
        "--fail-on-breach",
        action="store_true",
        default=True,
        help="Exit non-zero if the p95 or error-rate budget is breached (default: on). "
        "Useful for CI gating via `make benchmark`.",
    )
    args = parser.parse_args()
    params = load_params(args.params)
    bench_cfg = params["benchmark"]

    log.warning(
        "Latency results are environment-dependent (hardware, network, co-located "
        "load). Rerun this benchmark on the actual target deployment before using "
        "results for a go/no-go decision."
    )

    report = run_benchmark(bench_cfg)

    write_json_report(report, bench_cfg["output"]["json_path"])
    write_csv_report(report, bench_cfg["output"]["csv_path"])

    log.info(
        "Benchmark complete: p50=%.2fms p95=%.2fms p99=%.2fms throughput=%.1f req/s "
        "error_rate=%.4f passed=%s",
        report.p50_latency_ms,
        report.p95_latency_ms,
        report.p99_latency_ms,
        report.throughput_rps,
        report.error_rate,
        report.passed,
    )

    if args.fail_on_breach and not report.passed:
        log.error(
            "Benchmark FAILED budget: p95=%.2fms (budget %.2fms, ok=%s), "
            "error_rate=%.4f (budget %.4f, ok=%s)",
            report.p95_latency_ms,
            report.p95_budget_ms,
            report.passed_p95_budget,
            report.error_rate,
            report.max_error_rate_budget,
            report.passed_error_rate_budget,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
