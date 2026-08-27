"""Self-contained runner for the DVC `benchmark` stage.

`dvc.yaml`'s `deps:` list (e.g. `app/main.py`) only affects cache
invalidation — it does NOT start a server. The latency benchmark harness
(`payshap_ml.benchmarks.latency`) is a pure HTTP client: if nothing is
listening on `benchmark.target_url`, every request fails and you'll see
`error_rate=1.0000`, exactly as if the API were down in production.

This script makes `dvc repro` self-contained by:
  1. Launching `uvicorn app.main:app` as a subprocess on the host/port
     parsed from `benchmark.target_url` in params.yaml.
  2. Polling `/health` until it returns 200 (or timing out).
  3. Running the latency benchmark harness against it.
  4. Terminating the server subprocess (success or failure) and
     propagating the benchmark's exit code so CI/DVC gating still works.

Usage:
    python scripts/run_benchmark_stage.py --params params.yaml

For manual/local iteration where you want the server to keep running
between repeated benchmark runs, you can still do it the two-terminal way:
    make serve         # terminal 1
    make benchmark      # terminal 2
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from urllib.parse import urlparse

import httpx

# Make src/ importable when run directly (mirrors tests/conftest.py).
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from payshap_ml.utils.config import load_params  # noqa: E402
from payshap_ml.utils.logging import get_logger  # noqa: E402

log = get_logger("payshap_ml.scripts.run_benchmark_stage")

STARTUP_TIMEOUT_S = 30
POLL_INTERVAL_S = 0.5


def wait_for_health(health_url: str, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            resp = httpx.get(health_url, timeout=2.0)
            if resp.status_code == 200:
                return True
        except Exception:  # noqa: BLE001 — server likely just isn't up yet
            pass
        time.sleep(POLL_INTERVAL_S)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=str, default="params.yaml")
    args = parser.parse_args()

    params = load_params(args.params)
    bench_cfg = params["benchmark"]
    target = urlparse(bench_cfg["target_url"])
    host = target.hostname or "127.0.0.1"
    port = target.port or 8000
    health_url = f"{target.scheme}://{host}:{port}/health"

    log.info("Starting API server for benchmark stage: uvicorn on %s:%s", host, port)
    server_proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            host,
            "--port",
            str(port),
        ],
        cwd=str(ROOT),
    )

    try:
        log.info("Waiting up to %ss for %s to become healthy", STARTUP_TIMEOUT_S, health_url)
        if not wait_for_health(health_url, STARTUP_TIMEOUT_S):
            log.error(
                "Server did not become healthy within %ss. Check that a trained model "
                "artifact exists (models/model.pkl) — the app fails startup without one.",
                STARTUP_TIMEOUT_S,
            )
            return 1
        log.info("Server is healthy; running latency benchmark")

        # Import here (after path setup) and call the harness's own main()
        # so we get identical CLI behaviour (report writing, exit codes)
        # to running `python -m payshap_ml.benchmarks.latency` directly.
        from payshap_ml.benchmarks import latency as bench

        # Reuse argv-free entrypoint by calling the internals directly,
        # since bench.main() re-parses sys.argv (which holds this
        # script's own args, not the harness's).
        report = bench.run_benchmark(bench_cfg)
        bench.write_json_report(report, bench_cfg["output"]["json_path"])
        bench.write_csv_report(report, bench_cfg["output"]["csv_path"])

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

        if not report.passed:
            log.error(
                "Benchmark FAILED budget: p95=%.2fms (budget %.2fms), error_rate=%.4f (budget %.4f)",
                report.p95_latency_ms,
                report.p95_budget_ms,
                report.error_rate,
                report.max_error_rate_budget,
            )
            return 1
        return 0

    finally:
        log.info("Shutting down benchmark server subprocess")
        server_proc.terminate()
        try:
            server_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server_proc.kill()
            server_proc.wait(timeout=5)


if __name__ == "__main__":
    sys.exit(main())
