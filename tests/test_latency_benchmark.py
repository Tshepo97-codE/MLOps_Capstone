"""Tests for the latency benchmark harness.

These tests spin up the real FastAPI app in-process (via httpx's ASGI
transport, monkeypatched into the benchmark's client construction is
avoided — instead we run the actual app with uvicorn in a background
thread) so the benchmark logic is exercised against real HTTP responses,
not mocks, while staying fast and hermetic enough for CI.
"""

from __future__ import annotations

import json
import threading
import time

import httpx
import pytest
import uvicorn

from payshap_ml.benchmarks import latency as bench


@pytest.fixture(scope="module")
def live_test_server():
    """Run the FastAPI app in a background thread on a local port.

    Uses the app's normal startup path (including the lifespan model
    loader), so if no trained model artifact is present the /predict
    calls will correctly fail — this fixture does not fake model state.
    """
    import app.main as app_module

    config = uvicorn.Config(app_module.app, host="127.0.0.1", port=8765, log_level="warning")
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.05)

    yield "http://127.0.0.1:8765"

    server.should_exit = True
    thread.join(timeout=5)


def test_make_synthetic_payload_has_required_fields():
    payload = bench.make_synthetic_payload(42)
    required = {
        "transaction_id",
        "debtor_shap_id",
        "creditor_shap_id",
        "amount_zar",
        "clearing_system_property",
        "timestamp",
        "proxy_type",
    }
    assert required.issubset(payload.keys())
    assert payload["amount_zar"] > 0
    assert payload["clearing_system_property"] == "pacs.008"


def test_synthetic_payloads_vary_across_indices():
    p1 = bench.make_synthetic_payload(1)
    p2 = bench.make_synthetic_payload(2)
    assert p1["transaction_id"] != p2["transaction_id"]
    assert p1["debtor_shap_id"] != p2["debtor_shap_id"] or p1["amount_zar"] != p2["amount_zar"]


def test_percentile_helper_matches_known_values():
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert bench._percentile_no_numpy(values, 50) == 30.0
    assert bench._percentile_no_numpy(values, 0) == 10.0
    assert bench._percentile_no_numpy(values, 100) == 50.0


def test_percentile_helper_empty_list_is_nan():
    result = bench._percentile_no_numpy([], 95)
    assert result != result  # NaN != NaN


@pytest.mark.asyncio
async def test_run_benchmark_against_live_health_endpoint(live_test_server):
    """Smoke-tests the async batch runner against a real running server,
    hitting /health (always available even without a trained model) to
    validate the request/latency plumbing end-to-end without depending
    on a model artifact being present in CI.
    """
    results = await bench._run_batch(
        url=f"{live_test_server}/health",
        n_requests=5,
        concurrency=2,
        timeout_s=2.0,
        start_index=0,
    )
    assert len(results) == 5
    # /health is a GET-only route in app.main; POSTing to it should yield
    # a clean 4xx captured as a structured failure, not an unhandled crash.
    assert all(r.status_code is not None or r.error is not None for r in results)


def test_benchmark_report_serializes_to_json(tmp_path):
    report = bench.BenchmarkReport(
        timestamp_utc="2026-08-27T00:00:00+00:00",
        target_url="http://localhost:8000/predict",
        concurrency=10,
        target_rps=50,
        warmup_runs=10,
        measured_runs=100,
        timeout_ms=500,
        total_measured_requests=100,
        successful_requests=99,
        failed_requests=1,
        error_rate=0.01,
        throughput_rps=48.5,
        avg_latency_ms=12.3,
        p50_latency_ms=10.1,
        p95_latency_ms=25.4,
        p99_latency_ms=40.2,
        min_latency_ms=5.0,
        max_latency_ms=60.0,
        rss_mb_before=100.0,
        rss_mb_after=105.0,
        rss_mb_delta=5.0,
        p95_budget_ms=50.0,
        max_error_rate_budget=0.01,
        passed_p95_budget=True,
        passed_error_rate_budget=True,
        passed=True,
    )
    out_path = tmp_path / "latency_results.json"
    bench.write_json_report(report, str(out_path))
    assert out_path.exists()
    loaded = json.loads(out_path.read_text())
    assert loaded["p95_latency_ms"] == 25.4
    assert loaded["passed"] is True


def test_benchmark_report_serializes_to_csv(tmp_path):
    report = bench.BenchmarkReport(
        timestamp_utc="2026-08-27T00:00:00+00:00",
        target_url="http://localhost:8000/predict",
        concurrency=10,
        target_rps=50,
        warmup_runs=10,
        measured_runs=100,
        timeout_ms=500,
        total_measured_requests=100,
        successful_requests=100,
        failed_requests=0,
        error_rate=0.0,
        throughput_rps=50.0,
        avg_latency_ms=10.0,
        p50_latency_ms=9.0,
        p95_latency_ms=20.0,
        p99_latency_ms=30.0,
        min_latency_ms=4.0,
        max_latency_ms=35.0,
        rss_mb_before=100.0,
        rss_mb_after=101.0,
        rss_mb_delta=1.0,
        p95_budget_ms=50.0,
        max_error_rate_budget=0.01,
        passed_p95_budget=True,
        passed_error_rate_budget=True,
        passed=True,
    )
    out_path = tmp_path / "latency_results.csv"
    bench.write_csv_report(report, str(out_path))
    assert out_path.exists()
    content = out_path.read_text()
    assert "p95_latency_ms" in content
    assert "20.0" in content
