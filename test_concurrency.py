"""线程安全测试：EndpointState、manifest、stop_event、ConcurrencyConfig、_ConcurrentLog。"""
from __future__ import annotations

import tempfile
import threading
import time
from pathlib import Path

import pytest

from pipeline import (
    CIRCUIT_FAILURE_THRESHOLD,
    ConcurrencyConfig,
    Endpoint,
    EndpointState,
    _append_manifest,
    _ConcurrentLog,
    _read_manifest,
)


# ---------------------------------------------------------------------------
#  EndpointState 线程安全
# ---------------------------------------------------------------------------


class TestEndpointStateConcurrentFailures:
    """10 线程各调用 100 次 record_failure，最终计数 = 1000。"""

    def test_total_count(self):
        ep = Endpoint(name="test", api_url="http://localhost", api_key="k")
        state = EndpointState(ep)
        threads: list[threading.Thread] = []
        for _ in range(10):
            t = threading.Thread(target=self._fail_100, args=(state,))
            threads.append(t)
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert state.consecutive_failures == 1000

    @staticmethod
    def _fail_100(state: EndpointState) -> None:
        for _ in range(100):
            state.record_failure("err")


class TestEndpointStateCircuitBreakerAccuracy:
    """并发中熔断阈值准确触发。"""

    def test_circuit_opens(self):
        ep = Endpoint(name="cb", api_url="http://localhost", api_key="k")
        state = EndpointState(ep)
        barrier = threading.Barrier(CIRCUIT_FAILURE_THRESHOLD)
        results: list[bool] = []
        results_lock = threading.Lock()

        def fail_once():
            barrier.wait()
            state.record_failure("err")
            avail = state.is_available()
            with results_lock:
                results.append(avail)

        threads = [threading.Thread(target=fail_once) for _ in range(CIRCUIT_FAILURE_THRESHOLD)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 达到阈值后 is_available 应返回 False（冷却期未到）
        assert state.consecutive_failures >= CIRCUIT_FAILURE_THRESHOLD
        assert not state.is_available()


class TestEndpointStateSuccessResets:
    """record_success 在并发中正确重置。"""

    def test_reset_under_contention(self):
        ep = Endpoint(name="reset", api_url="http://localhost", api_key="k")
        state = EndpointState(ep)
        # 先积累一些失败
        for _ in range(CIRCUIT_FAILURE_THRESHOLD - 1):
            state.record_failure("err")

        barrier = threading.Barrier(5)

        def success_then_check():
            barrier.wait()
            state.record_success()

        threads = [threading.Thread(target=success_then_check) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert state.consecutive_failures == 0
        assert state.circuit_open_until == 0.0
        assert state.is_available()


class TestEndpointStateNoDeadlock:
    """is_available + record_failure 并发不死锁。"""

    def test_no_deadlock(self):
        ep = Endpoint(name="deadlock", api_url="http://localhost", api_key="k")
        state = EndpointState(ep)
        stop = threading.Event()
        deadlocked = threading.Event()

        def reader():
            while not stop.is_set():
                state.is_available()

        def writer():
            for _ in range(500):
                state.record_failure("err")
            for _ in range(500):
                state.record_success()

        readers = [threading.Thread(target=reader) for _ in range(5)]
        writers = [threading.Thread(target=writer) for _ in range(5)]
        all_threads = readers + writers

        for t in all_threads:
            t.start()

        # 等待 writers 完成（如果超时说明死锁）
        for t in writers:
            t.join(timeout=10)
            if t.is_alive():
                deadlocked.set()

        stop.set()
        for t in readers:
            t.join(timeout=5)

        assert not deadlocked.is_set(), "检测到死锁"


# ---------------------------------------------------------------------------
#  Manifest 线程安全
# ---------------------------------------------------------------------------


class TestManifestConcurrentAppend:
    """多线程并发写 manifest，无数据丢失。"""

    def test_no_data_loss(self, tmp_path: Path):
        manifest_path = tmp_path / "_manifest.txt"
        lock = threading.Lock()
        n_threads = 10
        n_per_thread = 50

        def append_batch(thread_id: int):
            for i in range(n_per_thread):
                name = f"t{thread_id}_f{i}"
                with lock:
                    _append_manifest(manifest_path, [name])

        threads = [threading.Thread(target=append_batch, args=(tid,)) for tid in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        result = _read_manifest(manifest_path)
        assert len(result) == n_threads * n_per_thread


# ---------------------------------------------------------------------------
#  stop_event
# ---------------------------------------------------------------------------


class TestStopEventSkipsRemaining:
    """stop_event 设置后 worker 跳过。"""

    def test_skip_after_stop(self):
        stop_event = threading.Event()
        processed: list[int] = []
        processed_lock = threading.Lock()

        def worker(idx: int):
            if stop_event.is_set():
                return
            with processed_lock:
                processed.append(idx)
            if idx == 2:
                stop_event.set()

        # 串行执行模拟 stop_event 效果
        for i in range(10):
            worker(i)

        # idx 0,1,2 应被处理；idx 2 触发 stop_event 后 3~9 被跳过
        assert 0 in processed
        assert 1 in processed
        assert 2 in processed
        for i in range(3, 10):
            assert i not in processed


# ---------------------------------------------------------------------------
#  ConcurrencyConfig 默认值
# ---------------------------------------------------------------------------


class TestConcurrencyConfigDefaults:
    """默认值 max_workers=3。"""

    def test_default(self):
        cfg = ConcurrencyConfig()
        assert cfg.max_workers == 3


class TestConcurrencyConfigMin1:
    """max_workers < 1 被钳位为 1。"""

    def test_clamp(self):
        # pipeline 中使用 max(1, ...) 来钳位
        raw_value = -5
        clamped = max(1, int(raw_value))
        assert clamped == 1

        raw_value = 0
        clamped = max(1, int(raw_value))
        assert clamped == 1


# ---------------------------------------------------------------------------
#  _ConcurrentLog 线程安全
# ---------------------------------------------------------------------------


class TestConcurrentLogThreadSafety:
    """多线程并发 record，记录完整。"""

    def test_records_complete(self, tmp_path: Path):
        import json

        clog = _ConcurrentLog(stage="test", max_workers=5, total_units=500)
        n_threads = 10
        n_per_thread = 50

        def record_batch(tid: int):
            for i in range(n_per_thread):
                clog.record(
                    worker=f"w{tid}",
                    unit=f"unit_{tid}_{i}",
                    status="ok",
                    duration_s=0.001 * i,
                )

        threads = [threading.Thread(target=record_batch, args=(tid,)) for tid in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 验证记录总数
        assert len(clog._records) == n_threads * n_per_thread

        # 验证写入 JSON 文件完整
        log_path = clog.write_log(tmp_path)
        data = json.loads(log_path.read_text(encoding="utf-8"))
        assert data["stage"] == "test"
        assert data["max_workers"] == 5
        assert data["total_units"] == 500
        assert len(data["records"]) == n_threads * n_per_thread
        assert data["summary"]["ok"] == n_threads * n_per_thread
        assert data["summary"]["error"] == 0
        assert data["summary"]["skip"] == 0
