# metrics.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import math
import time


def percentile(values: List[float], p: float) -> float:
    """Return the pth percentile (0-100) using a lightweight nearest-rank method."""
    if not values:
        return float("nan")
    xs = sorted(values)
    k = max(0, min(len(xs) - 1, math.ceil((p / 100.0) * len(xs)) - 1))
    return xs[k]


@dataclass
class EndpointStats:
    total_ms: List[float] = field(default_factory=list)
    model_ms: List[float] = field(default_factory=list)
    ok: int = 0
    err: int = 0
    items: int = 0  
    first_ts: Optional[float] = None
    last_ts: Optional[float] = None

    def record(self, total_ms: float, model_ms: Optional[float], ok: bool, items: int) -> None:
        """Store one request result (latencies, success/fail, and item count) and update the time window."""
        now = time.time()
        if self.first_ts is None:
            self.first_ts = now
        self.last_ts = now

        self.total_ms.append(float(total_ms))
        if model_ms is not None:
            self.model_ms.append(float(model_ms))

        if ok:
            self.ok += 1
        else:
            self.err += 1

        self.items += max(0, int(items))

    def _window_seconds(self) -> float:
        """Return elapsed seconds between first and last recorded request for throughput calculations."""
        if self.first_ts is None or self.last_ts is None:
            return 0.0
        dur = self.last_ts - self.first_ts
        return dur if dur > 0 else 0.0

    def summary(self) -> Dict[str, float]:
        """Compute aggregate stats (avg/p95 latency, error rate, and throughput) for this endpoint."""
        n = len(self.total_ms)

        avg_total = sum(self.total_ms) / n if n else float("nan")
        p95_total = percentile(self.total_ms, 95.0)

        nm = len(self.model_ms)
        avg_model = (sum(self.model_ms) / nm) if nm else float("nan")
        p95_model = percentile(self.model_ms, 95.0) if nm else float("nan")

        dur_s = self._window_seconds()
        rps = (n / dur_s) if dur_s > 0 else float("nan")
        ips = (self.items / dur_s) if dur_s > 0 else float("nan")
        err_rate = (self.err / n) if n else float("nan")

        return {
            "count": float(n),
            "ok": float(self.ok),
            "err": float(self.err),
            "err_rate": err_rate,
            "avg_total_ms": avg_total,
            "p95_total_ms": p95_total,
            "avg_model_ms": avg_model,
            "p95_model_ms": p95_model,
            "rps": rps,
            "ips": ips,
        }


class Metrics:
    def __init__(self) -> None:
        """Initialize the metrics registry that groups stats by endpoint."""
        self._by_endpoint: Dict[str, EndpointStats] = {}

    def record(
        self,
        endpoint: str,
        total_ms: float,
        model_ms: Optional[float],
        ok: bool,
        items: int = 1,
    ) -> None:
        """Record one request’s metrics under a specific endpoint key (e.g., '/detect')."""
        if endpoint not in self._by_endpoint:
            self._by_endpoint[endpoint] = EndpointStats()
        self._by_endpoint[endpoint].record(total_ms=total_ms, model_ms=model_ms, ok=ok, items=items)

    def format_summary(self) -> str:
        """Return a human-readable summary string for all endpoints (avg/p95 latency + throughput)."""
        lines = ["=== Performance Summary ==="]

        for ep in sorted(self._by_endpoint.keys()):
            st = self._by_endpoint[ep]
            s = st.summary()

            lines.append(ep)
            lines.append(
                f"  count={int(s['count'])}  ok={int(s['ok'])}  err={int(s['err'])}  err_rate={s['err_rate']*100:.1f}%"
            )
            lines.append(f"  total: avg={s['avg_total_ms']:.1f}ms  p95={s['p95_total_ms']:.1f}ms")

            if len(st.model_ms) > 0:
                lines.append(f"  model: avg={s['avg_model_ms']:.1f}ms  p95={s['p95_model_ms']:.1f}ms")

            # Avoid printing throughput if it can't be computed yet (NaN).
            if s["rps"] == s["rps"]:  # NaN check
                lines.append(f"  throughput={s['rps']:.2f} req/s")
            if s["ips"] == s["ips"] and s["ips"] > 0:
                lines.append(f"  items/sec={s['ips']:.2f}")

            lines.append("")

        return "\n".join(lines).rstrip()


metrics = Metrics()