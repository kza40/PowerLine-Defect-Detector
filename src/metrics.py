from dataclasses import dataclass, field
import math
import time
from typing import List, Optional, Dict

def percentile( values: List[float], p: float ):
    """Return the pth percentile 0-100."""
    if not values:
        return float("nan")
    sorted_values = sorted( values )

    target_position = ( p / 100.0 ) * len( sorted_values )
    rounded_position = math.ceil( target_position )
    zero_based_index = rounded_position - 1

    clamped_index = max( 0, min( len(sorted_values) - 1, zero_based_index ) )

    return sorted_values[clamped_index]

@dataclass
class PerEndpointMetrics:
    total_ms: List[float] = field( default_factory=list )
    model_ms: List[float] = field( default_factory=list )
    success_count: int = 0
    error_count: int = 0
    items: int = 0  # having times allows us to do batch detecting later if needed
    first_timestamp: Optional[float] = None
    last_timestamp: Optional[float] = None


    def record(self, total_ms: float, model_ms: Optional[float], was_successful: bool, items: int) -> None:
        """Record one request's timing + success/fail for this endpoint."""
        now = time.time()
        if self.first_timestamp is None:
            self.first_timestamp = now
        self.last_timestamp = now

        self.total_ms.append( float(total_ms) )
        if model_ms is not None:
            self.model_ms.append( float(model_ms) )

        if was_successful:
            self.success_count += 1
        else:
            self.error_count += 1

        self.items += max( 0, int(items) )

    def _window_seconds(self):
        """Return elapsed seconds between first and last recorded request for throughput calculations."""
        if self.first_timestamp is None or self.last_timestamp is None:
            return 0.0
        dur = self.last_timestamp - self.first_timestamp
        return dur if dur > 0 else 0.0
    
    def summary(self):
        """Compute aggregate avg/p95 latency, error rate, and throughput for this endpoint."""
        request_count = len( self.total_ms )

        average_total_latency_ms = (
            sum( self.total_ms ) / request_count
            if request_count
            else float("nan")
        )
        p95_total_latency_ms = percentile( self.total_ms, 95.0 )

        model_timing_sample_count = len( self.model_ms )
        average_model_latency_ms = (
            sum( self.model_ms ) / model_timing_sample_count
            if model_timing_sample_count
            else float("nan")
        )
        p95_model_latency_ms = (
            percentile( self.model_ms, 95.0 )
            if model_timing_sample_count
            else float("nan")
        )

        measurement_window_seconds = self._window_seconds()

        requests_per_second = (
            request_count / measurement_window_seconds
            if measurement_window_seconds > 0
            else float("nan")
        )
        items_per_second = (
            self.items / measurement_window_seconds
            if measurement_window_seconds > 0
            else float("nan")
        )
        error_rate = (
            self.error_count / request_count
            if request_count
            else float("nan")
        )

        return {
            "count": float(request_count),
            "success": float(self.success_count),
            "errors": float(self.error_count),
            "err_rate": error_rate,
            "avg_total_ms": average_total_latency_ms,
            "p95_total_ms": p95_total_latency_ms,
            "avg_model_ms": average_model_latency_ms,
            "p95_model_ms": p95_model_latency_ms,
            "rps": requests_per_second,
            "ips": items_per_second,
        }
    
class Metrics:

    def __init__(self):
        """Initialize the metrics registry that groups stats by endpoint."""
        self._by_endpoint: Dict[str, PerEndpointMetrics] = {}

    def format_summary(self) -> str:
        """Return a readable summary string for all endpoints ( avg/p95 latency + throughput )."""
        summary_lines = ["=== Performance Summary ==="]

        for endpoint_name in sorted( self._by_endpoint.keys() ):
            endpoint_stats = self._by_endpoint[endpoint_name]
            summary_values = endpoint_stats.summary()

            summary_lines.append( endpoint_name )
            summary_lines.append(
                f"  count={int(summary_values['count'])}  "
                f"success={int(summary_values['success'])}  "
                f"errors={int(summary_values['errors'])}  "
                f"err_rate={summary_values['err_rate'] * 100:.1f}%"
            )
            summary_lines.append(
                f"  total: avg={summary_values['avg_total_ms']:.1f}ms  "
                f"p95={summary_values['p95_total_ms']:.1f}ms"
            )

            if len( endpoint_stats.model_ms ) > 0:
                summary_lines.append(
                    f"  model: avg={summary_values['avg_model_ms']:.1f}ms  "
                    f"p95={summary_values['p95_model_ms']:.1f}ms"
                )

            # Avoiding printing throughput if it can't be computed yet (NaN).
            if summary_values["rps"] == summary_values["rps"]:  # NaN check
                summary_lines.append(
                    f"  throughput={summary_values['rps']:.2f} req/s"
                )

            if summary_values["ips"] == summary_values["ips"] and summary_values["ips"] > 0:
                summary_lines.append(
                    f"  items/sec={summary_values['ips']:.2f}"
                )

            summary_lines.append("")

        return "\n".join(summary_lines).rstrip()

    def record(
        self,
        endpoint: str,
        total_ms: float,
        model_ms: Optional[float],
        was_successful: bool,
        items: int = 1,
        ):
        """Record one request's metrics under a specific endpoint key."""
        if endpoint not in self._by_endpoint:
            self._by_endpoint[endpoint] = PerEndpointMetrics()
        self._by_endpoint[endpoint].record( total_ms=total_ms, model_ms=model_ms, was_successful=was_successful, items=items )

metrics = Metrics()