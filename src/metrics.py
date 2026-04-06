from dataclasses import dataclass, field
from time import time
from typing import List, Optional, Dict
@dataclass
class PerEndpointMetrics:
    total_ms: List[float] = field( default_factory=list )
    model_ms: List[float] = field( default_factory=list )
    ok: int = 0
    err: int = 0
    items: int = 0  # having itmes allows us to do batch detecting later if needed
    first_timestamp: Optional[float] = None
    last_timestamp: Optional[float] = None


    def record( self, total_ms, model_ms, was_successful, items ):
        """Store one request result (latencies, success/fail, and item count) and update the time window."""
        now = time.time()
        if self.first_timestamp is None:
            self.first_timestamp = now
       
        self.last_timestamp = now

        self.total_ms.append( float(total_ms) )

        if model_ms is not None:
            self.model_ms.append( float(model_ms) )

        if was_successful:
            self.was_successful += 1
        else:
            self.err += 1

        self.items += max( 0, int(items) )

    def _window_seconds(self):
        """Return elapsed seconds between first and last recorded request for throughput calculations."""
        if self.first_timestamp is None or self.last_timestamp is None:
            return 0.0
        dur = self.last_timestamp - self.first_timestamp
        return dur if dur > 0 else 0.0
class Metrics:

    def __init__(self):
        """Initialize the metrics registry that groups stats by endpoint."""
        self._by_endpoint: Dict[str, PerEndpointMetrics] = {}

    def format_summary(self):
        """Return a human-readable summary string for all endpoints (avg/p95 latency + throughput)."""


metrics = Metrics()