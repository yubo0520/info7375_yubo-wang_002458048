from contextlib import contextmanager
from .base import BaseTracer
try:
    from .agentops import AgentOpsTracer
except ImportError:
    AgentOpsTracer = None
from .triplet import TripletExporter


class NullTracer(BaseTracer):
    """No-op tracer used when agentops.sdk is not available."""

    @contextmanager
    def trace_context(self, name=None):
        yield self

    def get_last_trace(self):
        return []

    def init(self, *args, **kwargs):
        pass

    def teardown(self, *args, **kwargs):
        pass

    def init_worker(self, worker_id):
        pass

    def teardown_worker(self, worker_id):
        pass
