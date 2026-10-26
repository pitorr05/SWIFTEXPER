"""Non-synchronizing phase instrumentation; synchronize once at request end."""
import time
from contextlib import contextmanager
import torch


@contextmanager
def optimization_timer(statistics, model):
    device = next(model.parameters()).device
    events = None
    if device.type == 'cuda':
        with torch.cuda.device(device):
            events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            events[0].record()
    start = time.perf_counter()
    try:
        yield
    finally:
        statistics['_optimization_host_seconds'] += time.perf_counter() - start
        if events:
            with torch.cuda.device(device):
                events[1].record()
            statistics['_optimization_events'].append(events)


def begin_request(statistics):
    statistics['_optimization_host_seconds'] = 0.0
    statistics['_optimization_events'] = []


def finish_request(statistics):
    events = statistics.pop('_optimization_events', [])
    # The harness has synchronized all visible CUDA devices before calling this.
    return {'optimization_host_seconds': statistics.pop('_optimization_host_seconds', 0.0),
            'optimization_gpu_ms': sum(start.elapsed_time(end) for start, end in events)}
