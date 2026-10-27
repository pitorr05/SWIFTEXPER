"""Greedy SWIFT comb control. No native layer search or acceptance is replaced.

Allocation is exact for estimated rewards, not GPU throughput. The predictor is
Dirichlet-smoothed frequency estimation, not an anytime-valid confidence sequence.
No theoretical guarantee is claimed for learned transitions or latency fits.
"""
from collections import defaultdict, OrderedDict
from contextlib import contextmanager
import math
import time
import numpy as np

PROTOCOL = 'swift_tree_control_v1'
POLICIES = ('native', 'raw_dp', 'rank_dp', 'stop_only', 'joint', 'fixed')
TOPK = 10


def allocate(cdfs, survival, max_nodes):
    """Max-plus DP over sibling widths; every retained depth contains the spine.

    Returns optimal yield/plans for every at-most node budget, INCLUDING root.
    Ending at an earlier depth is allowed. Rank increments need not be concave.
    """
    if max_nodes < 1 or len(cdfs) != len(survival):
        raise ValueError('Invalid comb allocation')
    limit = max_nodes - 1
    previous = np.full(limit + 1, -np.inf)
    previous[0] = 1.0
    best = previous.copy()
    best_depth = np.zeros(limit + 1, dtype=int)
    parents = []
    for depth, (cdf, reach) in enumerate(zip(cdfs, survival), 1):
        current = np.full(limit + 1, -np.inf)
        parent = np.zeros(limit + 1, dtype=int)
        for width in range(1, min(TOPK, limit) + 1):
            candidate = previous[:-width] + float(reach) * cdf[width - 1]
            better = candidate > current[width:]
            current[width:][better] = candidate[better]
            parent[width:][better] = width
        parents.append(parent)
        better = current > best
        best[better], best_depth[better] = current[better], depth
        previous = current
    used = [int(np.argmax(best[:budget + 1])) for budget in range(limit + 1)]
    plans = []
    for budget in used:
        widths = []
        for index in range(int(best_depth[budget]) - 1, -1, -1):
            width = int(parents[index][budget])
            widths.append(width)
            budget -= width
        plans.append(list(reversed(widths)))
    return best[used], plans


def probabilities(top):
    top = np.asarray(top, dtype=float)
    if top.shape != (TOPK,) or not np.isfinite(top).all() or (top < 0).any():
        raise ValueError('Expected ten finite nonnegative probabilities')
    top = top / max(1.0, top.sum())
    return np.r_[top, max(0.0, 1.0 - top.sum())]


class RankEstimator:
    """Learn only on reached greedy prefixes. Unreached parents are censored.

    Per-mask tables prevent silently treating different native masks as identical.
    Back off from depth-specific to pooled-depth counts for the SAME mask/bin.
    """
    def __init__(self, minimum=32):
        self.minimum = minimum
        self.counts = defaultdict(lambda: np.zeros(11))
        self.pooled = defaultdict(lambda: np.zeros(11))

    @staticmethod
    def bin(top):
        return min(9, int(float(top[0]) * 10))

    def predict(self, mask, depth, top, learned):
        raw = probabilities(top)
        local = self.counts.get((mask, depth, self.bin(top)))
        pooled = self.pooled.get((mask, self.bin(top)))
        counts = local if local is not None and local.sum() >= self.minimum else pooled
        if not learned or counts is None or counts.sum() < self.minimum:
            return raw, 0.0
        # Fixed symmetric Dirichlet prior with total mass one.
        return (counts + 1.0 / 11) / (counts.sum() + 1), float(counts.sum())

    def update(self, mask, depth, top, rank):
        if not 0 <= rank <= TOPK:
            raise ValueError('Invalid rank label')
        self.counts[mask, depth, self.bin(top)][rank] += 1
        self.pooled[mask, self.bin(top)][rank] += 1


class CostModel:
    """Nonnegative ridge fit from normal inference CUDA events; no extra forwards."""
    def __init__(self, minimum=32):
        self.minimum = minimum
        self.samples, self.drafts, self.rates, self.host = [], [], [], []
        self.coefficients = None

    @staticmethod
    def features(nodes, context):
        return np.array([1.0, nodes / 64.0, nodes * context / 262144.0])

    def add_verify(self, nodes, context, milliseconds):
        if milliseconds <= 0 or not math.isfinite(milliseconds):
            return
        self.samples = (self.samples + [(nodes, context, milliseconds / 1000)])[-256:]
        if len(self.samples) < self.minimum:
            return
        x = np.stack([self.features(n, c) for n, c, _ in self.samples])
        y = np.array([s for _, _, s in self.samples])
        best = float('inf')
        # Exact enumeration of active sets for this 3-variable NNLS ridge fit.
        for bits in range(1, 8):
            active = [i for i in range(3) if bits & (1 << i)]
            a = x[:, active]
            fit = np.linalg.solve(a.T @ a + np.eye(len(active)) * 1e-6, a.T @ y)
            if (fit < 0).any():
                continue
            candidate = np.zeros(3)
            candidate[active] = fit
            loss = np.square(x @ candidate - y).sum()
            if loss < best:
                best, self.coefficients = loss, candidate

    @property
    def ready(self):
        return self.coefficients is not None and len(self.drafts) >= self.minimum and bool(self.rates)

    def verify(self, nodes, context):
        return max(1e-6, float(self.features(nodes, context) @ self.coefficients))

    def draft(self):
        return float(np.median(self.drafts[-128:]))

    def rho(self):
        return sum(n for n, _ in self.rates[-32:]) / sum(t for _, t in self.rates[-32:])


class TreeControl:
    def __init__(self, policy, max_nodes=251, lookahead=2, fixed_depth=4, fixed_width=3,
                 min_labels=32, cache_size=128):
        if policy not in POLICIES or not 1 <= max_nodes <= 251 or lookahead not in (1, 2):
            raise ValueError('Invalid tree policy/budget/lookahead')
        if not 1 <= fixed_depth <= 25 or not 1 <= fixed_width <= TOPK or min_labels < 1:
            raise ValueError('Invalid fixed tree/label minimum')
        self.policy, self.max_nodes, self.lookahead = policy, max_nodes, lookahead
        self.fixed_depth, self.fixed_width = fixed_depth, fixed_width
        self.estimator, self.cost = RankEstimator(min_labels), CostModel()
        self.transitions = defaultdict(lambda: np.zeros(10))
        self.top_sum = defaultdict(lambda: np.zeros(10))
        self.top_count = defaultdict(int)
        self.pending, self.cache, self.cache_size = [], OrderedDict(), cache_size
        self.begin_request()

    def begin_request(self):
        if self.pending:
            raise RuntimeError('Unresolved events from previous request')
        self.events, self.gpu_ms, self.host_seconds = [], defaultdict(float), defaultdict(float)
        self.controller_seconds = 0.0
        self.cache_hits = self.fallback_rounds = self.controlled_rounds = 0

    def resolve(self, wait=False):
        import torch
        if wait and self.pending:
            torch.cuda.synchronize()
        remaining = []
        for phase, start, end, nodes, context in self.pending:
            if not end.query():
                remaining.append((phase, start, end, nodes, context))
                continue
            milliseconds = start.elapsed_time(end)
            self.gpu_ms[phase] += milliseconds
            if phase == 'verify':
                self.cost.add_verify(nodes, context, milliseconds)
            elif phase == 'draft':
                self.cost.drafts = (self.cost.drafts + [milliseconds / 1000])[-256:]
        self.pending = remaining

    @contextmanager
    def phase(self, phase, nodes=0, context=0):
        import torch
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        begin = time.perf_counter()
        try:
            yield
        finally:
            end.record()
            self.host_seconds[phase] += time.perf_counter() - begin
            self.pending.append((phase, start, end, nodes, context))

    def begin_draft(self, model, context, max_depth):
        self.round_start = time.perf_counter()
        self.resolve()
        self.mask = tuple(tuple(sorted(map(int, x))) for x in model.get_skip_layers())
        self.top, self.ids, self.predictions, self.support = [], [], [], []
        # Native SWIFT always performs at least one draft forward, even when only
        # the root fits the remaining output budget. Preserve that in its control.
        self.max_depth = min(25, max(1, max_depth) if self.policy == 'native' else max_depth)
        self.context = context
        self.previous_bin, self.reason = None, 'native'

    def distribution(self, depth, top):
        return self.estimator.predict(self.mask, depth, top, self.policy in ('rank_dp', 'joint'))

    @staticmethod
    def rewards(predictions):
        cdfs, survival, reach = [], [], 1.0
        for p in predictions:
            cdfs.append(np.cumsum(p[:TOPK]))
            survival.append(reach)
            reach *= p[0]
        return cdfs, survival

    def terminal(self, predictions):
        cdfs, survival = self.rewards(predictions)
        yields, plans = allocate(cdfs, survival, min(self.max_nodes, 1 + TOPK * len(predictions)))
        nodes = np.array([1 + sum(p) for p in plans])
        features = np.stack([self.cost.features(int(n), self.context) for n in nodes])
        costs = np.maximum(1e-6, features @ self.cost.coefficients)
        costs += float(np.mean(self.cost.host[-32:])) if self.cost.host else 0.0
        utility = yields - self.cost.rho() * costs
        best = int(np.argmax(utility))
        return float(utility[best]), plans[best], float(yields[best])

    def future(self, predictions, depth, current_bin, horizon):
        # Reuse the DP frontier when appending a hypothetical step: do not rebuild
        # every old depth or reconstruct plans at each lookahead leaf.
        cap = min(self.max_nodes, 1 + TOPK * min(self.max_depth, depth + horizon))
        rho, draft_cost = self.cost.rho(), self.cost.draft()
        verify_costs = np.array([self.cost.verify(n, self.context) for n in range(1, cap + 1)])
        host = float(np.mean(self.cost.host[-32:])) if self.cost.host else 0.0
        price = rho * (verify_costs + host)

        def extend(previous, best, reach, prediction):
            current = np.full(cap, -np.inf)
            cdf = np.cumsum(prediction[:TOPK])
            for width in range(1, min(TOPK, cap - 1) + 1):
                current[width:] = np.maximum(current[width:], previous[:-width] + reach*cdf[width-1])
            return current, np.maximum(best, current), reach * prediction[0]

        last = np.full(cap, -np.inf)
        last[0] = 1.0
        best, reach = last.copy(), 1.0
        for prediction in predictions:
            last, best, reach = extend(last, best, reach, prediction)

        def value(last, best, reach, depth, current_bin, remaining):
            stop = float(np.max(best - price))
            if not remaining or depth >= self.max_depth:
                return stop
            weights = self.transitions.get((self.mask, min(depth, 5), current_bin))
            if weights is None or weights.sum() < 32:
                return stop
            expectation = 0.0
            for next_bin, probability in enumerate(weights / weights.sum()):
                key = self.mask, next_bin
                if probability == 0:
                    continue
                if self.top_count[key] < 4:
                    return stop
                top = self.top_sum[key] / self.top_count[key]
                prediction, _ = self.distribution(depth, top)
                state = extend(last, best, reach, prediction)
                expectation += probability * value(*state, depth+1, next_bin, remaining-1)
            return max(stop, expectation - rho*draft_cost)

        return float(np.max(best-price)), value(last, best, reach, depth, current_bin, horizon)

    def observe_draft(self, top, ids):
        begin = time.perf_counter()
        top = np.asarray(top, dtype=float)
        depth, current_bin = len(self.top), self.estimator.bin(top)
        key = self.mask, current_bin
        self.top_sum[key] += top
        self.top_count[key] += 1
        if self.previous_bin is not None:
            self.transitions[self.mask, min(depth, 5), self.previous_bin][current_bin] += 1
        self.previous_bin = current_bin
        prediction, support = self.distribution(depth, top)
        self.top.append(top)
        self.ids.append(list(map(int, ids)))
        self.predictions.append(prediction)
        self.support.append(support)
        if self.policy == 'fixed':
            stop, self.reason = len(self.top) >= self.fixed_depth, 'fixed'
        elif self.policy in ('stop_only', 'joint') and self.cost.ready:
            transitions = self.transitions.get((self.mask, min(len(self.top), 5), current_bin))
            if transitions is None or transitions.sum() < 32:
                stop, self.reason = top[0] < .8, 'transition_native'
            else:
                now, later = self.future(self.predictions, len(self.top), current_bin, self.lookahead)
                stop = later <= now + 1e-9
                self.reason = 'bellman_stop' if stop else 'bellman_continue'
        else:
            stop = top[0] < .8
            self.reason = 'cold_native' if self.policy in ('stop_only', 'joint') else 'native'
        self.controller_seconds += time.perf_counter() - begin
        return stop or len(self.top) >= self.max_depth

    def select(self, native_widths):
        begin = time.perf_counter()
        if self.policy == 'native':
            plan, expected = list(native_widths), None
        elif self.policy == 'fixed':
            plan, expected = [self.fixed_width] * min(len(self.top), self.fixed_depth), None
            if 1 + sum(plan) > self.max_nodes:
                raise ValueError('Fixed tree exceeds budget')
        elif self.policy in ('raw_dp', 'rank_dp'):
            cdfs, survival = self.rewards(self.predictions)
            values, plans = allocate(cdfs, survival, min(self.max_nodes, 1 + sum(native_widths)))
            plan, expected = plans[-1], float(values[-1])
        elif self.cost.ready:
            _, plan, expected = self.terminal(self.predictions)
        else:
            plan, expected = list(native_widths), None
        self.widths, self.predicted_yield = plan, expected
        if self.policy in ('stop_only', 'joint') and self.reason in ('cold_native', 'transition_native'):
            self.fallback_rounds += 1
        elif self.policy != 'native':
            self.controlled_rounds += 1
        choices = [[0] * d + [r] for d, b in enumerate(plan) for r in range(b)]
        self.controller_seconds += time.perf_counter() - begin
        return choices

    def buffers(self, choices, builder, device):
        key = tuple(tuple(p) for p in choices), str(device)
        if key in self.cache:
            self.cache_hits += 1
            self.cache.move_to_end(key)
            self.node_locations = self.cache[key]['_node_locations']
            return self.cache[key]
        if choices:
            result = builder(choices, device=device)
        else:
            import torch
            result = {'swift_attn_mask': torch.ones((1, 1, 1, 1), device=device),
                      'tree_indices': torch.zeros(1, dtype=torch.long, device=device),
                      'swift_position_ids': torch.zeros(1, dtype=torch.long, device=device),
                      'retrieve_indices': torch.zeros((1, 1), dtype=torch.long, device=device),
                      'p_indices': [[-1]], 'b_indices': [[[]]]}
        locations = {}
        for row, indices in enumerate(result['retrieve_indices'].tolist()):
            for column, node in enumerate(indices):
                if node >= 0:
                    locations[node] = (row, column)
        result['_node_locations'] = locations
        self.node_locations = locations
        self.cache[key] = result
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return result

    def feedback(self, logits, choices, committed, steps, elapsed, eos=False):
        begin = time.perf_counter()
        self.resolve()
        positions = {tuple(p): i + 1 for i, p in enumerate(sorted(choices, key=lambda p: (len(p), p)))}
        labels = []
        for depth, (top, ids, p) in enumerate(zip(self.top, self.ids, self.predictions)):
            if eos and depth >= committed - 1:
                break
            prefix = (0,) * depth
            if depth and prefix not in positions:
                break
            row, column = self.node_locations[0 if depth == 0 else positions[prefix]]
            target = int(logits[row, column].argmax().item())
            rank = ids.index(target) if target in ids else TOPK
            labels.append({'depth': depth, 'rank': rank, 'confidence': float(top[0]),
                           'prediction': p.tolist(), 'support': self.support[depth],
                           'brier': float(np.square(p - np.eye(11)[rank]).sum()),
                           'raw_brier': float(np.square(probabilities(top) - np.eye(11)[rank]).sum()),
                           'log_loss': -math.log(max(float(p[rank]), 1e-12))})
            self.estimator.update(self.mask, depth, top, rank)
            if rank != 0:
                break
        # This host round duration is a control surrogate, not the synchronized
        # request benchmark; include the actual feedback work before pricing time.
        elapsed = time.perf_counter() - self.round_start
        if elapsed > 0:
            self.cost.rates = (self.cost.rates + [(int(committed), elapsed)])[-64:]
        self.events.append({'draft_depth': len(self.top), 'widths': self.widths,
                            'verified_nodes': 1 + len(choices), 'committed': int(committed),
                            'draft_steps': int(steps), 'reason': self.reason,
                            'predicted_M': self.predicted_yield, 'rank_labels': labels,
                            'mask': [list(x) for x in self.mask], 'cost_ready': self.cost.ready})
        self.controller_seconds += time.perf_counter() - begin

    def finish(self):
        self.resolve(wait=True)
        return {'tree_control_protocol': PROTOCOL, 'tree_policy': self.policy,
                'controller_seconds': self.controller_seconds, 'phase_gpu_ms': dict(self.gpu_ms),
                'phase_host_seconds': dict(self.host_seconds), 'tree_buffer_hits': self.cache_hits,
                'controlled_rounds': self.controlled_rounds, 'fallback_rounds': self.fallback_rounds,
                'tree_events': self.events}
