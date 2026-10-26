"""Budgeted SWIFT mask selection. Pure scheduling above, isolated GPU evaluation below.

The utility is a linear-draft latency proxy, NOT SWIFT's measured tree speed.
No statistical guarantee for dependent/nonstationary contexts is assumed.
"""
from dataclasses import dataclass
import hashlib
import math
import random
import time


POLICIES = ('match_uniform', 'cost_uniform', 'match_halving',
            'cost_halving', 'cost_halving_refresh')
PROTOCOL = 'mask_search_v1'


@dataclass(frozen=True)
class Mask:
    attention: tuple
    mlp: tuple

    @classmethod
    def from_model(cls, model):
        return cls(*(tuple(sorted(map(int, values))) for values in model.get_skip_layers()))

    def apply(self, model):
        model.set_skip_layers(list(self.attention), list(self.mlp))

    def as_dict(self):
        return {'attention': list(self.attention), 'mlp': list(self.mlp)}


def candidate_pool(incumbent, layers, skip_ratio, seed, count=8):
    """Preserve boundary layers and use local RNG, independent of generation RNG."""
    if layers < 3 or not 0 < skip_ratio < 1 or count != 8:
        raise ValueError('Require >=3 layers, skip ratio in (0,1), and eight candidates')
    population = 2 * (layers - 2)
    skips = int(population * skip_ratio)
    if math.comb(population, skips) < count:
        raise ValueError('Insufficient unique masks for eight-candidate experiment')
    rng = random.Random(seed)
    masks = [incumbent]
    while len(masks) < count:
        chosen = rng.sample(range(population), skips)
        mask = Mask(tuple(sorted(i+1 for i in chosen if i < layers-2)),
                    tuple(sorted(i-(layers-2)+1 for i in chosen if i >= layers-2)))
        if mask not in masks:
            masks.append(mask)
    return masks


def prefix_yield(matches, horizon=5):
    """Mean 1+accepted-prefix length on all complete, overlapping replay windows."""
    if horizon < 1 or len(matches) < horizon:
        raise ValueError('Not enough matches for the requested horizon')
    total = 0
    for start in range(len(matches)-horizon+1):
        accepted = 0
        for match in matches[start:start+horizon]:
            if not match:
                break
            accepted += 1
        total += 1 + accepted
    return total / (len(matches)-horizon+1)


def schedule(policy):
    if policy not in POLICIES:
        raise ValueError(f'Unknown search policy: {policy}')
    # Each replay with n labels consumes n+1 input tokens. Both total 192.
    return ((8, 7), (4, 15), (2, 31)) if 'halving' in policy else ((8, 23),)


def select_mask(masks, policy, evaluate, order_seed=0):
    """Fresh scores each stage; ties favor the incumbent/stable candidate order."""
    survivors = list(range(len(masks)))
    observations = []
    rng = random.Random(order_seed)
    for size, labels in schedule(policy):
        if len(survivors) != size:
            raise ValueError('Schedule requires exactly eight initial masks')
        scores = {}
        rng.shuffle(survivors)
        for index in survivors:
            item = dict(evaluate(masks[index], labels))
            key = 'utility_proxy' if policy.startswith('cost') else 'matchness'
            score = item[key]
            if not math.isfinite(score):
                raise ValueError('Nonfinite mask score')
            scores[index] = score
            observations.append(dict(item, candidate=index, labels=labels))
        survivors.sort(key=lambda i: (-scores[i], i))
        survivors = survivors[:max(1, len(survivors)//2)]
    return masks[survivors[0]], observations


class MaskSearch:
    """Called only at accepted-prefix boundaries in a prepared research snapshot."""
    def __init__(self, model, policy, seed=2024, token_budget=824, interval=32,
                 horizon=5, time_budget_ms=0, diagnostic_samples=0):
        if policy not in POLICIES or token_budget < 206 or interval < 1 or horizon != 5:
            raise ValueError('Invalid mask-search configuration (horizon is fixed to 5 in v1)')
        if not math.isfinite(time_budget_ms) or time_budget_ms < 0 or diagnostic_samples < 0:
            raise ValueError('Negative search time budget/diagnostic sample count')
        self.model, self.policy, self.seed = model, policy, seed
        self.token_budget, self.interval, self.horizon = token_budget, interval, horizon
        self.time_budget_ms = time_budget_ms
        self.diagnostic_samples = diagnostic_samples
        self.request_number = 0

    def begin_request(self, context, statistics):
        self.request_number += 1
        self.task_id = str(context['task_id'])
        self.used_tokens = self.last_checkpoint = 0
        self.search_seconds = self.diagnostic_seconds = 0.0
        self.events = []
        self._commit_clock = time.perf_counter()
        self._search_at_commit = 0.0
        statistics['optimization'] = True

    def observe_commit(self, committed, draft_steps):
        """Track performance AFTER selection; never use it to retroactively pick output."""
        now = time.perf_counter()
        elapsed = now-self._commit_clock-(self.search_seconds-self._search_at_commit)
        if self.events:
            event = self.events[-1]
            event['future_committed_tokens'] += int(committed)
            event['future_draft_steps'] += int(draft_steps)
            event['future_rounds'] += 1
            event['future_host_seconds_excluding_search'] += max(0, elapsed)
        self._commit_clock = now
        self._search_at_commit = self.search_seconds

    def metrics(self):
        return {'mask_search_protocol': PROTOCOL, 'search_policy': self.policy,
                'candidate_input_tokens': self.used_tokens,
                'mask_search_seconds': self.search_seconds,
                'diagnostic_probe_seconds': self.diagnostic_seconds,
                'search_events': self.events}

    def ready(self, new_token_num, statistics):
        if (self.used_tokens + 206 > self.token_budget
                or (self.time_budget_ms and self.search_seconds*1000 >= self.time_budget_ms)):
            statistics['optimization'] = False
            return False
        return new_token_num >= 33 and (not self.events or new_token_num-self.last_checkpoint >= self.interval)

    def optimize(self, model, output_ids, input_past_key_values_data,
                 input_current_length_data, new_token_num, statistics,
                 optimizer=None, utility=None, position_ids=None,
                 verified_past_key_values_data=None, verified_current_length_data=None):
        import torch
        if not self.ready(new_token_num, statistics):
            return
        if output_ids.shape[1] < 33:
            return
        if verified_past_key_values_data is None or verified_current_length_data is None:
            raise ValueError('Research evaluation requires verified target KV and lengths')
        start = time.perf_counter()
        incumbent = Mask.from_model(model)
        old_tree = getattr(model.model, 'swift_mask', None)
        live_lengths = verified_current_length_data.clone()
        prompt_length = int(input_current_length_data[0])
        offset = output_ids.shape[1]-33 if self.policy.endswith('_refresh') else 0
        prefix_length = prompt_length + offset
        window = output_ids[:, offset:offset+33]
        seed_bytes = f'{self.seed}:{self.task_id}:{len(self.events)}'.encode('utf-8')
        seed = int.from_bytes(hashlib.sha256(seed_bytes).digest()[:8], 'big')
        pool = candidate_pool(incumbent, model.config.num_hidden_layers,
                              statistics['skip_ratio'], seed)
        event = {'checkpoint_tokens': int(new_token_num), 'context_offset': offset,
                 'proposal': 'seeded_random_plus_incumbent', 'candidate_seed': seed,
                 'incumbent': incumbent.as_dict(), 'candidates': [m.as_dict() for m in pool]}
        selected = incumbent
        try:
            model.model.swift_mask = None
            evaluator = ReplayEvaluator(model, verified_past_key_values_data,
                                        prefix_length, window, self.horizon)
            # Same dense profile and eight single-token profiles for all four ablations.
            evaluator.profile_verifier()
            profile_order = list(range(len(pool)))
            random.Random(seed ^ 0x5A17).shuffle(profile_order)
            for index in profile_order:
                evaluator.profile_drafter(pool[index])
            selected, observations = select_mask(pool, self.policy, evaluator.evaluate, order_seed=seed)
            event.update(selected=selected.as_dict(), observations=observations,
                         verifier_seconds=evaluator.verify_seconds,
                         profile_order=profile_order,
                         token_units=206, diagnostic=[], future_committed_tokens=0,
                         future_rounds=0, future_draft_steps=0,
                         future_host_seconds_excluding_search=0.0)
            self.used_tokens += 206
            if self.request_number <= self.diagnostic_samples and not self.events:
                diagnostic_start = time.perf_counter()
                event['diagnostic'] = [dict(mask=m.as_dict(), **evaluator.probe(m)) for m in pool]
                self.diagnostic_seconds += time.perf_counter()-diagnostic_start
            self.events.append(event)
            self.last_checkpoint = int(new_token_num)
            statistics['opt_iter'] += 1
        except Exception:
            selected = incumbent
            raise
        finally:
            selected.apply(model)
            model.model.swift_mask = old_tree
            if not torch.equal(live_lengths, verified_current_length_data):
                raise RuntimeError('Mask evaluation modified authoritative KV lengths')
            self.search_seconds += time.perf_counter()-start


class ReplayEvaluator:
    """Candidate replay/probes always use scratch KV. No probe output is committed."""
    def __init__(self, model, verified_data, prefix_length, window, horizon):
        import torch
        self.model, self.verified_data = model, verified_data
        self.prefix_length, self.window, self.horizon = prefix_length, window, horizon
        self.device = next(model.parameters()).device
        self.draft_seconds = {}
        self.verify_seconds = None
        self.torch = torch

    def scratch(self):
        from model.swift.kv_cache import clone_past_key_values
        data = [tensor.clone() for tensor in self.verified_data]
        lengths = self.torch.full((2*self.model.config.num_hidden_layers,),
                                  self.prefix_length, dtype=self.torch.long, device='cpu')
        return clone_past_key_values(self.model, data, lengths)

    def synchronize(self):
        if self.device.type == 'cuda':
            self.torch.cuda.synchronize(self.device)

    def forward(self, mask, ids, cache, draft):
        mask.apply(self.model)
        with self.torch.inference_mode(), self.model.self_draft(enabled=draft):
            output = self.model.model(input_ids=ids, attention_mask=None, past_key_values=cache)
            return self.model.lm_head(output[0])

    def timed_forward(self, mask, ids, draft):
        cache = self.scratch()
        self.synchronize()
        start = time.perf_counter()
        logits = self.forward(mask, ids, cache, draft)
        self.synchronize()
        elapsed = time.perf_counter()-start
        return logits, max(elapsed, 1e-12)

    def profile_verifier(self):
        dense = Mask((), ())
        _, self.verify_seconds = self.timed_forward(dense, self.window[:, :self.horizon+1], False)

    def profile_drafter(self, mask):
        _, self.draft_seconds[mask] = self.timed_forward(mask, self.window[:, :1], True)

    def evaluate(self, mask, labels):
        logits, replay_seconds = self.timed_forward(mask, self.window[:, :labels+1], True)
        matches = (logits[:, :-1].argmax(-1) == self.window[:, 1:labels+1]).squeeze(0).tolist()
        mean_prefix = prefix_yield(matches, self.horizon)
        denominator = self.horizon*self.draft_seconds[mask] + self.verify_seconds
        return {'matchness': sum(matches)/len(matches), 'prefix_yield_proxy': mean_prefix,
                'draft_single_seconds': self.draft_seconds[mask], 'replay_seconds': replay_seconds,
                'utility_proxy': mean_prefix/denominator}

    def probe(self, mask):
        """Actual linear draft/verify diagnostic on the SAME saved prefix for each mask.

        Includes cache cloning in diagnostic wall time. This is not SWIFT tree M.
        """
        self.synchronize()
        start = time.perf_counter()
        draft_cache = self.scratch()
        token = self.window[:, :1]
        drafted = []
        eos = getattr(self.model.config, 'eos_token_id', None)
        eos = set(eos if isinstance(eos, (list, tuple)) else ([] if eos is None else [eos]))
        for _ in range(self.horizon):
            logits = self.forward(mask, token, draft_cache, True)
            token = logits[:, -1:].argmax(-1)
            drafted.append(token)
            if int(token.item()) in eos:
                break
        ids = self.torch.cat([self.window[:, :1]] + drafted, dim=1)
        logits = self.forward(Mask((), ()), ids, self.scratch(), False)
        matches = (logits[:, :-1].argmax(-1) == ids[:, 1:]).squeeze(0).tolist()
        accepted = 0
        for match in matches:
            if not match:
                break
            accepted += 1
        committed = ids[0, 1:accepted+1].tolist() + [int(logits[0, accepted].argmax().item())]
        for index, value in enumerate(committed):
            if value in eos:
                committed = committed[:index+1]
                break
        self.synchronize()
        elapsed = time.perf_counter()-start
        return {'linear_committed_tokens': len(committed), 'seconds_including_copy': elapsed,
                'linear_tokens_per_second': len(committed)/elapsed,
                'scope': 'counterfactual_linear_probe_not_tree_speed'}
