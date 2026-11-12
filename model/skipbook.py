"""SkipBook: a curated, cross-request book of SWIFT skip-layer configurations.

SWIFT (Xia et al., ICLR 2025) runs search-then-freeze over one input stream: one
optimizer, one incumbent mask, frozen after early stopping. AdaptiveCache keyed
masks by an extra dense embedding forward. SkipBook separates three roles:

  Explorer  SWIFT's own random+Bayesian search (`swift_optimization`, unchanged),
            woken only for a new context or after a detected drift; each context
            owns a fresh optimizer so score histories from other domains do not mix.
  Judge     measured decode tokens-per-second of a request served by a frozen mask
            (the tokens-per-time objective of KnapSpec, measured online instead of
            estimated offline). Search requests never produce rewards.
  Curator   per-context arm set: search results are admitted as arms, arms are
            selected with Discounted UCB (Garivier & Moulines, 2011), dominated arms
            are removed by successive elimination (Even-Dar et al., 2006), and
            Page-Hinkley (Page, 1954; Hinkley, 1971) on accepted length triggers
            re-exploration.

Contexts are retrieved by cosine similarity of the mean-pooled last hidden state of
the dense prefill that SWIFT already computes (same representation family as
KNN-SSD / AdaptiveCache, but no additional forward pass).

Variant `prefill_cache` reproduces AdaptiveCache semantics exactly (stream optimizer,
miss -> SWIFT continues, hit -> reuse mask without search, admission on miss), with
the prefill key as the only difference.

The module is pure Python/NumPy except `pool_prefill` and the prefill hook; torch is
imported lazily so the bookkeeping is unit-testable on CPU.
"""
from collections import OrderedDict
import math
import time

import numpy as np

PROTOCOL = 'skipbook_v1'
VARIANTS = ('skipbook', 'prefill_cache')
SEARCH_KEYS = ('origin_score', 'opt_iter', 'tolerance_iter', 'optimization')


# ----------------------------------------------------------------------------- keys
def pool_prefill(hidden, mode='mean'):
    """hidden: [1, L, H] last hidden state of the dense prefill -> L2-normalised float32 key."""
    import torch
    with torch.inference_mode():
        states = hidden[0]
        if mode.startswith('last'):
            states = states[-int(mode[4:]):]
        elif mode != 'mean':
            raise ValueError(f'Unknown key pooling: {mode}')
        key = states.float().mean(0)
        key = torch.nn.functional.normalize(key, dim=0)
        return key.cpu().numpy().astype(np.float32)


def normalize(vector):
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm == 0:
        raise ValueError('Key must be finite and nonzero')
    return vector / norm


def as_mask(mask):
    return (tuple(sorted(int(x) for x in mask[0])), tuple(sorted(int(x) for x in mask[1])))


# ----------------------------------------------------------------------------- detectors
class PageHinkley:
    """One-sided Page-Hinkley test for a DECREASE of a positive signal.

    The signal is normalised by its running mean, so `delta` is a relative tolerance
    (0.05 = drops under 5% are ignored) and `threshold` is in relative units.
    """
    def __init__(self, delta=0.05, threshold=1.0, min_samples=5):
        self.delta, self.threshold, self.min_samples = delta, threshold, min_samples
        self.reset()

    def reset(self):
        self.count, self.mean, self.cumulative, self.minimum = 0, 0.0, 0.0, 0.0

    def update(self, value):
        if not math.isfinite(value) or value <= 0:
            return False
        self.count += 1
        self.mean += (value - self.mean) / self.count
        self.cumulative += 1.0 - value / self.mean - self.delta
        self.minimum = min(self.minimum, self.cumulative)
        if self.count >= self.min_samples and self.cumulative - self.minimum > self.threshold:
            self.reset()
            return True
        return False


class Arm:
    def __init__(self, mask, source):
        self.mask, self.source = as_mask(mask), source
        self.weight = 0.0      # discounted pull count N_t(gamma, i)
        self.total = 0.0       # discounted reward sum
        self.pulls = 0         # undiscounted pulls (for elimination warm-up)

    def mean(self):
        return self.total / self.weight if self.weight > 0 else None

    def as_dict(self):
        return {'attention': list(self.mask[0]), 'mlp': list(self.mask[1]), 'source': self.source,
                'pulls': self.pulls, 'mean_reward': self.mean()}


class Context:
    def __init__(self, ident, key, warm_mask, warm_source):
        self.ident, self.centroid, self.members = ident, normalize(key), 1
        self.arms = [Arm(warm_mask, warm_source)]
        self.search = None          # dict of SWIFT search state while exploring
        self.search_mask = None     # SWIFT incumbent inside the running search
        self.reward_scale = None    # running mean of rewards (scale-free UCB)
        self.rewards = 0
        self.drift = PageHinkley()
        self.explorations = 0

    def absorb(self, key):
        self.members += 1
        self.centroid = normalize(self.centroid + (normalize(key) - self.centroid) / self.members)


# ----------------------------------------------------------------------------- the book
class SkipBook:
    def __init__(self, model, statistics, make_optimizer, variant='skipbook', threshold=0.85,
                 max_arms=3, capacity=32, gamma=0.95, ucb_c=0.1, min_pulls=3, drift=True,
                 key_pool='mean', cache_capacity=100, stream_optimizer=None):
        if variant not in VARIANTS:
            raise ValueError(f'Unknown SkipBook variant {variant}')
        if not 0 < gamma <= 1 or not -1 <= threshold <= 1 or max_arms < 1 or capacity < 1:
            raise ValueError('Invalid SkipBook configuration')
        self.model, self.statistics, self.make_optimizer = model, statistics, make_optimizer
        self.variant, self.threshold, self.max_arms, self.capacity = variant, threshold, max_arms, capacity
        self.gamma, self.ucb_c, self.min_pulls, self.use_drift = gamma, ucb_c, min_pulls, drift
        self.key_pool, self.cache_capacity = key_pool, cache_capacity
        self.contexts = OrderedDict()   # skipbook: ident -> Context (LRU order)
        self.entries = OrderedDict()    # prefill_cache: ident -> (key, mask, source)
        self.next_ident = 0
        self.stream_optimizer = stream_optimizer  # SWIFT's own optimizer (prefill_cache keeps using it)
        self.current_optimizer = None
        self.optimizer = OptimizerProxy(self)
        self.request = None

    # ------------------------------------------------------------------ hooks
    def hook_initialize(self, swift_verify, swift_draft):
        """Drop-in replacement for model.swift.utils.initialize_swift.

        Identical computation; the only addition is `on_prefill` between the dense
        prefill (mask independent) and the first draft (mask dependent).
        """
        import torch
        book = self

        def initialize_swift(input_ids, model, max_new_tokens, past_key_values, past_key_values_data,
                             current_length_data, logits_processor=None):
            with torch.inference_mode():
                outputs, logits = swift_verify(model, input_ids, past_key_values=past_key_values)
                if logits_processor is not None:
                    last_logits = logits_processor(None, logits[:, -1])
                    probabilities = torch.nn.functional.softmax(last_logits, dim=1)
                    sample_token = torch.multinomial(probabilities, 1)
                else:
                    sample_token = torch.argmax(logits[:, -1])[None, None]
                start = time.perf_counter()
                key = pool_prefill(outputs[0], book.key_pool)
                book.on_prefill(key, key_seconds=time.perf_counter() - start)
                swift_logits, top1_prob = swift_draft(
                    model, input_ids=sample_token, past_key_values_data=past_key_values_data,
                    current_length_data=current_length_data, max_new_tokens=max_new_tokens,
                    logits_processor=logits_processor)
            return swift_logits, sample_token, top1_prob
        return initialize_swift

    def wrap(self, forward):
        book = self

        def forward_with_book(input_ids, model, tokenizer, max_new_tokens, **kwargs):
            forward.request_context = getattr(forward_with_book, 'request_context', {})
            book.request = {'start': time.perf_counter(), 'prefill_seen': False}
            try:
                result = forward(input_ids, model, tokenizer, max_new_tokens, **kwargs)
            except Exception:
                book.abort()
                raise
            _synchronize(model)
            metrics = book.finish(new_tokens=int(result[1]), accept_lengths=result[3])
            forward_with_book.last_metrics = dict(getattr(forward, 'last_metrics', {}), **metrics)
            return result
        return forward_with_book

    # ------------------------------------------------------------------ request lifecycle
    def on_prefill(self, key, key_seconds=0.0):
        request = self.request if self.request is not None else {}
        self.request = request
        request.update(prefill_seen=True, key_seconds=key_seconds)
        lookup_start = time.perf_counter()
        if self.variant == 'prefill_cache':
            self._prefill_cache_begin(key)
        else:
            self._skipbook_begin(key)
        request['lookup_seconds'] = time.perf_counter() - lookup_start
        _synchronize(self.model)
        request['decode_start'] = time.perf_counter()
        request['opt_before'] = self.statistics.get('_optimization_host_seconds', 0.0)

    def finish(self, new_tokens, accept_lengths):
        request = self.request
        self.request = None
        if not request or not request.get('prefill_seen'):
            raise RuntimeError('SkipBook prefill hook did not run; initialize_swift is not hooked')
        now = time.perf_counter()
        opt_seconds = self.statistics.get('_optimization_host_seconds', 0.0) - request['opt_before']
        decode_seconds = max(now - request['decode_start'] - opt_seconds, 1e-9)
        reward = new_tokens / decode_seconds
        mean_accept = sum(accept_lengths) / len(accept_lengths) if accept_lengths else None
        if self.variant == 'prefill_cache':
            info = self._prefill_cache_end(request)
        else:
            info = self._skipbook_end(request, reward, mean_accept)
        return {'skipbook_protocol': PROTOCOL, 'skipbook_variant': self.variant,
                'cache_enabled': True, 'cache_hit': request['hit'],
                'cache_similarity': request['similarity'],
                'embedding_seconds': request['key_seconds'],
                'search_seconds': request['lookup_seconds'],
                'used_layers': [list(request['mask'][0]), list(request['mask'][1])],
                'skipbook_decode_tokens_per_second': reward,
                'skipbook_decode_seconds': decode_seconds, **info}

    def abort(self):
        request, self.request = self.request, None
        if request and self.variant == 'prefill_cache' and request.get('restore'):
            self._restore(request['restore'])

    # ------------------------------------------------------------------ prefill_cache variant
    def _prefill_cache_begin(self, key):
        key = normalize(key)
        best, similarity = None, None
        for ident, (stored, _, _) in self.entries.items():
            score = float(stored @ key)
            if similarity is None or score > similarity:
                best, similarity = ident, score
        hit = best is not None and similarity >= self.threshold
        self.current_optimizer = self.stream_optimizer
        incumbent = as_mask(self.model.get_skip_layers())
        self.request.update(hit=hit, similarity=similarity, key=key, mask=incumbent,
                            restore={'mask': incumbent, 'optimization': self.statistics['optimization']})
        if hit:
            self.entries.move_to_end(best)
            mask = self.entries[best][1]
            self._apply(mask)
            self.statistics['optimization'] = False
            self.request.update(entry=best, mask=mask)

    def _prefill_cache_end(self, request):
        if request['hit']:
            self._restore(request['restore'])
            return {'skipbook_mode': 'hit', 'skipbook_entry': request['entry'],
                    'skipbook_book_size': len(self.entries)}
        mask = as_mask(self.model.get_skip_layers())
        request['mask'] = mask
        ident = self.next_ident
        self.next_ident += 1
        self.entries[ident] = (request['key'], mask, 'stream_incumbent')
        while len(self.entries) > self.cache_capacity:
            self.entries.popitem(last=False)
        return {'skipbook_mode': 'miss', 'skipbook_entry': ident, 'skipbook_book_size': len(self.entries)}

    # ------------------------------------------------------------------ skipbook variant
    def _nearest(self, key):
        best, similarity = None, None
        for context in self.contexts.values():
            score = float(context.centroid @ key)
            if similarity is None or score > similarity:
                best, similarity = context, score
        return best, similarity

    def _skipbook_begin(self, key):
        key = normalize(key)
        nearest, similarity = self._nearest(key)
        hit = nearest is not None and similarity >= self.threshold
        if hit:
            context = nearest
            context.absorb(key)
            self.contexts.move_to_end(context.ident)
        else:
            if nearest is not None:
                warm, source = self.best_arm(nearest).mask, f'warm_from_context_{nearest.ident}'
            else:
                warm, source = as_mask(self.model.get_skip_layers()), 'initial_mask'
            context = Context(self.next_ident, key, warm, source)
            self.next_ident += 1
            self.contexts[context.ident] = context
            while len(self.contexts) > self.capacity:
                self.contexts.popitem(last=False)
            self._start_search(context, reason='new_context')
        self.request.update(hit=hit, similarity=similarity, context=context)
        if context.search is not None:
            self._load_search(context)
            self.request.update(mode='explore', arm=None, mask=context.search_mask)
        else:
            arm = self.select_arm(context)
            self._apply(arm.mask)
            self.statistics['optimization'] = False
            self.current_optimizer = None
            self.request.update(mode='exploit', arm=arm, mask=arm.mask)

    def _skipbook_end(self, request, reward, mean_accept):
        context = request['context']
        info = {'skipbook_context': context.ident, 'skipbook_mode': request['mode'],
                'skipbook_contexts': len(self.contexts), 'skipbook_drift': False}
        if request['mode'] == 'explore':
            self._save_search(context)
            request['mask'] = context.search_mask
            if not context.search['optimization']:
                admitted = self.admit(context, context.search_mask, f'search_{context.explorations}')
                info['skipbook_admitted'] = admitted
                context.search = None
                context.search_mask = None
        else:
            arm = request['arm']
            self.update(context, arm, reward)
            info['skipbook_arm'] = context.arms.index(arm) if arm in context.arms else None
            if self.use_drift and mean_accept is not None and context.drift.update(mean_accept):
                info['skipbook_drift'] = True
                self._start_search(context, reason='drift', warm=self.best_arm(context).mask)
            self.eliminate(context)
        info['skipbook_arms'] = [a.as_dict() for a in context.arms]
        return info

    # ------------------------------------------------------------------ Explorer
    def _start_search(self, context, reason, warm=None):
        context.explorations += 1
        context.search = {'origin_score': 0, 'opt_iter': 0, 'tolerance_iter': 0, 'optimization': True,
                          'optimizer': self.make_optimizer(), 'reason': reason}
        context.search_mask = as_mask(warm) if warm is not None else context.arms[0].mask

    def _load_search(self, context):
        for key in SEARCH_KEYS:
            self.statistics[key] = context.search[key]
        self.current_optimizer = context.search['optimizer']
        self._apply(context.search_mask)

    def _save_search(self, context):
        for key in SEARCH_KEYS:
            context.search[key] = self.statistics[key]
        context.search_mask = as_mask(self.model.get_skip_layers())

    # ------------------------------------------------------------------ Judge / Curator
    def update(self, context, arm, reward):
        """Discounted UCB statistics; rewards are divided by the context's mean reward."""
        if not math.isfinite(reward) or reward <= 0:
            return
        context.rewards += 1
        scale = context.reward_scale
        context.reward_scale = reward if scale is None else scale + (reward - scale) / context.rewards
        for other in context.arms:
            other.weight *= self.gamma
            other.total *= self.gamma
        arm.weight += 1.0
        arm.total += reward / context.reward_scale
        arm.pulls += 1

    def _width(self, context, arm):
        total = sum(a.weight for a in context.arms)
        if arm.weight <= 0 or total <= 1:
            return math.inf
        return self.ucb_c * math.sqrt(max(math.log(total), 0.0) / arm.weight)

    def select_arm(self, context):
        for arm in context.arms:            # every arm is tried once before UCB
            if arm.pulls == 0:
                return arm
        return max(context.arms, key=lambda a: (a.mean() + self._width(context, a), -context.arms.index(a)))

    def best_arm(self, context):
        tried = [a for a in context.arms if a.mean() is not None]
        return max(tried, key=lambda a: a.mean()) if tried else context.arms[0]

    def admit(self, context, mask, source):
        mask = as_mask(mask)
        if any(a.mask == mask for a in context.arms):
            return False
        context.arms.append(Arm(mask, source))
        while len(context.arms) > self.max_arms:
            # Never evict the newcomer before it is tried; drop the worst tried arm.
            tried = [a for a in context.arms[:-1] if a.mean() is not None] or context.arms[:-1]
            context.arms.remove(min(tried, key=lambda a: a.mean() if a.mean() is not None else math.inf))
        return True

    def eliminate(self, context):
        if len(context.arms) < 2:
            return
        best = self.best_arm(context)
        if best.pulls < self.min_pulls:
            return
        floor = best.mean() - self._width(context, best)
        context.arms = [a for a in context.arms if a is best or a.pulls < self.min_pulls
                        or a.mean() + self._width(context, a) >= floor]

    # ------------------------------------------------------------------ model helpers
    def _apply(self, mask):
        self.model.set_skip_layers(np.array(mask[0], dtype=int), np.array(mask[1], dtype=int))

    def _restore(self, saved):
        self._apply(saved['mask'])
        self.statistics['optimization'] = saved['optimization']


class OptimizerProxy:
    """Routes SWIFT's optimizer calls to the optimizer of the context being explored."""
    def __init__(self, book):
        self._book = book

    def __getattr__(self, name):
        target = self._book.current_optimizer or self._book.stream_optimizer
        if target is None:
            raise RuntimeError('No active optimizer: SWIFT search ran outside an exploration episode')
        return getattr(target, name)


def _synchronize(model):
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        device = next(model.parameters()).device
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
