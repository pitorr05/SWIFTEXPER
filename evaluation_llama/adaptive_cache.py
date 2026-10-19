"""Online layer-set cache. This is contribution code, not original SWIFT."""
from collections import OrderedDict
import json
from pathlib import Path
import time
import numpy as np


class AdaptiveCache:
    def __init__(self, max_size=100, similarity_threshold=0.85):
        if max_size < 1 or not -1 <= similarity_threshold <= 1:
            raise ValueError('Invalid cache capacity or cosine threshold')
        self.max_size = max_size
        self.threshold = similarity_threshold
        self.clear()

    def clear(self):
        self.cache = OrderedDict()
        self.hit_count = self.miss_count = 0
        self.total_embedding_time = self.total_search_time = 0.0
        self.last_entry = None

    def get_embedding(self, input_ids, model):
        import torch
        from model.swift.utils import reset_swift_mode
        start = time.perf_counter()
        reset_swift_mode(model)
        with torch.inference_mode(), model.self_draft(enabled=False):
            output = model.model(input_ids=input_ids, output_hidden_states=False,
                                 use_cache=False, return_dict=True)
            embedding = output.last_hidden_state.float().mean(dim=1).squeeze(0).cpu().numpy()
        norm = np.linalg.norm(embedding)
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError('Cannot retrieve with a zero/nonfinite embedding')
        embedding = embedding / norm
        self.total_embedding_time += time.perf_counter() - start
        return embedding

    def find_similar(self, embedding):
        start = time.perf_counter()
        best_key, similarity = None, -1.0
        for key in self.cache:
            score = float(np.dot(embedding, np.asarray(key)))
            if best_key is None or score > similarity:
                best_key, similarity = key, score
        self.last_entry = None
        if best_key is not None and similarity >= self.threshold:
            self.hit_count += 1
            self.cache.move_to_end(best_key)
            entry = self.cache[best_key]
            self.last_entry = dict(entry['metadata'])
            result = (list(entry['attn_skip']), list(entry['mlp_skip']), similarity)
        else:
            self.miss_count += 1
            result = (None, None, similarity if best_key is not None else None)
        self.total_search_time += time.perf_counter() - start
        return result

    def add_to_cache(self, embedding, attn_skip, mlp_skip, metadata=None):
        key = tuple(float(x) for x in embedding)
        self.cache[key] = {'attn_skip': list(map(int, attn_skip)), 'mlp_skip': list(map(int, mlp_skip)),
                           'metadata': dict(metadata or {})}
        self.cache.move_to_end(key)
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

    def save_to_file(self, filepath, identity):
        target = Path(filepath)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, 'w', encoding='utf-8') as stream:
            json.dump({'schema_version': 1, 'identity': identity,
                       'max_size': self.max_size, 'threshold': self.threshold,
                       'entries': [{'embedding': list(k), **v} for k, v in self.cache.items()]}, stream)

    def load_from_file(self, filepath, identity):
        with open(filepath, encoding='utf-8') as stream:
            payload = json.load(stream)
        if payload.get('schema_version') != 1 or payload.get('identity') != identity:
            raise ValueError('Cache schema/model/protocol identity mismatch; use --cache-clear')
        if payload['max_size'] != self.max_size or payload['threshold'] != self.threshold:
            raise ValueError('Cache parameters differ from CLI')
        self.clear()
        for entry in payload['entries']:
            self.add_to_cache(np.asarray(entry['embedding']), entry['attn_skip'], entry['mlp_skip'], entry['metadata'])
