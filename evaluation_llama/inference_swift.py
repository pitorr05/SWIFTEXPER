"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse
import numpy as np
from collections import OrderedDict

from fastchat.utils import str_to_torch_dtype

from evaluation_llama.eval import run_eval

from transformers import AutoTokenizer
from bayes_opt import BayesianOptimization, UtilityFunction

from model.swift.utils import *
from model.swift.modeling_llama import LlamaForCausalLM
from model.swift.kv_cache import initialize_past_key_values


# ============================================
# ADAPTIVE CACHE CLASS
# ============================================

class AdaptiveCache:
    """
    Bộ nhớ đệm layer set thích ứng cho SWIFT.
    
    Design Principles:
    1. Online Learning: Cache được xây dựng trong lúc chạy, không cần pre-inference
    2. Plug-and-Play: Cache rỗng khi khởi tạo, tự động học
    3. LRU Eviction: Giới hạn kích thước cache để tránh tràn bộ nhớ
    4. Cosine Similarity: Dùng để tìm layer set tương tự nhất
    
    Attributes:
        cache (OrderedDict): Lưu {embedding_tuple: (attn_skip, mlp_skip)}
        max_size (int): Số lượng entry tối đa trong cache
        threshold (float): Ngưỡng cosine similarity để cache HIT
        hit_count, miss_count (int): Thống kê hiệu suất
    """
    
    def __init__(self, max_size: int = 100, similarity_threshold: float = 0.85):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.threshold = similarity_threshold
        self.hit_count = 0
        self.miss_count = 0
        self.total_embedding_time = 0.0
        self.total_search_time = 0.0
        
    def get_embedding(self, input_ids, model) -> np.ndarray:
        """
        Trích xuất vector đặc trưng từ input.
        
        Tham khảo từ KNN-SSD: sử dụng mean pooling của last hidden state.
        Đây là cách hiệu quả để capture domain information của input.
        
        Args:
            input_ids: Token IDs của input
            model: LlamaForCausalLM
            
        Returns:
            embedding: Vector đặc trưng đã được L2-normalized
        """
        import time
        start_time = time.time()
        
        with torch.no_grad():
            # Lấy hidden states từ model
            outputs = model.model(
                input_ids=input_ids,
                output_hidden_states=True,
                use_cache=False
            )
            # Mean pooling theo token dimension
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            # L2 normalization
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
        
        self.total_embedding_time += time.time() - start_time
        return embedding
    
    def find_similar(self, embedding: np.ndarray):
        """
        Tìm layer set tương tự nhất trong cache.
        
        Sử dụng cosine similarity để so sánh embedding của input 
        với các embedding đã được cache.
        
        Args:
            embedding: Vector đặc trưng của input mới
            
        Returns:
            tuple: (attn_skip, mlp_skip, similarity) hoặc (None, None, 0.0)
        """
        import time
        start_time = time.time()
        
        if not self.cache:
            self.total_search_time += time.time() - start_time
            return None, None, 0.0
        
        best_similarity = -1.0
        best_attn = None
        best_mlp = None
        
        for cached_emb_tuple, (attn, mlp) in self.cache.items():
            cached_emb = np.array(cached_emb_tuple)
            # Cosine similarity (các vector đã được normalize)
            similarity = np.dot(embedding, cached_emb)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_attn = attn
                best_mlp = mlp
        
        self.total_search_time += time.time() - start_time
        
        if best_similarity >= self.threshold and best_attn is not None:
            self.hit_count += 1
            # Move to end (LRU)
            key = tuple(embedding.tolist())
            if key in self.cache:
                self.cache.move_to_end(key)
            return best_attn, best_mlp, best_similarity
        else:
            self.miss_count += 1
            return None, None, best_similarity
    
    def add_to_cache(self, embedding: np.ndarray, attn_skip, mlp_skip):
        """
        Thêm layer set mới vào cache.
        
        Sử dụng LRU eviction: nếu cache đầy, xóa entry lâu nhất không được dùng.
        
        Args:
            embedding: Vector đặc trưng của input
            attn_skip: Attention layer set
            mlp_skip: MLP layer set
        """
        key = tuple(embedding.tolist())
        
        # Kiểm tra trùng lặp
        if key in self.cache:
            self.cache.move_to_end(key)
            return
        
        # LRU eviction
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)
        
        self.cache[key] = (attn_skip, mlp_skip)
    
    def get_stats(self) -> str:
        """Lấy thống kê hiệu suất cache"""
        total = self.hit_count + self.miss_count
        if total == 0:
            return "🔧 Cache: Empty (chưa có dữ liệu)"
        
        hit_rate = self.hit_count / total * 100
        return (f"📊 Cache Stats: {self.hit_count} hits, {self.miss_count} misses, "
                f"Hit Rate: {hit_rate:.1f}%, "
                f"Embedding Time: {self.total_embedding_time:.4f}s, "
                f"Search Time: {self.total_search_time:.4f}s")
    
    def clear(self):
        """Xóa toàn bộ cache"""
        self.cache.clear()
        self.hit_count = 0
        self.miss_count = 0
        self.total_embedding_time = 0.0
        self.total_search_time = 0.0
    
    def save_to_file(self, filepath: str):
        """Lưu cache xuống file để tái sử dụng sau"""
        import json
        # Convert numpy arrays to lists for JSON serialization
        cache_data = []
        for k, (attn, mlp) in self.cache.items():
            cache_data.append({
                'embedding': list(k),
                'attn_skip': list(attn) if isinstance(attn, (list, tuple, np.ndarray)) else attn,
                'mlp_skip': list(mlp) if isinstance(mlp, (list, tuple, np.ndarray)) else mlp
            })
        
        data = {
            'cache': cache_data,
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'threshold': self.threshold,
            'max_size': self.max_size
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"💾 Cache saved to {filepath}")
    
    def load_from_file(self, filepath: str):
        """Load cache từ file"""
        import json
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.cache = OrderedDict()
        for item in data['cache']:
            key = tuple(item['embedding'])
            attn = item['attn_skip']
            mlp = item['mlp_skip']
            self.cache[key] = (attn, mlp)
        
        self.hit_count = data.get('hit_count', 0)
        self.miss_count = data.get('miss_count', 0)
        self.threshold = data.get('threshold', self.threshold)
        self.max_size = data.get('max_size', self.max_size)
        print(f"📂 Cache loaded from {filepath} with {len(self.cache)} entries")


# ============================================
# SWIFT FORWARD WITH ADAPTIVE CACHE
# ============================================

def swift_forward(input_ids, model, tokenizer, max_new_tokens, statistics=None, optimizer=None, utility=None,
                  logits_processor=None, max_steps=512, cache_enabled=True, cache_threshold=0.85, 
                  cache_max_size=100, cache_file=None):
    """
    SWIFT forward với Adaptive Cache.
    
    Args:
        cache_enabled: Bật/tắt Adaptive Cache
        cache_threshold: Ngưỡng cosine similarity cho cache HIT
        cache_max_size: Kích thước tối đa của cache
        cache_file: Đường dẫn file để load/save cache
    """
    assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
    input_ids = input_ids.clone()
    accept_length_list = []
    
    # ============================================
    # 1. KHỞI TẠO ADAPTIVE CACHE
    # ============================================
    if not hasattr(swift_forward, "cache"):
        swift_forward.cache = AdaptiveCache(
            max_size=cache_max_size,
            similarity_threshold=cache_threshold
        )
        print("🔧 AdaptiveCache initialized!")
        
        # Load cache từ file nếu có
        if cache_file:
            try:
                swift_forward.cache.load_from_file(cache_file)
            except FileNotFoundError:
                print(f"⚠️ Cache file {cache_file} not found, starting fresh")
    
    # ============================================
    # 2. TRUY XUẤT CACHE
    # ============================================
    use_cache_layer = False
    cached_attn = None
    cached_mlp = None
    embedding = None
    
    if cache_enabled:
        embedding = swift_forward.cache.get_embedding(input_ids, model)
        cached_attn, cached_mlp, similarity = swift_forward.cache.find_similar(embedding)
        
        if cached_attn is not None:
            print(f"✅ Cache HIT! Similarity: {similarity:.3f}")
            model.set_skip_layers(cached_attn, cached_mlp)
            use_cache_layer = True
        else:
            print(f"🔍 Cache MISS! Best similarity: {similarity:.3f}")
            # Fallback: uniform skip (giống SWIFT gốc)
            _attn_skip_layer_id_set = np.arange(1, model.config.num_hidden_layers - 1, 2)
            _mlp_skip_layer_id_set = np.arange(1, model.config.num_hidden_layers - 1, 2)
            model.set_skip_layers(_attn_skip_layer_id_set, _mlp_skip_layer_id_set)
            use_cache_layer = False
    else:
        # Nếu cache bị tắt, dùng uniform skip như SWIFT gốc
        _attn_skip_layer_id_set = np.arange(1, model.config.num_hidden_layers - 1, 2)
        _mlp_skip_layer_id_set = np.arange(1, model.config.num_hidden_layers - 1, 2)
        model.set_skip_layers(_attn_skip_layer_id_set, _mlp_skip_layer_id_set)

    # ============================================
    # 3. PHẦN CÒN LẠI CỦA SWIFT (GIỮ NGUYÊN)
    # ============================================
    # Initialize the past key and value states
    (
        past_key_values,
        past_key_values_data,
        current_length_data,
    ) = initialize_past_key_values(model.model)
    model.past_key_values = past_key_values
    model.past_key_values_data = past_key_values_data
    model.current_length_data = current_length_data

    input_len = input_ids.shape[1]
    cur_length = input_len
    reset_swift_mode(model)
    swift_logits, sample_token, top1_prob = initialize_swift(input_ids, model, max_new_tokens,
                                                             past_key_values, past_key_values_data,
                                                             current_length_data, logits_processor=logits_processor)

    # Clone the prefilled past key and value states for swift optimization
    input_past_key_values_data = []
    for i in range(len(past_key_values_data)):
        input_past_key_values_data.append(past_key_values_data[i].clone())
    input_current_length_data = current_length_data.clone()

    new_token_num = 0
    draft_token_num = 0
    total_acc_num = 0
    
    for idx in range(max_steps):
        # drafted tokens + 1 bonus verified token
        draft_token_num += len(top1_prob)
        # Initialize the swift buffer
        swift_choices = eval(f"{get_choices_list(top1_prob, logits_processor=logits_processor)}")
        swift_buffers = generate_swift_buffers(swift_choices, device=model.model.layers[-1].self_attn.q_proj.weight.device)
        model.swift_buffers = swift_buffers
        model.swift_choices = swift_choices
        model.model.swift_mask = swift_buffers["swift_attn_mask"]

        candidates, cart_candidates_prob, tree_candidates = generate_candidates(
            swift_logits,
            swift_buffers["tree_indices"],
            swift_buffers["retrieve_indices"],
            sample_token,
            logits_processor
        )

        logits, outputs = tree_decoding(
            model,
            tree_candidates,
            past_key_values,
            swift_buffers["swift_position_ids"],
            input_ids,
            swift_buffers["retrieve_indices"],
        )

        best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor, cart_candidates_prob, swift_logits[2],
                swift_buffers["p_indices"], tree_candidates, swift_buffers["b_indices"]
            )

        input_ids, new_token_num, sample_token = update_inference_inputs(
            input_ids,
            candidates,
            best_candidate,
            accept_length,
            swift_buffers["retrieve_indices"],
            logits_processor,
            new_token_num,
            past_key_values_data,
            current_length_data,
            sample_p
        )

        # ============================================
        # 4. LAYER SET OPTIMIZATION - CHỈ CHẠY KHI CACHE MISS
        # ============================================
        if (not use_cache_layer) and (new_token_num > (statistics["context_window"] + 1) 
                and statistics["optimization"] and idx % statistics["opt_interval"] == 0):
            swift_optimization(
                model,
                input_ids[:, input_len:],
                input_past_key_values_data,
                input_current_length_data,
                new_token_num,
                statistics,
                optimizer=optimizer,
                utility=utility)

        # swift drafting
        swift_logits, top1_prob = swift_draft(
            model,
            input_ids=sample_token,
            new_token_num=new_token_num,
            past_key_values_data=past_key_values_data,
            current_length_data=current_length_data,
            max_new_tokens=max_new_tokens,
            logits_processor=logits_processor,
        )
        accept_length_tree = input_ids.shape[1] - cur_length
        cur_length = accept_length_tree + cur_length
        accept_length_list.append(accept_length_tree)
        total_acc_num += accept_length_tree - 1
        if tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
            break
        if new_token_num > max_new_tokens:
            break
    
    logging.info("token acceptance rate: {}".format(total_acc_num / draft_token_num))

    # ============================================
    # 5. LƯU LAYER SET VÀO CACHE
    # ============================================
    if (not use_cache_layer) and cache_enabled and embedding is not None:
        best_attn_skip, best_mlp_skip = model.get_skip_layers()
        # Chỉ lưu nếu layer set khác với default
        default_attn = np.arange(1, model.config.num_hidden_layers - 1, 2).tolist()
        default_mlp = np.arange(1, model.config.num_hidden_layers - 1, 2).tolist()
        
        if (best_attn_skip != default_attn) or (best_mlp_skip != default_mlp):
            swift_forward.cache.add_to_cache(embedding, best_attn_skip, best_mlp_skip)
            print(f"💾 Cached new layer set! {swift_forward.cache.get_stats()}")
    
    # In thống kê cache
    if cache_enabled:
        print(f"📊 {swift_forward.cache.get_stats()}")
    
    # Lưu cache xuống file nếu được yêu cầu
    if cache_file and cache_enabled:
        try:
            swift_forward.cache.save_to_file(cache_file)
        except Exception as e:
            print(f"⚠️ Failed to save cache: {e}")
    
    return input_ids, new_token_num, idx + 1, accept_length_list, draft_token_num


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
    )
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="The temperature for swift sampling.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.85,
        help="The top-p for sampling.",
    )
    parser.add_argument(
        "--skip-ratio",
        type=float,
        default=0.45,
        help="The skipped layer ratio of swift.",
    )
    parser.add_argument(
        "--opt-interval",
        type=int,
        default=1,
        help="The interval of swift optimization.",
    )
    parser.add_argument(
        "--bayes-interval",
        type=int,
        default=25,
        help="The interval of bayesian optimization.",
    )
    parser.add_argument(
        "--max-opt-iter",
        type=int,
        default=1000,
        help="The maximum layer set optimization iteration.",
    )
    parser.add_argument(
        "--max-tolerance-iter",
        type=int,
        default=300,
        help="The maximum tolerance of layer set search iteration.",
    )
    parser.add_argument(
        "--max-score",
        type=float,
        default=0.95,
        help="The early stop threshold of layer set search.",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=32,
        help="The context window of swift.",
    )
    parser.add_argument(
        "--optimization",
        action="store_true",
        default=False,
        help="Layer set optimization.",
    )
    parser.add_argument(
        "--bayes",
        action="store_true",
        default=False,
        help="Bayes Optimization of Layer set.",
    )
    parser.add_argument(
        "--cache-hit",
        action="store_true",
        default=False,
        help="Whether to use cached SWIFT configuration.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float32", "float64", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU.",
    )
    parser.add_argument(
        "--task-name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--data-num",
        type=int,
        default=10,
        help="The number of samples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2024,
        help="The sampling seed.",
    )
    
    # ============================================
    # ADAPTIVE CACHE ARGUMENTS
    # ============================================
    parser.add_argument(
        "--cache-enabled",
        action="store_true",
        default=False,
        help="Enable Adaptive Cache (tự động học layer set trong lúc chạy)"
    )
    parser.add_argument(
        "--cache-threshold",
        type=float,
        default=0.85,
        help="Similarity threshold for cache hit (0.0 - 1.0)"
    )
    parser.add_argument(
        "--cache-max-size",
        type=int,
        default=100,
        help="Maximum number of entries in cache"
    )
    parser.add_argument(
        "--cache-file",
        type=str,
        default=None,
        help="Path to save/load cache file (e.g., cache.json)"
    )
    parser.add_argument(
        "--cache-clear",
        action="store_true",
        default=False,
        help="Clear cache before running"
    )

    args = parser.parse_args()

    args.model_name = (args.model_id + "-swift-" + str(args.dtype)+ "-temp-" + str(args.temperature)
                       + "-top-p-" + str(args.top_p) + "-seed-" + str(args.seed) + "-max_new_tokens-" + str(args.max_new_tokens)+ "-opt_interval-" + str(args.opt_interval)
                       + "-bayes_interval-" + str(args.bayes_interval) + "-max_opt-" + str(args.max_opt_iter) + "-max_tolerance-" + str(args.max_tolerance_iter)
                       + "-max_score-" + str(args.max_score) + "-context_window-" + str(args.context_window) + "-skip_ratio-" + str(args.skip_ratio))
    
    if args.cache_enabled:
        args.model_name += "-cache-enabled"
    
    answer_file = f"outputs/{args.task_name}/{args.task_name}_{args.data_num}/model_answer/{args.model_id}/{args.model_name}.jsonl"
    set_logger()

    print(f"Output to {answer_file}")

    torch.nn.Linear.reset_parameters = lambda x: None

    model = LlamaForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=str_to_torch_dtype(args.dtype),
        low_cpu_mem_usage=True,
        device_map="auto")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    if args.temperature > 1e-5:
        logits_processor = prepare_logits_processor(temperature=args.temperature, top_p=args.top_p)
    else:
        logits_processor = None

    if args.cache_hit:
        # Load the cached layer set configuration
        args.optimization, args.bayes = False, False
        _attn_skip_layer_id_set, _mlp_skip_layer_id_set = get_cache_configuration(model_name=args.model_id,
                                                                                  task_name=args.task_name)
    else:
        # Unified layer set initialization
        _attn_skip_layer_id_set = np.arange(1, model.config.num_hidden_layers - 1, 2)  # keep the first and last layer
        _mlp_skip_layer_id_set = np.arange(1, model.config.num_hidden_layers - 1, 2)

    model.set_skip_layers(_attn_skip_layer_id_set, _mlp_skip_layer_id_set)

    # Bayes Optimization Settings
    pbounds = {f"x{i}": (0, 1) for i in range((model.config.num_hidden_layers - 2) * 2)} # keep the first and last layer
    optimizer = BayesianOptimization(f=None, pbounds=pbounds, random_state=1, verbose=1, allow_duplicate_points=True)
    optimizer.set_gp_params(alpha=1e-2)
    utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)

    statistics = {"origin_score": 0, "opt_iter": 0, "tolerance_iter": 0,
                  "skip_ratio": args.skip_ratio, "acceptance_rate_list": [], "opt_interval": args.opt_interval,
                  "bayes_interval": args.bayes_interval, "max_opt_iter": args.max_opt_iter,
                  "max_tolerance_iter": args.max_tolerance_iter, "max_score": args.max_score,
                  "context_window": args.context_window, "optimization": args.optimization, "bayes": args.bayes}

    # Clear cache nếu được yêu cầu
    if args.cache_clear and hasattr(swift_forward, "cache"):
        swift_forward.cache.clear()
        print("🧹 Cache cleared!")

    run_eval(
        model=model,
        tokenizer=tokenizer,
        forward_func=swift_forward,
        model_id=args.model_id,
        answer_file=answer_file,
        max_new_tokens=args.max_new_tokens,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        task_name=args.task_name,
        data_num=args.data_num,
        seed=args.seed,
        optimizer=optimizer,
        utility=utility,
        statistics=statistics,
        logits_processor=logits_processor,
        # Cache parameters
        cache_enabled=args.cache_enabled,
        cache_threshold=args.cache_threshold,
        cache_max_size=args.cache_max_size,
        cache_file=args.cache_file,
    )
