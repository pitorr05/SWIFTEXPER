# SWIFTEXPER

Repo thực nghiệm self-speculative decoding trên một GPU H100. Nhánh hiện tại: **research/swift-mask-search**.

## Bắt đầu từ đâu?

| Bạn cần làm gì? | File cần đọc hoặc chạy |
|---|---|
| Chạy AR, SWIFT và AdaptiveCache | `bash scripts/run_h100.sh` |
| Chạy nghiên cứu mask-search và ablation | `bash scripts/run_mask_search.sh` |
| Đọc thuật toán mới | `model/mask_search.py` |
| Đọc model, decoding và KV cache gốc | `model/swift/` |
| Đọc inference gốc | `evaluation_llama/inference_baseline.py`, `inference_swift.py` |
| Đọc logic tổ chức thí nghiệm | `scripts/run_experiments.py`, `run_mask_search.py` |
| Chấm HumanEval / chấm lại output | `scripts/evaluate_humaneval.py`, `recheck_humaneval.py` |
| Tính lại metric từ output đã lưu | `scripts/summarize_results.py` |
| Xem quyết định kỹ thuật và nghiên cứu | `EXPERIMENT_LOG.md` |
| Xem kết quả | `_local/runs/<run>/results.csv` |

```text
SWIFTEXPER/
|-- Readme.md                 # Hướng dẫn và bản đồ code
|-- EXPERIMENT_LOG.md         # Quyết định kỹ thuật, protocol, lịch sử sửa
|-- requirements.txt         # Môi trường SWIFT gốc
|-- evaluation_llama/        # Inference và đánh giá gốc
|-- model/
|   |-- swift/               # Model, decoding, KV cache gốc
|   `-- mask_search.py       # Thuật toán nghiên cứu mới
|-- scripts/
|   |-- run_h100.sh          # AR / SWIFT / AdaptiveCache
|   |-- run_mask_search.sh   # Nghiên cứu mới và ablation
|   |-- *.py                 # Điều phối, chuẩn bị source, chấm, tổng hợp
|   `-- runtime/             # Python hỗ trợ snapshot thực nghiệm
|-- assets/                  # Hình minh họa paper
|-- eval_llama.sh            # Ví dụ launcher gốc
|-- skip_layers.json         # Cấu hình layer tĩnh gốc
`-- _local/                  # Không đưa lên Git
    |-- runs/                # Output, điểm, source snapshot, provenance
    |-- papers/              # Paper tham khảo
    |-- references/knapspec/ # Code tham khảo cục bộ
    |-- worktrees/           # Git worktree: không tự di chuyển/xóa
    `-- ...                  # Archive, verification và cache
```

`EXPERIMENT_RESULTS.md`, nếu có ở máy của bạn, là báo cáo cục bộ được giữ nguyên; runner không phụ thuộc file này.

## Phân biệt code gốc và contribution

- **main** giữ source thuật toán SWIFT gốc cùng công cụ thực nghiệm. Runner áp dụng hiệu chỉnh harness/runtime có ghi nhận trên snapshot, không sửa working source gốc.
- **AdaptiveCache** chứa cache retrieval và lifecycle của contribution trước. Runner đọc source đã commit từ `origin/AdaptiveCache`.
- **research/swift-mask-search** thêm lựa chọn mask theo ngân sách, objective và cửa sổ context. Đây là prototype cần thực nghiệm; chưa xác lập speedup hay novelty.

`evaluation_llama/speed.py` là code tác giả giữ nguyên. `scripts/runtime/speed_adapter.py` là adapter metric research, được xuất thành `evaluation_llama/speed.py` trong snapshot. Không cần chạy thủ công cả hai để có bảng tổng hợp.

`runtime/` gồm `harness.py` (vòng đánh giá), `research_utils.py` (metric/output), `gsm8k.py` (dữ liệu/accuracy), `completion_decode.py`, `phase_timer.py` và `speed_adapter.py`. `source_patches.py` chuẩn bị snapshot bằng các module này.

## Lệnh nhanh trên H100: nhánh nghiên cứu

### Chạy riêng fixed_first 13B, đủ năm cấu hình và tự xuất CSV

```bash
cd /data/SWIFTEXPER
git switch research/swift-mask-search
git pull --ff-only origin research/swift-mask-search
git fetch origin main:refs/remotes/origin/main AdaptiveCache:refs/remotes/origin/AdaptiveCache
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=0 bash scripts/run_mask_search.sh \
  --stage evaluate --swift-ref origin/main \
  --models meta-llama/Llama-2-13b-hf \
  --code-models codellama/CodeLlama-13b-hf \
  --methods fixed_first \
  --tasks cnndm humaneval gsm8k \
  --cnndm-max-new-tokens 64 512 \
  --gsm8k-profiles paper64 quality512 \
  --data-num 1000 --data-seed 2024 --seeds 2024
```

Chạy đúng năm process inference tuần tự: CNN/DM-64, CNN/DM-512, HumanEval-512,
GSM8K-64 và GSM8K-512. HumanEval luôn đủ 164 task; CNN/DM và GSM8K theo
`--data-num`. Mỗi process tối ưu native trong request đầu rồi đóng băng mask
từ request tiếp theo. Không chạy thêm baseline/SWIFT hoặc ablation khác.

`results.csv` được cập nhật sau mỗi cấu hình, chứa tokens/s theo code tác giả,
global throughput, thời gian, M/alpha, latency p50/p95, optimization time,
HumanEval pass@1 và GSM8K accuracy/missing/token-limit rate. Không có AR/SWIFT
thì các cột speedup, mismatch và paired correctness để trống; không tái sử dụng
baseline 7B cho 13B. CNN/DM chưa có ROUGE trong runner này.

CNN/DM mặc định vẫn 64 token với key `cnndm`; 512 dùng `cnndm.quality512`.
Cột `max_new_tokens` ghi rõ ngân sách. CNN/DM-512 là mở rộng thực nghiệm,
không gọi là cấu hình 64-token của paper. Source và artifacts được lưu như cũ;
runner hiện vẫn cần ref AdaptiveCache khi chuẩn bị source, dù không chạy nó.

Dùng môi trường SWIFT đang có; model ID được tải qua Hugging Face nếu chưa cache. Llama-2 cần quyền truy cập và đăng nhập sẵn. Không cài requirements của KnapSpec vào môi trường này.

```bash
cd /data/SWIFTEXPER
git fetch origin
git switch research/swift-mask-search
git pull --ff-only origin research/swift-mask-search
git fetch origin AdaptiveCache:refs/remotes/origin/AdaptiveCache
source .venv/bin/activate

# Diagnostic: 16 câu train, 7B, chín phương pháp tuần tự.
CUDA_VISIBLE_DEVICES=0 bash scripts/run_mask_search.sh \
  --stage diagnose --models meta-llama/Llama-2-7b-hf \
  --tasks gsm8k --gsm8k-profiles paper64
```

Sau khi phân tích diagnostic và chốt protocol, đánh giá test:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_mask_search.sh \
  --stage evaluate \
  --models meta-llama/Llama-2-7b-hf meta-llama/Llama-2-13b-hf \
  --tasks cnndm gsm8k --gsm8k-profiles paper64 quality512 --seeds 2024
```

Mỗi lần chạy tạo thư mục mới trong `_local/runs/`; không resume tự động. Chạy một inference process mỗi thời điểm, in trực tiếp terminal, không ghi text runtime log. JSON/CSV và source snapshot vẫn được lưu để truy nguyên. Tốc độ ở diagnose có chi phí probe bổ sung nên không dùng làm benchmark. Kiểm tra local chưa thay thế inference thật trên H100.

Các mục dưới mô tả runner đối chứng, protocol và công cụ phân tích chi tiết.

## Run on the H100 server

Use the existing SWIFT Python environment. Model IDs are passed to Hugging Face; pre-downloaded model directories are not required. Access to the LLaMA-2 repository must already be configured in that environment.

```bash
cd /data/SWIFTEXPER
git switch main
git pull --ff-only origin main
git fetch origin AdaptiveCache:refs/remotes/origin/AdaptiveCache
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 bash scripts/run_h100.sh


note :
CUDA_VISIBLE_DEVICES=0 bash scripts/run_h100.sh \
  --llama-model meta-llama/Llama-2-7b-hf \
  --code-model codellama/CodeLlama-7b-hf
```

The launcher uses the current Python interpreter, performs the full experiments directly, and automatically evaluates HumanEval. It does not run a preliminary smoke experiment or warm-up, install packages, or modify the original algorithm files in the working repository.

| Dataset | Target model | Tasks | Maximum new tokens | Prompt |
|---|---|---:|---:|---|
| CNN/DailyMail | `meta-llama/Llama-2-13b-hf` | 1,000 | 64 | 1-shot |
| HumanEval | `codellama/CodeLlama-13b-hf` | 164 | 512 | Original function prompt |

Both datasets run baseline, SWIFT, and AdaptiveCache. Default settings are greedy decoding, float16, batch size 1, one GPU, and seed 2024. The script starts each AdaptiveCache process with an empty cache. Repeated runs keep data seed 2024 and rotate method order:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_h100.sh --seeds 2024 2025 2026
```

To override Hugging Face model IDs, interpreter, or Git refs, use `LLAMA_MODEL_PATH`, `CODE_MODEL_PATH`, `PYTHON_BIN`, `SWIFT_REF`, or `ADAPTIVE_REF`. Despite their names, the model variables accept either Hugging Face IDs or local paths. `--help` lists the remaining options.

### GSM8K: speed and answer accuracy

Run only GSM8K, without repeating CNN/DM or HumanEval:

```bash
# LLaMA-2-13B; both profiles, three methods per profile, sequentially.
CUDA_VISIBLE_DEVICES=0 bash scripts/run_h100.sh --tasks gsm8k

# LLaMA-2-7B; no CodeLLaMA is loaded for GSM8K.
CUDA_VISIBLE_DEVICES=0 bash scripts/run_h100.sh --tasks gsm8k \
  --llama-model meta-llama/Llama-2-7b-hf
```

By default this runs `paper64` (64 new tokens) followed by `quality512` (512 new tokens).
Select one with `--gsm8k-profiles paper64` or `--gsm8k-profiles quality512`.
Use `--tasks cnndm humaneval gsm8k` to run all datasets. Without `--tasks`, the original two-dataset default stays unchanged.

Both GSM8K profiles use the same seeded 1,000 test questions and five train demonstrations from `openai/gsm8k`, configuration `main`. Prompts request a final `#### number`. This is our explicit prompt/scoring protocol, not a claim of identical author prompts. Too-long GSM8K prompts fail instead of dropping demonstrations or question text. Target model IDs in JSONL now derive from the selected model path, including 7B overrides.

`results.csv` distinguishes `gsm8k.paper64` and `gsm8k.quality512`. It includes accuracy, accuracy delta, missing-answer rate, token-limit rate, and paired correct-to-wrong/wrong-to-correct counts relative to baseline. The corresponding `gsm8k.<profile>.<method>.accuracy.json` contains per-question predictions and IDs of changed correctness. Missing answers count as incorrect. Scoring uses the first `####` numeric answer before any subsequent `Question:`; commas, signs and decimal formatting are normalized, with no fallback to numbers in intermediate reasoning. Token-limit rate counts outputs whose length equals the budget (even if EOS is the last token); it does not prove an answer was truncated. Inspect it together with missing-answer rate. `quality512` is the main quality profile, but 512 tokens do not guarantee every solution finishes. The 64-token profile matches the paper's generation budget, not necessarily its exact full protocol or hardware results.

## Results and reproducibility

### Research branch: controlled mask-search ablations

On `research/swift-mask-search`, the three new source files are `model/mask_search.py` (selection and isolated replay), `scripts/run_mask_search.py` (snapshots, data splits, metrics), and `scripts/run_mask_search.sh` (launcher). Existing inference/model files and the original H100 launcher stay unchanged. `_local/references/knapspec/` is not imported or copied into this experiment.

```bash
# First diagnostic: 16 train questions, 7B, all nine methods sequentially.
CUDA_VISIBLE_DEVICES=0 bash scripts/run_mask_search.sh \
  --stage diagnose --models meta-llama/Llama-2-7b-hf \
  --tasks gsm8k --gsm8k-profiles paper64

# Larger development diagnostic on both model sizes; no test-set tuning.
CUDA_VISIBLE_DEVICES=0 bash scripts/run_mask_search.sh --stage diagnose --data-num 128

# Full evaluation only after selecting the protocol on development data.
CUDA_VISIBLE_DEVICES=0 bash scripts/run_mask_search.sh \
  --stage evaluate --seeds 2024 2025 2026
```

Default methods: AR, original SWIFT, `match_uniform`, `cost_uniform`, `match_halving`, `cost_halving`, `cost_halving_refresh`, AdaptiveCache and `fixed_first` (freeze the native SWIFT mask after request one, no embedding). `--methods` can select any nonempty subset, including only `fixed_first`; comparison columns are populated only for references selected in that run. Default models are LLaMA-2-7B/13B; default tasks are CNN/DM (64 tokens) and both GSM8K budgets. Use `--cnndm-max-new-tokens 64 512` for both CNN/DM budgets. HumanEval is opt-in with `--stage evaluate --tasks humaneval`; pair `--models` with `--code-models` when overriding sizes. Only one GPU and one inference subprocess run at a time. No installation, warm-up or GPU smoke test is performed.

Development uses train splits, excludes the one CNN/DM demonstration and five GSM8K demonstrations, and never uses HumanEval for tuning. Evaluation uses the existing test protocol. Each process starts fresh; method order rotates across seeds. Each run creates a new `_local/runs/mask_search_<timestamp>/` with original Git archives, executed source diffs, driver copies and hashes. Source edits are applied only inside those snapshots. Use `--prepare-only` to inspect them without inference.

**Controlled comparisons:** all five research policies use an incumbent plus seven unique seeded random masks from SWIFT's layer-skipping search space, keeping the first/last layers. They use a local RNG, not generation RNG. Unlike original SWIFT, this controlled proposal pool does not use Bayesian optimization; only objective/allocation/window change among the five policies. Incumbents can diverge after different selections, so only the random proposals and proposal seeds are paired, not the entire evolving pool. Candidate/profile order is randomized reproducibly.

Each race uses 192 replay input tokens: uniform = `8*(23+1)`; halving = `8*(7+1)+4*(15+1)+2*(31+1)`. Eight one-token draft profiles plus one six-token dense verification profile make **206 candidate-input units** per race. Default per-request budget 824 permits at most four races, separated by at least 32 committed tokens. This matches evaluated token units, not forward-call counts or GPU time; all actual profiling/copying/synchronization/selection costs remain inside inference timing. `--search-time-budget-ms` is an additional soft limit checked between races; an in-progress race is not interrupted. No early stop from a high in-window score is used in these research policies.

The cost objective is an explicitly approximate linear prefix-yield/latency proxy with horizon 5, inspired by cost-aware selection in KnapSpec. It is not measured SWIFT tree M or tree throughput. All policies pay the same profiling probes, including match-only variants. Fixed-window variants replay the first generated context; refresh uses the latest verified window with correctly cropped scratch target KV. No model weights, target acceptance rules or committed tokens are modified.

`diagnose` additionally pays for independent linear draft/full-target-verification probes of eight masks on the first race of the first four requests per process. These use private KV and never commit probe outputs. **Diagnostic speedups are not benchmark claims.** Per-method `*.diagnostics.csv` pairs proxy scores with linear probes and later observed tree M. Partial scores for halving losers have fewer evaluated labels, recorded explicitly. JSONL `forward_metrics.search_events` contains masks, context offsets, resource counts and post-selection observations. Those are research artifacts, not terminal text logs.

`results.csv` reports actual tree M/alpha, macro/global speeds, latency ratios, request p50/p95, mismatches, cache/embedding costs, candidate-token units, search wall time and diagnostic time; GSM8K also includes accuracy/missing/cap/paired correctness. Search/diagnostic time overlaps optimization wall time and must not be added twice. The post-selection host-time proxy subtracts search wall time; synchronized request wall time remains the authoritative end-to-end measurement. No speedup or novelty has been established by implementation or CPU checks alone.

### Re-score saved HumanEval tokens without GPU inference

```bash
python scripts/recheck_humaneval.py --run-dir \
  _local/runs/h100_20261019T183923223276Z \
  _local/runs/h100_20261019T205721904342Z
```

This loads only the cached tokenizer at the recorded model revision, reconstructs completions in prompt context, and evaluates all three methods for every recorded seed. Use the original environment/Transformers version. Add `--allow-download` only if tokenizer files are missing from the local cache; model weights are never loaded. Add `--diagnose-only` to decode and check syntax without running generated tests.

Each run receives a new `humaneval_context_<timestamp>/` containing samples, per-task diagnostics, evaluator results, scores and a comparison `results.csv`. Original JSONL, scores, timing and the original run's CSV are unchanged. The recheck CSV contains quality results only; pair it with the original speed results and explicitly identify the completion policy. Decoding changes do not guarantee a score increase. No indentation repair, cropping or answer selection is performed. New inference runs use the same `prompt_context_decode_v1` policy for HumanEval only; output token IDs and decoder algorithms are unchanged.

Each run creates `_local/runs/h100_<UTC timestamp>/`. Read `results.csv` for decoding speed, speedups, acceptance metrics, token mismatches, and HumanEval pass@1. Per-task code correctness is in `seed_2024/humaneval.<method>.samples.jsonl_results.jsonl`; aggregate scores are in `humaneval.<method>.pass1.json`. Commands, downloads, inference progress, optimization and evaluator output stream directly to the terminal with unbuffered Python output. The launcher creates no text log files and does not append operational messages to `EXPERIMENT_LOG.md`.

### Speed metric protocol

The primary protocol, `swift_author_code_v1`, follows the authors' [`evaluation_llama/speed.py`](https://github.com/hemingkx/SWIFT/blob/main/evaluation_llama/speed.py). For each method, with output count `n_i` and inference time `t_i` for task `i`:

```text
tokens_per_second   = mean(n_i / t_i)
speedup_vs_baseline = tokens_per_second(method) / tokens_per_second(baseline)
speedup_vs_swift    = tokens_per_second(method) / tokens_per_second(SWIFT)
```

The primary speedup is the ratio of mean speeds, not the mean of per-task speedups. Secondary fields are explicitly separate: `global_tokens_per_second = sum(n_i) / sum(t_i)`, `global_throughput_speedup_vs_*`, and `inference_time_speedup_vs_*`. Summary/comparison JSON, CSV, run configuration, and inference metadata record `metric_protocol` to distinguish the primary convention.

Earlier runner versions used global throughput in the primary CSV columns and did not record this metric protocol. Do not mix those primary columns with new results. For a completed old run, recompute the metrics from its per-task JSONL and reuse its existing HumanEval scores:

```bash
python scripts/summarize_results.py --run-dir _local/runs/h100_<timestamp>
```

This produces `results.swift_author_code_v1.csv` and `metrics.swift_author_code_v1.json`, keeping the original outputs, summary sidecars, CSV, and pass@1 files unchanged. It does not execute inference or HumanEval. If these versioned files already exist, select a new `--output-dir`. This protocol identifier specifies speed aggregation; it does not claim an unchanged upstream runtime. The source corrections documented below still apply.

Original Git snapshots are saved under `snapshots/`. The executed source copies are under `sources/`, with `SOURCE_PROVENANCE.json` and `SOURCE_CHANGES.patch` documenting the source commit and every applied correction. The exact launcher and patch assets are preserved under `driver/`.

Common corrections preserve HumanEval indentation, export evaluator-compatible task IDs/completions, honor output paths, and enforce output token/EOS boundaries. They are applied equally to both source copies. Cache behavior is defined by the AdaptiveCache branch itself, not by the launcher. Search and acceptance functions in the original model utilities are not rewritten. See [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) for the decisions and current execution status. The mask-search prototype has not yet been benchmarked on H100; supplied earlier measurements concern the existing SWIFT/AdaptiveCache experiments.

HumanEval executes generated Python; use the evaluation environment intended for generated code. A failed inference or evaluator step prints its traceback directly to the terminal, stops the run and leaves `failure.json` with partial result artifacts. A successful run produces `status.json` with `status: complete`. Token mismatches are reported for analysis and do not suppress pass@1 evaluation.

For failures, use the original subprocess traceback printed before the runner error, not exit status alone. LLaMA-2 access requires a Hugging Face account approved for `meta-llama/Llama-2-13b-hf` and authentication on the server. A 401/GatedRepoError at model download is an access issue; installing packages or changing the decoding algorithm does not grant model access.

## What to commit

Commit the experiment tools, documentation, ignore rules, and intended source changes. Do not commit `_local/`: it contains archived local work and generated artifacts, not source required by the runner. Old tracked `.pyc` files are removed; Python regenerates them when needed.

Publish both branches: the launcher/documentation commit on `main`, and the contribution fix commit on `AdaptiveCache`. The server must fetch the updated `origin/AdaptiveCache`; the runner rejects the older inline-cache implementation rather than silently replacing it. Keeping a worktree under `_local/` does not prevent its committed branch code from being pushed normally.

## Attribution

SWIFT is the work of its original authors. AdaptiveCache is an experimental extension; its scientific contribution must be supported by the measured results. The original codebase acknowledges [Self-SD](https://github.com/dilab-zju/self-speculative-decoding) and [EAGLE](https://github.com/SafeAILab/EAGLE). Original paper assets and the Apache-2.0 license are retained.

```bibtex
@inproceedings{xia2025swift,
  title={{SWIFT}: On-the-Fly Self-Speculative Decoding for {LLM} Inference Acceleration},
  author={Heming Xia and Yongqi Li and Jun Zhang and Cunxiao Du and Wenjie Li},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=EKJhH5D5wA}
}
```
