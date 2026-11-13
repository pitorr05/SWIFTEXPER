# SWIFTEXPER

Repo thực nghiệm self-speculative decoding trên một GPU H100. Nhánh hiện tại: **research/skipbook**.

## Bắt đầu từ đâu?

| Bạn cần làm gì? | File cần đọc hoặc chạy |
|---|---|
| Chạy AR, SWIFT và AdaptiveCache (protocol cũ) | `bash scripts/run_h100.sh` |
| Chạy SkipBook, PrefillCache, oracle và ablation | `bash scripts/run_mask_search.sh` |
| Đọc thuật toán mới | `model/skipbook.py` |
| Phân tích oracle headroom (CPU) | `scripts/analyze_headroom.py` |
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
|   |-- swift/               # Model, decoding, KV cache gốc (không sửa)
|   `-- skipbook.py          # SkipBook + PrefillCache
|-- scripts/
|   |-- run_h100.sh          # AR / SWIFT / AdaptiveCache
|   |-- run_mask_search.sh   # SkipBook, PrefillCache, fixed_mask, ablation
|   |-- analyze_headroom.py  # Bank mask + oracle headroom (CPU)
|   |-- *.py                 # Điều phối, chuẩn bị source, chấm, tổng hợp
|   `-- runtime/             # Python hỗ trợ snapshot thực nghiệm
|-- assets/                  # Hình minh họa paper
`-- _local/                  # Không đưa lên Git: runs/, papers/, references/, worktrees/
```

## Phân biệt code gốc và contribution

- **main** giữ source SWIFT gốc cùng công cụ thực nghiệm. Runner áp dụng hiệu chỉnh harness/runtime trên snapshot, không sửa working source.
- **AdaptiveCache** chứa cache retrieval của contribution trước. Từ a7d780f, key lấy từ prefill của SWIFT (`--cache-key prefill`, mặc định); `--cache-key embedding` giữ cách cũ. Runner đọc source đã commit từ `origin/AdaptiveCache`.
- **research/swift-mask-search** (lưu trữ): 5 policy chọn mask (`match_/cost_ uniform/halving/refresh`) đã chạy và **thua SWIFT**; chỉ `fixed_first` có lợi (+1–4%). Các policy này đã bị xoá khỏi nhánh hiện tại; code và kết quả vẫn ở nhánh đó và trong `_local/runs`.
- **research/skipbook** (nhánh này): `prefill_cache`, `skipbook`, các ablation có tên và oracle `fixed_mask`. Chưa có kết quả GPU; chưa xác lập speedup hay novelty.

`runtime/` gồm `harness.py` (vòng đánh giá, task `mixed`), `research_utils.py` (metric/output/stream), `gsm8k.py`, `completion_decode.py`, `phase_timer.py` và `speed_adapter.py`. `source_patches.py` chuẩn bị snapshot bằng các module này.

## Các phương pháp

| `--methods` | Là gì |
|---|---|
| `baseline` | Autoregressive (AR) |
| `swift` | SWIFT gốc: search mask online suốt stream |
| `adaptive` | AdaptiveCache (nhánh `AdaptiveCache`, từ a7d780f): key lấy từ prefill dense SWIFT đã chạy, không tốn forward thêm |
| `adaptive_embed` | AdaptiveCache với key cũ (`--cache-key embedding`: thêm 1 forward); dùng để đo phần tiết kiệm và so với số cũ |
| `fixed_first` | SWIFT search ở request đầu rồi đóng băng mask |
| `prefill_cache` | Cùng ngữ nghĩa với `adaptive` nhưng chạy qua `model/skipbook.py` (kiểm tra chéo hai bản cài đặt) |
| `skipbook` | **Phương pháp chính** (xem dưới) |
| `skipbook_k1` | Ablation: 1 mask mỗi context (bỏ bandit) |
| `skipbook_nodrift` | Ablation: tắt Page-Hinkley (không search lại khi M giảm) |
| `skipbook_onectx` | Ablation: mọi request chung 1 context (bỏ phân cụm prompt) |
| `skipbook_q64` | Ablation: key chỉ từ 64 token cuối của prompt |
| `fixed_mask` | Oracle: mỗi mask trong `--mask-bank` chạy cố định cả stream |

**SkipBook, mỗi request:**
1. Prefill dense (SWIFT vốn chạy).
2. Mean-pool hidden state → key (~0 giây).
3. Tìm context theo cosine ≥ `--sb-threshold`.
   - Context mới → **Explore**: chạy search gốc của SWIFT với optimizer Bayes riêng của context, warm-start từ mask tốt nhất của context gần nhất. Mask tìm được thành một arm.
   - Context cũ → **Exploit**: Discounted UCB chọn 1 trong ≤ `--sb-max-arms` arm, khoá mask, không search. Reward là tokens/s decode đo thật.
4. Page-Hinkley theo dõi M; khi M giảm thì Explore lại. Arm thua rõ bị loại (successive elimination).

Mặc định `--methods` = `baseline swift fixed_first prefill_cache skipbook`.

## Chạy trên H100 (theo thứ tự)

```bash
cd /data/SWIFTEXPER
git fetch origin
git switch research/skipbook
git pull --ff-only origin research/skipbook
git fetch origin main:refs/remotes/origin/main AdaptiveCache:refs/remotes/origin/AdaptiveCache

# Môi trường (một lần): Python 3.10, torch 2.1.2 + CUDA 12.1, transformers 4.37.1 như SWIFT.
pip install uv   # nếu chưa có
uv venv --python 3.10 .venv
source .venv/bin/activate
uv pip install -r requirements-h100.txt --build-constraint build-constraints-h100.txt
uv pip install pip "setuptools<70" wheel   # human-eval phải cài bằng pip (uv từ chối entry point của nó)
python -m pip install --no-build-isolation git+https://github.com/openai/human-eval
python -c "import torch,transformers;print(torch.__version__, transformers.__version__, torch.cuda.is_available())"

export HF_HOME=/data/hf_cache    # cache model/dataset trên /data
huggingface-cli login            # token có quyền meta-llama/Llama-2-7b-hf
```

Các lần sau chỉ cần `source .venv/bin/activate && export HF_HOME=/data/hf_cache`. `requirements.txt` là file gốc của SWIFT, giữ để tái lập; runner dùng `requirements-h100.txt`.

**Bước 1. E1: stream trộn CNN/DM + GSM8K (thí nghiệm chính).** Mix ratio 0 = hai khối liền nhau, 1 = đổi domain nhiều nhất.

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_mask_search.sh --stage diagnose --data-num 200 \
  --swift-ref origin/main --models meta-llama/Llama-2-7b-hf --tasks mixed --mix-ratios 0 0.3 1 \
  --methods baseline swift fixed_first adaptive adaptive_embed skipbook \
  --output-dir _local/runs/e1_mixed_7b
```

**Bước 2. Stream đơn domain** (SkipBook không được thua SWIFT):

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_mask_search.sh --stage diagnose --data-num 128 \
  --swift-ref origin/main --models meta-llama/Llama-2-7b-hf --tasks cnndm gsm8k --gsm8k-profiles quality512 \
  --methods baseline swift fixed_first prefill_cache skipbook \
  --output-dir _local/runs/e1_single_7b
```

Xem `results.csv`: cột `speedup_vs_swift`, `skipbook_contexts` (≈ 2 trên stream trộn là đúng), `skipbook_explore_share`, `cache_hit_rate`. Nếu `skipbook_contexts` = 1 trên stream trộn thì key gộp CNN và GSM8K; khi đó thêm `skipbook_q64` vào bước 3.

**Ngưỡng đi tiếp:** SkipBook ≥ SWIFT + 5% ở mix ratio ≥ 0.3 **và** không thua SWIFT ở bước 2. Không đạt thì dừng và gửi CSV để phân tích, chưa chạy bước 3–4.

**Bước 3. Ablation** (một lệnh, một bảng):

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_mask_search.sh --stage diagnose --data-num 200 \
  --swift-ref origin/main --models meta-llama/Llama-2-7b-hf --tasks mixed --mix-ratios 0.3 \
  --methods baseline swift prefill_cache skipbook skipbook_k1 skipbook_nodrift skipbook_onectx skipbook_q64 \
  --output-dir _local/runs/e1_ablation_7b
```

**Bước 4. E2: oracle headroom** ("SkipBook đạt bao nhiêu % so với chọn mask hoàn hảo"):

```bash
python scripts/analyze_headroom.py bank --runs _local/runs/e1_mixed_7b --random 4 --replicate \
  --output _local/runs/bank_e1_7b.json
CUDA_VISIBLE_DEVICES=0 bash scripts/run_mask_search.sh --stage diagnose --data-num 200 \
  --swift-ref origin/main --models meta-llama/Llama-2-7b-hf --tasks mixed --mix-ratios 0.3 1 \
  --methods baseline swift fixed_first prefill_cache skipbook fixed_mask \
  --mask-bank _local/runs/bank_e1_7b.json --output-dir _local/runs/e2_headroom_7b
python scripts/analyze_headroom.py analyze --run _local/runs/e2_headroom_7b
```

`analysis_headroom/d2_d4_headroom.csv` có `skipbook_vs_oracle_request`, `best_single_speedup`, `oracle_domain_speedup`, `noise_selection_inflation` (nhiễu đo thời gian; trừ đi trước khi kết luận) và `M_within_request_spearman` (M có dự báo tốc độ không).

**Bước 5. Bảng paper** (chỉ sau khi bước 1–4 tốt): lặp bước 1–3 với `--stage evaluate` (split test), thêm `meta-llama/Llama-2-13b-hf` và `--seeds 2024 2025 2026`.

`diagnose` dùng split train để chỉnh tham số; số trong paper lấy từ `evaluate`. Mỗi lần chạy tạo thư mục mới trong `_local/runs/` (không resume). Một inference process mỗi thời điểm, in trực tiếp terminal. Kiểm tra CPU chưa thay thế inference thật trên H100.

### Các tuỳ chọn khác

- `--stream-order-seeds 1 2 3 ...`: hoán vị thứ tự request (đo độ nhạy của SWIFT với request đầu).
- `--sb-threshold`, `--sb-max-arms`, `--sb-gamma`, `--sb-ucb-c`, `--sb-drift`, `--sb-key-pool`: tham số mặc định của SkipBook/PrefillCache; ablation có tên ghi đè từng tham số.
- Chạy riêng `fixed_first` 13B trên năm cấu hình cũ:
  `--stage evaluate --swift-ref origin/main --models meta-llama/Llama-2-13b-hf --code-models codellama/CodeLlama-13b-hf --methods fixed_first --tasks cnndm humaneval gsm8k --cnndm-max-new-tokens 64 512 --gsm8k-profiles paper64 quality512 --data-num 1000`

Dùng môi trường SWIFT đang có; Llama-2 cần quyền truy cập Hugging Face sẵn. Runner cần ref AdaptiveCache khi chuẩn bị source, kể cả khi không chạy `adaptive`.

Các mục dưới mô tả runner protocol cũ và công cụ phân tích chi tiết.

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

### Research branch: controlled mask-search ablations (archived)

The five controlled policies (`match_uniform`, `cost_uniform`, `match_halving`, `cost_halving`, `cost_halving_refresh`) were evaluated on `research/swift-mask-search` and lost to SWIFT; they are removed from this branch. Code, protocol and results remain on that branch, in `EXPERIMENT_LOG.md` and in `_local/runs`.

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
