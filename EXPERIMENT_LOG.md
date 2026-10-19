# Nhật ký thực nghiệm SWIFT và AdaptiveCache

> Bố cục hiện hành: bắt đầu từ `Readme.md`, chạy bằng `scripts/run_h100.sh`. Toàn bộ tài liệu và artifact local ở `_local/` (không đưa lên Git); output mới mặc định ở `_local/runs/`. Các đường dẫn `research/`, `runs/`, `Paper/` trong những mục cũ dưới đây là lịch sử trước khi dọn repo.

> Cập nhật kiến trúc mới nhất: code cache đã được chuyển trực tiếp vào nhánh `AdaptiveCache` (commit local `fec8d32`). `scripts/assets/` chỉ còn harness đánh giá chung, không còn bản cache hoặc wrapper thay thế. Cần push cả nhánh contribution và main trước khi server chạy phiên bản runner này.

## 2026-10-20 — Phạm vi và trạng thái

Mục tiêu là một lệnh chạy trên H100: CNN/DailyMail 1.000 bài với LLaMA-2-13B, rồi HumanEval 164 bài với CodeLLaMA-13B; mỗi bộ chạy baseline, SWIFT từ `main` và AdaptiveCache từ nhánh contribution. HumanEval được chấm pass@1 tự động. Mọi ghi chú triển khai và kết quả sẽ nối tiếp trong file này; không tạo báo cáo Markdown riêng.

Code tracked của repository gốc chưa bị sửa. Lần triển khai trước đã tạo hai local clone trong `research/` và các thay đổi chưa hoàn tất. Runner chính mới không sử dụng các clone đó: mỗi run xuất source từ Git ref gốc rồi áp dụng các sửa đổi có ghi diff trong thư mục run. Snapshot nguồn và code sau sửa đều được lưu để truy nguyên.

Các ghi chú trước đây đã được hợp nhất: đã xác định output HumanEval thiếu schema evaluator, `.strip()` làm mất indentation, prompt bị đổi spaces thành tab, `--answer-file` bị bỏ qua, điều kiện dừng có thể trả token quá ngân sách/EOS. Nhánh AdaptiveCache có vấn đề reset layer set nhưng giữ optimizer, LRU cập nhật key query thay vì key tìm trúng và embedding có thể gặp tree mask cũ. Đây là các vấn đề implementation; chưa phải bằng chứng thuật toán SWIFT cốt lõi sai.

Quyết định hiện hành: sửa chung phần dữ liệu/output và giới hạn token/EOS ở cả hai bản làm việc; giữ thuật toán search/acceptance gốc. Sửa vòng đời cache riêng phía contribution. Cache miss tiếp nối incumbent của stream; cache hit chỉ thay cấu hình cho request hiện tại. Tính đầy đủ embedding/search vào thời gian. Giữ BO random_state=1 của tác giả; data seed cố định 2024, run seed mặc định 2024. Không thêm smoke test, warm-up hoặc chặn chạy đầy đủ vì token mismatch.

Máy local có torch 2.7.1+cu118, transformers 5.5.0 và không có CUDA; chưa chạy inference H100, chưa có pass@1 hoặc speedup. Môi trường server đang dùng được giữ nguyên, script không tự nâng dependency.

## 2026-10-20 — Runner và đánh giá đã triển khai

Đã tạo `scripts/run_h100.sh`, runner Python, helper chấm HumanEval và assets áp dụng vào source tạm. Mặc định chạy đủ 1.000/164 bài, không smoke test/warm-up, seed 2024; `--seeds 2024 2025 2026` dùng cho đợt lặp và tự luân phiên thứ tự phương pháp. Cache bắt đầu rỗng mỗi process, không dùng persistence. Sau mỗi giai đoạn runner ghi tiếp vào chính file này.

Các sửa chung ở SWIFT và AdaptiveCache: output có task ID/token IDs, indentation nguyên bản, manifest dữ liệu, metadata môi trường/model, đường dẫn output thực sự được sử dụng, summary sidecar riêng; cắt prefix verified tại EOS/ngân sách và không draft thêm khi đã kết thúc. Giữ nguyên `model/swift/utils.py` của hai Git ref (không đổi search, acceptance hoặc điều kiện dừng tối ưu của tác giả). Baseline đặt eager attention cho phù hợp backend custom SWIFT. Những thay đổi này được xuất đầy đủ ở `SOURCE_CHANGES.patch`, không gán là novelty của contribution.

AdaptiveCache được tách lớp và bọc quanh decoder chung để sửa nhất quán state trên hit/miss, LRU và tree mask; source nhánh trước sửa vẫn lưu trong original ZIP. Timer bao gồm forward embedding, search và update cache. Metadata ghi request nguồn của entry và việc incumbent có cải thiện ở request đó hay không. Vẫn dùng một forward embedding bổ sung, chưa thực hiện ablation tái sử dụng prefill.

Helper chấm gọi trực tiếp `human_eval.evaluation.evaluate_functional_correctness(k=[1])`. Nó kiểm tra đủ 164 ID duy nhất, bảo toàn completion, đối chiếu pass@1 với số boolean `passed` trong kết quả từng bài, rồi lưu `humaneval.<method>.pass1.json`. Không có điểm hoặc thiếu kết quả sẽ báo lỗi. Có kiểm tra tĩnh lời gọi exec trong evaluator cài đặt để không vô tình báo điểm từ bản đã vô hiệu hóa thực thi. Mã nguồn tham khảo: https://github.com/openai/human-eval/blob/master/human_eval/evaluation.py và https://github.com/openai/human-eval/blob/master/human_eval/execution.py .

`results.csv` được cập nhật sau mỗi dataset gồm throughput tổng, speedup với baseline/SWIFT, mean accepted length, acceptance rate, token mismatch, cache metrics và pass@1 cho HumanEval. Token mismatch được ghi lại để nghiên cứu, không chặn evaluator. Không tự kết luận độ chính xác không giảm.

### Cách chạy trên server

Copy thư mục `scripts/` cùng file log này vào repository trên H100, có Git refs `main` và `origin/AdaptiveCache`. Không cần copy thư mục `research/` cũ hoặc output local. Kích hoạt môi trường SWIFT đã cài, bảo đảm có `human-eval` (đã nằm trong requirements của repo) và quyền tải/đọc model.

```bash
cd /data/SWIFTEXPER
CUDA_VISIBLE_DEVICES=0 bash scripts/run_h100.sh
```

Nếu model đã nằm trong ổ đĩa server:

```bash
CUDA_VISIBLE_DEVICES=0 \
LLAMA_MODEL_PATH=/data/models/Llama-2-13b-hf \
CODE_MODEL_PATH=/data/models/CodeLlama-13b-hf \
bash scripts/run_h100.sh
```

Lặp ba seed sau đợt đầu:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_h100.sh --seeds 2024 2025 2026
```

Các lựa chọn khác: `PYTHON_BIN` chọn interpreter, `SWIFT_REF`/`ADAPTIVE_REF` chọn Git ref, `OUTPUT_DIR` chọn thư mục run mới, `LLAMA_REVISION`/`CODE_REVISION` khóa model revision. Không dùng output đã tồn tại. Mặc định output là `runs/h100_<UTC timestamp>/`, từng seed có log và JSONL riêng. Log chung duy nhất là file này. Điểm HumanEval là tỷ lệ 0–1, ví dụ 0.311 tương đương 31.1%; chưa có số thực nghiệm ở thời điểm viết đoạn này. Evaluator thực thi code model sinh ra, nên dùng môi trường đánh giá cô lập trên server.

## 2026-10-20 — Bàn giao script

Đã kiểm tra cú pháp Python của scripts và source được xuất; `bash -n scripts/run_h100.sh` thành công với Git Bash. Đã chạy chế độ chỉ chuẩn bị source bằng `python scripts/run_experiments.py --prepare-only --output-dir runs/source_review_final_20261020`: xuất đúng hai commit, tạo original ZIP và diff; không gọi inference hay evaluator. Một lần kiểm tra local trước đó gặp lỗi encoding console Windows, đã sửa stdout/stderr UTF-8 và ghi rõ run thất bại trong log. Không có kết quả giả lập được báo cáo thành số liệu nghiên cứu.

Đã giữ tương thích phần ghi source với Python 3.9 của môi trường SWIFT gốc. `git diff --exit-code -- evaluation_llama model eval_llama.sh requirements.txt` xác nhận code tracked ban đầu không bị sửa. Các file báo cáo riêng tạo trước đây đã bỏ sau khi gộp nội dung vào file này; entrypoint cũ trong `research/` chuyển tiếp sang runner mới để tránh dùng quy trình smoke/warm-up đã hủy. `.gitignore` loại các bản local clone, output và bytecode khỏi việc commit nhầm.

Trạng thái cuối ở máy local: **đã chuẩn bị script và kiểm tra source/cú pháp; chưa chạy H100, chưa có pass@1 hoặc speedup thực tế**. Chạy lệnh server ở trên để thực hiện toàn bộ inference và chấm điểm. Mỗi run mới dùng source Git refs gốc cùng bản patch đi kèm, không cần copy `research/` hoặc `runs/` từ local.

## Chuyển bộ chạy lên Git để server pull

Chỉ đưa `.gitignore`, `scripts/` (gồm cả `assets/` và `.gitattributes`) và `EXPERIMENT_LOG.md` vào commit. Không đưa `research/`, `runs/`, `Paper/` hay `desktop.ini` vào commit này. Tại lần kiểm tra, các file bàn giao vẫn untracked; chưa tạo commit hoặc push.

Model mặc định là ID Hugging Face `meta-llama/Llama-2-13b-hf` và `codellama/CodeLlama-13b-hf`; không yêu cầu model có sẵn dưới `/data/models`. Thư viện tự tải hoặc dùng Hugging Face cache của môi trường hiện có. Trên server cần có cả ref `main` và `origin/AdaptiveCache`; runner không tự fetch nhánh để tránh thay đổi source ngoài ý muốn.

```bash
# Local: chỉ stage bộ bàn giao, không dùng git add .
git add -- .gitignore scripts EXPERIMENT_LOG.md
git commit -m "Add H100 experiment runner and HumanEval pass@1 evaluation"
git push origin main

# Server: cập nhật main và ref contribution, dùng môi trường đã cài
cd /data/SWIFTEXPER
git switch main
git pull --ff-only origin main
git fetch origin AdaptiveCache:refs/remotes/origin/AdaptiveCache
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 bash scripts/run_h100.sh
```

### 2026-10-19T17:29:50+00:00

Bắt đầu run `C:\Users\Admin\SWIFTEXPER\runs\source_review_20261020`. Chuẩn bị source từ `main` và `origin/AdaptiveCache`; repository gốc không bị sửa.

### 2026-10-19T17:29:50+00:00

Run `C:\Users\Admin\SWIFTEXPER\runs\source_review_20261020` thất bại tại prepare sources: 'charmap' codec can't encode character '\u1eaf' in position 1: character maps to <undefined>. Giữ nguyên artifact và log; không coi là run đầy đủ.

### 2026-10-19T17:30:01+00:00

Bắt đầu run `C:\Users\Admin\SWIFTEXPER\runs\source_review_20261020_v2`. Chuẩn bị source từ `main` và `origin/AdaptiveCache`; repository gốc không bị sửa.

### 2026-10-19T17:30:01+00:00

Đã chuẩn bị source và lưu original ZIP, SOURCE_CHANGES.patch, metadata cấu hình. SWIFT `a0b0256f42de24e6ce052d350fcb7905ee8f68d6`, AdaptiveCache `bed868364a43aec122154b865735878b4230fc75`. Không chạy smoke test hoặc warm-up.

### 2026-10-19T17:30:01+00:00

Run `C:\Users\Admin\SWIFTEXPER\runs\source_review_20261020_v2` chỉ chuẩn bị source; chưa chạy GPU, chưa có điểm.

### 2026-10-19T17:31:32+00:00

Bắt đầu run `C:\Users\Admin\SWIFTEXPER\runs\source_review_final_20261020`. Chuẩn bị source từ `main` và `origin/AdaptiveCache`; repository gốc không bị sửa.

### 2026-10-19T17:31:33+00:00

Đã chuẩn bị source và lưu original ZIP, SOURCE_CHANGES.patch, metadata cấu hình. SWIFT `a0b0256f42de24e6ce052d350fcb7905ee8f68d6`, AdaptiveCache `bed868364a43aec122154b865735878b4230fc75`. Không chạy smoke test hoặc warm-up.

### 2026-10-19T17:31:33+00:00

Run `C:\Users\Admin\SWIFTEXPER\runs\source_review_final_20261020` chỉ chuẩn bị source; chưa chạy GPU, chưa có điểm.

## 2026-10-20 — Sắp xếp repository để bàn giao

Theo yêu cầu gom phần không đưa lên Git vào một folder, đã tạo `_local/` và di chuyển (không xóa dữ liệu): `Paper/` sang `_local/papers/`, `research/` sang `_local/archive/research/`, `runs/` sang `_local/runs/`, `desktop.ini` sang `_local/system/`. Bytecode ở source/scripts được gom vào `_local/archive/bytecode/`. Các local clone cũ chỉ còn là bản lưu, không dùng làm entrypoint hoặc source chạy chính. Đường dẫn tuyệt đối bên trong metadata của run local cũ giữ nguyên giá trị lịch sử, không phải đường dẫn dùng cho run mới.

Đã viết lại `Readme.md` thành trang giới thiệu project, phân biệt upstream và contribution, sơ đồ thư mục, một lệnh chạy H100, cấu hình benchmark, cách đọc pass@1/results.csv và attribution/citation. Chỉ một file nhật ký công việc vẫn là file này; README là tài liệu sử dụng repo, không phải một báo cáo việc làm khác.

Runner mặc định ghi vào `_local/runs/h100_<timestamp>/`; launcher chuyển bytecode mới vào `_local/python_cache/` bằng PYTHONPYCACHEPREFIX, tương thích Python 3.9. `.gitignore` bỏ qua toàn bộ `_local/`, thêm guard cho output legacy và desktop.ini. Vì 6 file `.pyc` đã bị commit trong repo gốc, việc di chuyển chúng thể hiện thành 6 deletion trong Git; cần đưa các deletion này vào commit dọn repo. Không đổi `.py` của thuật toán SWIFT gốc, requirements, eval_llama.sh, skip_layers.json hoặc LICENSE.

Lệnh PowerShell script dọn ban đầu bị execution policy máy local chặn trước khi di chuyển; đã thực hiện bằng các cmdlet native trong phiên PowerShell hiện hành, không thay đổi execution policy. Mọi source/destination được kiểm tra nằm trong workspace và `_local/` trước khi Move-Item; không overwrite thư mục đã tồn tại.

### Commit và chạy sau khi sắp xếp

```bash
# Local: stage đúng tài liệu, bộ chạy và việc bỏ bytecode đã tracked
git add -- .gitignore Readme.md EXPERIMENT_LOG.md scripts
git add -u -- evaluation_llama/__pycache__ model/swift/__pycache__
git diff --cached --stat
git commit -m "Organize experiment tooling and add automatic HumanEval scoring"
git push origin main

# Server
cd /data/SWIFTEXPER
git switch main
git pull --ff-only origin main
git fetch origin AdaptiveCache:refs/remotes/origin/AdaptiveCache
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 bash scripts/run_h100.sh
```

Không cần copy hoặc push `_local/`; script tự tạo thư mục output khi chạy. Model vẫn gọi bằng Hugging Face ID. Chưa commit/push thay người dùng và chưa có kết quả inference H100.

Xác minh sau sắp xếp: `git check-ignore` xác nhận papers/archive/runs bên trong `_local/` đều bị bỏ qua; diff của các file `.py` upstream và các cấu hình gốc vẫn rỗng; Bash syntax và Python 3.9 syntax của 8 file Python trong `scripts/` đều hợp lệ. Các kiểm tra này không chạy inference hoặc evaluator.

## 2026-10-20 — Đưa implementation về đúng nhánh AdaptiveCache

Theo xác nhận của người dùng, đã tạo local branch `AdaptiveCache` từ `origin/AdaptiveCache` và worktree ở `_local/worktrees/AdaptiveCache`, không checkout đè lên main đang có thay đổi. Git cần quyền ghi metadata `.git`; sandbox ban đầu chặn tạo ref, sau đó đã được cấp phép cho thao tác worktree và commit local. Không sửa quyền sandbox hoặc push remote.

Commit local **`fec8d32` — Fix AdaptiveCache retrieval and stream state lifecycle** chỉ chứa hai file contribution: `evaluation_llama/adaptive_cache.py` và `evaluation_llama/inference_swift.py`. Lớp cache sửa LRU/empty miss/embedding mask; forward wrapper giữ incumbent trên miss, tạm dùng cấu hình hit rồi khôi phục state bằng finally. Cache khởi tạo trước vòng chạy; clear bỏ qua load file cũ. Có provenance của entry và thời gian embedding/search/update. Decoder bên trong branch giữ nguyên body của SWIFT main, chỉ đổi tên `_swift_forward_impl`; kiểm tra AST xác nhận điều này trước khi commit. Sửa token/EOS và xuất HumanEval vẫn do phần chuẩn hóa chung ở runner áp dụng tương đương cho hai bên.

Đã bỏ `scripts/assets/adaptive_cache.py` và `adaptive_forward.txt`. `source_patches.py` nay đọc chính decoder từ branch contribution, không dựng implementation AdaptiveCache mới từ main. File cache của branch được giữ nguyên khi xuất run. Runner từ chối ref contribution cũ và hướng dẫn push/fetch branch mới. Nhánh remote `origin/AdaptiveCache` tại máy local chưa cập nhật vì chưa push; chuẩn bị source local dùng `--adaptive-ref AdaptiveCache`.

### Đưa cả hai nhánh lên server

```bash
# Từ checkout main: commit bộ chạy/tài liệu và việc bỏ bytecode nếu chưa commit
git add -- .gitignore Readme.md EXPERIMENT_LOG.md scripts
git add -u -- evaluation_llama/__pycache__ model/swift/__pycache__
git commit -m "Use branch-owned AdaptiveCache with shared H100 evaluation"
git push origin main AdaptiveCache

# Server
cd /data/SWIFTEXPER
git switch main
git pull --ff-only origin main
git fetch origin AdaptiveCache:refs/remotes/origin/AdaptiveCache
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 bash scripts/run_h100.sh
```

Commit contribution đã lưu local; bộ chạy trên main vẫn chưa commit. Không cần stage thư mục worktree: Git push branch sẽ gửi code đã commit dù worktree nằm trong `_local/` bị ignore. Chưa có kết quả H100 hoặc pass@1 thực tế.

Đã xuất source với `--prepare-only --adaptive-ref AdaptiveCache` tại `_local/runs/branch_owned_source_review_20261020`. Hai commit được lấy đúng là main `a0b0256` và contribution `fec8d32`. File cache được runner giữ byte-for-byte như trong Git archive; nội dung cũng khớp `git show` sau chuẩn hóa CRLF/LF (archive trên máy Windows dùng CRLF, Git blob dùng LF). Worktree AdaptiveCache sạch sau commit; source `.py` upstream main không đổi. Đây chỉ là xác minh source/cú pháp, không có inference hoặc số điểm giả lập.

### 2026-10-19T17:44:56+00:00

Bắt đầu run `C:\Users\Admin\SWIFTEXPER\_local\runs\branch_owned_source_review_20261020`. Chuẩn bị source từ `main` và `AdaptiveCache`; repository gốc không bị sửa.

### 2026-10-19T17:44:57+00:00

Đã chuẩn bị source và lưu original ZIP, SOURCE_CHANGES.patch, metadata cấu hình. SWIFT `a0b0256f42de24e6ce052d350fcb7905ee8f68d6`, AdaptiveCache `fec8d32dd805314a185ad245db40ca1c9dd31bb7`. Không chạy smoke test hoặc warm-up.

### 2026-10-19T17:44:57+00:00

Run `C:\Users\Admin\SWIFTEXPER\_local\runs\branch_owned_source_review_20261020` chỉ chuẩn bị source; chưa chạy GPU, chưa có điểm.

## Công bố bộ chạy lên GitHub theo yêu cầu người dùng

Người dùng đã yêu cầu thực thi commit/push cả hai nhánh. Đã kiểm tra remote: `main` đang ở `a0b0256`, `AdaptiveCache` ở `bed8683`, phù hợp ancestry của các thay đổi local; không cần force push. Source `.py` SWIFT gốc và các cấu hình upstream không đổi. Bộ chạy gồm 7 file Python có syntax Python 3.9 hợp lệ, shell launcher có Bash syntax hợp lệ. `_local/` không được stage; chỉ đưa tài liệu, ignore rules, scripts và việc bỏ 6 bytecode đã tracked lên main. Contribution đã nằm trong commit `fec8d32` trên AdaptiveCache.

Git trong sandbox không đọc được credentials Windows; kiểm tra remote đã thực hiện ngoài sandbox với quyền được cấp. Không thay đổi credential, remote URL hoặc lịch sử Git. Kết quả commit/push và xác minh remote sẽ được ghi nối tiếp sau khi lệnh hoàn thành. Chưa chạy thực nghiệm H100 ở bước công bố source này.

### Đã push thành công

Đã tạo commit main `bb07794` — Add organized H100 experiments with automatic HumanEval pass@1 — và push atomic cùng contribution `fec8d32` lên `https://github.com/pitorr05/SWIFTEXPER`. GitHub đã nhận cả hai nhánh; `git ls-remote --heads origin main AdaptiveCache` xác nhận đúng hai commit sau push. Không có force push; `_local/` và dữ liệu local không nằm trong commit. Mục xác nhận này được lưu bằng commit tài liệu tiếp theo trên main.

Lệnh trên server để lấy cả hai nhánh và chạy (model tự tải qua Hugging Face ID):

```bash
cd /data/SWIFTEXPER
git switch main
git pull --ff-only origin main
git fetch origin AdaptiveCache:refs/remotes/origin/AdaptiveCache
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 bash scripts/run_h100.sh
```

Chỉ hoàn thành công bố source; chưa chạy job trên server H100 hoặc báo điểm pass@1. Output thực tế sẽ nằm trong `_local/runs/` và được ghi tiếp vào nhật ký khi script chạy trên server.
