# SWIFT và AdaptiveCache — thiết kế, logic sửa code và diễn giải thực nghiệm

Tài liệu này giải thích implementation đang được đo, lý do thay đổi và những kết luận dữ liệu có thể hỗ trợ. Đây là ngữ cảnh để tiếp tục phát triển code và viết paper; không phải nhật ký thao tác Git hoặc output terminal. Chỉ cập nhật khi có quyết định thiết kế, đổi protocol hay kết quả thực nghiệm có ý nghĩa. Tiến trình inference/evaluator in trực tiếp ra terminal; code không tạo file log văn bản và không tự nối thông báo vào tài liệu này.

## 1. Câu hỏi nghiên cứu và ba phương pháp

Mục tiêu là kiểm tra khả năng tái sử dụng cấu hình skipped layers giữa các request để cải thiện hiệu năng self-speculative decoding. Cần đo lợi ích sau toàn bộ chi phí cache và đánh giá HumanEval pass@1 để phát hiện suy giảm functional correctness.

| Phương pháp | Nguồn và vai trò |
|---|---|
| Autoregressive | `evaluation_llama/inference_baseline.py` từ main; mốc target model dùng `model.generate()` |
| SWIFT | Code của tác giả paper trên main; mốc self-speculative decoding, có tối ưu layer set xuyên input stream |
| AdaptiveCache | Nhánh contribution; truy hồi layer set bằng embedding đầu vào, cosine similarity và cache online |

Mốc đối chiếu source upstream là commit `a0b0256`; contribution trước sửa là `bed8683`, sau sửa vòng đời cache là `fec8d32`. Code Python của thuật toán upstream trên main được giữ nguyên. Tuy nhiên, runner xuất source từ Git rồi áp dụng chuẩn hóa vào bản thực thi. **Source nguyên bản được lưu không có nghĩa phép đo chạy implementation nguyên trạng.**

Phần model, random/Bayesian search và acceptance trong `model/swift/` không bị runner viết lại. AdaptiveCache đã được sửa trực tiếp trên nhánh contribution; scripts không chèn một implementation cache khác vào nhánh đó. Bản thực thi và diff của từng run là bằng chứng xác định chính xác code đã đo.

## 2. Protocol thực nghiệm hiện tại

| Tác vụ | Model Hugging Face | Số bài | Prompt | Max new tokens |
|---|---|---:|---|---:|
| CNN/DailyMail | `meta-llama/Llama-2-13b-hf` | 1.000 | 1-shot; test set 3.0.0, exemplar từ train | 64 |
| HumanEval | `codellama/CodeLlama-13b-hf` | 164 | Function prompt gốc, giữ indentation | 512 |

CNN/DailyMail giới hạn **64 token**. Hai model là target model cho hai tác vụ, không phải cặp target/draft; cả ba phương pháp trên cùng dataset đều dùng cùng target model.

Cấu hình chung: một GPU H100, batch size 1, float16, greedy (`temperature=0`). Top-p được truyền là 0.85 cho CNN/DM và 0.95 cho HumanEval nhưng không ảnh hưởng lựa chọn token khi greedy. Model được tải qua ID Hugging Face hoặc cache của môi trường.

Data seed cố định 2024, quyết định tập CNN/DM, thứ tự và exemplar. Run seed mặc định 2024, có thể lặp 2025/2026 trong khi giữ data seed. HumanEval chạy đủ 164 task. Manifest lưu ID, one-shot prompt và fingerprint; từng bản ghi lưu input token IDs/hash để xác nhận prompt thực tế giống nhau. Trường hợp CNN/DM vượt context, harness dùng cùng quy tắc cắt cho cả ba phương pháp; HumanEval quá dài sẽ báo lỗi thay vì cắt mất function prompt.

SWIFT: skip ratio 0.45, context window 32, opt interval 1, Bayesian interval 25, tối đa 1.000 bước tối ưu, tolerance 300, max score 0.95. Giữ BO `random_state=1` của tác giả, nên đổi run seed không đồng nghĩa đổi mọi nguồn ngẫu nhiên của optimizer.

AdaptiveCache: threshold cosine 0.85, capacity 100, cache rỗng đầu mỗi process, không load cache từ run trước. Các giá trị này là cấu hình khởi đầu, chưa có bằng chứng tối ưu. Hiện không warm-up; số đo gồm giai đoạn khởi đầu của stream. Lặp nhiều seed sẽ luân phiên thứ tự phương pháp. Paper dùng phần cứng khác H100, nên không kỳ vọng khớp tuyệt đối tốc độ hoặc speedup trong bảng tác giả.

## 3. Các sửa chung so với implementation nguyên bản

| Hành vi cũ | Bản thực nghiệm hiện tại | Lý do và ảnh hưởng |
|---|---|---|
| Output HumanEval chỉ có choices, thiếu task ID/completion theo evaluator | Lưu task ID; export raw completion theo schema chính thức | Cho phép chấm đúng bài, tránh coi inference hoàn thành là đã có điểm |
| Prompt đổi spaces thành tab; completion bị `.strip()` | Giữ prompt và indentation gốc; không sửa đáp án sau sinh | Tránh lỗi Python do xử lý output; thay prompt có thể thay generation so với script cũ |
| SWIFT thêm cả nhóm verified rồi kiểm tra EOS và `>` max tokens | Chỉ commit prefix đến EOS/ngân sách; kết thúc trước optimization/draft tiếp | Cùng ràng buộc đầu ra; có thể thay số token, số bước tối ưu cuối request và latency so với upstream |
| Baseline dùng attention backend mặc định của Transformers | Đặt eager attention cho baseline | Phù hợp backend custom SWIFT; có thể ảnh hưởng hiệu năng và sai khác số học của baseline |
| Khai báo `--answer-file` nhưng vẫn tự đặt đường dẫn, ghi nối; summary lẫn bản ghi | Dùng đường dẫn chỉ định, từ chối output đã tồn tại, summary sidecar riêng | Tránh trộn run; không thay logic dự đoán |
| Seed chọn data gắn với run seed | Tách data seed, xác nhận manifest/prompt giữa phương pháp | Giữ workload tương đương khi lặp seed |
| Thông tin đo và source chưa đủ để truy nguyên | Lưu token IDs, cấu hình, model revision nếu có, source diff và timer optimization | Tăng khả năng audit; instrumentation vẫn có overhead và không được giả định miễn phí |

**Cách mô tả baseline trong paper:** autoregressive dùng target model qua Hugging Face generate và eager attention; SWIFT dùng thuật toán gốc trong pipeline đánh giá đã chuẩn hóa, có sửa output/termination và instrumentation. Không viết “chạy code gốc hoàn toàn không thay đổi”. Phép tổng hợp metric đúng tác giả cũng không tự loại bỏ các khác biệt runtime này.

Những sửa chung không phải novelty của AdaptiveCache. Nếu cần tái lập runtime upstream chính xác, phải thực hiện đối chứng riêng; kết quả runner hiện tại không tự cung cấp đối chứng đó. Chưa có đo H100 định lượng ảnh hưởng từng sửa chung.

## 4. AdaptiveCache: logic cũ, quyết định sửa và hệ quả

### 4.1. Representation và lookup

Mỗi prompt đi qua full model để lấy hidden state cuối. Mean pooling theo token, tích lũy ở float32 rồi L2-normalize; cosine similarity tìm entry gần nhất. Layer set gồm hai danh sách attention/MLP skip; hit khi similarity đạt threshold. LRU giới hạn số entry. Model weights không được huấn luyện thêm, nhưng hệ thống có bộ nhớ trạng thái bổ sung.

**Vẫn có một forward embedding riêng trước prefill decoding.** Chưa tái sử dụng representation từ prefill. Nó có thể tốn nhiều thời gian với prompt dài; không được bỏ chi phí này khỏi speedup.

### 4.2. Vì sao sửa quản lý trạng thái

Incumbent là cấu hình skipped layers tốt nhất đang được SWIFT giữ cho stream; statistics/optimizer lưu điểm và ngân sách tìm kiếm tương ứng. Các thành phần này phải được quản lý nhất quán.

| Vấn đề ở contribution cũ | Quyết định hiện tại | Hệ quả cần hiểu |
|---|---|---|
| Reset uniform layer mỗi request nhưng vẫn giữ score/optimizer và cờ dừng | Miss tiếp nối incumbent cùng trạng thái tối ưu xuyên request | Tránh layer set và điểm tối ưu không tương ứng; thay chính sách thực thi so với contribution cũ |
| Hit thay cấu hình mà không tách rõ với incumbent của stream | Hit tạm dùng layer truy hồi, bỏ search request đó; finally khôi phục incumbent và cờ optimization | Cấu hình phục vụ query không ghi đè incumbent và state search của stream |
| Embedding chạy trước reset tree mask | Reset mask và tắt draft mode trước full-model embedding | Tránh representation dùng mask còn từ request trước |
| LRU move bằng key query | Move đúng key entry tìm trúng | Eviction phản ánh entry thực sự được dùng |
| Empty cache không tăng miss; clear có thể diễn ra khi cache chưa tồn tại | Khởi tạo trước vòng chạy; tính empty miss; clear bỏ qua load | Thống kê và trạng thái đầu run có nghĩa rõ ràng |
| Entry chỉ ghi embedding/layer set | Ghi task nguồn, số bước tối ưu trên request nguồn và incumbent có cải thiện hay không | Không diễn giải mọi entry là cấu hình đã tối ưu riêng cho prompt nguồn |

Miss sau khi SWIFT đã early-stop **không tự khởi động tối ưu lại**. Cache có thể lưu incumbent kế thừa dù request nguồn không tối ưu. Vì vậy, implementation hiện tại chưa chứng minh khả năng tự phát hiện domain shift hoặc tìm layer set tối ưu cho từng query. Hit cũng chưa có bước kiểm tra cấu hình truy hồi có lợi hơn incumbent về latency hay không.

Khi cache tắt, wrapper đi vào decoder mà không reset uniform mỗi request. SWIFT baseline chính vẫn chạy từ nhánh main, không được thay bằng tên gọi “AdaptiveCache tắt cache”.

Chi phí embedding, tìm kiếm, thêm entry và khôi phục trạng thái nằm trong timer ngoài. Persistence tùy chọn có schema/model/protocol identity; benchmark mặc định không dùng persistence. Cache metrics giúp giải thích overhead, không thay thế số đo tổng thể.

## 5. Chỉ số và phạm vi thời gian đo

### 5.1. Chỉ số tốc độ chính theo code tác giả

Với task i, `n_i` là số token đầu ra thực sự được trả về (kể cả EOS nếu có), `t_i` là thời gian forward được đồng bộ CUDA:

```text
V(method) = (1 / N) * sum_i(n_i / t_i)
speedup_vs_baseline = V(method) / V(autoregressive)
speedup_vs_swift    = V(method) / V(SWIFT)
```

Đây là công thức trong `evaluation_llama/speed.py` của tác giả; không phải trung bình speedup từng bài. Protocol có marker `swift_author_code_v1`. Paper nêu tốc độ và wall-time speedup nhưng phần metric không ghi rõ phép tổng hợp tốc độ qua từng bài; bằng chứng cho công thức trên là code công khai. Không khẳng định mọi paper dùng cùng phép tổng hợp.

Chỉ số bổ sung được đặt tên riêng: `global_tokens_per_second = sum(n_i) / sum(t_i)`; `global_throughput_speedup_vs_*` là tỷ số các throughput tổng; `inference_time_speedup_vs_*` là tỷ số tổng thời gian. Chúng có trọng số/ý nghĩa khác chỉ số chính. Nếu lượng token khác nhau, tỷ số thời gian tổng cũng không đồng nhất với tỷ số throughput tổng.

Phiên bản runner cũ chưa có marker từng dùng throughput tổng làm cột chính. Cần tính lại từ JSONL và xuất file versioned bằng `scripts/summarize_results.py`; không ghép cột chính cũ/mới như cùng protocol. Công cụ này giữ nguyên dữ liệu và điểm pass@1 cũ, không chạy lại model/evaluator.

### 5.2. Timer và chỉ số giải thích

Timer mỗi request đồng bộ CUDA trước/sau forward, gồm prefill, drafting, verification, optimization và toàn bộ xử lý cache. Không gồm tải model, tải dataset, tokenization, decode thành text, ghi JSONL hay chạy HumanEval evaluator. Vì vậy, đây là thời gian inference của request, không phải thời gian end-to-end của toàn bộ job và cũng không phải decode-only đã loại prefill.

Mean accepted length tính bằng tổng token commit qua các vòng verification chia số vòng đó. Acceptance rate dùng `sum(accept_length - 1) / sum(draft_steps)` theo cách đếm của implementation, không lấy tổng node trên token tree làm mẫu số. Với baseline, accepted length là 1 và acceptance rate để trống trong CSV. Các prefix cuối bị cắt theo EOS/ngân sách được tính theo token thực sự commit; số node tree đã verify và số accepted draft trước cắt được lưu riêng để thấy công việc đã làm.

Timer optimization có host seconds và GPU-event milliseconds; không cộng hai số vì có thể chồng lấp. Hit rate tính trên số request lookup, gồm miss đầu khi cache rỗng. Các timer con không phải phân rã đầy đủ, không chồng lấp của toàn bộ latency; số đo quyết định vẫn là timer ngoài.

### 5.3. HumanEval pass@1

Chạy một completion greedy cho mỗi bài trong đủ 164 bài. Export `task_id` và completion không chứa prompt, giữ indentation; không sửa code theo kết quả test. Evaluator chính thức kiểm tra chức năng, timeout mặc định 3 giây/bài và 4 worker. Điểm trong cấu hình này là `số bài đạt / 164`; helper đối chiếu với boolean passed của kết quả từng bài và yêu cầu đầy đủ ID.

Cần phân biệt evaluator thất bại với model có pass@1 thấp. Mục tiêu là tạo được điểm hợp lệ, không bảo đảm điểm dương hoặc bằng paper. Pass@1 bằng baseline là bằng chứng thực nghiệm trên benchmark này; riêng nó không chứng minh bảo toàn phân phối hoặc xác nhận mọi token giống nhau. Token mismatch được báo cáo độc lập, cần phân tích nguyên nhân; không chặn evaluator để vẫn thấy ảnh hưởng đến chất lượng.

## 6. Bằng chứng cần có trước khi viết contribution

Câu hỏi chính: **truy hồi layer set có cải thiện thời gian sau toàn bộ overhead, và trong điều kiện nào?** Lợi ích có thể đến từ giảm search hoặc cấu hình draft phù hợp hơn; chi phí có thể đến từ embedding, lookup và cấu hình truy hồi kém. Không dùng hit rate cao để suy ra speedup.

| Giả thuyết cần kiểm chứng | Bằng chứng/đối chứng cần có |
|---|---|
| Lookup tiết kiệm chi phí thực | Speedup với cả AR và SWIFT; đo embedding/search/update và optimization, tính cả cache rỗng đầu stream |
| Cosine phản ánh khả năng chuyển giao cấu hình | Đối chiếu similarity với lợi ích latency/acceptance từng request; so với giữ incumbent và chọn entry ngẫu nhiên |
| Cache không chỉ nhớ một cấu hình của domain đồng nhất | Phân tích số layer set khác nhau, nguồn entry, vị trí trong stream và miss sau early-stop; sau đó mới khảo sát domain shift |
| Thay đổi giúp nhiều chế độ workload | Phân tích theo độ dài prompt/output và model; không coi so sánh khác họ kiến trúc là scaling thuần theo số tham số |
| Chất lượng được duy trì trong benchmark | Báo pass@1 cùng số bài đạt, token mismatch và chênh lệch với baseline; điều tra chênh lệch thay vì chỉ nhìn speedup |

Phải phân biệt sửa lỗi kỹ thuật (LRU, reset mask, state nhất quán), quyết định thiết kế phương pháp (reuse incumbent, threshold/capacity, miss/hit policy) và contribution khoa học đã có bằng chứng. Retrieval theo representation đã có trong KNN-SSD; không tuyên bố chỉ riêng cosine + cache là mới. Việc tối ưu cấu hình theo chi phí cũng cần đối chiếu KnapSpec. Novelty của nghiên cứu này chưa được xác nhận.

Các rủi ro diễn giải hiện tại: cấu hình tối ưu sớm có thể chi phối toàn stream; thứ tự bài ảnh hưởng cache và SWIFT; không warm-up khiến phần khởi đầu có trọng số trong kết quả; sửa eager backend/termination/instrumentation có thể thay baseline latency. Những hạn chế này phải được công bố, không dùng nhãn protocol metric để che khác biệt runtime.

## 7. Truy nguyên kết quả và trạng thái bằng chứng

Mỗi run dưới `_local/runs/` giữ original source ZIP, commit nguồn, bản thực thi, `SOURCE_CHANGES.patch`, driver/assets, run config, manifest, model revision nếu có và phiên bản môi trường. JSONL lưu ID, input/output token IDs, thời gian và cache/source metadata. `results.csv` là bảng tổng hợp; file pass1/results chứa functional correctness từng bài. Đây là dữ liệu kết quả và truy nguyên cần cho nghiên cứu; không phải file log văn bản.

Runner cho process con kế thừa stdout/stderr trực tiếp và đặt Python unbuffered. Người chạy thấy download, progress bar, tối ưu và traceback ngay trên terminal; không chuyển output sang `.log`, không dùng tee/pipe để thu rồi phát lại. Thay đổi đường xuất này không đổi công thức metric hoặc thuật toán; thông báo bên trong forward vẫn có chi phí I/O và không được giả định overhead bằng 0. Exit status 1 chỉ báo process thất bại; traceback con mới chỉ rõ nguyên nhân. Lần chạy H100 được cung cấp dừng ở tải config LLaMA-2 với HTTP 401/GatedRepoError: cần quyền model và đăng nhập Hugging Face, không phải bằng chứng decoding sai. Chưa có kết quả inference từ lần đó.

Đã kiểm tra logic tổng hợp metric bằng dữ liệu tổng hợp và đối chiếu hàm speed gốc; đã kiểm tra cú pháp và xuất source. Những kiểm tra đó không phải thực nghiệm model. **Chưa có kết quả H100, pass@1 hoặc speedup thực tế được cung cấp trong phiên làm việc này.** Khi có dữ liệu, chỉ cập nhật các kết luận có liên kết tới run/artifact và ghi rõ cấu hình, số lần lặp, độ biến thiên và các sai khác protocol.

Tài liệu tham chiếu: [paper SWIFT](https://proceedings.iclr.cc/paper_files/paper/2025/file/d74d002a9154b4cc433a234feb27c5f4-Paper-Conference.pdf), [speed.py tác giả](https://github.com/hemingkx/SWIFT/blob/main/evaluation_llama/speed.py), [HumanEval](https://github.com/openai/human-eval), [KNN-SSD](https://arxiv.org/abs/2505.16162), [KnapSpec](https://arxiv.org/abs/2602.20217). Hướng dẫn chạy và bố cục repo nằm trong `Readme.md`.


## GSM8K: tách phép đo tốc độ và chất lượng

Bổ sung `openai/gsm8k`, config `main`: lấy 1.000 câu test và 5 ví dụ train bằng data seed 2024. ID được gán theo vị trí split trước shuffle; manifest lưu few-shot IDs, prompt và fingerprint. Đáp án test không được đưa vào prompt; reference chỉ lưu trong JSONL để chấm sau inference. Baseline, SWIFT và AdaptiveCache dùng cùng input và thứ tự. GSM8K dùng target LLaMA-2 (`--llama-model`).

Hai profile độc lập: `paper64` (64 token, ngân sách sinh theo SWIFT) và `quality512` (512 token, kết quả chất lượng chính). Mặc định vẫn chạy CNN/DM và HumanEval; chọn `--tasks gsm8k` để chỉ chạy GSM8K, `--gsm8k-profiles` để chọn một hoặc cả hai profile. Mỗi phương pháp/profile tạo process mới, optimizer/cache khởi tạo lại; không mang cache từ 64 sang 512. Prompt/demonstrations giống nhau giữa hai profile; nếu vượt context thì báo lỗi, không cắt prompt GSM8K. Không thay đổi decoder/search/acceptance để thêm dataset.

Protocol `gsm8k_5shot_strict_hash_v1`: prompt yêu cầu kết thúc bằng `#### number`; ví dụ train giữ reasoning và đáp án gốc. Lấy số sau marker `####` đầu tiên trong completion trước `Question:` tiếp theo; chuẩn hóa dấu phẩy, dấu +/- và số thập phân bằng Decimal. Không lấy số cuối trong reasoning làm fallback, không sửa lỗi đáp án. Thiếu marker/đáp án hợp lệ tính sai, vẫn giữ trong mẫu số. Đây là strict-format exact-match accuracy, bao gồm khả năng tuân thủ định dạng. Prompt được công khai và băm hash, chưa xác nhận trùng prompt tác giả; không gọi đây là tái lập tuyệt đối paper.

CSV tách dataset key `gsm8k.paper64` và `gsm8k.quality512`, cùng speed protocol `swift_author_code_v1`. Báo cáo accuracy/delta, missing-answer rate, token-limit rate và số câu baseline đúng→sai / sai→đúng. JSON accuracy lưu từng prediction, reference và paired IDs. Chấm điểm nằm ngoài thời gian inference; không dùng model judge hoặc thực thi code để chấm toán. Offline resummary tính lại trực tiếp từ JSONL, không chạy GPU.

Token-limit rate đếm output dài đúng ngân sách, kể cả EOS ở token cuối; không đồng nghĩa mọi output đó đều bị cắt đáp án. Lưu termination reason để kiểm tra. Accuracy thấp ở 64 token có thể do thiếu ngân sách, không chỉ do decoder. 512 token cũng không bảo đảm giải xong: cần đọc cả missing-answer và token-limit rate trước khi kết luận. Accuracy tổng bằng nhau không chứng minh từng câu giống nhau hoặc bảo toàn phân phối. Kết quả GSM8K không tự chứng minh cache có lợi khi chuyển miền: mỗi dataset hiện là stream riêng.

Model ID trong JSONL nay lấy từ model path thay vì cố định nhãn 13B; thay đổi này sửa nhãn provenance, không thay trọng số được load. Script vẫn in trực tiếp terminal, không tạo runtime text log. Thêm GSM8K không thay đổi default CNN/DM + HumanEval và không ảnh hưởng job đã khởi chạy từ source snapshot cũ. Chưa chạy benchmark GSM8K trên H100; cần kết quả thật để kết luận về speedup hoặc accuracy.

Kiểm tra CPU bằng dữ liệu giả: chuẩn hóa số, đáp án thiếu, ID trùng, lựa chọn ví dụ chỉ từ train, hai ngân sách và nhãn model 7B; mô phỏng đủ sáu lượt chạy tuần tự, paired correctness và CSV offline/live thống nhất. Source preparation thành công cho cả hai nhánh, không inference hoặc GPU warm-up. Các kiểm tra này xác nhận plumbing/chấm điểm, không xác nhận chất lượng hay hiệu năng model.

## Diễn giải kết quả H100 7B/13B cho báo cáo

`BaoCao.tex` tổng hợp hai CSV do người thực nghiệm cung cấp: 13B `h100_20261019T183923223276Z`, 7B `h100_20261019T205721904342Z`, seed 2024. Bảng chính đặt M, alpha, tokens/s, speedup và pass@1 cạnh nhau theo SWIFT; bảng phân tích tách chi phí embedding và hiệu quả cache. AdaptiveCache so với SWIFT: CNN/DM-7B -11,24%, CNN/DM-13B -13,34%, HumanEval-7B -0,46%, HumanEval-13B +10,23% theo tokens/s chính. Một seed chưa cho phép kết luận ý nghĩa thống kê hoặc quy luật scaling; skip ratio 0,45 chưa được tối ưu riêng theo kích thước.

Motivation là chọn truy hồi theo lợi ích latency sau toàn bộ chi phí, không chỉ cosine/hit rate. Cần đối chứng retrieved layer set với incumbent trên cùng input, ablation chi phí embedding và đánh giá nhiều seed. Chưa kết luận cấu hình truy hồi là nguyên nhân duy nhất của thay đổi M hay speedup. HumanEval-7B có M/alpha tổng hợp bằng SWIFT không chứng minh từng output hay layer set giống nhau.

Pass@1 hiện tại 7B 4/164, 13B 2/164 ở cả ba phương pháp được đánh dấu chưa xác minh pipeline, không diễn giải là 7B có năng lực tốt hơn hoặc chứng minh losslessness. Lỗi decode/indentation là giả thuyết cần kiểm tra từ token IDs. Tổng token 13B đúng 164*512; tổng 7B khác giữa AR và speculative, không suy tỷ lệ chạm giới hạn từ tổng. Báo cáo giữ ranh giới giữa speedup ratio-of-means, global throughput và tỷ số tổng thời gian. Số liệu này chưa được tái lập độc lập tại máy phát triển; không sửa inference hoặc raw artifacts để viết báo cáo.

## GSM8K: kiểm chứng giá trị cache và ngân sách sinh

Bổ sung log run `h100_20261020T042351056502Z` vào `BaoCao.tex`: 1.000 câu, 64/512 token. Model ID chưa có trong log người dùng gửi, không suy từ tốc độ. Chỉ có M/alpha/chi phí chi tiết cho AdaptiveCache-512; các ô khác để chưa báo cáo, cần CSV hoàn chỉnh. Bảng chính gộp 18 dòng, hai bảng phụ tách chi phí cache và chẩn đoán chất lượng. Accuracy AR 4,8%/26,3%, SWIFT và AdaptiveCache 4,7%/26,2%; chênh tổng đúng một câu không đồng nghĩa chỉ một câu đổi đúng/sai. Cap 100% ở cả hai ngân sách không khẳng định mọi lời giải bị cắt hoặc chưa có đáp án.

AdaptiveCache chậm hơn SWIFT khoảng 6,16%/6,62% (tính từ tốc độ làm tròn), dù quality512 hit 999/1.000 và embedding chiếm 1,05% thời gian. Với cơ chế cập nhật cache trên miss và tắt tối ưu trên hit, có giả thuyết khóa sớm cấu hình đầu stream; cần mask hash/provenance và đối chứng fixed-first-mask để xác nhận. Phần few-shot chung có thể chi phối prompt embedding, chưa có đo để kết luận. Chưa có M/alpha SWIFT trong log này nên không suy giảm drafting từ tỷ lệ tốc độ. Optimization host/GPU có thể chồng lấp, không cộng hai số đo thành overhead.

Đọc PDF trong `_local/papers`, dùng SWIFT, KNN-SSD, CLaSp, KnapSpec, BANDITSPEC và When Drafts Evolve làm trục đối chiếu. KNN-SSD đã truy hồi layer set theo hidden states/cosine; KnapSpec đã tối ưu theo chi phí và các công trình online đã dùng feedback. Motivation vì vậy tập trung vào quyết định reuse/revalidate/relearn trong cold-start stream và giá trị ròng sau mọi chi phí, chưa tuyên bố novelty. Đối chứng cần tách chi phí embedding, chất lượng mask và khả năng thích nghi; đo thời gian đến đáp án theo fixed stop protocol để tránh coi sinh dư là hiệu quả giải toán. Chỉ cập nhật tài liệu, không sửa code hoặc kết quả đã đo.

### Hoàn thiện số liệu từ bốn CSV

Người thực nghiệm đã cung cấp CSV đầy đủ: GSM8K-7B `h100_20261019T224233022537Z`, GSM8K-13B `h100_20261020T042351056502Z`, cùng hai run CNN/DM và HumanEval trước đó. Thông tin thiếu ở mục trên đã được giải quyết; `BaoCao.tex` có bảng chính 24 dòng, bảng chi phí 8 dòng và bảng chẩn đoán GSM8K 12 dòng, không còn ô chưa báo cáo. Không diễn giải nguyên nhân pass@1 trong bản báo cáo theo yêu cầu người dùng; chỉ giữ kết quả chấm trong bảng.

GSM8K có hit 99,9% ở cả hai kích thước và hai ngân sách. Ở 7B, M/alpha bằng SWIFT trong từng ngân sách, tỷ trọng embedding giảm từ 5,43% xuống 0,77% khi tăng 64 lên 512 token và chênh tốc độ macro thu hẹp từ -7,63% xuống -0,54%. Tuy nhiên, tổng thời gian 7B-512 giảm khoảng 0,085%: đây là gần hòa, không chọn metric phụ để tuyên bố thắng. Ở 13B, M giảm 6,86%/18,54%, tốc độ giảm 6,16%/6,62%, dù chi phí embedding được amortize. Cần phân biệt overhead với chất lượng cấu hình truy hồi, không suy nhân quả từ hit rate hoặc M tổng hợp.

Paired correctness GSM8K-7B không đổi so với AR (R/G=0/0). GSM8K-13B-64: SWIFT và AdaptiveCache đều 1/0; 13B-512: SWIFT 4/3, AdaptiveCache 2/1. Cùng chênh ròng một câu không có nghĩa cùng các bài đúng/sai; cần task IDs để so trực tiếp hai phương pháp. Motivation cập nhật là kiểm chứng chất lượng cấu hình tại thời điểm cache admission và cơ chế tái kiểm tra, cùng điều kiện amortization, thay vì chỉ tăng cache hit. Chưa có nhiều seed hoặc đối chứng fixed-mask để kết luận khóa sớm.

### Công cụ chấm lại HumanEval từ token đã lưu

Thêm `scripts/recheck_humaneval.py` chạy CPU/tokenizer và evaluator trên các run chỉ định. Policy `prompt_context_decode_v1` tái tạo continuation từ prompt+output token IDs, kiểm tra prompt và ranh giới trước khi chấm, không sửa code/indentation, không crop hoặc chọn đáp án theo kết quả test. Dùng revision và phiên bản Transformers được ghi nhận; mặc định chỉ đọc tokenizer cache. Công cụ lưu export, diagnostics, score và CSV mới dưới thư mục timestamp riêng, không thay raw JSONL, điểm cũ hoặc thời gian inference. `--diagnose-only` không thực thi tests. Future harness áp dụng cùng policy cho cả ba phương pháp HumanEval; thuật toán/token sinh không đổi. Đây là thay đổi representation/chấm điểm, không phải số đo inference mới; chưa có điểm chấm lại thực tế trên server để cập nhật báo cáo. Không bổ sung phần giải thích điểm pass@1 vào `BaoCao.tex`.


## 2026-10-27 — SWIFT tree-control: implementation của hai cơ chế mục tiêu

Nhánh research/swift-tree-control tạo từ main. Chỉ ba source entry mới: model/swift_tree_control.py, scripts/run_swift_tree.py và launcher .sh. Không nhập AdaptiveCache hay mask_search, không đổi layer-search native. prepare_single() xuất SWIFT/AR với harness chung mà không cần ref AdaptiveCache.

Cơ chế 1 dùng rank label của greedy target trong top-10 draft (và ngoài top-10), chỉ trên parent prefix hợp lệ. Dirichlet smoothing và backoff cùng mask; minimum 32 nhãn, không học hậu tố sau sai khác đầu. DP max-plus chọn sibling widths trong at-most node budget; công thức reward dựa trên survival main spine và rank coverage, không bảo đảm xấp xỉ đó đúng với mọi workload.

Cơ chế 2 là stopping/budget dùng finite lookahead (mặc định hai bước), transition bin từ draft quan sát và NNLS latency từ CUDA events của inference bình thường. Mô hình thiếu dữ liệu quay về native stopping; cập nhật model/timing nằm trong request wall time. Không gọi là confidence sequence hoặc Bellman tối ưu cho runtime thực. Cả hai contribution mới là mục tiêu nghiên cứu, chưa được xác nhận bằng benchmark.

Bảy đối chứng mặc định gồm AR, SWIFT native, native_control (cùng cây với instrumentation/buffer cache), raw_dp, rank_dp, stop_only và joint. Fixed sweep là tùy chọn. So native_control giúp tách gain kỹ thuật; log nhãn raw/learned paired cùng learned-label/fallback counts để phát hiện controller chưa thực sự hoạt động.

Phát hiện rủi ro padding: retrieve_indices=-1 lấy Cartesian dummy ID 0 trong code cũ; 0 là ID vocabulary hợp lệ nên có thể bị greedy chấp nhận. Snapshot mới dùng dummy -1 (không vào model) cho cả reference/controlled. Posterior/commit function không viết lại, source working SWIFT không sửa. Không coi metric protocol là tái lập runtime nguyên trạng paper.

Diagnostic/train và evaluate/test riêng; loại demonstrations, HumanEval evaluation-only. Không có paid counterfactual target probes tự động; không dùng oracle từ dữ liệu hậu nghiệm để điều khiển output. CSV có speed protocol gốc, M/alpha, mismatches, chất lượng, latency, rank scoring và controller phases; timer host/GPU không cộng chồng lấp.

Kiểm tra CPU với target giả đã qua: DP so exhaustive enumeration cả rank không concave; lookahead ưu tiên continue/stop khi đổi chi phí; estimator tách masks; sáu policy trên các hàm SWIFT snapshot thực; EOS, budget 1/2/64, root-only, optimization hook và KV commit. Pipeline CSV/GSM8K với bảy methods synthetic cũng qua. Prepare-only không tải model. Chưa chạy Llama thật/H100; không có speedup hoặc accuracy mới.

Lookahead tái sử dụng frontier DP khi nối bước giả định, không dựng lại mọi đường ở mỗi lá. Telemetry tách GPU draft/KV copy/candidate/verify/posterior/commit; phase không instrument để trống, không giả thành 0. Có diagnostic-only --layer-search fixed để cô lập thay đổi lịch native optimizer khi số vòng verification khác nhau; evaluate vẫn bắt buộc native search. Các hàm posterior/commit/optimization và model/KV source đã đối chiếu AST/bytes với gốc; chỉ guard Cartesian padding và draft hooks được áp dụng trên snapshot.
