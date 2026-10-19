# SWIFT và AdaptiveCache — thiết kế, logic sửa code và diễn giải thực nghiệm

Tài liệu này giải thích implementation đang được đo, lý do thay đổi và những kết luận dữ liệu có thể hỗ trợ. Đây là ngữ cảnh để tiếp tục phát triển code và viết paper; không phải nhật ký thao tác Git hoặc output terminal. Chỉ cập nhật khi có quyết định thiết kế, đổi protocol hay kết quả thực nghiệm có ý nghĩa. Tiến trình từng lần chạy nằm trong `execution.log` của run, không tự nối vào tài liệu này.

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

Mỗi run dưới `_local/runs/` giữ original source ZIP, commit nguồn, bản thực thi, `SOURCE_CHANGES.patch`, driver/assets, run config, manifest, model revision nếu có và phiên bản môi trường. JSONL lưu ID, input/output token IDs, thời gian và cache/source metadata. `results.csv` là bảng tổng hợp; file pass1/results chứa functional correctness từng bài. `execution.log` và log từng lệnh dùng điều tra lỗi, không phải nội dung tự động thêm vào tài liệu này.

Đã kiểm tra logic tổng hợp metric bằng dữ liệu tổng hợp và đối chiếu hàm speed gốc; đã kiểm tra cú pháp và xuất source. Đã xác minh tiến trình của cả run chuẩn bị thành công/thất bại và lệnh tính lại metric được ghi vào execution.log, không sửa tài liệu này. Những kiểm tra đó không phải thực nghiệm model. **Chưa có kết quả H100, pass@1 hoặc speedup thực tế được cung cấp trong phiên làm việc này.** Khi có dữ liệu, chỉ cập nhật các kết luận có liên kết tới run/artifact và ghi rõ cấu hình, số lần lặp, độ biến thiên và các sai khác protocol.

Tài liệu tham chiếu: [paper SWIFT](https://proceedings.iclr.cc/paper_files/paper/2025/file/d74d002a9154b4cc433a234feb27c5f4-Paper-Conference.pdf), [speed.py tác giả](https://github.com/hemingkx/SWIFT/blob/main/evaluation_llama/speed.py), [HumanEval](https://github.com/openai/human-eval), [KNN-SSD](https://arxiv.org/abs/2505.16162), [KnapSpec](https://arxiv.org/abs/2602.20217). Hướng dẫn chạy và bố cục repo nằm trong `Readme.md`.
