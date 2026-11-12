# SWIFT / AdaptiveCache — Experimental Results

## 1. Phạm vi và thiết lập

Kết quả sơ bộ từ bốn lượt chạy, 24 dòng kết quả; mỗi cấu hình có một seed. Nguồn là CSV/log được cung cấp trong báo cáo, không phải lượt chạy mới hoặc kiểm chứng độc lập.

- Hardware: 1 × NVIDIA H100 80GB; FP16; greedy decoding; batch size 1.
- Seed = data seed = 2024; không warm-up. Mỗi cấu hình dùng cùng dữ liệu, prompt và thứ tự giữa ba phương pháp.
- AR: autoregressive target baseline. SWIFT: baseline trực tiếp. AdaptiveCache: SWIFT kết hợp tái sử dụng cấu hình layer.
- SWIFT: skip ratio 0.45; context window 32; max optimization iterations 1000; Bayesian interval 25; tolerance 300; max score 0.95.
- AdaptiveCache: cosine threshold 0.85; capacity 100; cache rỗng đầu mỗi process. Các stream và ngân sách chạy độc lập.

| Dataset | Model | N | Prompt | Max new tokens |
|---|---|---:|---|---:|
| CNN/DailyMail | meta-llama/Llama-2-7b-hf / Llama-2-13b-hf | 1000 | 1-shot | 64 |
| HumanEval | codellama/CodeLlama-7b-hf / CodeLlama-13b-hf | 164 | Function prompt gốc | 512 |
| GSM8K | meta-llama/Llama-2-7b-hf / Llama-2-13b-hf | 1000 | 5-shot từ train | 64 / 512 |

## 2. Định nghĩa metric

- `M`: token commit trung bình mỗi vòng verification, gồm token bổ sung của target; không đếm forward phụ cho embedding/tối ưu.
- `alpha`: draft token commit / draft steps ghi nhận. EOS và clipping ngân sách có thể ảnh hưởng vòng cuối; đây là định nghĩa instrumentation của lượt chạy.
- `v = mean(n_i / t_i)`: trung bình Token/s từng mẫu; `S_AR = v_method / v_AR`.
- `t_i`: synchronized forward wall time, gồm prefill, decoding, embedding/cache và tối ưu; không gồm tải model/dataset, tokenization, output I/O hoặc chấm điểm.
- Global throughput `sum(n_i) / sum(t_i)` khác `v`. Không đổi macro speedup thành phần trăm giảm tổng thời gian.
- `GSM8K Acc.`: exact match đáp án số sau chuẩn hóa; parser yêu cầu `#### number`, lấy đáp án đầu trước `\nQuestion:` tiếp theo. Không trích được đáp án tính sai.
- `ΔM (%) = 100 × (M_AdaptiveCache / M_SWIFT − 1)`; `Δv (%) = 100 × (v_AdaptiveCache / v_SWIFT − 1)`.
- `Embedding share (%) = 100 × T_embedding / T_inference_AdaptiveCache`. Embedding đã nằm trong inference time, không cộng lại.

## 3. Kết quả chính

Chỉ giữ chất lượng GSM8K. HumanEval chỉ báo cáo hiệu năng sinh token do pipeline completion đang được kiểm tra; CNN/DM chưa chấm chất lượng tóm tắt. In đậm tốc độ cao nhất trong mỗi nhóm.

| Dataset / budget | Size | Method | M ↑ | alpha ↑ | Token/s ↑ | S_AR ↑ | GSM8K Acc. (%) ↑ |
|---|---|---|---:|---:|---:|---:|---:|
|  | 7B | AR | 1.000 | — | 71.055 | 1.000× | — |
|  | 7B | SWIFT | 2.693 | 0.9398 | **83.839** | **1.180×** | — |
|  | 7B | AdaptiveCache | 2.525 | 0.9323 | 74.418 | 1.047× | — |
|  | 13B | AR | 1.000 | — | 50.203 | 1.000× | — |
|  | 13B | SWIFT | 3.065 | 0.9432 | **61.259** | **1.220×** | — |
|  | 13B | AdaptiveCache | 2.486 | 0.9443 | 53.086 | 1.057× | — |
|  | 7B | AR | 1.000 | — | 73.185 | 1.000× | — |
|  | 7B | SWIFT | 2.457 | 0.9677 | **83.125** | **1.136×** | — |
|  | 7B | AdaptiveCache | 2.457 | 0.9677 | 82.743 | 1.131× | — |
|  | 13B | AR | 1.000 | — | 57.976 | 1.000× | — |
|  | 13B | SWIFT | 2.942 | 0.9757 | 62.913 | 1.085× | — |
|  | 13B | AdaptiveCache | 4.499 | 0.9723 | **69.347** | **1.196×** | — |
|  | 7B | AR | 1.000 | — | 72.784 | 1.000× | — |
|  | 7B | SWIFT | 2.273 | 0.9166 | **86.251** | **1.185×** | — |
|  | 7B | AdaptiveCache | 2.273 | 0.9166 | 79.673 | 1.095× | — |
|  | 13B | AR | 1.000 | — | 55.237 | 1.000× | — |
|  | 13B | SWIFT | 2.430 | 0.9411 | **61.742** | **1.118×** | — |
|  | 13B | AdaptiveCache | 2.263 | 0.9355 | 57.937 | 1.049× | — |
|  | 7B | AR | 1.000 | — | 77.841 | 1.000× | — |
|  | 7B | SWIFT | 2.482 | 0.9236 | **91.257** | **1.172×** | — |
|  | 7B | AdaptiveCache | 2.482 | 0.9236 | 90.764 | 1.166× | — |
|  | 13B | AR | 1.000 | — | 56.951 | 1.000× | — |
|  | 13B | SWIFT | 3.756 | 0.9816 | **76.590** | **1.345×** | — |
|  | 13B | AdaptiveCache | 3.059 | 0.9796 | 71.523 | 1.256× | — |

## 4. Chi phí và hiệu quả tái sử dụng

Các thay đổi so với SWIFT được tính trước khi làm tròn. Cache hit rate đo việc tìm được entry đạt ngưỡng, không đo lợi ích của entry.

| Dataset / budget | Size | Hit (%) | Embedding (s) | Embedding share (%) | ΔM (%) | Δv (%) |
|---|---|---:|---:|---:|---:|---:|
| CNN/DM–64 | 7B | 98.10 | 82.607 | 9.29 | -6.26 | -11.24 |
| CNN/DM–64 | 13B | 98.30 | 137.355 | 11.11 | -18.89 | -13.34 |
| HumanEval–512 | 7B | 71.34 | 2.984 | 0.30 | 0.00 | -0.46 |
| HumanEval–512 | 13B | 97.56 | 4.305 | 0.34 | +52.96 | +10.23 |
| GSM8K–64 | 7B | 99.90 | 43.760 | 5.43 | 0.00 | -7.63 |
| GSM8K–64 | 13B | 99.90 | 74.899 | 6.77 | -6.86 | -6.16 |
| GSM8K–512 | 7B | 99.90 | 43.701 | 0.77 | 0.00 | -0.54 |
| GSM8K–512 | 13B | 99.90 | 75.349 | 1.05 | -18.54 | -6.62 |

## 5. GSM8K: chẩn đoán chất lượng

Mỗi dòng có 1000 câu. `Missing` = parser không trích được đáp án theo format, không đồng nghĩa model không đưa ra đáp án. `Cap` = output dài đúng ngân sách, không chứng minh lời giải bị cắt trước đáp án. `Mismatch` = số câu khác toàn bộ chuỗi output token so với AR, không phải số token khác hoặc số câu toán sai. `R/G` = AR đúng → sai / AR sai → đúng; kiểm tra `correct_method = correct_AR − R + G`.

| Budget | Size | Method | Correct | Missing (%) | Cap (%) | Mismatch vs AR | R/G |
|---:|---|---|---:|---:|---:|---:|---|
| 64 | 7B | AR | 41 | 71.30 | 100 | 0 | 0/0 |
| 64 | 7B | SWIFT | 41 | 71.80 | 100 | 26 | 0/0 |
| 64 | 7B | AdaptiveCache | 41 | 71.80 | 100 | 26 | 0/0 |
| 64 | 13B | AR | 48 | 82.00 | 100 | 0 | 0/0 |
| 64 | 13B | SWIFT | 47 | 82.00 | 100 | 24 | 1/0 |
| 64 | 13B | AdaptiveCache | 47 | 82.10 | 100 | 19 | 1/0 |
| 512 | 7B | AR | 140 | 5.80 | 100 | 0 | 0/0 |
| 512 | 7B | SWIFT | 140 | 5.70 | 100 | 126 | 0/0 |
| 512 | 7B | AdaptiveCache | 140 | 5.70 | 100 | 126 | 0/0 |
| 512 | 13B | AR | 263 | 10.30 | 100 | 0 | 0/0 |
| 512 | 13B | SWIFT | 262 | 10.30 | 100 | 99 | 4/3 |
| 512 | 13B | AdaptiveCache | 262 | 10.20 | 100 | 96 | 2/1 |

## 6. Insight có bằng chứng

1. **Tái sử dụng chưa tạo lợi thế ổn định.** AdaptiveCache tăng macro Token/s so với AR ở 8/8 cấu hình, nhưng chỉ vượt SWIFT ở HumanEval–13B (+10.23%). Đây là tốc độ sinh token, chưa là tăng tốc tạo lời giải đúng.
2. **Chi phí và hiệu quả cấu hình là hai yếu tố riêng.** CNN/DM có embedding share 9.29–11.11% đồng thời M giảm. GSM8K–13B–512 chỉ có embedding share 1.05% nhưng M giảm 18.54%, tốc độ giảm 6.62%; không thể quy toàn bộ giảm tốc cho embedding. Chưa xác lập quan hệ nhân quả của mask từ các thống kê tổng hợp.
3. **Output dài giúp amortize chi phí.** GSM8K–7B: embedding gần cố định (~43.7 s), share giảm 5.43% → 0.77%, bất lợi tốc độ thu hẹp -7.63% → -0.54%; M và alpha trùng SWIFT trong từng budget. Đây là quan sát phù hợp giả thuyết amortization, chưa là ablation nhân quả.
4. **Gần hòa phụ thuộc cách tổng hợp.** GSM8K–7B–512: AdaptiveCache 5659.145 s, SWIFT 5663.950 s (tổng thời gian giảm ~0.085%), nhưng macro Token/s giảm 0.54%. Không chọn metric thuận lợi để tuyên bố thắng.
5. **Acceptance rate cao chưa đủ.** GSM8K–13B–512: alpha 0.9816 → 0.9796 nhưng M 3.756 → 3.059. Cần đọc alpha cùng M, draft length, số verification steps và latency.
6. **Ngân sách thay đổi cả chất lượng lẫn workload.** GSM8K AR Accuracy tăng 4.10% → 14.00% (7B), 4.80% → 26.30% (13B) khi tăng 64 → 512. Cap 100% nên tốc độ có thể tính cả token sau đáp án; chưa đồng nhất throughput với hiệu suất giải toán.
7. **Accuracy bằng nhau không chứng minh đầu ra giống nhau.** GSM8K–7B có R=G=0 nhưng vẫn có token mismatch. GSM8K–13B–512: SWIFT 4/3, AdaptiveCache 2/1; cùng mất ròng một câu nhưng hành vi từng câu khác nhau.

## 7. Giả thuyết chưa được xác nhận và đối chứng ưu tiên

- **Khóa cấu hình sớm:** 999/1000 hit ở GSM8K phù hợp khả năng tái sử dụng entry đầu phần lớn stream, nhưng chưa chứng minh mask không đổi. Kiểm tra mask hash, nguồn entry, tuổi entry, số lần refresh và tối ưu tích lũy.
- **Embedding bị phần prompt chung chi phối:** năm ví dụ cố định có thể làm full-prompt keys gần nhau. So full-prompt với question-only key; chưa gọi đây là cache collapse nếu chưa đo.
- **Đối chứng quan trọng:** SWIFT; fixed-mask sau request đầu không embedding; AdaptiveCache hiện tại; AdaptiveCache có refresh bằng feedback. Giữ cùng stream, thử nhiều thứ tự và chạy lặp.
- **Tách chi phí:** so key hiện tại với key tái sử dụng từ prefill/verification nếu khả thi; tính đủ chi phí lấy feature. Đo paired latency của retrieved mask và incumbent trên cùng prefix.
- **Đánh giá tác vụ:** dùng fixed stop protocol sau đáp án hợp lệ, giống nhau cho các phương pháp, không dựa trên đúng/sai; đo thời gian đến đáp án. Đổi stop khi sinh phải đo lại tốc độ; chấm lại output không tạo lượt đo inference mới.

## 8. Giới hạn diễn giải

- Một seed, một GPU; chưa có khoảng tin cậy, kiểm chứng độc lập hoặc kết luận scaling law.
- Không khẳng định lossless/distribution preservation: có token mismatch với AR. M/alpha bằng nhau không chứng minh từng output giống nhau.
- Không gộp speedup giữa budget 64 và 512; không trộn model LLaMA-2 với CodeLLaMA khi so chất lượng hoặc quy nguyên nhân theo kích thước.
- Không kết luận thích ứng chuyển miền từ các stream riêng; cần mixed-domain và miền quay lại.
- Các timer host/GPU có thể chồng lấp; không cộng các pha nếu chưa xác nhận chúng rời nhau. Lookup và embedding là hai chi phí khác nhau.
- Các insight hiện tại định hướng thực nghiệm tiếp theo; chưa chứng minh cơ chế mới hoặc novelty so với toàn bộ prior work.

## 9. Provenance và cách dùng trong repo

Protocol: `swift_author_code_v1`. Run root gốc trên server: `/data/SWIFTEXPER/_local/runs/`.

| Run ID | Nội dung |
|---|---|
| h100_20261019T183923223276Z | CNN/DM và HumanEval, 13B |
| h100_20261019T205721904342Z | CNN/DM và HumanEval, 7B |
| h100_20261019T224233022537Z | GSM8K, 7B, 64/512 token |
| h100_20261020T042351056502Z | GSM8K, 13B, 64/512 token |

Các run ID trên là nguồn của số liệu trong tài liệu, không phải kết quả mới nhất mặc định của repo. Đối chiếu `results.csv`, `run_config.json`, metadata và per-task JSONL trước khi sửa code hoặc viết claim. Không thay thế kết quả gốc bằng kết quả chấm lại; lưu riêng policy và provenance. Tài liệu chủ động không giữ pass@1 HumanEval theo yêu cầu phạm vi báo cáo.


## 2026-11-02 ? Ph?n v?ng t? output t?ng request (k?t qu? m?i)

Ngu?n: 5 run trong `visualization_selected_outputs_v1`, 32 JSONL / 20.116 request;
?? ki?m tra pairing b?ng task ID v? input token hash, t?i t?nh N/token/time/TPS/M
v? ??i chi?u CSV. C?c ph??ng ph?p ngo?i Fixed-first, AdaptiveCache, tree-control
v? AR/SWIFT ??i ch?ng b? lo?i kh?i ph?n t?ch.

Xem `_local/analysis/index.html` ho?c `_local/analysis/mechanism_report.pdf`.
H?nh motivation kh?ng v? accuracy/pass@1. B?ng `region_characteristics.csv`
cung c?p median/IQR ??c ?i?m trong m?i v?ng; kh?ng coi quan h? l? nguy?n nh?n.

| C?u h?nh AdaptiveCache | Hit | Hit nhanh h?n SWIFT | S? mask d?ng cu?i request kh?c nhau |
|---|---:|---:|---:|
| CNN/DM-64, 7B | 981 | 25 | 1 |
| HumanEval-512, 7B | 117 | 1 | 1 |
| GSM8K-512, 7B | 999 | 85 | 1 |
| HumanEval-512, 13B | 160 | 153 | 1 |

Quan s?t: HumanEval-13B c? 151/164 request v?a t?ng M v?a gi?m latency; CNN-64
AdaptiveCache c? 693/1000 request v?a gi?m M v?a t?ng latency. Fixed CNN-512
c? 533/1000 request M gi?m nh?ng latency gi?m. ??y l? c?c nh?m c?n ??i chi?u
??c ?i?m, ch?a ch?ng minh m?t router/c? ch? m?i s? th?ng.

GSM8K-512 v? HumanEval-7B c?a AdaptiveCache c? M/output gi?ng SWIFT tr?n to?n b?
request. Fixed HumanEval v? hai budget GSM8K c?ng c? t?nh ch?t n?y; latency v?n
kh?c. SWIFT th??ng ng?ng optimization r?t s?m: Fixed GSM512 t?i request 3/1000,
CNN512 t?i 4/1000, HumanEval t?i 2/164. Do ?? kh?ng th? quy to?n b? ch?nh l?ch
cu?i stream cho optimization c?n ch?y. C?n interleave/??o th? t? ph??ng ph?p,
l?p nhi?u seed v? ki?m so?t c?ng runtime tr??c khi k?t lu?n nh?n qu?.

B?n AdaptiveCache run ghi nh?n m?t mask d?ng cu?i request m?i c?u h?nh; nhi?u
cache entry c? th? ch? tr? c?ng mask. Ch?a c? b?ng ch?ng ??nh tuy?n ?a d?ng mask.
??i ch?ng ?u ti?n ti?p theo: c?ng mask v? runtime, b?t/t?t ri?ng embedding v?
lookup; ki?m tra ph? thu?c th? t? request/mask ??u; ch? sau ?? thi?t k? ?i?u ki?n
reuse. Kh?ng d?ng nh?n ground truth ?? ?i?u khi?n inference.

CNN64 v? CNN512 c? c?ng 1000 task ID nh?ng ch? 999 input hash gi?ng nhau; c?n
lo?i ri?ng/ghi r? tr??ng h?p clipping khi ph?n t?ch t?c ??ng budget.
Tree-control l? train/128, kh?ng gh?p hi?u n?ng v?i test. Brier tr?n c?ng 43.380
nh?n ?? support trong Rank DP: learned th?p h?n raw; c?y/M g?n nh? kh?ng ??i,
ch?a t?o l?i ?ch t?c ??. C?n ki?m tra ?? nh?y quy?t ??nh thay v? ch? c?i thi?n d? ?o?n.

C?c m?c k?t qu? c? ph?a tr?n l? l?ch s? ? nh?ng run kh?c; kh?ng g?p TPS gi?a c?c
run ?? t?nh speedup m?i. Model revision thi?u pin v?n l? gi?i h?n provenance.
