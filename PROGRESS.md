# Tiến độ cải thiện train baseline

**Cập nhật:** 2026-11-01
**Mốc so sánh:** run `baseline`, `seed=42`, `SAMPLE_N=200`
**Trạng thái:** đã sửa code và kiểm tra local; lần inference Kaggle mới nhất dùng các module không đồng bộ nên các câu được log trong lần chạy đó đều abstain do lỗi tương thích, chưa phải kết quả benchmark.

## Baseline

- PA mẫu: **-0.175**; PA ước lượng có trọng số trên train: **-0.1771**.
- Đúng / sai / abstain: **1 / 36 / 163**.
- Có **72 câu** bị context overflow, **52 câu** gặp `NameError` với `orders_df`, và **19 câu** gặp `KeyError` với `order_date` (các nhóm có thể giao nhau).
- `expected_type` đúng **97/195 (49.74%)** khi loại câu `unanswerable`.

## Đã triển khai

- Giới hạn prompt và retry theo tokenizer/context budget; khi compact prompt sẽ ưu tiên giữ schema và câu hỏi, đồng thời dừng retry lặp lại prompt khi gặp context overflow.
- Làm tên DataFrame được tạo ổn định, hợp lệ và không ghi đè biến nội bộ; bổ sung thông tin biến/cột hiện có khi retry sau `NameError`/`KeyError`.
- Chuẩn hóa thêm cột ngày, gồm Excel serial và một số định dạng ngày tiếng Việt.
- Cải thiện bộ phân loại loại câu hỏi và thêm regression tests dựa trên lỗi/nhãn trong mẫu.
- Cập nhật notebook để truyền tokenizer và ngân sách token; đặt run tag mới `fix_context_v1` để không dùng lẫn kết quả baseline.
- Cập nhật README hướng dẫn cấu hình liên quan.

### File code và kiểm thử liên quan

- `src/modules/prompt_builder.py`, `src/predict.py`: giới hạn token và truyền cấu hình budget.
- `src/modules/retry_engine.py`: xử lý retry/context overflow và thông tin biến/cột.
- `src/modules/code_executor.py`: mapping biến DataFrame.
- `src/modules/data_loader.py`: chuyển đổi dữ liệu ngày.
- `src/modules/question_understanding.py`: phân loại `expected_type`.
- `tests/`: thêm và cập nhật kiểm thử cho prompt, retry, executor, loader và classifier.
- `kaggle_train_baseline.ipynb`, `README.md`: cấu hình chạy và hướng dẫn.

## Kiểm tra local

- `pytest tests/`: **85 passed**.
- Notebook JSON hợp lệ; đã parse **7 code cells** sau khi bỏ các dòng IPython magic/shell.
- Trên cùng CSV dùng để phát triển classifier, kết quả tăng từ **97/195** lên **195/195** câu không phải `unanswerable`. Vì đây là mẫu đã dùng để chẩn đoán/tinh chỉnh, con số này **không phải kết quả holdout** và không chứng minh PA inference đã tăng.

## Chưa xác nhận / việc tiếp theo

1. Cập nhật Kaggle Input bằng notebook và source mới; khởi động lại kernel/session để tránh import module cũ.
2. `datamate_code.zip` trong workspace hiện đã khớp với source `src/` và `prompts/` (đã đối chiếu nội dung 18 file). Tuy nhiên, notebook không tự giải nén archive; cần đưa source lên Kaggle Model hoặc Dataset dưới dạng thư mục để notebook tìm được.
3. Giữ nguyên model, seed, cỡ mẫu và cấu hình baseline còn lại; sau khi đồng bộ source, chạy 200 câu với `RUN_TAG="fix_context_v3"`.
4. Tải `eval_summary.json` và `error_analysis.csv`, rồi đối chiếu PA, số lỗi context/biến/cột/ngày, abstention và thời gian với baseline.
5. Chỉ sau khi kết quả được xác nhận mới cân nhắc sample/seed validation khác hoặc chạy đủ 3.000 câu.

Batch processing, timeout thực sự cho code sinh, resume/chia batch nâng cao và so sánh data-parallel với TP=2 chưa được triển khai trong vòng này.

### Kaggle module mismatch during inference — 2026-11-01

Log mới nhất cho biết Kaggle đã import:

```text
/kaggle/input/models/jcoolvn/datamate-code/pytorch/default/1/src/predict.py
```

Pipeline ở `predict.py` gọi `PromptBuilder.build(..., token_counter=..., max_tokens=...)`, nhưng `PromptBuilder` được import từ một phiên bản cũ không chấp nhận `token_counter`. Lỗi lặp lại cho từng câu và được pipeline ghi thành `abstain`; các kết quả đó không phải dự đoán của model và không được dùng để đánh giá PA. Do notebook có thể ghi abstain rồi tiếp tục, không resume trong thư mục run bị lỗi.

Notebook được cập nhật preflight để kiểm tra các tham số bắt buộc của cả `DataMatePipeline.__init__`, `PromptBuilder.build` và `RetryEngine.run`, đồng thời in đường dẫn `predict.py` và `prompt_builder.py`. Kiểm tra diễn ra trước khi khởi tạo vLLM. Run tag mặc định đổi thành `fix_context_v3` để không đọc nhầm submission/manifest từ các lần chạy trước.

**Việc cần làm:** cập nhật đồng bộ toàn bộ thư mục `src/` và `prompts/` trong Kaggle Input đang mount dưới tên `datamate-code` (không chỉ thay `predict.py`), restart session/kernel, upload notebook mới, xác nhận cả hai đường dẫn import trỏ vào cùng source version và preflight qua trước khi bắt đầu inference.

## Kaggle run update — 2026-11-01

Lần chạy gần nhất dừng với:

```text
TypeError: DataMatePipeline.__init__() got an unexpected keyword argument 'prompt_token_counter'
```

Trong source hiện tại, `DataMatePipeline.__init__` có cả `prompt_token_counter` và `max_prompt_tokens`. Do đó notebook đã nạp một `src.predict` cũ/khác với source vừa cập nhật (hoặc module cũ còn trong kernel); traceback chưa cho biết chính xác Kaggle đã chọn thư mục source nào. Papermill lỗi ở cell inference, sau khoảng 424 giây, trước khi xử lý câu hỏi.

Notebook đã được cập nhật để xóa các module `src.*` đã cache, ưu tiên `CODE_DIR` khi import và in đường dẫn `predict.py` thực tế. Nó cũng kiểm tra chữ ký constructor **trước khi khởi tạo vLLM** và dừng với thông báo hướng dẫn nếu source thiếu tham số cần thiết.

Khi chạy lại:

1. Upload/cập nhật Kaggle Input chứa source mới nhất (ít nhất `src/` và `prompts/`); chỉ cập nhật notebook là chưa đủ.
2. Restart session/kernel.
3. Xem dòng `Imported predict.py:` để xác nhận đường dẫn; nếu báo thiếu tham số, sửa `CODE_DIR` hoặc cập nhật đúng Kaggle Dataset.
4. Chạy lại với run tag mới nếu lần chạy đã tạo `run_config.json` trong output hiện tại; không xóa output cũ một cách mù quáng.

Các cảnh báo `resource_tracker` về shared memory và `SyntaxWarning` của `mistune`/`nbconvert` không phải exception làm Papermill dừng; lỗi dừng chính là `TypeError` ở trên. Hiện chưa có kết quả inference mới để so PA.

### Xác nhận source Kaggle — 2026-11-01

Traceback mới cho biết chính xác module được import:

```text
/kaggle/input/models/jcoolvn/datamate-model/pytorch/default/1/src/predict.py
```

Chữ ký tại Kaggle chỉ có `data_dir`, `output_dir`, `model_dir`, `max_retries`, `confidence_threshold` và `llm_generate_fn`; thiếu `prompt_token_counter` cùng `max_prompt_tokens`. Đây là phiên bản Kaggle Model đang được gắn vào notebook, không phải lỗi chữ ký trong source ở workspace. Đã kiểm tra `datamate_code.zip`: 18 file trong `src/` và `prompts/` khớp byte với source hiện tại, và `src/predict.py` trong archive có hai tham số cần thiết.

**Việc cần làm:** tạo/publish phiên bản mới của Kaggle Model `jcoolvn/datamate-model` từ nội dung `datamate_code.zip` mới nhất (giữ `src/` và `prompts/` ở thư mục gốc), rồi gắn phiên bản mới vào notebook. Chỉ tải notebook mới lên nhưng vẫn giữ Model version cũ sẽ tiếp tục gặp lỗi này. Sau đó restart session/kernel và chạy cell preflight; chỉ chạy tiếp khi `Imported predict.py:` trỏ tới phiên bản mới và constructor có hai tham số trên. Lần chạy vừa rồi đã dừng đúng ở preflight trước khi nạp vLLM/inference.

### Follow-up sau context-v4 — 2026-11-01

Phân tích `context-v4` cho thấy các lỗi thực thi lặp lại và một số đáp án sai giá trị. Kiểm tra prompt phát hiện few-shot examples dùng `.dt.date` và so với `date(...)`, mâu thuẫn với chỉ dẫn system prompt yêu cầu giữ pandas timestamps. Một trace cũng cho thấy model sao chép mẫu này, đổi sang cột ngày thực tế, rồi so giá trị date với `pd.Timestamp`; code chạy nhưng bộ lọc có thể trả rỗng.

Đã cập nhật local:

- Few-shot date examples giữ Timestamp và so với `pd.Timestamp(...)`.
- Prompt nhấn mạnh schema hiện tại là nguồn chuẩn, không đoán tên biến/cột, không đọc lại file; cảnh báo schema trong ví dụ chỉ mang tính minh họa.
- Retry feedback cho `FileNotFoundError` kèm danh sách bảng/cột đã nạp; feedback lỗi so sánh ngày kèm dtype và hướng dẫn dùng `pd.to_datetime`/`pd.Timestamp`.
- Thêm regression tests cho các hướng dẫn và retry context này.

Kiểm tra: `python -m pytest tests/ -q` — **89 passed**. Các thay đổi này mới được kiểm tra local; cần đồng bộ `src/` và `prompts/` lên Kaggle rồi chạy lại cùng sample 200 câu với run tag mới để đo PA, abstention và thời gian.

### Sửa pipeline và đối chiếu sai giá trị — 2026-11-02

Đã điều tra các ví dụ đại diện trong 41 câu `wrong_value`, đối chiếu trace với workbook và README của business:

- `biz_102`, doanh thu tháng 5 (`0.86` so với `2.11`): tính lại từ sheet tháng cho thấy chỉ tính giao dịch hoàn thành (loại hủy và hoàn hàng) cho kết quả tham chiếu `2.11%`.
- `biz_102`, xếp hạng khách tháng 4: kết quả có các nhãn “khách lẻ”/biến thể của khách không lưu danh tính; README xác nhận đây không phải khách được định danh. Lọc đơn hoàn thành và loại khách không định danh khớp danh sách tham chiếu.
- `biz_103`, tỷ lệ hủy tháng 4: code lọc đơn hủy ra trước rồi mới đếm đơn hủy, khiến tử số bằng 0. Dữ liệu có 22 đơn hủy trên 1.165 đơn, đúng `1.89%`.
- `biz_103`, số đơn từ đầu năm đến 13/2: code chỉ dùng sheet tháng 1. Đối chiếu cả hai sheet cho 1.871 mã đơn; loại 58 đơn hủy và 28 đơn hoàn/đổi trả còn 1.785 đơn như đáp án tham chiếu.

Đã cập nhật:

- Prompt yêu cầu giữ mọi thực thể/bộ lọc trong câu hỏi, gộp các sheet tháng cùng schema khi phạm vi ngày đi qua nhiều tháng, đếm mã đơn duy nhất, dùng trạng thái thực có trong dữ liệu, xử lý đúng tử số/mẫu số tỷ lệ, loại giao dịch hủy/hoàn khi tính “đã bán”, và không xếp khách vãng lai như khách có danh tính.
- Loader Excel điền tiếp mã đơn/ngày/trạng thái/khách hàng qua các dòng sản phẩm thuộc cùng đơn khi workbook dùng ô merge. ISO date được ưu tiên để tránh `dayfirst` đảo tháng/ngày.
- Few-shot chỉ giữ ví dụ đúng loại câu hỏi và tương thích schema. Executor gửi lỗi kiểu ngày thực tế vào retry có hướng dẫn Timestamp; retry dừng khi code và lỗi lặp, và trả đúng số lượt retry.

Kiểm thử local: `python -m pytest tests/ -q` — **101 passed**. Notebook đã đặt `RUN_TAG="fix_context_v5"`; `datamate-code.zip` được tạo lại gồm 18 file `src/` và `prompts/`, đã đối chiếu hash và không chứa `__pycache__`/`.pyc`.

**Chưa có benchmark mới:** chưa chạy được inference trên Kaggle từ môi trường local. Vì vậy chưa thể kết luận PA, số sai/abstain hay thời gian có cải thiện. Cần publish/gắn archive mới vào Kaggle Model, chạy đúng sample 200 câu với seed 42 và cấu hình model baseline, rồi mới so với baseline (`PA -0.175`) và `context-v4` (`PA -0.225`).

### Benchmark 14B INT8 — context-14b — 2026-11-01

Đã đọc `context-14b/eval_summary.json` và `error_analysis.csv`. Run tag là `context_v5a_14b_int8`; sample 200/3.000 câu, seed 42. Summary không ghi model ID, nên checkpoint chỉ được nhận diện theo run tag/notebook config chứ chưa xác minh độc lập từ artifact.

| Run | PA mẫu | PA có trọng số | Đúng / sai / abstain | Thời gian TB/câu |
|---|---:|---:|---:|---:|
| Baseline | -0.175 | -0.1771 | 1 / 36 / 163 | 14.79 giây |
| context-v5 (3B) | -0.305 | -0.3048 | 5 / 66 / 129 | 23.19 giây |
| context-14b INT8 | -0.765 | -0.7618 | 11 / 164 / 25 | 40.08 giây |

So với context-v5, số abstain giảm 104 và số câu đúng tăng 6, nhưng số câu sai tăng 98; thời gian trung bình tăng khoảng 73%. Error categories: 25 `abstain_exec_error`, 153 `wrong_value`, 8 `false_unanswerable`, 3 `missed_unanswerable`, 11 `correct`. `expected_type_accuracy_excluding_unanswerable` là 100%, nhưng không phản ánh đúng tính toán.

Trong 25 lỗi thực thi cuối cùng có 4 `DateTypeError`, 3 `KeyError: 'Thời gian'`, 2 lỗi import `time`, cùng một số lỗi tên cột/biến, kiểu dữ liệu và cú pháp. Đáng chú ý, lỗi runtime giảm mạnh nhưng 153 câu sai giá trị cho thấy điểm nghẽn hiện tại chuyển sang diễn giải câu hỏi và nghiệp vụ, không chỉ khả năng sinh code chạy được. Ví dụ trong CSV: câu doanh thu tháng 5/2025 ở `biz_102` đúng `2.11%`; câu xếp hạng top khách tháng 4 vẫn đưa các biến thể “khách lẻ” vào kết quả.

**Kết luận:** trên sample này 14B INT8 chạy code thành công/ít abstain hơn nhưng PA thấp hơn đáng kể và chậm hơn; chưa nên chạy toàn bộ 3.000 câu. Ưu tiên phân tích/fix các sai lệch kỳ lọc, trạng thái và mặt hàng cụ thể, sau đó đánh giá lại cùng sample. Nguồn: `context-14b/eval_summary.json`, `context-14b/error_analysis.csv`.

### Triển khai sau error analysis 14B

Đã kiểm tra lại `error_analysis.csv`: có **30** đáp án number/percent bằng 0 và cả 30 bị chấm sai (không phải 28 như ước lượng trong plan). Mẫu này chứng minh zero là tín hiệu rủi ro đáng kiểm tra, nhưng không đủ để cấm mọi đáp án zero.

Đã triển khai:

- Bản thử nghiệm lúc đầu bật ba candidate code lấy mẫu độc lập ở temperature 0.2; từng candidate có retry riêng và chỉ chấp nhận khi đa số khớp. **Cơ chế này đã được gỡ** ở mục “Theo dõi latency và bỏ ensemble candidate” bên dưới; không dùng để chạy tiếp.
- Nếu code tạo zero number/percent từ DataFrame đã lọc rỗng, executor báo lỗi để retry; retry cho model danh sách giá trị categorical thực tế. Không có bộ lọc rỗng hoặc kết quả khác zero thì không bị ảnh hưởng.
- Schema và retry context đưa các giá trị phân loại có cardinality thấp vào prompt và gợi ý giá trị gần nhất cho literal filter bị sai. Prompt nhấn mạnh kiểm tra giá trị lọc, dùng ngày chính xác và không từ chối câu hỏi chỉ vì kết quả lọc rỗng.
- Loader giữ nguyên từng sheet và tạo thêm bảng hợp nhất cho các sheet cùng schema có tên tháng rõ ràng, kèm `__source_month`/`__source_sheet`; prompt yêu cầu lọc theo tháng nguồn khi thiếu cột ngày. Đã xác minh trên workbook thật của `biz_102`: 18 sheet, 86.029 dòng hợp nhất, từ `2025-01` đến `2026-06`.
- Cho phép import `time`; bỏ chặn tĩnh toàn bộ `.dt.date`, vì grouping theo ngày hợp lệ. Lỗi so sánh date/Timestamp vẫn đi qua xử lý exception/retry.

Kiểm thử trước khi bỏ ensemble: `python -m pytest tests/ -q` — 112 passed. Sau khi gỡ multi-candidate, `python -m pytest tests/ -q` — **107 passed**. Notebook JSON và code cells parse được; workbook `biz_102` thật đã xác nhận tạo bảng hợp nhất đúng 18 kỳ tháng. Đã làm mới `datamate_code.zip` (18 file source/prompt, không chứa bytecode). Chưa tự động xóa giao dịch hủy/hoàn trên mọi bảng (sẽ làm hỏng câu hỏi về tỷ lệ hủy), không học ngưỡng từ nhãn 200 câu đánh giá và không khẳng định các zero khác đều do filter sai.

### Theo dõi latency và bỏ ensemble candidate — 2026-11-01

Run 14B trước (`context_v5a_14b_int8`) đã xử lý 200 câu trong tổng cộng **8.015,73 giây (~2 giờ 13 phút)** theo `eval_summary.json`, tương đương **40,08 giây/câu**.

Với run `context_v5b_14b_int8_ensemble3`, các log đầu hiện hiển thị khoảng **51–114 giây/câu**; riêng các lượt sinh/retry được log tốn khoảng **11–38 giây**, còn mỗi lần chạy code tốn khoảng **0.0–0.1 giây**. `LLM generation attempt 1 took 0.0s` trong RetryEngine không bao gồm sinh candidate ban đầu: candidate đã được tạo trước theo batch, nên đây không phải thời gian inference bằng 0.

Lưu ý: ngoại suy sơ bộ 14–32 giờ cho 1.000 câu từ vài câu đầu (51–114 giây/câu) đã bị thay thế bởi số liệu wall-clock hoàn chỉnh bên dưới; không dùng dải ngoại suy ngắn đó làm ETA.

**Việc tối ưu trước khi chạy 1.000 câu:**

1. Đo wall-clock trên batch đại diện sau khi quay về một candidate, rồi so trực tiếp cùng model/run cũ.
2. Đo thời gian từng retry và số lượt retry; tối ưu lại `MAX_RETRIES` riêng nếu execution errors vẫn chiếm đáng kể.
3. Với prompt lớn, đánh giá giảm schema/examples hoặc cache prompt theo business mà không loại bỏ cột/giá trị cần để trả lời chính xác.

**Quyết định tại thời điểm ghi log:** ensemble có vẻ chậm hơn run cũ theo các câu đầu, nhưng chưa đủ dữ liệu để kết luận mức tăng latency ổn định. Phần khuyến nghị dưới đây đã được cập nhật sau khi gỡ ensemble.

Sau khi xem log tiếp và cân nhắc mục tiêu chạy lớn, đã **gỡ hẳn tính năng 3 candidate** thay vì chỉ đặt count về 1: pipeline quay về một chuỗi generate/retry cho mỗi câu; notebook bỏ hàm batch candidate, cấu hình/metadata candidate và preflight chữ ký tương ứng. Run tag đổi sang `context_v5c_14b_int8_single` để không đọc tiếp output của run ensemble. Cấu hình sampling đơn trong notebook là temperature `0.0` như run 14B trước. Giữ nguyên các cải tiến khác (empty-filter retry, fuzzy category suggestions, combined monthly sheets, cache-module preflight).

Đây là thay đổi cấu hình cho lần chạy tiếp theo; run Kaggle đang chạy không tự đổi cấu hình. Nếu đã có output dưới tag ensemble, không dùng output đó làm kết quả run single. Chưa có số đo wall-clock cuối của ensemble và chưa benchmark run single mới.

### Kết quả benchmark single sau cải tiến — 2026-11-02

Đã đọc artifacts mới trong `context-14b/`. `eval_summary.json` ghi run tag `context_v5c_14b_int8_single`, sample 200/3.000, seed 42. Đây là lần chạy **single candidate** sau khi bỏ ensemble; summary không lưu model ID hay hash danh sách sample, nên so sánh dưới đây dựa trên run tag, seed và kích thước sample.

| Run | PA mẫu | PA có trọng số | Đúng / sai / abstain | Thời gian TB/câu | Tổng thời gian |
|---|---:|---:|---:|---:|---:|
| 14B INT8 trước cải tiến (`context_v5a`) | -0.765 | -0.7618 | 11 / 164 / 25 | 40,08 giây | 8.015,73 giây |
| 14B INT8 single mới (`context_v5c`) | **-0.575** | **-0.5786** | **24 / 139 / 37** | **39,50 giây** | **7.899,65 giây** |

PA tăng **0,190 điểm**; đúng tăng 13, sai giảm 25, abstain tăng 12. Đây là cải thiện rõ trên sample, dù PA vẫn âm và số câu sai vẫn cao. Thời gian gần như không đổi: giảm khoảng **116 giây trên cả 200 câu** (~1,45%), không cho thấy ensemble là nguyên nhân của toàn bộ latency trước đó.

Thay đổi rõ nhất theo loại câu là date: run cũ có 2 đúng, 0 sai, 8 abstain; run mới có **8 đúng, 2 sai, 0 abstain**. Điều này phù hợp với việc bỏ chặn `.dt.date` và để lỗi kiểu ngày thực sự đi vào retry, nhưng đây là so sánh hai run tổng thể, không phải ablation chứng minh một sửa đổi đơn lẻ gây ra toàn bộ mức tăng.

Các lỗi vẫn còn đáng kể: run mới có 135 `wrong_value`, 32 `abstain_exec_error`, 5 `abstain_gate`, 3 `false_unanswerable` và 1 `missed_unanswerable`. Trong 32 execution abstains, 14 là `EmptyFilterError` và 12 là `KeyError`. Có 4 đáp án numeric zero, cả 4 đều sai, giảm từ 30 zero sai trong run trước; không đủ để kết luận các thay đổi empty-filter đã sửa 26 câu cụ thể vì chưa đối chiếu được output cũ từng ID và sample hash không có trong summary.

Thời gian thực tế 39,50 giây/câu tương đương phép nhân tuyến tính khoảng **11 giờ cho 1.000 câu**; đây vẫn chỉ là ngoại suy, không phải cam kết runtime. Kết quả này thay cho ước tính 14–32 giờ dựa trên vài câu đầu.

**Kết luận:** PA được cải thiện rõ và latency trung bình giữ gần mức run cũ. Giữ cấu hình single cho vòng kế tiếp; ưu tiên xử lý `EmptyFilterError`/`KeyError` và 135 sai giá trị. Cần xác nhận cùng chính xác question IDs bằng sample signature ở lần benchmark sau, đồng thời chạy ablation nếu muốn xác định sửa đổi nào đóng góp bao nhiêu.

### Kaggle preflight stale module cache — 2026-11-01

Notebook từng báo `RetryEngine.run` thiếu `initial_response`, dù source lúc đó có tham số này. Nguyên nhân trong preflight cũ: cell chỉ xóa cache `src.*`, nhưng `src.predict` import các module con bằng tên `modules.*` (`from modules.retry_engine import RetryEngine`). Nếu `modules.retry_engine` cũ vẫn còn trong `sys.modules`, `importlib.reload(src.predict)` tiếp tục gắn class RetryEngine cũ vào pipeline. Vì vậy lỗi đó không tự nó chứng minh upload sai. Tham số `initial_response` sau đó được gỡ cùng tính năng multi-candidate; phần cache fix vẫn còn hữu ích cho đồng bộ module.

Đã sửa `kaggle_train_baseline.ipynb`: trước import, preflight xóa cả `src.*` lẫn `modules.*`; sau import, in đường dẫn `retry_engine.py` và kiểm tra các module đều nằm dưới `CODE_DIR`. Đã tái hiện Python stale cache và xác nhận cache cũ bị xóa, module nạp từ source mới. Cell notebook compile được.
