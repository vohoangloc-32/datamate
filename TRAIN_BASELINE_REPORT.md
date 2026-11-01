# DataMate train baseline — tình hình và kế hoạch triển khai

**Ngày lập:** 2026-10-31
**Nguồn:** run `baseline`, `seed=42`, `SAMPLE_N=200`; `eval_summary.json` và `error_analysis.csv` do người dùng cung cấp.
**Mục tiêu:** xác định nguyên nhân PA thấp, cải thiện pipeline theo bằng chứng từ baseline và đánh giá lại có kiểm chứng. Baseline bên dưới vẫn giữ nguyên làm mốc; trạng thái implementation hiện tại được cập nhật ở mục 6.

## 1. Tóm tắt tình hình

Notebook đã chạy inference trên mẫu 200 câu lấy từ 3.000 câu train; đây không phải fine-tuning. Kết quả hiện tại:

| Chỉ số | Kết quả |
|---|---:|
| PA thô trên mẫu | -0.175 |
| PA ước lượng có trọng số trên train | -0.1771 |
| Đúng / sai / abstain / missing | 1 / 36 / 163 / 0 |
| Tỷ lệ abstain | 81.5% |
| Thời gian tổng / trung bình mỗi câu | 2,957.74 giây / 14.79 giây |
| Độ chính xác `expected_type` baseline, bỏ `unanswerable` | 49.74% |

PA dùng công thức `(đúng - sai) / tổng`; vì vậy kết quả mẫu là `(1 - 36) / 200 = -0.175`. Con số weighted PA hữu ích để tham khảo nhưng chưa phải phép đo ổn định: một số answer type hiếm chỉ có 1 câu trong mẫu, và mẫu train sẽ được dùng để chẩn đoán/tuning.

### Phân bố kết quả theo loại đáp án chuẩn

| Loại | Đúng | Sai | Abstain | Tổng |
|---|---:|---:|---:|---:|
| number | 0 | 32 | 94 | 126 |
| percent | 1 | 1 | 36 | 38 |
| list | 0 | 2 | 11 | 13 |
| date | 0 | 0 | 10 | 10 |
| string | 0 | 0 | 6 | 6 |
| unanswerable | 0 | 0 | 5 | 5 |
| table | 0 | 0 | 1 | 1 |
| boolean | 0 | 1 | 0 | 1 |

### Nguyên nhân lỗi nổi bật

Phân tích attempts trong CSV cho thấy nhiều lỗi lặp lại; cùng một câu có thể gặp một lỗi ở nhiều lượt retry, do đó số attempts không đồng nghĩa số câu độc lập.

| Lỗi | Số câu bị ảnh hưởng | Số lần xuất hiện trong attempts | Nhận định |
|---|---:|---:|---|
| Prompt vượt giới hạn 16,384 tokens | 72 | 213 | Tất cả 72 câu cuối cùng đều abstain do lỗi thực thi. |
| `NameError: orders_df` | 52 | 149 | Model tham chiếu tên DataFrame không có trong execution environment. 51 câu cuối cùng abstain; một câu cho kết quả sai loại. |
| `KeyError: 'order_date'` | 19 | 50 | Model giả định cột ngày không tồn tại; 18 câu cuối abstain, một câu sai giá trị. |
| `TypeError` | 11 | 22 | Bao gồm lỗi so sánh `Timestamp` với `datetime.date` và dữ liệu kiểu không đồng nhất. |
| `ValueError` | 8 | 18 | Có trường hợp parse ngày Excel dạng số như `45689`. |
| `DateParseError` | 8 | 10 | Có trường hợp chuỗi ngày tiếng Việt như `1 thg 6 2025`. |
| `SyntaxError` | 6 | 15 | Code sinh không hợp lệ; các câu này cuối cùng abstain. |
| `FileNotFoundError` | 2 | 6 | Code thử đọc lại `DuLieuBanHang.xlsx` bằng tên file không có trong runtime, thay vì dùng dữ liệu đã nạp. |

Các nhóm lỗi có thể giao nhau ở cấp câu hỏi. Ngoài lỗi thực thi: có 17 câu sai loại, 17 câu sai giá trị và 2 câu dự đoán `unanswerable` sai; chỉ có 1 câu đúng.

## 2. Chẩn đoán kỹ thuật

1. **Prompt context là ưu tiên số một.** `PromptBuilder.build()` đưa schema, giá trị mẫu, tối đa 3 dòng mẫu của từng bảng, thông tin README/rulebook và ví dụ vào prompt. Schema/sample rows được thêm trực tiếp cho mọi bảng; riêng `build_retry_prompt()` nối toàn bộ prompt ban đầu với code lỗi và thông báo lỗi. Điều này có thể lặp lại payload lớn qua từng retry. Cần đo token thật, không suy luận giới hạn từ độ dài ký tự.
2. **Tên biến và schema chưa đủ hiệu quả với model.** Prompt có bảng ánh xạ tên bảng sang biến, nhưng lỗi `orders_df` và `order_date` cho thấy chỉ đưa mapping/schema vào prompt chưa đảm bảo model sử dụng đúng. Cần củng cố chỉ dẫn, kiểm tra mapping thực tế theo business và đo lại theo từng câu.
3. **Chuẩn hóa ngày/kiểu dữ liệu cần xử lý có chủ đích.** CSV chỉ ra các dạng ngày không chuẩn và lỗi trộn `Timestamp`/`date`. Cần xác minh dữ liệu gốc và quy tắc nạp dữ liệu trước khi thêm chuyển đổi; tránh chuẩn hóa mù làm thay đổi ý nghĩa hoặc dữ liệu.
4. **Retry hiện không chống được lỗi prompt quá dài.** Khi lỗi generation do giới hạn context, retry có thể lặp lại gần như cùng payload. Nên phân loại lỗi này và chuyển sang prompt rút gọn có kiểm soát hoặc abstain rõ ràng thay vì lặp vô ích.
5. **Độ chính xác loại đáp án là vấn đề độc lập.** `expected_type` đúng khoảng một nửa trên mẫu, nhưng cần xem confusion matrix và lỗi từng loại trước khi chỉnh classifier. Không nên quy toàn bộ 17 `wrong_type` cho classifier nếu chưa đối chiếu `expected_type`, `prediction_type` và reference.

## 3. Implementation plan

### Giai đoạn 0 — Đóng băng phép so sánh

- Giữ nguyên run baseline và file kết quả làm mốc; không ghi đè.
- Mỗi thử nghiệm dùng `RUN_TAG` mới, nhưng giữ `SEED=42`, `SAMPLE_N=200`, model và tham số còn lại không đổi để so cùng câu hỏi.
- Ghi lại hash code/data, cấu hình, số câu hoàn tất, PA, thời gian và nhóm lỗi cho từng run.
- Xác nhận các lần chạy có đúng cùng 200 ID; nếu code/data/config thay đổi ngoài phần đang thử, ghi rõ thay đổi.

**Đạt khi:** có baseline bất biến và manifest chứng minh các lần so sánh dùng cùng mẫu.

### Giai đoạn 1 — Giảm prompt và chặn retry vô ích

- Đo số token của prompt thực tế bằng tokenizer đang dùng trong vLLM; log token count theo câu, business và từng attempt.
- Tạo ngân sách prompt nhỏ hơn `MAX_MODEL_LEN`, có phần dự phòng cho output; không chỉ cắt chuỗi theo số ký tự.
- Ưu tiên giữ: chỉ dẫn nền tảng cần thiết, tên biến chính xác, schema tên/cột/kiểu dữ liệu, rulebook liên quan, câu hỏi và mốc thời gian.
- Chỉ đưa sample rows hoặc ví dụ khi còn trong ngân sách; rút gọn sample rows và loại thông tin lặp/không liên quan theo quy tắc rõ ràng.
- Với lỗi context overflow, không retry nguyên prompt lớn. Tạo một bản compact prompt có giới hạn; nếu vẫn không vừa, ghi lỗi cụ thể và abstain có kiểm soát.
- Ghi token count trước/sau compact để biết câu nào đã được cứu và chất lượng có bị ảnh hưởng không.

**Kiểm thử/tiêu chí đạt:**

- Unit tests cho prompt ngắn, prompt nhiều bảng, prompt vượt ngân sách và retry sau context overflow.
- Không có prompt gửi tới model vượt ngân sách token cấu hình.
- Trên cùng mẫu 200 câu: mục tiêu giảm rõ rệt lỗi context so với 72 câu baseline; báo cáo số còn lại, PA và thời gian. Không coi việc giảm token là thành công nếu PA giảm.

### Giai đoạn 2 — Bám đúng tên biến và schema

- Kiểm tra `get_table_var_mapping()` và execution globals trên từng business; xác nhận mọi biến được nêu trong prompt đúng là biến khả dụng, không bị trùng sau sanitize.
- Làm nổi bật mapping `tên bảng -> tên biến Python` và yêu cầu chỉ sử dụng đúng tên có trong mapping; cấm tự đặt tên như `orders_df` nếu không được liệt kê.
- Giữ schema tên cột chính xác; cân nhắc đưa các cột liên quan câu hỏi lên trước, nhưng không bịa cột hoặc bỏ cột cần thiết.
- Với `NameError`/`KeyError`, retry prompt nên kèm danh sách biến và cột khả dụng liên quan, thay vì chỉ gửi traceback.
- Phân tích riêng 52 câu `orders_df` và 19 câu `order_date`; xác định business, bảng/cột đúng, và lỗi có chung nguyên nhân hay không.

**Kiểm thử/tiêu chí đạt:**

- Test mapping cho tên bảng có dấu cách, dấu chấm, gạch nối và trường hợp tên sanitize bị trùng.
- Kiểm tra code sinh dùng tên bảng/cột tồn tại; đo tỷ lệ `NameError`/`KeyError` trên cùng sample.
- PA không giảm trong khi tỷ lệ lỗi biến/cột giảm.

### Giai đoạn 3 — Chuẩn hóa ngày và kiểu dữ liệu

- Chọn các câu tiêu biểu có `45689`, chuỗi ngày tiếng Việt và lỗi `Timestamp`/`date`; đối chiếu từng giá trị với file nguồn và loader.
- Xác định cột nào đã là datetime, cột nào còn Excel serial/text; xử lý ở data loader hoặc cung cấp quy tắc chuyển đổi rõ cho code model.
- Tránh chuyển cột sang `.dt.date` khi code sau đó so với `Timestamp`; thống nhất một kiểu so sánh trong hướng dẫn/ví dụ.
- Chỉ thêm helper dùng chung sau khi xác định được các định dạng thật và có test cho giá trị hợp lệ, thiếu, sai định dạng, Excel serial.

**Kiểm thử/tiêu chí đạt:**

- Unit tests tái hiện từng định dạng ngày lỗi trong CSV và xác nhận ngày chuyển đổi đúng.
- Không phát sinh lỗi `DateParseError`, `ValueError` do Excel serial hoặc `TypeError` ngày trên các ca kiểm thử.
- Chạy lại sample để xem tác động tới đáp án và PA, không chỉ số lỗi parse.

### Giai đoạn 4 — Cải thiện phân loại loại đáp án và câu trả lời sai

- Tạo confusion matrix `expected_type` so với reference type, bỏ `unanswerable` khỏi accuracy của classifier như báo cáo hiện tại.
- Đọc 17 lỗi `wrong_type`, 17 `wrong_value` và 2 `false_unanswerable` theo câu hỏi, đối chiếu code/answer/ref.
- Chỉnh quy tắc classifier hoặc chỉ dẫn sinh code theo các mẫu lỗi có bằng chứng; không tinh chỉnh chỉ để khớp một ít câu.
- Đánh giá confidence gate riêng: abstain hiện có PA 0 nhưng sai có -1; chỉ điều chỉnh threshold sau khi có dữ liệu đúng/sai đủ để so sánh.

**Kiểm thử/tiêu chí đạt:**

- Có confusion matrix theo loại, ví dụ lỗi được phân loại và unit tests cho quy tắc mới.
- Theo dõi PA, wrong rate, abstention rate và kết quả từng answer type trên cùng sample.

### Giai đoạn 5 — Xác nhận và mở rộng

- Chạy lại 200 câu cùng sample, lưu `eval_summary.json`, `error_analysis.csv`, logs và cấu hình ở run tag mới.
- So sánh baseline và candidate theo PA, lỗi context, lỗi biến/cột, lỗi ngày, wrong type/value, abstention và thời gian.
- Nếu một thay đổi cải thiện, chạy thêm seed/mẫu khác để kiểm tra độ ổn định; sample dùng để phát triển không thay thế validation độc lập.
- Chỉ chạy toàn bộ 3.000 câu khi pipeline ổn định, không còn lỗi hệ thống chiếm tỷ trọng lớn và thời gian/GPU đã được chấp nhận.

## 4. Quy tắc quyết định sau mỗi vòng

- **Không tăng `MAX_MODEL_LEN` như giải pháp duy nhất:** model và runtime vẫn có giới hạn; prompt phải nằm trong ngân sách thực tế.
- **Không thay nhiều yếu tố trong cùng một run:** nếu context, classifier và retry cùng đổi thì khó biết thay đổi nào tạo ra kết quả.
- **Không đánh giá chỉ bằng abstention rate:** giảm abstain nhưng tăng câu sai có thể làm PA tệ hơn do câu sai bị trừ điểm.
- **Ưu tiên PA**, đồng thời theo dõi lỗi theo nguyên nhân để bảo đảm cải thiện không phải ngẫu nhiên.
- **Giữ các lỗi chưa xử lý hiển thị rõ:** không biến lỗi sinh/thực thi thành kết quả thành công giả.

## 5. Phạm vi dự kiến của thay đổi code

Các điểm cần xem xét khi triển khai plan:

- `src/modules/prompt_builder.py`: token budget, prompt compact và retry prompt không lặp payload quá lớn.
- `src/modules/schema_registry.py`: schema compact/ưu tiên thông tin cần thiết nếu cần hỗ trợ prompt budget.
- `src/modules/code_executor.py`: mapping biến khả dụng và test các trường hợp tên biến.
- `src/modules/retry_engine.py`: phân loại context overflow, ghi attempt và tránh retry không hiệu quả.
- `src/modules/data_loader.py`: chỉ khi xác minh lỗi ngày/kiểu bắt nguồn từ cách loader đọc dữ liệu.
- `src/modules/question_understanding.py`: chỉ sau khi phân tích confusion matrix của loại đáp án.
- `tests/`: thêm test hồi quy cho các lỗi thực tế nêu trong báo cáo.

Nếu gói code Kaggle còn chứa bản sao `datamate_code/src/`, cần đồng bộ các thay đổi tương ứng và xác minh notebook đang import đúng bản đã cập nhật.

## 6. Implementation status (local)

Các thay đổi sau đã được áp dụng trong working tree; chưa chạy inference lại trên Kaggle nên **chưa có kết luận về PA, tỷ lệ abstain, tốc độ hoặc lỗi thực tế sau sửa**.

| Hạng mục | Trạng thái |
|---|---|
| Giới hạn prompt và retry theo tokenizer/context budget | Đã triển khai; compact prompt ưu tiên giữ schema và câu hỏi, đồng thời dừng retry lặp khi gặp context overflow. |
| Hỗ trợ đúng tên biến/schema trong retry | Đã triển khai; tên DataFrame được chuẩn hóa, tránh đè biến nội bộ; lỗi `NameError`/`KeyError` có thêm gợi ý biến/cột hiện có. |
| Chuẩn hóa dữ liệu ngày | Đã triển khai và có test cho Excel serial và ngày tiếng Việt. |
| Phân loại `expected_type` | Đã bổ sung quy tắc và regression cases dựa trên lỗi thấy trong CSV. Trên cùng CSV: 195/195 câu không phải `unanswerable` khớp nhãn reference, so với baseline 97/195. Đây là phép đo trên chính mẫu dùng để chẩn đoán/tuning, **không phải kết quả holdout**. |
| Notebook và hướng dẫn cấu hình | Notebook truyền tokenizer/budget và dùng run tag `fix_context_v1`; README mô tả các biến cấu hình. |
| Kiểm tra local | `pytest tests/`: 85 passed. Notebook JSON hợp lệ và 7 code cells parse được sau khi bỏ các dòng IPython magic/shell. |

### Các bước tiếp theo trên Kaggle

1. Cập nhật notebook và thư mục `src/` trên Kaggle bằng phiên bản mới; xác nhận notebook import đúng source đã cập nhật.
2. Restart kernel/session để không dùng module Python cũ, giữ nguyên model, seed, cỡ mẫu và các cấu hình baseline còn lại.
3. Chạy cùng 200 câu với run tag `fix_context_v1`, rồi tải `eval_summary.json` và `error_analysis.csv`.
4. So sánh PA trước/sau, số câu context overflow, `NameError`/`KeyError`, lỗi ngày, abstention và thời gian. Chỉ khi kết quả tốt mới cân nhắc chạy seed/mẫu validation khác hoặc toàn bộ 3.000 câu.

Batch processing, timeout thực sự cho code sinh, resume/chia batch nâng cao và so sánh data-parallel với TP=2 chưa nằm trong vòng sửa này; cần đánh giá riêng sau khi xác nhận lần chạy fix-context.

### Cập nhật lần chạy Kaggle — 2026-11-01

Lần chạy mới dừng ở cell inference với `TypeError: DataMatePipeline.__init__() got an unexpected keyword argument 'prompt_token_counter'`. Source trong báo cáo có tham số này và `max_prompt_tokens`; vì vậy Kaggle đang import source cũ/khác hoặc module cũ còn trong kernel. Log chưa xác định được chính xác thư mục nguồn đã import. Papermill dừng trước khi inference câu hỏi, nên chưa có PA mới.

Notebook đã được sửa để xóa module `src.*` cache, ưu tiên `CODE_DIR`, in đường dẫn `predict.py` thực tế và xác thực chữ ký `DataMatePipeline` trước khi nạp vLLM. Khi chạy lại, cần cập nhật Kaggle Input bằng source mới, restart kernel/session, xác nhận `Imported predict.py:` trỏ đúng source; nếu run cũ đã tạo config thì dùng run tag mới. Cảnh báo shared-memory `resource_tracker` và `SyntaxWarning` của `mistune`/`nbconvert` không phải nguyên nhân Papermill dừng; lỗi gây dừng là `TypeError`.

Traceback tiếp theo xác nhận source cũ cụ thể là `/kaggle/input/models/jcoolvn/datamate-model/pytorch/default/1/src/predict.py`; chữ ký tại đó thiếu cả `prompt_token_counter` lẫn `max_prompt_tokens`. Đã kiểm tra `datamate_code.zip`: 18 file thuộc `src/` và `prompts/` khớp byte với source hiện tại, và archive chứa constructor mới. Vì vậy cần publish một phiên bản Kaggle Model mới từ archive này rồi gắn phiên bản mới vào notebook; chỉ thay notebook không cập nhật code của Model Input. Restart kernel/session và xác nhận đường dẫn `Imported predict.py:` sau khi đổi version. Preflight mới dừng trước khi load vLLM nếu chữ ký vẫn cũ; chưa có inference hay PA mới.

Lần chạy sau dừng ở bước đối chiếu `run_config.json`: thư mục `fix_context_v1` đã có manifest từ lần chạy trước, nhưng code/dữ liệu/cấu hình hiện tại khác nên notebook từ chối resume để tránh ghép kết quả không tương thích. Tag sau đó được chuyển sang `fix_context_v2`.

Log Kaggle mới nhất cho thấy `/kaggle/input/models/jcoolvn/datamate-code/pytorch/default/1/src/predict.py` gọi `PromptBuilder.build` với `token_counter`, nhưng bản `PromptBuilder` được import không nhận tham số đó. Lỗi được bắt ở cấp từng câu và ghi câu trả lời `abstain`; các kết quả này không phải inference hợp lệ và không dùng để tính benchmark. Nguyên nhân là các module source Kaggle không đồng bộ, không phải lỗi từ model output.

Notebook hiện kiểm tra chữ ký `DataMatePipeline.__init__`, `PromptBuilder.build` và `RetryEngine.run` trước khi nạp vLLM; đồng thời in đường dẫn file của hai module đầu tiên. Run tag mặc định đổi thành `fix_context_v3` để không resume submission chứa abstain do lỗi tương thích. Cần publish/đính kèm toàn bộ `src/` và `prompts/` đồng bộ vào Kaggle Input `datamate-code`, restart session, upload notebook mới và xác nhận preflight qua trước khi inference.
