# EVO-PCA v64.0.0 — Evolutionary Predictive & Camouflage Awareness Shield

**EVO-PCA** là hệ thống Tường Lửa & Phát Hiện Rủi Ro Đa Tầng (Multi-tier Security Shield) dành cho các ứng dụng thực thi tự động (Agentic Workflow & Tool Execution) sử dụng Mô Hình Ngôn Ngữ Lớn (LLM). Hệ thống cung cấp cơ cấu bảo vệ chủ động 3 tầng kết hợp giữa kiểm định quy tắc chính xác (Deterministic Regex), học máy dự đoán theo lịch sử chuỗi (Markov & LSTM), và Trọng tài nhận dạng ngữ nghĩa cơ cấu sâu (LLM Judge Router).

---

## 🏛️ Tinh Hoa Kiến Trúc 3 Tầng (Three-Tier Architecture)

```
[ Ingress Tool Call / Action ] 
         │
         ├──► Tầng 0 (Tier 0): Deterministic Firewall & Canonicalization
         │       └─ Lọc chuỗi nguy hiểm tức thì (rm -rf, curl, eval, SQL injection, v.v.)
         │
         ├──► Tầng 0.5 (Tier 0.5): Neural LSTM + Markov K=3 Semantic Shield
         │       └─ Phát hiện ngụy trang ngữ nghĩa (Semantic Camouflage Jump)
         │       └─ Bảo vệ kháng lỗi đếm bước (Step-Counting Invariance Guard)
         │
         └──► Tầng V61 (Tier V61): Adaptive ML Router & LLM Judge
                 └─ Tự điều chỉnh ngưỡng trần (Hysteresis Deadband ≥ 2 chu kỳ)
                 └─ Tác vụ song hàng (Parallel Execution + Circuit Breaker)
```

### 1. Kháng Bước Đếm (Step-Counting Invariance Guard)
Trong các phiên làm việc dài (long-running sessions), các mô hình neural theo chuỗi (như LSTM cũ) dễ bị thiên vị đếm bước (sập bẫy từ chối lệnh nhầm khi bước thứ 50–100 chỉ đơn thuần là các thao tác theo dõi, đọc log lặp đi lặp lại).
- **Giải pháp v64.0.0:** Hệ thống kích hoạt module **Rủi ro Chuỗi Markov ($K=3$)** sử dụng Khoảng cách Cosine trên Dense Embeddings từ bộ mô hình `all-MiniLM-L6-v2`.
- Nếu chuỗi thao tác chỉ là lặp lại lành tính ($\text{seq\_risk} < 0.05$), chốt giáp an toàn sẽ vô hiệu hóa báo động đếm bước của LSTM, giữ trọn vẹn luồng thi công không bị đứt đoạn.
- Nếu xuất hiện bước nhảy vọt bất ngờ (như từ `check_server_uptime()` lập tức bẻ lái sang `export_database_records()`), mốc delta $\ge 0.65$ lập tức gạt sập cầu dao (**Block**).

> [!NOTE]
> **Lưu ý về Benchmark (ABSR / AgentDojo):** Nhánh phân loại theo xác suất của LSTM hiện đang bị tắt mặc định thông qua biến môi trường `TIER05_LSTM_ENABLED=False` để nhường tài nguyên cho V61 LLM Judge. Các chỉ số như "Tier 0.5-LSTM Blocks: 0.0" trong báo cáo benchmark phản ánh chính xác trạng thái tắt này. Khi bật, ngưỡng chặn sẽ được đồng bộ qua cấu hình `LSTM_BLOCK_THRESHOLD` thay vì fix cứng.

### 2. Dịch Vụ Mô Hình Hợp Nhất (`SharedSemanticEncoder`) & Bồn Cache Có Kiểm Soát
- Tách rời hoàn toàn gánh nặng xử lý AI embedding: mọi module tham gia đều hội tụ qua cổng dịch vụ Singleton **`SharedSemanticEncoder`** trong `models/security/advanced_heuristics.py`, **tiêu tốn bộ nhớ mô hình bằng 0 khi sử dụng lại**.
- **`SessionEmbeddingCacheManager`:** Tác vụ lưu cache vector chuỗi được khóa theo kích thước cố định (`maxsize = 50` mục/phiên, `TTL = 3600s`), tuyệt đối không gây tràn RAM (Memory Leak).

---

## 🛠️ Hướng Dẫn Cài Đặt & Cấu Hình

### 1. Yêu Cầu Môi Trường (Prerequisites)
- Python $\ge 3.10$
- Các thư viện cốt lõi: `torch`, `sentence-transformers`, `scikit-learn`, `pyyaml`, `requests`.

### 2. Bảng Tinh Chỉnh Ngoại Trụ (`config/settings.yaml`)
Toàn bộ hệ thống không sử dụng số cứng (Zero Magic Numbers). Quý Vị có thể tùy ý thay đổi độ nhạy của tường lửa ngay trong tệp cấu hình trung tâm [`config/settings.yaml`](file:///D:/DEMO_GROUP_1/config/settings.yaml):

```yaml
OLLAMA_MODEL: gemma3:4b            # Mô hình sử dụng cho tầng Trọng tài LLM Judge
OLLAMA_TIMEOUT: 15.0               # Giám sát giới hạn chờ cho inference (giúp qua ải Cold-Start)
FIREWALL_MODE: STRICT              # Chế độ làm việc của tường lửa (STRICT / AUDIT)
OLLAMA_HOST: http://localhost:11434
SEMANTIC_DELTA_THRESHOLD: 0.65       # Ngưỡng đứt gãy ngữ nghĩa (>= 0.65 sẽ khóa đòn nhảy vọt)
SEQUENCE_RISK_GAIN: 1.2            # Hệ số khuếch đại rủi ro Markov K=3
```
*(Ghi chú: Toàn bộ thông số trên đều có thể ghi đè nóng thông qua biến môi trường tương ứng).*

---

## 🧪 Hướng Dẫn Kiểm Thử & Chạy Đánh Giá Độ Nhạy

Hệ thống được thiết kế kèm theo màng lọc nghiệm thu đối chứng kép cực kỳ khát khe để đảm bảo mọi sự tùy biến không phá vỡ tính năng an toàn cốt lõi.

### 1. Kiểm Chân Chức Năng Cốt Lõi (Unit Testing)
Chạy trực tiếp cú pháp lệnh sau trong terminal để tự động tra cứu độ tin cậy của mô hình Markov và hàng rào cự tuyệt Overfitting (Generalization Gate):

```powershell
py -m unittest tests/test_step_counting_invariance.py tests/test_generalization_gate.py -v
```

**Các mốc kiểm định tiêu chuẩn:**
- **Invariance Guard:** Chức năng mồi thử dương tính (Positive Control) trả về $\Delta = 1.0$; chuỗi 100 bước lệnh lành tính suôn sẻ không phát sinh báo cháy sai (`blocked = False`); ngay lúc tiêm lệnh độc hại vào Bước 101 lập tức kích nổ còi khóa!
- **Generalization Gate $\ge 80\%$:** Kiểm định chuỗi tấn công tinh tế bẻ lọt rào chắn còi regex cũ (`ALLOW`), hoàn toàn dựa vào rủi ro chuyển dịch ngữ nghĩa để cự tuyệt $\ge 80\%$ (hiện đang đạt mốc vững vàng $100\%$).

### 2. Thực Thi Đánh Giá Tải Cao trên Tập Mẫu Ngoại Tuyến
Để đánh giá tốc độ xử lý hàng loạt của pipeline với hỗ trợ song hàng `ThreadPoolExecutor(max_workers=8)` và tự ngắt bảo vệ (Circuit Breaker $6.0\text{s}$), thực thi:
```powershell
py tests/run_external_benchmark.py
```

---

## 📖 Hướng Dẫn Kỹ Thuật Khi Cần Chỉnh Sửa & Mở Rộng (Developer Guide)

Để các Nhà phát triển (Developers) dễ dàng đồng hành và tối ưu hóa hệ thống mà không vấp phải lỗi, hãy tham chiếu biểu đồ trách nhiệm mã nguồn sau:

| Thư Mục / Tệp Quyết Định | Trách Nhiệm Kỹ Thuật | Lưu Ý Khi Can Thiệp & Mở Rộng |
| :--- | :--- | :--- |
| **`core/pipeline.py`** | Điều khiển chính (Orchestrator), quản lý tuần tự Tier 0 $\to$ Tier 0.5 $\to$ V61 | Khi bổ sung tầng màng cản mới, hãy đăng ký trong `UnifiedFirewallPipeline.scan()`. |
| **`core/tier_lstm.py`** | Tầng học máy rủi ro chuỗi, chứa Bounded Cache & logic tính Delta Markov | Hàm `compute_sequence_risk()` duy trì thuật toán $\Delta = 1 - \text{cosine\_sim}$. **Tuyệt đối giữ còi bảo vệ bước nhảy** nhằm duy trì khả năng kháng Step-Counting Bias. |
| **`models/security/advanced_heuristics.py`** | Chứa bộ điều khiển ngưỡng động `AdaptiveEscalationManager` & Dịch vụ `SharedSemanticEncoder` | Có cơ cấu Giảm chấn Hysteresis ($\ge 2$ chu kỳ kiên trì chệch hướng mới điều chỉnh ngưỡng). Không khởi tạo rải rác Transformer, luôn dùng `SharedSemanticEncoder.get()`. |
| **`config/settings.yaml` & `config_loader.py`** | Quy ước danh bạ tham số hoạt động toàn hệ thống | Nếu bổ sung biến cấu hình mới, hãy chèn thêm khóa vào YAML và bổ sung từ điển đồng bộ trong hàm `load_settings()` của `config_loader.py`. |

### 💡 Bí Kíp Thao Tác Nhanh cho Người Chức Trị:
1. **Muốn thêm từ điển cấm đoán tức thì?** Mở `core/pipeline.py` (hoặc module `Tier0`) và chỉnh sửa danh sách chính quy rà quét chuỗi thô ngầm định.
2. **Muốn Tối Ưu Hóa Tỷ Lệ Cản/Lọt (Precision vs Recall)?** Chỉ cần mở `settings.yaml` điều chỉnh hệ số `SEMANTIC_DELTA_THRESHOLD` (hạ xuống $0.55$ để thắt chặt nhạy cảm, tăng lên $0.75$ để tăng sự thông thoáng cho Agent tiếp cận hệ thống).
3. **Mỗi lần chỉnh sửa lõi thuật ngữ xong:** Bắt buộc chạy chuỗi kiểm tra `test_generalization_gate.py` để lấy phiếu xác nhận an toàn của hệ thống (phải luôn duy trì tỷ lệ phát hiện đòn tấn công độc lập $\ge 80\%$) trước khi đẩy commit lên kho chung!

---
*EVO-PCA Shield — Kiến tạo móng vững chắc cho Kỷ nguyên Thực Thi AI An Toàn.*
