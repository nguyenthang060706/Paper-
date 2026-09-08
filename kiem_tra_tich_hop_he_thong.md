# Kiểm tra Tích hợp Toàn hệ thống (System Integration Audit) — v8

**Thay đổi so với v7:**
1. **Ghi nhận hoàn tất Bước 0a & Bước 0b:**
   - **Bước 0b (Race Condition & Hardening):** Đã vá race condition `HeuristicStateTracker.evaluate()`, lock scope tinh gọn hoàn toàn CPU in-memory, benchmark contention p95 chỉ tăng +1.7% (so với trần 10%), AST gatekeeping tự động chặn phình to critical section ($p99 \le 100\mu s$), revert LSTM threshold về 0.98 chuẩn và rerun `benchmark_3000.py` thành công (ABSR multi-step 95.42%, 338ms).
   - **Bước 0a (Code Review tĩnh quan hệ Threshold):** Xác nhận **Phát biểu A là sự thật duy nhất** của active pipeline (`core/pipeline.py`): `adaptive_threshold` trực tiếp override $T3$ (`block_threshold` của V61 ML Model). Phát biểu B là tàn tích tài liệu cũ từ `tests/run_benchmark.py` (nơi từng nhân 100 vào Heuristics Voting, nay đã bị loại bỏ hoàn toàn trong pipeline chuẩn).
2. **BÁO ĐỘNG ĐỎ — Nguy cơ tái diễn thảm họa đầu độc E7:** Phát hiện dòng code `core/pipeline.py:389` đang hardcode gom `MultiStep-Heuristics` và `LLM-Session-Judge` vào `fpr_manager.record_decision(is_escalated=True)`. Nâng **Bước 2 (Signal Registry)** thành **HARD BLOCKER** bắt buộc phải merge trước khi Bước 3 (BM25) chạy benchmark thực tế.
3. **Khai tử nhãn gộp $T1/T2/T3$ — Chuẩn hóa Bảng 5 Ngưỡng Độc Lập:** Tách bạch 5 biến ngưỡng thực tế trong code (`v61_structure_escalation_floor=0.35`, `session_judge_band_floor=0.40`, `v61_model_review_threshold=0.5385/0.5577`, `session_judge_band_ceil=0.5577`, `v61_adaptive_block_threshold=[0.58, 0.95]`).
4. **Đóng 2 Điểm mở ở Mục 4:**
   - Khẳng định không thể "revisit" một action đã thực thi; chuyển hướng `re_evaluate_action()` sang **Data Interception + Taint Session State + Emergency Kill-switch (`EGRESS_KILL_SWITCH`)**.
   - Đăng ký `EGRESS_KILL_SWITCH` vào Signal Registry với chính sách `feeds_escalation_managers = set()` (tập rỗng, không nuôi escalation).

---

## 0. Sơ đồ kiến trúc tổng hợp chuẩn hóa (v8)

```
Action → Tier 0 (Regex 32-pattern) ─BLOCK──────────────────────────▶ Decision: BLOCK
              │ PASS
              ▼
       Tier 0.5: Heuristic State-Machine  ◀────────────────────────┐
       ┌─────────────────────────────────┐                          │ Egress Feedback (dùng chung lock Bước 0b)
       │ _tag_action() waterfall:         │                          │ SanitizerResult.decision
       │  Regex → BM25 → Embedding        │                          │ (QUARANTINE / STRIP_AND_WRAP)
       │ (session state, TTL, kill-chain) │                          │ -> (a) Data Interception: drop payload
       │ (session_length >= N -> 1 lan,   │                          │ -> (b) Taint session state ngay lập tức
       │  sau do throttle moi K action)   │                          │ -> (c) EGRESS_KILL_SWITCH nếu cực kỳ nguy hiểm
       └─────────────────────────────────┘                          │
              │ Kill-chain triggered ──────────────────────────────▶ │
              │ PASS / Warning Flag / session_length trigger          │
              ▼                                                      │
       V61 ML Action Risk Model (ml_score)                           │
              │                                                      │
   ┌──────────┴──────────────────────┬───────────────────────────────┤
   │                                 │                               │
ml_score < 0.40           0.40 <= score < 0.5577            score >= v61_adaptive_block_t
(or ALLOW structure)      (or Warning Flag/session_len)     (miền biến thiên [0.58, 0.95])
   │                                 │                               │
   ▼                                 ▼                               ▼
 ALLOW                      LLM Session Judge                      BLOCK ───▶ feeds_escalation_managers?
(Fast Path)             (Nhận context: recent_egress_flags)      (Fast Path)  (Chỉ V61_ML được nuôi
                                     │                                         V61_ADAPTIVE, triệt tiêu
                                     ▼                                         hoàn toàn nguy cơ E7!)
                          V61 LLM Action Judge
                       (Triggered khi score >= 0.5385
                        hoặc prompt có code/intent >= 0.35)
```

### Bảng 5 Ngưỡng Chuẩn Hóa (Thay thế hoàn toàn nhãn T1/T2/T3)

| Tên biến chuẩn hóa mới | File & Vị trí | Giá trị thực tế | Ý nghĩa kiến trúc |
| :--- | :--- | :--- | :--- |
| **`v61_structure_escalation_floor`** | `models/security/v61_inference_router.py:309` | **`0.3500`** | **Fast-to-Slow Heuristic Override:** Prompt có cấu trúc code hoặc động từ mệnh lệnh thì ép từ `ALLOW -> REVIEW`. |
| **`session_judge_band_floor`** | `core/pipeline.py:26` (`suspicious_band_min`) | **`0.4000`** | **Ngưỡng sàn kích hoạt Layer 2B:** Vùng xám dưới review threshold kích hoạt LLM Session Judge. |
| **`v61_model_review_threshold`** | `config/thresholds.json` / `v61_inference_router.py:278` | **`0.5385`** (prompt)<br>**`0.5577`** (action) | **Ngưỡng REVIEW nội bộ V61 ML:** Ranh giới phân định fast-path ALLOW vs slow-path LLM Action Judge. |
| **`session_judge_band_ceil`** | `core/pipeline.py:24` (`_action_review_threshold`) | **`0.5577`** | **Ngưỡng trần kích hoạt Layer 2B:** Trên mức này chuyển giao quyền cho V61 LLM Action Judge. |
| **`v61_adaptive_block_threshold`** | `pipeline.fpr_manager.adaptive_threshold` | **`[0.5800, 0.9500]`** | **Ngưỡng BLOCK trực tiếp của V61 ML:** Bị biến thiên bởi `AdaptiveEscalationManager`. Sàn 0.58, Trần 0.95. |

---

## 1. Các vấn đề cốt lõi đã giải quyết & Báo động khẩn cấp

### 🔴 1.1. Xác nhận dứt điểm quan hệ Adaptive Threshold vs Block Threshold
- **Code review tĩnh xác nhận:** **Chỉ Phát biểu A đúng trong active pipeline**. `adaptive_threshold` trực tiếp gán vào `current_block_t` của V61 ML Model.
- **Phát biểu B là tài liệu lỗi thời:** `VotingAggregator` trong pipeline hiện hành hoàn toàn không nhận `adaptive_threshold`, sử dụng 4 mốc tĩnh trên thang 0-100 (MONITOR=40, REVIEW=68, DENY=82, QUARANTINE=90).

### 🔴 1.2. [BÁO ĐỘNG ĐỎ] Nguy cơ tái diễn vòng lặp đầu độc E7 với Heuristics & Session Judge
- **Hiện trạng nguy hiểm trong code (`core/pipeline.py:389`):**
  ```python
  if decision_layer in ("V61", "Heuristics", "MultiStep-Heuristics", "LLM-Session-Judge"):
      self.fpr_manager.record_decision(is_escalated=bool(is_real_block), layer=decision_layer)
  ```
- **Hậu quả:** Cả `MultiStep-Heuristics` và `LLM-Session-Judge` đều đang trực tiếp nuôi feedback loop của `fpr_manager`. Khi Bước 3 mở rộng BM25, số lượng block từ State-Machine tăng mạnh sẽ lập tức kích hoạt cơ chế tự vệ sai lầm của `fpr_manager`, đẩy `v61_adaptive_block_threshold` lên trần 0.95 $\to$ vô hiệu hóa V61 direct blocking $\to$ ABSR sụp đổ.
- **Quyết định:** **Nâng Bước 2 (Signal Registry Schema) thành HARD BLOCKER.** Bắt buộc cô lập chính sách nguồn trước khi thực hiện Bước 3.

### 🟡 1.3. Chuẩn hóa Signal Registry Schema & Chính sách feeds_escalation_managers
- Mỗi signal/source đăng ký tường minh tập `feeds_escalation_managers: Set[EscalationManagerType]`:
  - `V61_ML` $\to$ `{EscalationManagerType.V61_ADAPTIVE}`
  - `TIER0`, `TIER05_STATE_MACHINE`, `TIER05_LSTM` $\to$ `set()` (tập rỗng)
  - `LLM_SESSION_JUDGE`, `V61_LLM_ACTION_JUDGE` $\to$ `set()` (tập rỗng)
  - `LLM_JUDGE_TIMEOUT_FAILSAFE` $\to$ `set()` (tập rỗng)
  - `EGRESS_KILL_SWITCH` $\to$ `set()` (tập rỗng)

### 🟡 1.4. Đóng vấn đề kiến trúc `re_evaluate_action()` (Mục 1.6)
- **Thực tế kiến trúc:** Ingress `pipeline.scan()` hoạt động đồng bộ, per-action. Một khi đã trả ALLOW thì tool call đã chạy trên môi trường thực tế, không thể "rollback/un-execute".
- **Giải pháp thay thế chuẩn hóa cho Bước 8:**
  1. **Data Interception:** Drop hoặc làm sạch output từ tool khi Sanitizer trả QUARANTINE.
  2. **Taint Session State:** Đánh dấu cờ `EGRESS_QUARANTINE_DETECTED` vào `SessionState` ngay lập tức (dùng lock Bước 0b).
  3. **Emergency Kill-Switch (`EGRESS_KILL_SWITCH`):** Ngắt phiên ngay lập tức đối với payload cực độc, đăng ký source riêng trong Registry với `feeds_escalation_managers = set()`.

---

## 2. Thứ tự xử lý chuẩn hóa (v8)

```
Bước 0a [ĐÃ HOÀN TẤT]: Code review tĩnh xác nhận Phát biểu A đúng, dải [0.58, 0.95], bóc tách 5 ngưỡng.
Bước 0b [ĐÃ HOÀN TẤT]: Vá race condition, lock scope p99 <= 100us, threshold 0.98, rerun benchmark 3000.
   ▼
[HARD BLOCKER - BẮT BUỘC XONG VÀ MERGE TRƯỚC BƯỚC 3]
Bước 2: Thiết kế & Triển khai Signal Registry Schema (models/security/signal_registry.py)
        - Enum EscalationManagerType & SignalSource
        - Policy field feeds_escalation_managers: Set[EscalationManagerType]
        - Gỡ bỏ dòng hardcode "if decision_layer in (...)" tại core/pipeline.py:389
        - Triệt tiêu hoàn toàn nguy cơ tái diễn vòng lặp đầu độc E7
   ▼
Bước 3: Triển khai BM25 coverage (H1) - đăng ký đúng chuẩn Registry
   ▼
Bước 4: Chạy lại test TTL (H2) trên nền tagging mở rộng + calibrate throttle K
   ▼
Bước 5: Chốt quyết định LSTM (A/B test trên nền pipeline đã bảo vệ bởi Registry)
   ▼
Bước 6: Hoàn thiện Signal Registry FULL IMPLEMENTATION (migrate toàn bộ module)
   ▼
Bước 7: Chuẩn hóa hardening dùng chung 2 LLM Judge + đăng ký LLM_JUDGE_TIMEOUT_FAILSAFE
   ▼
Bước 8: Thêm Egress Feedback Loop (Data Interception + Taint Session + EGRESS_KILL_SWITCH)
   ▼
Bước 9: Mở rộng startup sanity-check (bao phủ dải [0.58, 0.95] và 5 ngưỡng mới)
   ▼
Bước 10: Cập nhật E5 Latency Breakdown
   ▼
Bước 11: Unit test xác nhận quan hệ 5 ngưỡng chuẩn hóa
   ▼
Bước 12: Full-scale Benchmark rerun cuối cùng (8 threads song song)
```
