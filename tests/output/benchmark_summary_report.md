# Báo Cáo Kết Quả Benchmark EVO-PCA Dual Shield

**Mô hình đánh giá**: `EVO-PCA Dual Shield (Tier0/0.5 + V61 + Tier0.5-LSTM)`  
**Tập dữ liệu**: `Ablation Dataset` (v61/v64 action risk model + AgentDojo scenarios)  
**Thời gian thực thi**: `04/08/2026`

---

## 1. Tổng Quan Chỉ Số Hiệu Năng (Performance Metrics)

| Chỉ số (Metric) | Giá trị (Value) | Đánh giá & Mục tiêu |
|---|---|---|
| **FPR (%)** *(False Positive Rate)* | **6.11%** | Tỷ lệ báo động nhầm tổng thể |
| **FPR Steady-State (%)** | **4.07%** | Tỷ lệ báo nhầm sau giai đoạn warm-up (Đạt mục tiêu $\le 5\%$) |
| **ABSR Total (%)** *(Attack Block Success Rate)* | **9.57%** | Tỷ lệ chặn thành công tổng thể tất cả các bước |
| **ABSR Single-step (%)** | **9.18%** | Tỷ lệ chặn các tấn công đơn lẻ (Prompt injection / RCE 1 bước) |
| **ABSR Multi-step Action (%)** | **23.46%** | Tỷ lệ chặn theo từng hành động trong chuỗi đa bước |
| **ABSR Multi-step Session (any-step)** | **33.10%** | **Tỷ lệ chặn thành công ít nhất 1 bước trong session** |
| **ABSR Multi-step Session (step1-only)** | **33.10%** | Tỷ lệ chặn ngay từ bước đầu tiên của session |
| **Độ trễ trung bình (Avg Latency)** | **224.44 ms** | Độ trễ xử lý mỗi hành động |

---

## 2. Phân Bổ Số Lượng Chặn Theo Tầng Security (Block Distribution by Layer)

| Tầng bảo vệ (Layer) | Số lượng hành động bị chặn | Tỷ lệ trong tổng số chặn | Ghi chú |
|---|---|---|---|
| **Tier 0 (Stateless Regex Rules)** | **699** | **40.1%** | Chặn các từ khóa injection / RCE cứng |
| **Heuristics (Advanced Heuristics)** | **689** | **39.6%** | Phát hiện theo rule ngữ cảnh / chuỗi hành động |
| **V61 (ML Action Risk Model)** | **354** | **20.3%** | Dự đoán rủi ro dựa trên mô hình XGBoost/RandomForest |
| **Tier 0.5-LSTM (Temporal Risk)** | **0** | **0.0%** | *Lưu ý: IAT bị trung hòa (fixed ~5s) do dataset không có timestamp thực* |
| **Tổng số lượt chặn** | **1,742** | **100.0%** | |

---

## 3. Phân Tích Chi Tiết & Nhận Xét

1. **Khả năng kiểm soát Báo động giả (FPR Control)**:
   - **FPR Steady-State (4.07%)** nằm trong giới hạn cho phép ($\le 5\%$). Điều này bảo đảm mô hình ít gây gián đoạn công việc của người dùng hợp lệ.

2. **Khả năng chặn tấn công theo Session (Session-level ABSR)**:
   - Ở cấp độ Session đa bước (**Multi-step Session**), hệ thống đạt tỷ lệ phát hiện **33.10%**, và đặc biệt là chặn ngay từ **bước 1 (Step 1)** đạt **33.10%**, giúp ngăn chặn cuộc tấn công từ sớm trước khi xảy ra thiệt hại.

3. **Ghi chú về tầng LSTM (Tier 0.5-LSTM)**:
   - Tầng LSTM đạt 0 block trong bài test này do dữ liệu `AgentDojo / Ablation Dataset` không chứa thông tin thời gian thực (`Inter-Arrival Time - IAT`), khiến tính năng phát hiện bất thường về tốc độ thao tác bị vô hiệu hóa tạm thời.

---

## 4. Các File Kết Quả Đã Được Xuất

- 📄 **Báo cáo tổng hợp (CSV)**: [benchmark_results.csv](file:///c:/Users/ADMIN/jupyternotebook/EVO_PCA/tests/output/benchmark_results.csv)
- 📄 **Báo cáo dạng JSON**: [benchmark_results.json](file:///c:/Users/ADMIN/jupyternotebook/EVO_PCA/tests/output/benchmark_results.json)
- 📄 **Báo cáo chi tiết mẫu FP/FN**: [fp_fn_report_EVO-PCA_Dual_Shield_(Tier0_0.5_+_V61_+_Tier0.5-LSTM).txt](file:///c:/Users/ADMIN/jupyternotebook/EVO_PCA/fp_fn_report_EVO-PCA_Dual_Shield_%28Tier0_0.5_%2B_V61_%2B_Tier0.5-LSTM%29.txt)
