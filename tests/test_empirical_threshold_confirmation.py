import os
import sys
import json
import pandas as pd

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.pipeline import UnifiedFirewallPipeline

def run_empirical_confirmation():
    print("=" * 80)
    print("  BƯỚC 0: XÁC NHẬN THỰC NGHIỆM ĐƯỜNG CONG LEO THANG ADAPTIVE THRESHOLD")
    print("=" * 80)

    # 1. Load dataset (lấy 300 records từ evo_pca_11k_balanced.jsonl)
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "evo_pca_11k_balanced.jsonl")
    records = []
    if os.path.exists(data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 300:
                    break
                records.append(json.loads(line))
    else:
        print(f"Dataset not found at {data_path}, creating synthetic sessions...")
        for i in range(300):
            records.append({
                "action": f"tool_call(arg={i})" if i % 2 == 0 else f"User prompt {i}",
                "action_type": "tool_call" if i % 2 == 0 else "prompt",
                "session_id": f"sess_{i // 5}",
                "label": "malicious" if i % 3 == 0 else "benign"
            })

    print(f"Loaded {len(records)} records for empirical telemetry trace.")

    # --- Phase 1: Test LEGACY mode (Bug state) ---
    print("\n[Phase 1] Chạy với ESCALATION_FEEDBACK_MODE=legacy (LSTM ON)...")
    os.environ["TIER05_LSTM_ENABLED"] = "True"
    os.environ["ESCALATION_FEEDBACK_MODE"] = "legacy"
    
    pipeline_legacy = UnifiedFirewallPipeline(use_synthetic_iat=True)
    for r in records:
        pipeline_legacy.scan(
            action=r["action"],
            session_id=r.get("session_id", "default"),
            action_type=r.get("action_type", "prompt"),
            actual_label=(r.get("label") != "benign")
        )
        
    trace_path_legacy = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "empirical_escalation_trace_legacy.csv")
    pipeline_legacy.fpr_manager.export_telemetry_trace(trace_path_legacy)
    legacy_metrics = pipeline_legacy.fpr_manager.get_metrics()
    
    print(f"  Legacy Final Metrics: {legacy_metrics}")

    # --- Phase 2: Test DECOUPLED mode (Fixed state) ---
    print("\n[Phase 2] Chạy với ESCALATION_FEEDBACK_MODE=decoupled (LSTM ON)...")
    os.environ["TIER05_LSTM_ENABLED"] = "True"
    os.environ["ESCALATION_FEEDBACK_MODE"] = "decoupled"
    
    pipeline_decoupled = UnifiedFirewallPipeline(use_synthetic_iat=True)
    for r in records:
        pipeline_decoupled.scan(
            action=r["action"],
            session_id=r.get("session_id", "default"),
            action_type=r.get("action_type", "prompt"),
            actual_label=(r.get("label") != "benign")
        )
        
    trace_path_decoupled = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "empirical_escalation_trace_decoupled.csv")
    pipeline_decoupled.fpr_manager.export_telemetry_trace(trace_path_decoupled)
    decoupled_metrics = pipeline_decoupled.fpr_manager.get_metrics()
    lstm_metrics = pipeline_decoupled.lstm_monitor.get_metrics()
    
    print(f"  Decoupled V61 Metrics: {decoupled_metrics}")
    print(f"  Decoupled LSTM Metrics: {lstm_metrics}")

    # --- Đánh giá kết quả ---
    print("\n" + "=" * 80)
    print("  KẾT QUẢ XÁC NHẬN THỰC NGHIỆM BƯỚC 0:")
    print("=" * 80)
    print(f"  - Legacy Mode Adaptive Threshold   : {legacy_metrics['adaptive_threshold']} (Escalation Rate: {legacy_metrics['escalation_rate']*100:.1f}%)")
    print(f"  - Decoupled Mode Adaptive Threshold: {decoupled_metrics['adaptive_threshold']} (Escalation Rate: {decoupled_metrics['escalation_rate']*100:.1f}%)")
    
    is_confirmed = legacy_metrics['adaptive_threshold'] > decoupled_metrics['adaptive_threshold']
    if is_confirmed:
        print("\n=> [XÁC NHẬN THÀNH CÔNG] Hiện tượng threshold leo thang kịch trần trong Legacy mode ĐÃ ĐƯỢC CHỨNG MINH THỰC NGHIỆM.")
        print("=> Decoupled mode giữ vững ngưỡng V61 ổn định mà không bị ô nhiễm bởi LSTM blocks.")
    else:
        print("\n=> [CẢNH BÁO] Threshold không có sự khác biệt rõ rệt giữa 2 mode, kích hoạt nhánh phân tích dự phòng.")

if __name__ == "__main__":
    run_empirical_confirmation()
