"""
tier05_standalone_benchmark.py
================================
Benchmark độc lập cho SessionAwareTier05 (core/tier05.py) — KHÔNG dùng bất kỳ
ML model nào (PromptRiskModel / ActionRiskModel bị loại bỏ hoàn toàn khỏi vòng lặp).

Mục đích: cô lập lỗi Multi-step ABSR = 0% để xác định nguyên nhân nằm ở:
  - Tầng 1: Regex không extract được flag nào từ action text.
  - Tầng 2: Flag có extract nhưng không đủ để match combo nguy hiểm nào.
  - Tầng 3: Combo có match nhưng bị kẹt ở MONITOR do ESCALATION_THRESHOLD.

Đồng thời đo FPR trên toàn bộ session benign để làm cơ sở quyết định
Hướng A (hạ ESCALATION_THRESHOLD) hay Hướng B (nâng risk_score gốc của combo).

Cách chạy:
    python tier05_standalone_benchmark.py --input /path/to/evo_pca_full.jsonl

Nếu không truyền --input, script sẽ thử đường dẫn mặc định bên dưới rồi báo lỗi
rõ ràng nếu không tìm thấy (không âm thầm dùng dữ liệu rỗng).
"""

import argparse
import json
import sys
import os
from collections import defaultdict, Counter
from pathlib import Path

# --- Import Tier 0.5 thật từ project, KHÔNG mock/fallback ---
# Nếu import lỗi, dừng ngay và báo rõ, tránh benchmark "chạy được" trên bản fallback rỗng.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from core.tier05 import SessionAwareTier05, Tier05Config
except ImportError as e:
    print(f"[FATAL] Không import được core.tier05.SessionAwareTier05: {e}")
    print("        Hãy chạy script này từ thư mục gốc của project (nơi có thư mục core/).")
    sys.exit(1)

DEFAULT_INPUT_PATH = r"D:\DEMO_GROUP_1\Benchmark_Datasets\output\evo_pca_full.jsonl"
ESCALATION_THRESHOLD = Tier05Config.ESCALATION_THRESHOLD  # đọc trực tiếp từ config thật, không hardcode lại


def load_dataset(path: str) -> list:
    p = Path(path)
    if not p.exists():
        print(f"[FATAL] Không tìm thấy file dataset: {path}")
        sys.exit(1)
    records = []
    with open(p, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] Bỏ qua dòng {line_no} lỗi JSON: {e}")
    return records


def group_and_sort_by_session(records: list) -> dict:
    """Gom theo session_id, sort theo step_num tăng dần trong mỗi session.
    Không tin thứ tự sẵn có trong file — luôn sort tường minh để tránh
    tier05.action_count bị lệch nếu JSONL bị ghi/đọc không đúng thứ tự."""
    sessions = defaultdict(list)
    for r in records:
        sid = r.get("session_id")
        if sid is None:
            continue
        sessions[sid].append(r)
    for sid in sessions:
        sessions[sid].sort(key=lambda r: r.get("step_num", 0))
    return sessions


def scan_session(tier05: SessionAwareTier05, session_id: str, steps: list) -> dict:
    """Chạy toàn bộ các bước của 1 session qua tier05.scan(), trả về thống kê chi tiết
    theo 3 tầng thất bại. Mỗi session_id trong dataset đã là unique (kiểm tra ở main),
    nên không cần reset_session() thủ công — nhưng vẫn gọi để an toàn tuyệt đối nếu
    script bị chạy lại / session_id trùng do augment dữ liệu sau này."""
    tier05.reset_session(session_id)

    any_flag = False
    any_combo = False
    ever_blocked = False
    max_trigger_per_combo = Counter()   # combo_description -> max trigger_count đạt được trong session này
    decisions = []                      # lịch sử quyết định từng bước, để debug thủ công khi cần

    for step in steps:
        action = step.get("action", "")
        result = tier05.scan(action, session_id=session_id)
        decisions.append({
            "step_num": step.get("step_num"),
            "decision": result.decision.value if hasattr(result.decision, "value") else str(result.decision),
            "is_blocked": result.is_blocked,
            "rule_fired": result.rule_fired,
            "confidence": result.confidence,
        })
        if result.is_blocked:
            ever_blocked = True

    # Đọc trạng thái tích lũy cuối cùng của session để biết flag/combo nào đã từng fire
    report = tier05.get_session_report(session_id)
    if report.get("active_flags"):
        any_flag = True
    combo_counts = report.get("combo_trigger_counts", {})
    if combo_counts:
        any_combo = True
        for desc, cnt in combo_counts.items():
            max_trigger_per_combo[desc] = max(max_trigger_per_combo[desc], cnt)

    return {
        "any_flag": any_flag,
        "any_combo": any_combo,
        "ever_blocked": ever_blocked,
        "max_trigger_per_combo": dict(max_trigger_per_combo),
        "decisions": decisions,
        "active_flags": report.get("active_flags", []),
    }


def run_group(tier05: SessionAwareTier05, sessions: dict, group_name: str) -> dict:
    total = len(sessions)
    sessions_with_any_flag = 0
    sessions_with_any_combo = 0
    sessions_capped_at_monitor = 0   # any_combo=True nhưng ever_blocked=False
    sessions_blocked = 0
    sessions_near_escalation = 0     # có combo với trigger_count == THRESHOLD - 1 (chỉ có ý nghĩa cho benign)
    combo_fire_histogram = defaultdict(list)  # combo_desc -> list các max_trigger_count trên từng session

    per_session_detail = {}

    for sid, steps in sessions.items():
        res = scan_session(tier05, sid, steps)
        per_session_detail[sid] = res

        if res["any_flag"]:
            sessions_with_any_flag += 1
        if res["any_combo"]:
            sessions_with_any_combo += 1
        if res["ever_blocked"]:
            sessions_blocked += 1
        elif res["any_combo"]:
            # Có combo match nhưng session KHÔNG hề bị block -> đúng dấu hiệu "kẹt ở MONITOR"
            sessions_capped_at_monitor += 1

        near_escalation_this_session = False
        for desc, cnt in res["max_trigger_per_combo"].items():
            combo_fire_histogram[desc].append(cnt)
            if cnt == ESCALATION_THRESHOLD - 1:
                near_escalation_this_session = True
        if near_escalation_this_session:
            sessions_near_escalation += 1

    return {
        "group": group_name,
        "total_sessions": total,
        "sessions_with_any_flag": sessions_with_any_flag,
        "sessions_with_any_combo": sessions_with_any_combo,
        "sessions_capped_at_monitor": sessions_capped_at_monitor,
        "sessions_blocked": sessions_blocked,
        "sessions_near_escalation (trigger_count == threshold-1)": sessions_near_escalation,
        "combo_fire_histogram": {
            desc: dict(Counter(counts)) for desc, counts in combo_fire_histogram.items()
        },
        "_detail": per_session_detail,  # giữ lại để debug, không in ra report tóm tắt
    }


def print_summary(result: dict):
    g = result["group"]
    t = result["total_sessions"]
    print(f"\n{'=' * 70}")
    print(f" NHÓM: {g}  (tổng {t} session)")
    print(f"{'=' * 70}")
    if t == 0:
        print("  [WARN] Không có session nào trong nhóm này — kiểm tra lại filter attack_type/label.")
        return

    def pct(n):
        return f"{n} ({100 * n / t:.1f}%)"

    print(f"  Tầng 1 - Có flag nào được extract:     {pct(result['sessions_with_any_flag'])}")
    print(f"  Tầng 2 - Có combo nào được match:      {pct(result['sessions_with_any_combo'])}")
    print(f"  Tầng 3 - Kẹt ở MONITOR (combo match")
    print(f"           nhưng KHÔNG BAO GIỜ block):    {pct(result['sessions_capped_at_monitor'])}")
    print(f"  Thực sự bị BLOCK ít nhất 1 lần:         {pct(result['sessions_blocked'])}")
    near_key = "sessions_near_escalation (trigger_count == threshold-1)"
    print(f"  Gần escalation (trigger_count == {ESCALATION_THRESHOLD - 1}):    {pct(result[near_key])}")

    print(f"\n  Phân phối trigger_count theo từng combo:")
    if not result["combo_fire_histogram"]:
        print("    (không có combo nào fire trong nhóm này)")
    for desc, hist in result["combo_fire_histogram"].items():
        hist_str = ", ".join(f"{k}x{v}" for k, v in sorted(hist.items()))
        print(f"    - {desc[:60]}")
        print(f"        trigger_count phân phối: {{{hist_str}}}  (đơn vị: giá_trị x số_session)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH,
                         help="Đường dẫn tới evo_pca_full.jsonl")
    parser.add_argument("--output-json", default=None,
                         help="Nếu truyền, ghi report đầy đủ (kèm _detail per-session) ra file JSON này")
    args = parser.parse_args()

    print(f"[INFO] Đọc dataset từ: {args.input}")
    records = load_dataset(args.input)
    print(f"[INFO] Tổng số record (mỗi record = 1 action/step): {len(records)}")

    # --- Filter 2 nhóm theo đúng field thật trong dataset ---
    multistep_records = [r for r in records if r.get("attack_type") == "malicious_multistep"]
    benign_records = [r for r in records if r.get("label") == "benign"]

    multistep_sessions = group_and_sort_by_session(multistep_records)
    benign_sessions = group_and_sort_by_session(benign_records)

    # Kiểm tra unique session_id giữa 2 nhóm (đề phòng dataset lỗi gán trùng id)
    overlap = set(multistep_sessions) & set(benign_sessions)
    if overlap:
        print(f"[WARN] Phát hiện {len(overlap)} session_id trùng giữa 2 nhóm malicious/benign — "
              f"đây là lỗi dữ liệu, cần kiểm tra lại dataset builder. Ví dụ: {list(overlap)[:3]}")

    print(f"[INFO] Số session malicious_multistep: {len(multistep_sessions)}")
    print(f"[INFO] Số session benign:              {len(benign_sessions)}")

    # Mỗi nhóm dùng 1 instance tier05 riêng để tránh flag rò rỉ chéo giữa 2 nhóm
    tier05_malicious = SessionAwareTier05()
    tier05_benign = SessionAwareTier05()

    result_malicious = run_group(tier05_malicious, multistep_sessions, "malicious_multistep")
    result_benign = run_group(tier05_benign, benign_sessions, "benign_all")

    print_summary(result_malicious)
    print_summary(result_benign)

    # --- Kết luận tự động (gợi ý, không thay quyết định của Security Engineer) ---
    print(f"\n{'=' * 70}")
    print(" GỢI Ý CHẨN ĐOÁN (dựa trên số liệu trên, cần review thủ công trước khi áp dụng)")
    print(f"{'=' * 70}")
    t = result_malicious["total_sessions"]
    if t > 0:
        flag_rate = result_malicious["sessions_with_any_flag"] / t
        combo_rate = result_malicious["sessions_with_any_combo"] / t
        capped_rate = result_malicious["sessions_capped_at_monitor"] / t
        if flag_rate < 0.3:
            print(" -> Flag extraction rate THẤP: nghi vấn regex trong _FLAG_PATTERNS không khớp"
                  " với văn phong/cú pháp thật của AgentDojo. Cần xem lại regex trước khi đụng vào"
                  " ESCALATION_THRESHOLD — sửa threshold sẽ không giúp gì nếu flag còn chưa fire được.")
        elif combo_rate < flag_rate * 0.5:
            print(" -> Flag fire được nhưng combo hiếm khi match: nghi vấn _DANGEROUS_COMBOS chưa"
                  " bao phủ đúng pattern thật (VD 2 flag cần thiết ít khi cùng xuất hiện trong 1 session).")
        elif capped_rate > 0.3:
            print(" -> XÁC NHẬN giả thuyết ESCALATION_THRESHOLD: nhiều session có combo match nhưng"
                  " không bao giờ đạt BLOCK. Đây là bằng chứng data-driven ủng hộ Hướng A/B đã đề xuất.")
        else:
            print(" -> Không thấy dấu hiệu rõ ràng nào ở 3 tầng trên — cần xem lại _detail per-session"
                  " để debug thủ công từng case.")

    if args.output_json:
        # Loại bỏ _detail khi ghi ra để file JSON gọn, trừ khi cần debug sâu thì tự bật lại
        out = {
            "malicious_multistep": {k: v for k, v in result_malicious.items() if k != "_detail"},
            "benign_all": {k: v for k, v in result_benign.items() if k != "_detail"},
            "escalation_threshold_used": ESCALATION_THRESHOLD,
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\n[INFO] Đã ghi report ra: {args.output_json}")


if __name__ == "__main__":
    main()
