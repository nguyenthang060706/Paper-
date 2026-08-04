"""
core/config_loader.py
======================
Đọc config/settings.yaml và set các biến môi trường tương ứng, TRƯỚC KHI
bất kỳ module nào khởi tạo V61SecurityRouter (Singleton) hay đọc FIREWALL_MODE.

Vì V61SecurityRouter là Singleton, config chỉ có tác dụng đầy đủ nếu
load_settings() được gọi ở dòng ĐẦU TIÊN của entrypoint (benchmark script,
pipeline test, server main), trước bất kỳ `from core.pipeline import ...`
hay `V61SecurityRouter()` nào.

Không dùng PyYAML để tránh thêm dependency — settings.yaml hiện tại chỉ là
key: value phẳng (không nesting, không list), nên parser tay đơn giản, an toàn
và không phụ thuộc package ngoài.

Nếu sau này settings.yaml cần cấu trúc phức tạp hơn (nesting, list), hãy
thay bằng `import yaml; yaml.safe_load(...)` và cài `pyyaml`.
"""
import os
from pathlib import Path

# Ánh xạ key trong settings.yaml -> tên biến môi trường tương ứng.
# Chỉ set env var nếu nó CHƯA tồn tại (không override nếu người dùng đã
# tự export tay trong shell) — tôn trọng thứ tự ưu tiên: env var thủ công
# > settings.yaml > default hardcode trong code.
_KEY_TO_ENV = {
    "ollama_model": "OLLAMA_MODEL",
    "ollama_timeout": "OLLAMA_TIMEOUT",
    "firewall_mode": "FIREWALL_MODE",
    "ollama_base_url": "OLLAMA_HOST",
    "semantic_delta_threshold": "SEMANTIC_DELTA_THRESHOLD",
    "sequence_risk_gain": "SEQUENCE_RISK_GAIN",
}

_DEFAULT_SETTINGS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "settings.yaml"
)


def _parse_flat_yaml(text: str) -> dict:
    """Parse key: value phẳng, bỏ qua comment (#) và dòng rỗng.
    Tự động strip quotes bao quanh string value ("gemma3:4b" -> gemma3:4b).
    KHÔNG hỗ trợ nesting/list — nếu thấy indent hoặc '-' đầu dòng, raise rõ ràng
    thay vì âm thầm parse sai."""
    result = {}
    for line_no, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if raw_line[:1] in (" ", "\t") or line.strip().startswith("-"):
            raise ValueError(
                f"settings.yaml dòng {line_no}: phát hiện nesting/list. "
                f"Parser phẳng này không hỗ trợ — cần chuyển sang PyYAML thật."
            )
        if ":" not in line:
            raise ValueError(f"settings.yaml dòng {line_no}: thiếu dấu ':' — '{raw_line}'")
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        result[key] = value
    return result


def load_settings(path: str = None, override_existing: bool = False) -> dict:
    """Đọc settings.yaml, set env var tương ứng, trả về dict đã parse để log/debug.

    Args:
        path: đường dẫn tới settings.yaml. Mặc định: <project_root>/config/settings.yaml
        override_existing: nếu True, GHI ĐÈ env var đã tồn tại (dùng khi benchmark
            cần đảm bảo đúng 1 config duy nhất, bất kể shell đang có gì).
            Mặc định False để tôn trọng env var người dùng tự set tay.

    Returns:
        dict các key đã parse được từ file (để in ra log/metadata benchmark).

    Raises:
        FileNotFoundError nếu không tìm thấy settings.yaml — KHÔNG âm thầm dùng
        default rỗng, vì đây chính là loại lỗi "config trôi dạt" đã gặp nhiều lần
        trong dự án này (nhiều bản thresholds.json/advanced_heuristics.py khác nhau).
    """
    path = path or _DEFAULT_SETTINGS_PATH
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"[config_loader] Không tìm thấy {path}. "
            f"Không dùng default ngầm — hãy tạo file này hoặc truyền path đúng."
        )

    parsed = _parse_flat_yaml(p.read_text(encoding="utf-8"))

    applied = {}
    for key, value in parsed.items():
        env_name = _KEY_TO_ENV.get(key)
        if env_name is None:
            # Key lạ, không map được -> cảnh báo thay vì bỏ qua âm thầm
            print(f"[config_loader] WARNING: key '{key}' trong settings.yaml không có mapping env var, bỏ qua.")
            continue
        if override_existing or env_name not in os.environ:
            os.environ[env_name] = value
            applied[env_name] = value
        else:
            applied[env_name] = f"{os.environ[env_name]} (kept existing, ignored file value: {value})"

    print(f"[config_loader] Loaded {path}:")
    for k, v in applied.items():
        print(f"    {k} = {v}")

    return parsed


if __name__ == "__main__":
    # Cho phép chạy `python core/config_loader.py` để debug nhanh xem
    # settings.yaml sẽ áp ra env var gì, không cần chạy cả benchmark.
    load_settings(override_existing=True)
