"""Optional Transformers compatibility hooks for model configs shipped by SGLang.

The current slime image has transformers 4.57.1, which does not register the
Qwen3.5 `qwen3_5` config. SGLang already vendors the config classes; importing
its helper registers them with transformers.AutoConfig. This module is safe to
import for non-Qwen3.5 runs because it only extends AutoConfig's registry.
"""


def register_extra_configs() -> None:
    try:  # pragma: no cover - depends on the runtime image.
        import sglang.srt.utils.hf_transformers_utils  # noqa: F401
    except Exception:
        return
