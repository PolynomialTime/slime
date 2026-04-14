import unicodedata
from collections import Counter


_ALLOWED_CONTROL_WHITESPACE = {"\n", "\r", "\t"}


def is_non_printing_char(ch: str) -> bool:
    if ch in _ALLOWED_CONTROL_WHITESPACE:
        return False
    return unicodedata.category(ch).startswith("C")


def count_non_printing_chars(text: str) -> int:
    return sum(1 for ch in text or "" if is_non_printing_char(ch))


def strip_non_printing_chars(text: str) -> str:
    if not text:
        return text
    return "".join(ch for ch in text if not is_non_printing_char(ch))


def non_printing_char_ratio(text: str) -> float:
    if not text:
        return 0.0
    return count_non_printing_chars(text) / max(len(text), 1)


def summarize_non_printing_chars(text: str, top_k: int = 5) -> dict:
    counts = Counter(ch for ch in text or "" if is_non_printing_char(ch))
    top = [
        {
            "codepoint": f"U+{ord(ch):04X}",
            "count": count,
            "category": unicodedata.category(ch),
        }
        for ch, count in counts.most_common(top_k)
    ]
    total = sum(counts.values())
    length = len(text or "")
    return {
        "non_printing_chars": total,
        "text_length": length,
        "ratio": (total / max(length, 1)) if length else 0.0,
        "top": top,
    }
