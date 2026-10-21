"""Text helpers shared by caching, retrieval and evaluation (English + Persian)."""
from __future__ import annotations

import re
import unicodedata

_CHAR_FIX = str.maketrans({"ي": "ی", "ك": "ک", "ۀ": "ه", "ة": "ه", "أ": "ا", "إ": "ا", "ؤ": "و"})
_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
_DIACRITICS = re.compile("[ً-ٰٟـ]")  # harakat + tatweel
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?؟؛。])\s+|\n+")
_HEADING = re.compile(r"^=+\s*(.*?)\s*=+$")


def normalize(text: str) -> str:
    """Lowercase, unify Persian/Arabic letters and digits, drop punctuation."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.translate(_CHAR_FIX).translate(_DIGITS)
    text = _DIACRITICS.sub("", text)
    text = text.replace("‌", " ").replace("‍", "")  # ZWNJ -> space
    out = []
    for ch in text:
        cat = unicodedata.category(ch)
        out.append(ch if cat[0] in "LN" or cat == "Mn" else " ")
    return " ".join("".join(out).split())


def tokenize(text: str) -> list[str]:
    return normalize(text).split()


def detect_lang(text: str) -> str:
    """'fa' if most letters are Arabic-script, else 'en'."""
    arabic = sum(1 for c in text if "؀" <= c <= "ۿ")
    latin = sum(1 for c in text if c.isascii() and c.isalpha())
    return "fa" if arabic > latin else "en"


def split_sentences(text: str) -> list[str]:
    """Split Wikipedia-style text into sentences; turns '== Heading ==' into plain lines."""
    sentences: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        m = _HEADING.match(line)
        if m:
            line = m.group(1)
        if len(line) < 3:
            continue
        sentences.extend(s.strip() for s in _SENTENCE_SPLIT.split(line) if len(s.strip()) >= 3)
    return sentences
