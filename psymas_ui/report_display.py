"""Presentation-only cleanup; never mutate the original model or audit text."""

import html
import re


def clean_report_display(text: str) -> str:
    cleaned = html.unescape(str(text or ""))
    cleaned = cleaned.replace("\u00a0", " ").replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"[ \t]+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return "\n".join(line.rstrip() for line in cleaned.split("\n")).strip()
