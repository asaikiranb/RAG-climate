"""
WhatsApp message formatter.

Converts raw LLM output (which contains Markdown + source citations)
into WhatsApp-compatible plain text.

WhatsApp markdown dialect:
  *text*        → bold
  _text_        → italic
  ~text~        → strikethrough
  ```text```    → monospace
  1. 2. 3.      → numbered list (rendered natively)
  Bullets       → WhatsApp does NOT render `-` or `*` as bullets;
                  we use the unicode bullet character • instead.
"""

import re


def _strip_citations(text: str) -> str:
    """Remove source citation markers like [1], [2][3], [1, 2], etc."""
    # Matches [1], [2][3], [1][2][3], [1, 2, 3]
    text = re.sub(r'(\[\d+(?:,\s*\d+)*\])+', '', text)
    return text


def _strip_html(text: str) -> str:
    """Remove any residual HTML tags."""
    return re.sub(r'<[^>]+>', '', text)


def _convert_bullets(text: str) -> str:
    """
    Convert markdown bullets (- item or * item at line start)
    to unicode bullet • which WhatsApp renders clearly.
    """
    text = re.sub(r'^\s*[-*]\s+', '• ', text, flags=re.MULTILINE)
    return text


def _convert_bold(text: str) -> str:
    """
    Convert **text** (double asterisk markdown bold) to *text*
    (single asterisk WhatsApp bold).
    Leaves already-single-asterisk bold untouched.
    """
    text = re.sub(r'\*\*(.+?)\*\*', r'*\1*', text)
    return text


def _clean_whitespace(text: str) -> str:
    """Collapse 3+ consecutive blank lines into 2, strip trailing spaces."""
    lines = [line.rstrip() for line in text.splitlines()]
    result = []
    blank_count = 0
    for line in lines:
        if line == '':
            blank_count += 1
            if blank_count <= 1:       # allow at most one blank line between blocks
                result.append(line)
        else:
            blank_count = 0
            result.append(line)
    return '\n'.join(result).strip()


def format_for_whatsapp(raw_answer: str) -> str:
    """
    Full formatting pipeline for a raw LLM answer → WhatsApp message.

    Steps (order matters):
      1. Strip HTML tags
      2. Strip source citations [1][2]
      3. Convert **bold** → *bold*
      4. Convert bullet markers - / * → •
      5. Clean up whitespace
    """
    text = _strip_html(raw_answer)
    text = _strip_citations(text)
    text = _convert_bold(text)
    text = _convert_bullets(text)
    text = _clean_whitespace(text)
    return text
