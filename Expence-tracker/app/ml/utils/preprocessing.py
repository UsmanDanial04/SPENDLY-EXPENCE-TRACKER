import re, string

_CURRENCY_RE  = re.compile(r"[\$£€¥₹]\s*\d[\d,\.]*|\d[\d,\.]*\s*[\$£€¥₹]")
_NUMBER_RE    = re.compile(r"\b\d+[\d,\.]*\b")
_WHITESPACE_RE = re.compile(r"\s+")

def clean_description(text: str) -> str:
    text = text.lower()
    text = _CURRENCY_RE.sub(" ", text)
    text = _NUMBER_RE.sub(" ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text

def build_feature_text(description: str, notes: str = "") -> str:
    parts = [clean_description(description)]
    if notes:
        parts.append(clean_description(notes))
    return " ".join(parts)
