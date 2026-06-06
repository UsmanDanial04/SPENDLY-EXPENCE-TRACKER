"""
preprocessor.py — Parse and clean bank statement CSV files for the expense tracker.

Handles multiple bank CSV formats, normalises descriptions for ML input,
validates amounts and dates, and separates clean rows from failed ones.
"""

import csv
import io
import re
from datetime import datetime
from typing import Any

# ── Constants ────────────────────────────────────────────────────────────────

# Candidate column names, in priority order, for each logical field
DATE_ALIASES: list[str] = [
    "date", "transaction date", "txn date", "value date",
    "posting date", "trans date", "dated",
]

DESCRIPTION_ALIASES: list[str] = [
    "description", "narration", "details", "transaction details",
    "particulars", "memo", "reference", "remarks", "transaction description",
    "trans description", "narrative",
]

AMOUNT_ALIASES: list[str] = [
    "amount", "debit", "credit", "transaction amount", "txn amount",
    "value", "net amount", "withdrawal", "deposit",
]

# Date formats tried in order when parsing
DATE_FORMATS: list[str] = [
    "%Y-%m-%d",   # 2024-03-15
    "%d/%m/%Y",   # 15/03/2024
    "%m/%d/%Y",   # 03/15/2024
    "%d-%m-%Y",   # 15-03-2024
    "%d %b %Y",   # 15 Mar 2024
    "%d %B %Y",   # 15 March 2024
    "%b %d, %Y",  # Mar 15, 2024
    "%Y/%m/%d",   # 2024/03/15
]

# Patterns stripped from descriptions before ML normalisation
_MERCHANT_CODE_RE  = re.compile(r"#\w+")           # #4829, #REF01
_SPECIAL_CHARS_RE  = re.compile(r"[^A-Z0-9 .&'/-]")  # keep safe punctuation
_WHITESPACE_RE     = re.compile(r"\s+")


# ── Description cleaner ──────────────────────────────────────────────────────

def clean_description(text: str) -> str:
    """
    Normalise a single transaction description for ML input.

    Steps applied in order:
    1. Uppercase
    2. Strip merchant/reference codes like ``#4829`` or ``#REF01``
    3. Remove special characters (keep alphanumerics, spaces, and ``.&'/-``)
    4. Collapse runs of whitespace to a single space
    5. Strip leading/trailing whitespace

    Args:
        text: Raw description string from the bank CSV.

    Returns:
        Cleaned, normalised description string.

    Examples:
        >>> clean_description("McDonald's #4829 - Payment OK")
        "MCDONALDS - PAYMENT OK"
        >>> clean_description("UBER * TRIP  eats_98712")
        "UBER  TRIP  EATS98712"
    """
    text = text.upper()
    text = _MERCHANT_CODE_RE.sub("", text)
    text = _SPECIAL_CHARS_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


# ── Internal helpers ─────────────────────────────────────────────────────────

def _normalise_header(name: str) -> str:
    """Lowercase and strip a column header for fuzzy matching."""
    return name.strip().lower()


def _detect_column(headers: list[str], aliases: list[str]) -> str | None:
    """
    Return the first header that matches any alias (case-insensitive).

    Args:
        headers: Column names from the CSV.
        aliases: Ordered list of candidate names to look for.

    Returns:
        The matched header string as it appears in *headers*, or ``None``.
    """
    normalised = {_normalise_header(h): h for h in headers}
    for alias in aliases:
        if alias in normalised:
            return normalised[alias]
    return None


def _parse_date(raw: str) -> datetime | None:
    """
    Try each known date format and return the first successful parse.

    Args:
        raw: Raw date string from the CSV cell.

    Returns:
        Parsed ``datetime`` object, or ``None`` if all formats fail.
    """
    raw = raw.strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


def _parse_amount(raw: str) -> float | None:
    """
    Convert a bank amount string to a float.

    Handles:
    - Currency symbols (``$``, ``£``, ``€``, ``₨``)
    - Thousand separators (``,``)
    - Parentheses for negative values ``(123.45)`` → ``-123.45``
    - Trailing/leading whitespace

    Args:
        raw: Raw amount string from the CSV cell.

    Returns:
        Float value, or ``None`` if conversion fails.
    """
    raw = raw.strip()
    negative = raw.startswith("(") and raw.endswith(")")
    # Strip currency symbols, commas, parentheses
    raw = re.sub(r"[£$€₨,()]", "", raw).strip()
    try:
        value = float(raw)
        return -abs(value) if negative else value
    except ValueError:
        return None


# ── Main parser ──────────────────────────────────────────────────────────────

def parse_csv(
    file_bytes: bytes,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Parse and clean a bank statement CSV supplied as raw bytes.

    The function auto-detects column names, cleans descriptions, validates
    dates and amounts, and separates rows that pass validation from those
    that do not.

    Args:
        file_bytes: Raw bytes of the uploaded CSV file (UTF-8 or latin-1).

    Returns:
        A two-element tuple ``(transactions, failures)`` where:

        * **transactions** – list of dicts with keys:
          ``date`` (``datetime``), ``description`` (``str``),
          ``amount`` (``float``)
        * **failures** – list of dicts with keys:
          ``row`` (original row dict), ``row_number`` (1-based, header = 0),
          ``error`` (human-readable reason string)

    Raises:
        ValueError: If required columns cannot be detected or the file is empty.
    """
    # Decode bytes — try UTF-8 first, fall back to latin-1
    try:
        text = file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        text = file_bytes.decode("latin-1")

    reader = csv.DictReader(io.StringIO(text))

    if reader.fieldnames is None:
        raise ValueError("CSV file appears to be empty or has no header row.")

    headers: list[str] = list(reader.fieldnames)

    # ── Column detection ────────────────────────────────────────────────────
    date_col   = _detect_column(headers, DATE_ALIASES)
    desc_col   = _detect_column(headers, DESCRIPTION_ALIASES)
    amount_col = _detect_column(headers, AMOUNT_ALIASES)

    missing = [
        label
        for label, col in [("date", date_col), ("description", desc_col), ("amount", amount_col)]
        if col is None
    ]
    if missing:
        raise ValueError(
            f"Could not detect columns for: {', '.join(missing)}. "
            f"Found headers: {headers}"
        )

    transactions: list[dict[str, Any]] = []
    failures:     list[dict[str, Any]] = []

    for row_number, row in enumerate(reader, start=2):  # row 1 = header
        errors: list[str] = []

        # ── Date ───────────────────────────────────────────────────────────
        raw_date = (row.get(date_col) or "").strip()
        parsed_date = _parse_date(raw_date) if raw_date else None
        if not raw_date:
            errors.append("date is empty")
        elif parsed_date is None:
            errors.append(f"unparseable date {raw_date!r}")

        # ── Amount ─────────────────────────────────────────────────────────
        raw_amount = (row.get(amount_col) or "").strip()
        parsed_amount = _parse_amount(raw_amount) if raw_amount else None
        if not raw_amount:
            errors.append("amount is empty")
        elif parsed_amount is None:
            errors.append(f"non-numeric amount {raw_amount!r}")

        # ── Description ────────────────────────────────────────────────────
        raw_desc = (row.get(desc_col) or "").strip()
        if not raw_desc:
            errors.append("description is empty")
        cleaned_desc = clean_description(raw_desc) if raw_desc else ""

        # ── Route row ──────────────────────────────────────────────────────
        if errors:
            failures.append({
                "row":        dict(row),
                "row_number": row_number,
                "error":      "; ".join(errors),
            })
        else:
            transactions.append({
                "date":        parsed_date,
                "description": cleaned_desc,
                "amount":      parsed_amount,
            })

    return transactions, failures


# ── __main__ smoke-test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    SAMPLE_CSV = """\
Transaction Date,Narration,Amount
15/03/2024,McDONALD'S #4829 Payment,12.50
2024-03-16,UBER * TRIP  eats_98712,8.75
17/03/2024,AMAZON MKTPLACE  PMT,34.99
18/03/2024,SHELL PETROL STN #112,55.00
19/03/2024,NETFLIX.COM,15.99
20/03/2024,STARBUCKS STORE 0042  ,6.80
21/03/2024,SPOTIFY AB,9.99
22/03/2024,,12.00
23/03/2024,TESCO SUPERSTORE,not-a-number
24/03/2024,WATER UTILITY PMT,(87.40)
""".encode("utf-8")

    print("=" * 56)
    print("  preprocessor.py — smoke test")
    print("=" * 56)

    txns, fails = parse_csv(SAMPLE_CSV)

    print(f"\n✓  Clean transactions ({len(txns)}):\n")
    for t in txns:
        print(
            f"  [{t['date'].strftime('%Y-%m-%d')}]  "
            f"{t['description']:<36}  £{t['amount']:>8.2f}"
        )

    print(f"\n✗  Failed rows ({len(fails)}):\n")
    for f in fails:
        print(f"  Row {f['row_number']:>2}: {f['error']}")
        print(f"         raw → {f['row']}")

    print("\n── clean_description examples ──────────────────────────")
    samples = [
        "McDONALD'S #4829 - Payment",
        "UBER * TRIP  eats_98712",
        "SHELL PETROL STN #112",
        "AMAZON.COM*AB12CD3EF  Seattle WA",
        "  multiple   spaces   here  ",
    ]
    for s in samples:
        print(f"  {s!r}")
        print(f"    → {clean_description(s)!r}\n")

    print("=" * 56)
