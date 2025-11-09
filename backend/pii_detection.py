# backend/pii_detection.py

import re
from typing import List, Dict, Any

"""
PII detection logic for RegDoc Classifier
-----------------------------------------
Detects personally identifiable information (PII) across document pages.
Adds contextual tags to separate personal vs. business contact info.
"""

# === Regular Expressions for Common PII ===

EMAIL_PATTERN = re.compile(
    r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
)

PHONE_PATTERN = re.compile(
    r"(\+?\d{1,2}\s?)?(\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4})"
)

SSN_PATTERN = re.compile(
    r"\b\d{3}-\d{2}-\d{4}\b"
)

CREDIT_CARD_PATTERN = re.compile(
    r"\b(?:\d[ -]*?){13,16}\b"
)

ADDRESS_PATTERN = re.compile(
    r"\b\d{1,5}\s+[A-Za-z0-9.\s]+\s+(Street|St|Road|Rd|Avenue|Ave|Boulevard|Blvd|Lane|Ln|Drive|Dr|Way|Court|Ct|Place|Pl)\b",
    re.IGNORECASE,
)


# === Helper: detect business vs. personal emails ===

def is_business_email(email: str) -> bool:
    """Heuristically identify non-personal, public-facing email addresses."""
    email = email.lower().strip()
    local_part = email.split("@")[0]
    return any(
        local_part.startswith(prefix)
        for prefix in [
            "info",
            "contact",
            "support",
            "sales",
            "help",
            "team",
            "hello",
            "admin",
            "office",
            "service",
        ]
    )


# === Main Detection Function ===

def find_pii(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Scans all pages and returns structured PII findings.

    Each finding: {
        "type": "email" | "phone" | "ssn" | "credit_card" | "address",
        "value": "...",
        "page": int,
        "is_business": bool   # only for emails
    }
    """

    results: List[Dict[str, Any]] = []

    for page in pages:
        page_num = page.get("page_num", -1)
        text = (page.get("text") or "").strip()

        # === Email ===
        for email in EMAIL_PATTERN.findall(text):
            results.append({
                "type": "email",
                "value": email,
                "page": page_num,
                "is_business": is_business_email(email)
            })

        # === Phone ===
        for phone in PHONE_PATTERN.findall(text):
            # phone is a tuple due to capture groups; join into string
            clean_phone = " ".join([p for p in phone if p]).strip()
            results.append({
                "type": "phone",
                "value": clean_phone,
                "page": page_num
            })

        # === SSN ===
        for ssn in SSN_PATTERN.findall(text):
            results.append({
                "type": "ssn",
                "value": ssn,
                "page": page_num
            })

        # === Credit Card ===
        for cc in CREDIT_CARD_PATTERN.findall(text):
            results.append({
                "type": "credit_card",
                "value": cc.strip(),
                "page": page_num
            })

        # === Address ===
        for addr in ADDRESS_PATTERN.findall(text):
            # Since we used capture groups, re-run search to extract full match
            for match in re.finditer(ADDRESS_PATTERN, text):
                results.append({
                    "type": "address",
                    "value": match.group(0),
                    "page": page_num
                })
            break  # avoid duplicate address per page

    return results

