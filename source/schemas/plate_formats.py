"""Regional license plate syntax rules, Bangladesh BRTA format registry, and character disambiguation."""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

# Mapping of Bangla numerals to Arabic numerals
BANGLA_TO_ARABIC_DIGITS = {
    "০": "0",
    "১": "1",
    "২": "2",
    "৩": "3",
    "৪": "4",
    "৫": "5",
    "৬": "6",
    "৭": "7",
    "৮": "8",
    "৯": "9",
}

ARABIC_TO_BANGLA_DIGITS = {v: k for k, v in BANGLA_TO_ARABIC_DIGITS.items()}

# Common OCR confusion matrix
CHAR_TO_DIGIT_MAP = {
    "O": "0", "o": "0", "Q": "0", "D": "0",
    "I": "1", "l": "1", "|": "1", "L": "1", "i": "1",
    "Z": "2", "z": "2",
    "A": "4",
    "S": "5", "s": "5",
    "G": "6", "b": "6",
    "T": "7",
    "B": "8",
}

DIGIT_TO_CHAR_MAP = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "4": "A",
    "5": "S",
    "6": "G",
    "8": "B",
}

# Regional regex patterns
PLATE_FORMAT_REGISTRY = {
    # 1. Bangladesh BRTA Formats
    "BD_METRO_DHAKA_BANGLA": {
        "pattern": r"ঢাকা\s*মেট্রো\s*[-–—]?\s*([ক-হ])\s*([০-৯]{2})[-–—]?([০-৯]{4})",
        "description": "Dhaka Metro Bangla Plate",
        "script": "bangla",
        "region": "Dhaka Metro",
    },
    "BD_METRO_CTG_BANGLA": {
        "pattern": r"চট্টগ্রাম\s*মেট্রো\s*[-–—]?\s*([ক-হ])\s*([০-৯]{2})[-–—]?([০-৯]{4})",
        "description": "Chittagong Metro Bangla Plate",
        "script": "bangla",
        "region": "Chittagong Metro",
    },
    "BD_DISTRICT_BANGLA": {
        "pattern": r"([ক-হ]{2,15})\s*[-–—]?\s*([ক-হ])\s*([০-৯]{2})[-–—]?([০-৯]{4})",
        "description": "District Bangla Plate",
        "script": "bangla",
        "region": "District",
    },
    "BD_DIGITAL_METRO": {
        "pattern": r"(DHAKA|CTG|CHITTAGONG|SYLHET|RAJSHAHI|KHULNA|BARISHAL|RANGPUR)\s*METRO\s*[-–—]?\s*([A-Z]{1,2})\s*[-–—]?\s*(\d{2})[-–—]?(\d{4})",
        "description": "Bangladesh Digital Metro Plate (English)",
        "script": "latin",
        "region": "Digital Metro",
    },
    "BD_DIGITAL_DISTRICT": {
        "pattern": r"([A-Z]{4,15})\s*[-–—]?\s*([A-Z]{1,2})\s*[-–—]?\s*(\d{2})[-–—]?(\d{4})",
        "description": "Bangladesh Digital District Plate",
        "script": "latin",
        "region": "Digital District",
    },

    # 2. North America / USA Formats
    "US_STANDARD_7CHAR": {
        "pattern": r"\b(\d[A-Z]{3}\d{3})\b",
        "description": "US Standard Plate (7-character, e.g. California)",
        "script": "latin",
        "region": "USA / California",
    },
    "US_STANDARD_6CHAR": {
        "pattern": r"\b([A-Z]{3}[-\s]?\d{3,4})\b",
        "description": "US Standard Plate (3-letter 3/4-digit)",
        "script": "latin",
        "region": "USA",
    },
    "US_CAN_ALPHANUMERIC": {
        "pattern": r"\b([A-Z0-9]{1,3}[-\s]+[A-Z0-9]{3,4})\b",
        "description": "US / Canada Standard Alphanumeric Plate",
        "script": "latin",
        "region": "North America",
    },

    # 3. UK & European Formats
    "UK_STANDARD": {
        "pattern": r"\b([A-Z]{2}\d{2}\s?[A-Z]{3})\b",
        "description": "UK Standard Plate (e.g. AB12 CDE)",
        "script": "latin",
        "region": "United Kingdom",
    },
    "EURO_STANDARD": {
        "pattern": r"\b([A-Z]{1,3}[-\s]?\d{1,4}[-\s]?[A-Z]{1,3})\b",
        "description": "European Standard Plate",
        "script": "latin",
        "region": "Europe",
    },

    # 4. Standard 3-segment International
    "STANDARD_ALPHANUMERIC": {
        "pattern": r"([A-Z0-9]{2,4})[-–—\s]+([A-Z0-9]{2,4})[-–—\s]+([0-9]{3,4})",
        "description": "Generic Standard Alphanumeric Plate",
        "script": "latin",
        "region": "International",
    },

    # 5. US & International Vanity / Custom Plates
    "US_VANITY_CUSTOM": {
        "pattern": r"\b([A-Z0-9]{4,8})\b",
        "description": "US / International Vanity & Custom Plate",
        "script": "latin",
        "region": "Vanity / Custom",
    },
}


def normalize_bangla_digits(text: str) -> str:
    """Replace all Bangla numeral characters with Arabic 0-9."""
    res = []
    for ch in text:
        res.append(BANGLA_TO_ARABIC_DIGITS.get(ch, ch))
    return "".join(res)


def disambiguate_plate_text(raw_text: str) -> str:
    """Clean common CCTV OCR noises from license plate text."""
    # Remove leading/trailing quotes, brackets, and noise symbols
    cleaned = re.sub(r"[`'\"~^*\[\]{}()|<>]+", "", raw_text).strip()
    # Normalize common keyword digit confusions in words
    cleaned = re.sub(r"METR0", "METRO", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"DHAK4", "DHAKA", cleaned, flags=re.IGNORECASE)
    # Ensure spacing around METRO and district names if concatenated
    cleaned = re.sub(r"(DHAKA|CTG|CHITTAGONG|SYLHET|RAJSHAHI|KHULNA|BARISHAL|RANGPUR)(METRO)", r"\1 \2", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"(METRO)([A-Z]{1,2})(\d{2})", r"\1 \2 \3", cleaned, flags=re.IGNORECASE)
    # Normalize multiple whitespaces and hyphens
    cleaned = re.sub(r"\s*[-–—_]+\s*", "-", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)

    # Handle vanity plates with leading symbol artifact (e.g. California Kids Plate heart ♥ read as O/V/0)
    vanity_match = re.match(r"^[OV0♥★\*\^~]([A-Z]{4,7})$", cleaned)
    if vanity_match:
        cleaned = vanity_match.group(1)

    return cleaned


def match_plate_format(plate_text: str) -> Tuple[bool, Optional[str], Optional[Dict[str, str]]]:
    """Check text against registered plate formats and return matched pattern and groups."""
    cleaned = disambiguate_plate_text(plate_text)

    # First check cleaned text against patterns
    for format_key, info in PLATE_FORMAT_REGISTRY.items():
        pattern = info["pattern"]
        match = re.search(pattern, cleaned, re.IGNORECASE)
        if match:
            groups = match.groups()
            meta = {
                "format_name": format_key,
                "region": info["region"],
                "script": info["script"],
                "normalized": "-".join(groups) if groups else cleaned,
            }
            return True, format_key, meta

    return False, None, None
