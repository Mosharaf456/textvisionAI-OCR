"""Unit tests for Pydantic schemas, invoice arithmetic validation, and plate format rules."""

from __future__ import annotations

import pytest
from source.schemas.id_card_schema import IDCardExtraction, IDCardType, MRZData
from source.schemas.invoice_schema import InvoiceExtraction, InvoiceTotals, LineItem
from source.schemas.plate_formats import (
    disambiguate_plate_text,
    match_plate_format,
    normalize_bangla_digits,
)


def test_invoice_arithmetic_validation_valid():
    inv = InvoiceExtraction(
        invoice_number="INV-2026-001",
        line_items=[
            LineItem(description="Item A", quantity=2.0, unit_price=50.0, amount=100.0),
            LineItem(description="Item B", quantity=1.0, unit_price=100.0, amount=100.0),
        ],
        totals=InvoiceTotals(
            subtotal=200.0,
            tax=20.0,
            discount=10.0,
            total=210.0,
        ),
    )
    assert inv.is_math_valid is True
    assert len(inv.validation_notes) == 0


def test_invoice_arithmetic_validation_mismatch():
    inv = InvoiceExtraction(
        invoice_number="INV-2026-002",
        line_items=[
            LineItem(description="Item A", quantity=1.0, unit_price=100.0, amount=100.0),
        ],
        totals=InvoiceTotals(
            subtotal=100.0,
            tax=15.0,
            total=150.0,  # Should be 115.0
        ),
    )
    assert inv.is_math_valid is False
    assert len(inv.validation_notes) > 0
    assert any("Arithmetic mismatch" in note for note in inv.validation_notes)


def test_bangla_plate_format_matching():
    # Bangla Metro Dhaka plate
    plate_bn = "ঢাকা মেট্রো গ ১১-২২৩৩"
    matched, fmt_name, meta = match_plate_format(plate_bn)
    assert matched is True
    assert fmt_name == "BD_METRO_DHAKA_BANGLA"
    assert meta["region"] == "Dhaka Metro"


def test_bangla_digital_plate_format_matching():
    # English Digital Metro plate
    plate_en = "DHAKA METRO GA 11-2233"
    matched, fmt_name, meta = match_plate_format(plate_en)
    assert matched is True
    assert fmt_name == "BD_DIGITAL_METRO"


def test_plate_noise_disambiguation():
    raw_noisy = " `DHAKA - METRO - GA  11-2233\" "
    cleaned = disambiguate_plate_text(raw_noisy)
    assert cleaned == "DHAKA-METRO-GA 11-2233"


def test_normalize_bangla_digits():
    bn_digits = "১২৩৪৫৬৭৮৯০"
    ar_digits = normalize_bangla_digits(bn_digits)
    assert ar_digits == "1234567890"


def test_us_and_vanity_plate_format_matching():
    # California vanity plate with heart symbol
    plate_vanity = "OMYFIGY"
    matched, fmt_name, meta = match_plate_format(plate_vanity)
    assert matched is True
    assert fmt_name == "US_VANITY_CUSTOM"
    assert meta["normalized"] == "MYFIGY"

    # US Standard 7-char California
    plate_us7 = "7XYZ789"
    matched, fmt_name, meta = match_plate_format(plate_us7)
    assert matched is True
    assert fmt_name == "US_STANDARD_7CHAR"

    # UK plate
    plate_uk = "BD51 SMR"
    matched, fmt_name, meta = match_plate_format(plate_uk)
    assert matched is True
    assert fmt_name == "UK_STANDARD"

