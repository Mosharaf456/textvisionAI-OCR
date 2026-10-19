"""Pydantic schemas for structured invoice entity extraction and mathematical validation."""

from __future__ import annotations

import re
from typing import List, Optional
from pydantic import BaseModel, Field, model_validator


class LineItem(BaseModel):
    """An individual line item within an invoice table."""
    description: str = Field(..., description="Product or service description")
    quantity: Optional[float] = Field(None, description="Purchased quantity")
    unit_price: Optional[float] = Field(None, description="Price per unit")
    amount: Optional[float] = Field(None, description="Calculated total line item amount")

    @model_validator(mode="after")
    def compute_or_validate_amount(self) -> "LineItem":
        # If amount is missing but quantity and unit_price are provided, compute amount
        if self.amount is None and self.quantity is not None and self.unit_price is not None:
            self.amount = round(self.quantity * self.unit_price, 2)
        return self


class InvoiceTotals(BaseModel):
    """Invoice monetary totals and currency details."""
    subtotal: Optional[float] = Field(None, description="Sum before taxes and discounts")
    tax: Optional[float] = Field(None, description="Tax / VAT / GST amount")
    discount: Optional[float] = Field(0.0, description="Discount amount")
    total: Optional[float] = Field(None, description="Grand total payable amount")
    currency: Optional[str] = Field("USD", description="Currency code or symbol")


class InvoiceExtraction(BaseModel):
    """Structured and validated invoice data."""
    invoice_number: Optional[str] = Field(None, description="Unique invoice ID / number")
    invoice_date: Optional[str] = Field(None, description="Invoice issue date (YYYY-MM-DD)")
    due_date: Optional[str] = Field(None, description="Payment due date (YYYY-MM-DD)")
    vendor_name: Optional[str] = Field(None, description="Supplier / seller business name")
    vendor_address: Optional[str] = Field(None, description="Supplier business address")
    customer_name: Optional[str] = Field(None, description="Recipient / buyer name")
    customer_address: Optional[str] = Field(None, description="Recipient address")
    line_items: List[LineItem] = Field(default_factory=list, description="Extracted table items")
    totals: InvoiceTotals = Field(default_factory=InvoiceTotals)
    is_math_valid: bool = Field(True, description="True if subtotal + tax - discount matches total")
    validation_notes: List[str] = Field(default_factory=list)
    confidence: float = Field(1.0, description="Confidence score of overall extraction")

    @model_validator(mode="after")
    def validate_arithmetic_consistency(self) -> "InvoiceExtraction":
        notes: List[str] = []
        t = self.totals

        # 1. Check if sum of line items matches subtotal
        if self.line_items and t.subtotal is not None:
            items_sum = sum(item.amount for item in self.line_items if item.amount is not None)
            if abs(items_sum - t.subtotal) > 0.05:
                notes.append(
                    f"Line items sum ({items_sum:.2f}) does not match subtotal ({t.subtotal:.2f})"
                )

        # 2. Check if subtotal + tax - discount == total
        if t.subtotal is not None and t.total is not None:
            tax = t.tax or 0.0
            disc = t.discount or 0.0
            expected_total = round(t.subtotal + tax - disc, 2)
            if abs(expected_total - t.total) > 0.05:
                notes.append(
                    f"Arithmetic mismatch: subtotal ({t.subtotal}) + tax ({tax}) - disc ({disc}) = {expected_total}, but grand total is {t.total}"
                )
                self.is_math_valid = False

        self.validation_notes.extend(notes)
        return self
