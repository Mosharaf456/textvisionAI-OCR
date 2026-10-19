"""Use Case 3: Structured Information Extraction from Invoices and ID Cards / Passports.

Extracts key-value pairs, tabular line items, MRZ passport zones, and validates schemas.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from source.core.factory import get_ocr_engine
from source.core.logger import create_request_id, get_logger
from source.core.types import BoundingBox, OCRResult, TextLine
from source.preprocessing.perspective import find_document_contour, four_point_transform
from source.schemas.id_card_schema import IDCardExtraction, IDCardType, MRZData
from source.schemas.invoice_schema import InvoiceExtraction, InvoiceTotals, LineItem

logger = get_logger("use_cases.structured_extractor")


class StructuredExtractor:
    """Extracts structured entities and validates schemas from invoices and identity documents."""

    def __init__(self):
        self.engine = get_ocr_engine()

    # =========================================================================
    # INVOICE EXTRACTION PIPELINE
    # =========================================================================

    def extract_invoice(
        self,
        image_input: Union[str, Path, np.ndarray],
        request_id: Optional[str] = None,
    ) -> InvoiceExtraction:
        """Extract structured invoice fields, line items, and validate arithmetic."""
        req_id = request_id or create_request_id()

        if isinstance(image_input, (str, Path)):
            img = cv2.imread(str(image_input))
            if img is None:
                raise ValueError(f"Could not read invoice image: {image_input}")
        else:
            img = image_input

        ocr_result = self.engine.predict(img, request_id=req_id)
        lines = ocr_result.lines

        # 1. Extract Invoice Number
        inv_num = self._find_invoice_number(lines)

        # 2. Extract Dates
        inv_date = self._find_date(lines, anchor_patterns=[r"invoice\s*date", r"date", r"dated"])
        due_date = self._find_date(lines, anchor_patterns=[r"due\s*date", r"payment\s*due"])

        # 3. Extract Vendor and Customer
        vendor = self._find_vendor(lines)
        customer = self._find_customer(lines)

        # 4. Extract Totals (Subtotal, Tax, Discount, Grand Total)
        totals = self._find_totals(lines)

        # 5. Extract Line Items Table
        line_items = self._find_line_items(lines)

        return InvoiceExtraction(
            invoice_number=inv_num,
            invoice_date=inv_date,
            due_date=due_date,
            vendor_name=vendor,
            customer_name=customer,
            line_items=line_items,
            totals=totals,
            confidence=round(ocr_result.mean_confidence, 4),
        )

    def _find_invoice_number(self, lines: List[TextLine]) -> Optional[str]:
        patterns = [
            r"invoice\s*(?:no|number|#|id)?[:\s#]*([A-Z0-9\-_/]+)",
            r"bill\s*(?:no|number|#)[:\s#]*([A-Z0-9\-_/]+)",
            r"inv\s*#?[:\s]*([A-Z0-9\-_/]+)",
        ]
        # Search combined text on same line
        for l in lines:
            for pat in patterns:
                m = re.search(pat, l.text, re.IGNORECASE)
                if m:
                    candidate = m.group(1).strip(" :-#")
                    if len(candidate) >= 2 and any(c.isdigit() for c in candidate):
                        return candidate

        # Search value to right of anchor
        anchor_val = self._find_value_near_anchor(lines, [r"invoice\s*(?:no|number|#)", r"inv\s*#"])
        if anchor_val:
            cleaned = re.sub(r"[^A-Z0-9\-_/]", "", anchor_val)
            if cleaned:
                return cleaned
        return None

    def _find_date(self, lines: List[TextLine], anchor_patterns: List[str]) -> Optional[str]:
        date_regex = r"(\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})"

        for l in lines:
            for pat in anchor_patterns:
                if re.search(pat, l.text, re.IGNORECASE):
                    # Check same line
                    m = re.search(date_regex, l.text)
                    if m:
                        return self._normalize_date(m.group(1))

        # Check adjacent value
        val = self._find_value_near_anchor(lines, anchor_patterns)
        if val:
            m = re.search(date_regex, val)
            if m:
                return self._normalize_date(m.group(1))

        return None

    def _normalize_date(self, date_str: str) -> str:
        """Convert varied date strings into standardized YYYY-MM-DD."""
        cleaned = re.sub(r"[/.]", "-", date_str.strip())
        parts = cleaned.split("-")
        if len(parts) == 3:
            if len(parts[0]) == 4:  # YYYY-MM-DD
                return f"{parts[0]}-{int(parts[1]):02d}-{int(parts[2]):02d}"
            elif len(parts[2]) == 4:  # DD-MM-YYYY
                return f"{parts[2]}-{int(parts[1]):02d}-{int(parts[0]):02d}"
        return cleaned

    def _find_vendor(self, lines: List[TextLine]) -> Optional[str]:
        # Typically the top prominent text or near "From:"
        for l in lines:
            if re.search(r"from[:\s]", l.text, re.IGNORECASE):
                val = re.sub(r"from[:\s]*", "", l.text, flags=re.IGNORECASE).strip()
                if val:
                    return val

        # Fallback: first prominent line in top 20% of page
        if lines:
            top_lines = sorted(lines[:5], key=lambda x: x.bbox.y_min)
            for l in top_lines:
                t = l.text.strip()
                if not re.search(r"invoice|bill|receipt|tax", t, re.IGNORECASE) and len(t) > 3:
                    return t
        return None

    def _find_customer(self, lines: List[TextLine]) -> Optional[str]:
        anchor_patterns = [r"bill\s*to[:\s]*", r"billed\s*to[:\s]*", r"customer[:\s]*", r"client[:\s]*"]
        for idx, l in enumerate(lines):
            for pat in anchor_patterns:
                m = re.search(pat, l.text, re.IGNORECASE)
                if m:
                    # Same line
                    rem = l.text[m.end():].strip(" :-")
                    if rem:
                        return rem
                    # Next line
                    val = self._find_value_below_anchor(lines, l)
                    if val:
                        return val
        return None

    def _find_totals(self, lines: List[TextLine]) -> InvoiceTotals:
        subtotal = self._extract_amount(lines, [r"\bsub\s*total\b", r"\bsubtotal\b", r"\bnet\s*amount\b"])
        tax = self._extract_amount(lines, [r"\btax\b", r"\bvat\b", r"\bgst\b"])
        discount = self._extract_amount(lines, [r"\bdiscount\b", r"\bless\b"]) or 0.0
        total = self._extract_amount(lines, [r"\bgrand\s*total\b", r"\btotal\s*amount\b", r"^\s*total\b", r"\bbalance\s*due\b", r"\btotal[:\s]"])

        return InvoiceTotals(
            subtotal=subtotal,
            tax=tax,
            discount=discount,
            total=total,
        )

    def _extract_amount(self, lines: List[TextLine], anchor_patterns: List[str]) -> Optional[float]:
        money_regex = r"[\$€£¥৳]?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})|\d+(?:\.\d{1,2})?)"

        for l in lines:
            for pat in anchor_patterns:
                if re.search(pat, l.text, re.IGNORECASE):
                    # Check text after anchor
                    matches = list(re.finditer(money_regex, l.text))
                    if matches:
                        last_m = matches[-1].group(1).replace(",", "")
                        try:
                            return float(last_m)
                        except ValueError:
                            pass

        # Try spatial proximity (to right or below)
        val = self._find_value_near_anchor(lines, anchor_patterns)
        if val:
            m = re.search(money_regex, val)
            if m:
                try:
                    return float(m.group(1).replace(",", ""))
                except ValueError:
                    pass

        return None

    def _find_line_items(self, lines: List[TextLine]) -> List[LineItem]:
        """Detect tabular items matching description, qty, price, amount."""
        items: List[LineItem] = []
        # Find header line (contains 'Description', 'Item', 'Qty', 'Price')
        header_idx = -1
        for idx, l in enumerate(lines):
            if re.search(r"description|item", l.text, re.IGNORECASE) and re.search(r"qty|amount|price|rate", l.text, re.IGNORECASE):
                header_idx = idx
                break

        if header_idx != -1:
            # Lines following header up until Subtotal/Total
            for l in lines[header_idx + 1:]:
                t = l.text.strip()
                if re.search(r"subtotal|total|tax|vat|notes|thank", t, re.IGNORECASE):
                    break

                # Try to parse line format: "<Desc> <Qty> <Price> <Amount>" or "<Desc> <Amount>"
                numbers = re.findall(r"\d+(?:\.\d{1,2})?", t)
                if numbers:
                    amt = float(numbers[-1])
                    desc_part = re.sub(r"[\$€£¥৳]?\s*\d+(?:\.\d{1,2})?", "", t).strip()
                    qty = float(numbers[0]) if len(numbers) >= 3 else 1.0
                    price = float(numbers[1]) if len(numbers) >= 3 else amt
                    items.append(LineItem(description=desc_part or t, quantity=qty, unit_price=price, amount=amt))

        return items

    def _find_value_near_anchor(self, lines: List[TextLine], anchor_patterns: List[str]) -> Optional[str]:
        for anchor_line in lines:
            for pat in anchor_patterns:
                if re.search(pat, anchor_line.text, re.IGNORECASE):
                    # Find a line horizontally to the right
                    candidates = []
                    for other in lines:
                        if other == anchor_line:
                            continue
                        # Same vertical band
                        if abs(other.bbox.center[1] - anchor_line.bbox.center[1]) < (anchor_line.bbox.height * 0.8):
                            # Must be to the right
                            if other.bbox.x_min >= anchor_line.bbox.x_max - 5:
                                dist = other.bbox.x_min - anchor_line.bbox.x_max
                                candidates.append((dist, other.text))

                    if candidates:
                        candidates.sort(key=lambda x: x[0])
                        return candidates[0][1]

                    # Fallback: check line directly below
                    val_below = self._find_value_below_anchor(lines, anchor_line)
                    if val_below:
                        return val_below

        return None

    def _find_value_below_anchor(self, lines: List[TextLine], anchor_line: TextLine) -> Optional[str]:
        candidates = []
        for other in lines:
            if other == anchor_line:
                continue
            # Below anchor
            if other.bbox.y_min >= anchor_line.bbox.y_max - 5:
                # Horizontally overlapping
                if abs(other.bbox.x_min - anchor_line.bbox.x_min) < (anchor_line.bbox.width * 0.8):
                    dist = other.bbox.y_min - anchor_line.bbox.y_max
                    candidates.append((dist, other.text))

        if candidates:
            candidates.sort(key=lambda x: x[0])
            return candidates[0][1]
        return None

    # =========================================================================
    # ID CARD AND PASSPORT EXTRACTION PIPELINE
    # =========================================================================

    def extract_id_card(
        self,
        image_input: Union[str, Path, np.ndarray],
        request_id: Optional[str] = None,
    ) -> IDCardExtraction:
        """Extract structured fields from National ID cards, Passports, or Driving Licenses."""
        req_id = request_id or create_request_id()

        if isinstance(image_input, (str, Path)):
            img = cv2.imread(str(image_input))
            if img is None:
                raise ValueError(f"Could not read ID card image: {image_input}")
        else:
            img = image_input

        # 1. Perspective rectification if 4 corners are clearly found
        corners = find_document_contour(img)
        if corners is not None:
            logger.info("4-corner boundary found, rectifying perspective")
            processed_img = four_point_transform(img, corners)
        else:
            processed_img = img

        ocr_result = self.engine.predict(processed_img, request_id=req_id)
        lines = ocr_result.lines

        # 2. Check for MRZ (Passport / TD1)
        mrz_data = self._parse_mrz_lines(lines)

        # 3. Detect Card Type
        card_type = IDCardType.UNKNOWN
        full_text_lower = ocr_result.full_text.lower()

        if mrz_data or "passport" in full_text_lower:
            card_type = IDCardType.PASSPORT
        elif any(k in full_text_lower for k in ["national id", "nid", "identity card", "citizenship"]):
            card_type = IDCardType.NATIONAL_ID
        elif any(k in full_text_lower for k in ["driving license", "driver license", "driving licence"]):
            card_type = IDCardType.DRIVING_LICENSE

        # 4. Extract Key Fields
        doc_number = (
            mrz_data.document_number if mrz_data else self._find_id_number(lines)
        )
        dob = mrz_data.date_of_birth if mrz_data else self._find_dob(lines)
        name = mrz_data.names if mrz_data else self._find_name(lines)
        blood = self._find_blood_group(lines)

        return IDCardExtraction(
            card_type=card_type,
            document_number=doc_number,
            full_name=name,
            date_of_birth=dob,
            blood_group=blood,
            mrz_data=mrz_data,
            confidence=round(ocr_result.mean_confidence, 4),
        )

    def _parse_mrz_lines(self, lines: List[TextLine]) -> Optional[MRZData]:
        """Detect and parse ICAO Doc 9303 MRZ strings (containing '<<')."""
        mrz_candidates = [l.text.replace(" ", "").upper() for l in lines if "<<" in l.text]
        if len(mrz_candidates) >= 2:
            # 2-line Passport MRZ (Type TD3, 44 chars)
            line1 = mrz_candidates[-2]
            line2 = mrz_candidates[-1]
            if len(line1) >= 30 and len(line2) >= 30:
                doc_type = line1[0]
                country = line1[2:5]
                names = line1[5:].replace("<", " ").strip()
                doc_num = line2[:9].replace("<", "")
                nat = line2[10:13]
                dob_raw = line2[13:19]
                sex = line2[20]
                exp_raw = line2[21:27]

                return MRZData(
                    format="TD3_2Line",
                    document_type=doc_type,
                    country_code=country,
                    document_number=doc_num,
                    nationality=nat,
                    date_of_birth=f"19{dob_raw[:2]}-{dob_raw[2:4]}-{dob_raw[4:6]}",
                    sex=sex,
                    expiry_date=f"20{exp_raw[:2]}-{exp_raw[2:4]}-{exp_raw[4:6]}",
                    names=names,
                )
        return None

    def _find_id_number(self, lines: List[TextLine]) -> Optional[str]:
        # NID numbers: typically 10, 13, or 17 continuous digits
        for l in lines:
            t = l.text.replace(" ", "")
            m = re.search(r"(?:NID|NO|ID)?[:\s]*(\d{10}|\d{13}|\d{17})\b", t, re.IGNORECASE)
            if m:
                return m.group(1)

        # Passport alphanumeric: 1 letter followed by 7-8 digits
        for l in lines:
            m = re.search(r"\b([A-Z][0-9]{7,8})\b", l.text)
            if m:
                return m.group(1)

        return None

    def _find_dob(self, lines: List[TextLine]) -> Optional[str]:
        dob_patterns = [r"date\s*of\s*birth", r"dob", r"birth\s*date"]
        date_regex = r"(\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})"
        for l in lines:
            for pat in dob_patterns:
                if re.search(pat, l.text, re.IGNORECASE):
                    m = re.search(date_regex, l.text)
                    if m:
                        return self._normalize_date(m.group(1))

        # Check adjacent value
        val = self._find_value_near_anchor(lines, dob_patterns)
        if val:
            m = re.search(date_regex, val)
            if m:
                return self._normalize_date(m.group(1))
        return None

    def _find_name(self, lines: List[TextLine]) -> Optional[str]:
        name_patterns = [r"name[:\s]", r"full\s*name[:\s]"]
        for l in lines:
            for pat in name_patterns:
                m = re.search(pat, l.text, re.IGNORECASE)
                if m:
                    cand = l.text[m.end():].strip(" :-")
                    if cand and len(cand) > 2:
                        return cand

        # Value near anchor
        val = self._find_value_near_anchor(lines, name_patterns)
        if val:
            return val
        return None

    def _find_blood_group(self, lines: List[TextLine]) -> Optional[str]:
        for l in lines:
            m = re.search(r"\b(A|B|AB|O)[+-]\b", l.text)
            if m:
                return m.group(0)
        return None
