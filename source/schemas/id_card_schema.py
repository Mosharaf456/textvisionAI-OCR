"""Pydantic schemas for ID cards, passports, and MRZ data extraction."""

from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class IDCardType(str, Enum):
    NATIONAL_ID = "national_id"
    PASSPORT = "passport"
    DRIVING_LICENSE = "driving_license"
    UNKNOWN = "unknown"


class MRZData(BaseModel):
    """Machine Readable Zone (ICAO 9303) parsed fields."""
    format: str = Field(..., description="TD1 (3 lines), TD2 (2 lines), or TD3 (2 lines)")
    document_type: str = Field("P", description="P for passport, I for identity card")
    country_code: str = Field(..., description="ISO 3-letter country code")
    document_number: str = Field(..., description="Passport or ID number")
    document_number_check: bool = Field(True, description="Doc number checksum validity")
    nationality: str = Field(..., description="ISO 3-letter nationality code")
    date_of_birth: str = Field(..., description="DOB in YYYY-MM-DD")
    dob_check: bool = Field(True, description="DOB checksum validity")
    sex: str = Field(..., description="M, F, or <")
    expiry_date: str = Field(..., description="Expiry date in YYYY-MM-DD")
    expiry_check: bool = Field(True, description="Expiry date checksum validity")
    names: str = Field(..., description="Primary and secondary names")
    composite_check: bool = Field(True, description="Overall MRZ checksum validity")


class IDCardExtraction(BaseModel):
    """Structured ID document representation."""
    card_type: IDCardType = Field(IDCardType.UNKNOWN, description="Detected document category")
    document_number: Optional[str] = Field(None, description="National ID or Passport number")
    full_name: Optional[str] = Field(None, description="Citizen full name")
    father_name: Optional[str] = Field(None, description="Father's name if present")
    mother_name: Optional[str] = Field(None, description="Mother's name if present")
    date_of_birth: Optional[str] = Field(None, description="Date of birth in YYYY-MM-DD")
    issue_date: Optional[str] = Field(None, description="Card issuance date")
    expiry_date: Optional[str] = Field(None, description="Card expiration date")
    blood_group: Optional[str] = Field(None, description="Blood group (e.g. A+, B+, O+, AB-)")
    address: Optional[str] = Field(None, description="Full registered residential address")
    pin_or_zip: Optional[str] = Field(None, description="Postal / ZIP code")
    mrz_data: Optional[MRZData] = Field(None, description="Parsed MRZ if passport or TD1 card")
    confidence: float = Field(1.0, description="Confidence score of recognition")
