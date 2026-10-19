"""Model manager for downloading, verifying SHA-256 hashes, and caching ONNX models."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import requests

from source.core.logger import get_logger

logger = get_logger("model_manager")


class ModelIntegrityError(Exception):
    """Raised when downloaded model SHA-256 does not match pinned manifest."""
    pass


class ModelNotFoundError(Exception):
    """Raised when a required model file is missing and cannot be downloaded."""
    pass


class ModelManager:
    """Manages model lifecycle, caching, and cryptographic verification."""

    def __init__(
        self,
        models_dir: Optional[Union[str, Path]] = None,
        manifest_path: Optional[Union[str, Path]] = None,
    ):
        base_dir = Path(__file__).resolve().parent.parent.parent
        self.models_dir = Path(models_dir) if models_dir else base_dir / "resources" / "models"
        self.manifest_path = (
            Path(manifest_path) if manifest_path else self.models_dir / "model_manifest.json"
        )
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> Dict[str, Any]:
        if not self.manifest_path.exists():
            logger.warning("Manifest not found", path=str(self.manifest_path))
            return {"models": []}
        try:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error("Failed to parse model manifest", error=str(e))
            return {"models": []}

    @staticmethod
    def calculate_sha256(file_path: Union[str, Path]) -> str:
        """Compute the SHA-256 hex digest of a file in streaming chunks."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def verify_model(self, model_meta: Dict[str, Any]) -> bool:
        """Check if local model file exists and matches the pinned SHA-256 hash."""
        filename = model_meta["filename"]
        target_path = self.models_dir / filename
        if not target_path.exists():
            return False

        expected_hash = model_meta.get("sha256")
        if not expected_hash:
            return True

        actual_hash = self.calculate_sha256(target_path)
        if actual_hash.lower() != expected_hash.lower():
            logger.error(
                "Model integrity mismatch",
                filename=filename,
                expected=expected_hash,
                actual=actual_hash,
            )
            return False
        return True

    def get_model_path(
        self,
        name_or_type: str,
        auto_download: bool = True,
    ) -> Optional[Path]:
        """Get verified path to a model by name or type ('detection', 'recognition', 'classification').

        If not present and auto_download is True, downloads and verifies SHA-256.
        """
        model_meta = None
        for m in self.manifest.get("models", []):
            if m.get("name") == name_or_type or m.get("type") == name_or_type:
                model_meta = m
                break

        if not model_meta:
            # Check if file exists directly by filename
            direct_path = self.models_dir / name_or_type
            if direct_path.exists():
                return direct_path
            return None

        filename = model_meta["filename"]
        local_path = self.models_dir / filename

        if local_path.exists():
            if self.verify_model(model_meta):
                return local_path
            else:
                logger.warning("Corrupted model detected, removing", path=str(local_path))
                local_path.unlink(missing_ok=True)

        if not auto_download:
            return None

        # Download
        url = model_meta.get("source_url")
        if not url:
            raise ModelNotFoundError(f"No source_url for model {name_or_type}")

        logger.info("Downloading model", name=model_meta["name"], url=url)
        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            temp_path = local_path.with_suffix(".tmp")
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)

            # Verify before renaming
            actual_hash = self.calculate_sha256(temp_path)
            expected_hash = model_meta.get("sha256")
            if expected_hash and actual_hash.lower() != expected_hash.lower():
                temp_path.unlink(missing_ok=True)
                raise ModelIntegrityError(
                    f"Integrity check failed for {filename}! Expected {expected_hash}, got {actual_hash}"
                )

            temp_path.rename(local_path)
            logger.info("Model verified and cached", path=str(local_path))
            return local_path
        except Exception as e:
            logger.error("Download failed", model=filename, error=str(e))
            raise
