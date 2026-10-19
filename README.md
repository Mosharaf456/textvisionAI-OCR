# TextVision AI — Intelligent Text Recognition and Extraction

> **TextVision AI** is an enterprise-grade, CPU-optimized OCR and document intelligence platform that unifies document digitization, hybrid PDF extraction, financial table parsing, identity verification, and CCTV vehicle surveillance into a single high-performance engine.
>
> **Target Platform:** Multi-Core Intel/AMD CPUs (Tested on Intel Core i5-12400)  
> **Core Backend:** ONNX Runtime (CPU Execution Provider with AVX2/VNNI) + OpenCV  
> **Platform Capabilities:** [docs/CAPABILITIES.md](docs/CAPABILITIES.md)  
> **Architecture Blueprint:** [docs/PLAN.md](docs/PLAN.md)  
> **API & CLI Manual:** [docs/API_REFERENCE.md](docs/API_REFERENCE.md)

---

## Features & Workflows

### 1. Scanning Printed Documents into Editable Text
- **Advanced Preprocessing:** Automatic deskewing via contour minimum bounding rectangles, morphological illumination normalization / shadow cancellation, and Sauvola adaptive thresholding.
- **Multi-Column Reading Order:** Implementation of the **Recursive XY-Cut algorithm** to correctly reconstruct reading flow across multi-column newspapers, reports, and brochures without column interleaving.
- **Searchable PDF Exporter:** Generates standardized PDF documents overlaying invisible, selectable text above the original scan.

### 2. Reading Text from Images & Multi-Page PDFs
- **Hybrid PDF Processing:** Extracts embedded digital vector text in < 15ms per page when present; falls back to multi-threaded raster OCR for scanned pages.
- **Multi-Format Support:** PNG, JPG, JPEG, TIFF, BMP, WebP.

### 3. Extracting Structured Information from Invoices & ID Cards
- **Invoices:** Semantic anchor detection, spatial proximity pairing (horizontal right & vertical below), line-item tabular extraction, and **Pydantic mathematical integrity cross-checks** (`sum(items) == subtotal`, `subtotal + tax - discount == total`).
- **ID Cards & Passports:** Four-point perspective rectification, ICAO Doc 9303 MRZ parser with checksum verification, national ID number extraction, DOB, and blood group detection.

### 4. Vehicle License Plate Recognition for CCTV Surveillance (ALPR)
- **Surveillance-Grade Pipeline:** CLAHE contrast adjustment for night glare, aspect ratio filtering, and character disambiguation (`0` vs `O`, `8` vs `B`).
- **Regional Syntax Matching:** Supports **Bangladesh BRTA plate formats** (Dhaka Metro, CTG Metro, District, Digital English plates, motorcycles) with Bangla numeral normalization, plus international alphanumeric formats.
- **Temporal Majority Voting:** Aggregates readings across video stream frames using `TemporalPlateTracker` to eliminate transient motion-blur errors.

---

## Quickstart

### 1. Launching the Application & Web UI
Run [`./start.sh`](file:///media/mosharaf/New%20Volume6/brotecs/kathalAI/kathal_AI_new/1.prod/other-tts/OCR/start.sh) to start the system:

```bash
# Start FastAPI REST API & Web UI (default: http://0.0.0.0:8000)
./start.sh
```

Open **[http://localhost:8000](http://localhost:8000)** in your browser to access the **Interactive Testing Dashboard**:
* **📄 Document Scanner:** Upload documents or click *Load Sample Document* to see real-time bounding box overlays on HTML5 canvas, formatted Markdown, plain text, and download Searchable PDFs.
* **📑 PDF & Image Reader:** Test hybrid PDF fast-path (<15ms) vs multi-threaded raster OCR.
* **🧾 Invoices & ID Cards:** Extract structured key-values, line-item tables, and test arithmetic balance verification.
* **🚗 CCTV ALPR:** Recognize vehicle number plates with Bangladesh BRTA syntax matching and character disambiguation.
* **📖 Swagger API Docs:** Available at `http://localhost:8000/docs`.

### 2. Run Tests
```bash
pytest -v tests/
```

### 3. Run CLI
```bash
# Benchmark CPU performance
python -m source.cli.main benchmark

# Scan document to Searchable PDF
python -m source.cli.main scan sample.png --pdf output_searchable.pdf

# Parse invoice to JSON
python -m source.cli.main invoice sample_invoice.jpg --json output.json

# Parse CCTV License Plate
python -m source.cli.main alpr cctv_frame.jpg
```

### 4. Run REST API
```bash
uvicorn source.api.app:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger documentation will be available at `http://localhost:8000/docs`.

---

## Directory Structure
```
OCR/
├── docs/
│   ├── PLAN.md                     # Comprehensive architecture and engineering plan
│   └── API_REFERENCE.md            # Programmatic and CLI documentation
├── resources/
│   ├── sample_documents/           # Sample test documents
│   ├── sample_cctv/                # Sample CCTV frames
│   └── models/
│       └── model_manifest.json     # Pinned model versions with SHA-256 hashes
├── source/
│   ├── core/                       # Base engine, types, ONNX engine, logger, model manager
│   ├── preprocessing/              # Deskew, binarization, perspective warp, Recursive XY-Cut
│   ├── schemas/                    # Pydantic schemas (Invoice, ID Card, ALPR, Plate formats)
│   ├── use_cases/                  # 4 Core industrial OCR workflows
│   ├── cli/                        # Rich/Typer command line interface
│   └── api/                        # FastAPI REST microservice
├── tests/                          # Automated Pytest suite (22 unit & integration tests)
├── pyproject.toml
├── requirements.txt
└── README.md
```
