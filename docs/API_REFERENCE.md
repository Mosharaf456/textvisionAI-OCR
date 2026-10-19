# TextVision AI: API & CLI Reference Manual

> **TextVision AI** — Intelligent text recognition and extraction.  
> Comprehensive reference for executing TextVision AI via Command-Line Interface (CLI) and REST API.

---

## 1. Command-Line Interface (CLI)

The CLI is powered by Typer and Rich, providing formatted console outputs and tables.

### 1.1 Running the CLI
```bash
# Using the active virtual environment
source .venv/bin/activate

# View all commands
python -m source.cli.main --help
```

---

### 1.2 Use Case 1: Document Scanner (Printed to Text & Searchable PDF)

Digitizes printed papers, deskews, removes shadows, reorders columns via Recursive XY-Cut, and exports searchable PDF.

```bash
# Output structured Markdown to console
python -m source.cli.main scan path/to/document.png --markdown

# Output plain text
python -m source.cli.main scan path/to/document.png --plain

# Export Searchable PDF with invisible text layer
python -m source.cli.main scan path/to/document.png --pdf output_searchable.pdf
```

---

### 1.3 Use Case 2: Multi-Format Image & Hybrid PDF Reader

Reads PDFs (using < 15ms digital vector fast-path where present, falling back to multi-threaded raster OCR for scanned pages) and diverse image formats (`.png`, `.jpg`, `.tiff`, `.webp`).

```bash
# Read digital or scanned PDF
python -m source.cli.main read path/to/document.pdf

# Force raster OCR on all pages (bypassing digital text stream)
python -m source.cli.main read path/to/document.pdf --force-ocr
```

---

### 1.4 Use Case 3: Invoice & ID Card Extraction

Extracts key-values, line-item tables, and verifies arithmetic consistency.

```bash
# Parse invoice and display table & totals
python -m source.cli.main invoice path/to/invoice.jpg

# Export validated Pydantic JSON
python -m source.cli.main invoice path/to/invoice.jpg --json invoice_extracted.json

# Parse National ID Card or Passport MRZ
python -m source.cli.main id-card path/to/id_card.png --json id_extracted.json
```

---

### 1.5 Use Case 4: Vehicle License Plate Recognition (ALPR)

Detects plates in CCTV frames, performs contrast enhancement, character disambiguation, and regional format matching (Bangladesh Metro/District/Digital & International).

```bash
python -m source.cli.main alpr path/to/cctv_frame.jpg
```

---

### 1.6 CPU Benchmark

Measures latency and throughput on the host CPU.

```bash
python -m source.cli.main benchmark --iterations 5
```

---

## 2. FastAPI REST Microservice

### 2.1 Starting the API Server
```bash
# Development mode with reload
.venv/bin/uvicorn source.api.app:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger UI documentation is available at: `http://localhost:8000/docs`

---

### 2.2 Endpoints Summary

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Health check, ONNX runtime status, CPU info |
| `GET` | `/metrics` | Request counter and average latency metrics |
| `POST` | `/api/v1/scan` | Upload image -> returns text, markdown, boxes; optional `?export_pdf=true` downloads PDF |
| `POST` | `/api/v1/extract-invoice` | Upload invoice -> returns validated `InvoiceExtraction` JSON |
| `POST` | `/api/v1/extract-id` | Upload ID -> returns validated `IDCardExtraction` JSON |
| `POST` | `/api/v1/alpr` | Upload CCTV frame -> returns `List[PlateDetection]` with syntax check |

---

### 2.3 Example cURL Requests

#### Scan Document:
```bash
curl -X POST "http://localhost:8000/api/v1/scan?deskew=true" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample_doc.png"
```

#### Extract Invoice:
```bash
curl -X POST "http://localhost:8000/api/v1/extract-invoice" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@invoice_sample.png"
```

#### CCTV ALPR:
```bash
curl -X POST "http://localhost:8000/api/v1/alpr" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@cctv_car.jpg"
```
