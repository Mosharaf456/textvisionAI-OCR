"""Command Line Interface for CPU-based Production OCR Pipeline."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import typer

from source.use_cases.document_scanner import DocumentScanner
from source.use_cases.pdf_image_reader import PDFImageReader
from source.use_cases.structured_extractor import StructuredExtractor
from source.use_cases.vehicle_alpr import VehicleALPR

app = typer.Typer(
    name="textvision",
    help="TextVision AI — Intelligent text recognition and extraction.",
    add_completion=False,
)
console = Console()


@app.command()
def scan(
    input_file: Path = typer.Argument(..., help="Path to document image file"),
    output_pdf: Optional[Path] = typer.Option(None, "--pdf", "-p", help="Export searchable PDF path"),
    markdown: bool = typer.Option(True, "--markdown/--plain", help="Print structured Markdown vs plain text"),
    deskew: bool = typer.Option(True, "--deskew/--no-deskew", help="Enable automatic deskewing"),
):
    """Use Case 1: Scan printed document into editable text and Searchable PDF."""
    console.print(f"[bold cyan]Scanning document:[/bold cyan] {input_file}")
    scanner = DocumentScanner(auto_deskew=deskew)
    result = scanner.process(input_file)

    console.print(Panel(
        f"[green]Lines Extracted:[/green] {result['total_lines']} | "
        f"[green]Confidence:[/green] {result['confidence']:.2%} | "
        f"[green]Deskew Angle:[/green] {result['skew_angle']:.2f}°",
        title="Extraction Summary"
    ))

    if markdown:
        console.print("\n[bold yellow]--- Formatted Markdown Output ---[/bold yellow]")
        console.print(result["markdown"])
    else:
        console.print("\n[bold yellow]--- Plain Text Output ---[/bold yellow]")
        console.print(result["plain_text"])

    if output_pdf:
        pdf_path = scanner.export_searchable_pdf(input_file, output_pdf, result["ocr_result"])
        console.print(f"\n[bold green]✓ Searchable PDF saved to:[/bold green] {pdf_path}")


@app.command()
def read(
    input_file: Path = typer.Argument(..., help="Path to PDF or image file"),
    force_ocr: bool = typer.Option(False, "--force-ocr", help="Bypass digital fastpath and force raster OCR"),
):
    """Use Case 2: Read text from images and multi-page PDFs (Hybrid digital + OCR)."""
    console.print(f"[bold cyan]Reading document:[/bold cyan] {input_file}")
    reader = PDFImageReader()
    res = reader.read(input_file, force_ocr=force_ocr)

    table = Table(title=f"Document Pages ({res['file_name']})")
    table.add_column("Page", justify="center")
    table.add_column("Extraction Method", style="cyan")
    table.add_column("Lines", justify="right")
    table.add_column("Confidence", justify="right")

    for p in res["pages"]:
        table.add_row(
            str(p.get("page_number", 1)),
            p.get("extraction_method", "unknown"),
            str(p.get("lines_count", 0)),
            f"{p.get('confidence', 0.0):.2%}",
        )
    console.print(table)
    console.print(f"\n[bold green]Total Text Length:[/bold green] {len(res['full_text'])} characters")


@app.command()
def invoice(
    input_file: Path = typer.Argument(..., help="Path to invoice image"),
    output_json: Optional[Path] = typer.Option(None, "--json", "-j", help="Save extracted JSON to file"),
):
    """Use Case 3: Extract structured key-values and line items from invoices."""
    console.print(f"[bold cyan]Parsing invoice:[/bold cyan] {input_file}")
    extractor = StructuredExtractor()
    inv = extractor.extract_invoice(input_file)

    meta_table = Table(title="Invoice Details")
    meta_table.add_column("Field", style="cyan")
    meta_table.add_column("Extracted Value", style="bold white")
    meta_table.add_row("Invoice Number", inv.invoice_number or "N/A")
    meta_table.add_row("Invoice Date", inv.invoice_date or "N/A")
    meta_table.add_row("Due Date", inv.due_date or "N/A")
    meta_table.add_row("Vendor", inv.vendor_name or "N/A")
    meta_table.add_row("Customer", inv.customer_name or "N/A")
    meta_table.add_row("Subtotal", f"{inv.totals.subtotal}" if inv.totals.subtotal else "N/A")
    meta_table.add_row("Tax", f"{inv.totals.tax}" if inv.totals.tax else "N/A")
    meta_table.add_row("Grand Total", f"{inv.totals.total}" if inv.totals.total else "N/A")
    meta_table.add_row("Math Valid?", "[green]✓ YES[/green]" if inv.is_math_valid else "[red]✗ NO[/red]")
    console.print(meta_table)

    if inv.line_items:
        item_table = Table(title="Line Items")
        item_table.add_column("Description", style="white")
        item_table.add_column("Qty", justify="right")
        item_table.add_column("Unit Price", justify="right")
        item_table.add_column("Amount", justify="right", style="green")
        for itm in inv.line_items:
            item_table.add_row(
                itm.description,
                str(itm.quantity or ""),
                str(itm.unit_price or ""),
                str(itm.amount or ""),
            )
        console.print(item_table)

    if output_json:
        with open(output_json, "w", encoding="utf-8") as f:
            f.write(inv.model_dump_json(indent=2))
        console.print(f"[bold green]✓ Invoice JSON saved to:[/bold green] {output_json}")


@app.command()
def id_card(
    input_file: Path = typer.Argument(..., help="Path to ID card or passport image"),
    output_json: Optional[Path] = typer.Option(None, "--json", "-j", help="Save extracted JSON to file"),
):
    """Use Case 3: Extract structured fields from National ID cards or Passports."""
    console.print(f"[bold cyan]Parsing identity document:[/bold cyan] {input_file}")
    extractor = StructuredExtractor()
    doc = extractor.extract_id_card(input_file)

    table = Table(title="ID Document Details")
    table.add_column("Attribute", style="cyan")
    table.add_column("Value", style="bold white")
    table.add_row("Document Type", doc.card_type.value)
    table.add_row("Document Number", doc.document_number or "N/A")
    table.add_row("Full Name", doc.full_name or "N/A")
    table.add_row("Date of Birth", doc.date_of_birth or "N/A")
    table.add_row("Blood Group", doc.blood_group or "N/A")
    if doc.mrz_data:
        table.add_row("MRZ Checksum", "[green]Valid[/green]" if doc.mrz_data.document_number_check else "[red]Invalid[/red]")
    console.print(table)

    if output_json:
        with open(output_json, "w", encoding="utf-8") as f:
            f.write(doc.model_dump_json(indent=2))
        console.print(f"[bold green]✓ ID JSON saved to:[/bold green] {output_json}")


@app.command()
def alpr(
    input_file: Path = typer.Argument(..., help="Path to CCTV video frame image"),
):
    """Use Case 4: Vehicle License Plate Recognition for CCTV Surveillance."""
    console.print(f"[bold cyan]Processing CCTV frame for license plates:[/bold cyan] {input_file}")
    alpr_engine = VehicleALPR()
    detections = alpr_engine.process_frame(input_file)

    if not detections:
        console.print("[yellow]No license plates detected in frame.[/yellow]")
        return

    table = Table(title="Detected License Plates")
    table.add_column("Plate Text", style="bold white")
    table.add_column("Normalized", style="cyan")
    table.add_column("Format Type", style="magenta")
    table.add_column("Syntax Valid?", justify="center")
    table.add_column("Confidence", justify="right")

    for d in detections:
        table.add_row(
            d.plate_text,
            d.normalized_text,
            d.format_name or "Custom/Unknown",
            "[green]✓ YES[/green]" if d.is_format_valid else "[yellow]No[/yellow]",
            f"{d.confidence:.2%}",
        )
    console.print(table)


@app.command()
def benchmark(
    iterations: int = typer.Option(5, "--iterations", "-n", help="Number of benchmark iterations"),
):
    """Run CPU inference speed and latency benchmark."""
    console.print(f"[bold cyan]Running CPU OCR Benchmark ({iterations} iterations)...[/bold cyan]")
    from source.core.factory import get_ocr_engine
    engine = get_ocr_engine()

    # Synthetic realistic document image
    img = np.full((800, 600, 3), 255, dtype=np.uint8)
    for i in range(15):
        cv2.putText(img, f"Document Line {i+1}: Testing CPU OCR Performance", (40, 50 + i * 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    latencies = []
    for i in range(iterations):
        t0 = time.perf_counter()
        res = engine.predict(img)
        dt = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt)
        console.print(f"Iteration {i+1}: {dt:.2f} ms ({len(res.lines)} lines recognized)")

    p50 = float(np.median(latencies))
    mean = float(np.mean(latencies))
    fps = 1000.0 / mean

    console.print(Panel(
        f"[green]P50 Latency:[/green] {p50:.2f} ms\n"
        f"[green]Mean Latency:[/green] {mean:.2f} ms\n"
        f"[green]Throughput:[/green] {fps:.2f} pages/sec",
        title="Benchmark Summary"
    ))


if __name__ == "__main__":
    app()
