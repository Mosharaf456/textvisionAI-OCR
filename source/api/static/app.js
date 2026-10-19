// ==============================================================================
// CPU-Based OCR Suite - Frontend Application Logic
// ==============================================================================

document.addEventListener("DOMContentLoaded", () => {
    initTabs();
    initHealthCheck();
    initDocumentScanner();
    initPdfReader();
    initStructuredExtractor();
    initAlpr();
});

// ------------------------------------------------------------------------------
// Global Helpers
// ------------------------------------------------------------------------------
const spinner = {
    show: (msg = "Processing with CPU OCR...") => {
        const el = document.getElementById("loadingOverlay");
        const txt = document.getElementById("loadingMessage");
        if (txt) txt.textContent = msg;
        if (el) el.classList.add("active");
    },
    hide: () => {
        const el = document.getElementById("loadingOverlay");
        if (el) el.classList.remove("active");
    }
};

function initHealthCheck() {
    const dot = document.getElementById("statusIndicator");
    const text = document.getElementById("statusText");
    const latencyPill = document.getElementById("latencyBadge");

    async function check() {
        const t0 = performance.now();
        try {
            const res = await fetch("/health");
            const dt = Math.round(performance.now() - t0);
            if (res.ok) {
                const data = await res.json();
                dot.style.backgroundColor = "var(--accent-emerald)";
                dot.style.boxShadow = "0 0 10px var(--accent-emerald)";
                text.textContent = "Online (CPU)";
                latencyPill.textContent = `${dt} ms`;
            } else {
                throw new Error();
            }
        } catch {
            dot.style.backgroundColor = "var(--accent-rose)";
            dot.style.boxShadow = "0 0 10px var(--accent-rose)";
            text.textContent = "Offline";
            latencyPill.textContent = "-- ms";
        }
    }
    check();
    setInterval(check, 10000);
}

function initTabs() {
    const tabBtns = document.querySelectorAll(".tab-btn");
    const contents = document.querySelectorAll(".tab-content");

    tabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            tabBtns.forEach(b => b.classList.remove("active"));
            contents.forEach(c => c.classList.remove("active"));

            btn.classList.add("active");
            const targetId = btn.getAttribute("data-tab");
            const targetContent = document.getElementById(targetId);
            if (targetContent) targetContent.classList.add("active");
        });
    });
}

function drawImageAndBoxes(canvas, imageSource, boxes = [], showBoxes = true) {
    const ctx = canvas.getContext("2d");
    const img = new Image();

    img.onload = () => {
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0);

        if (showBoxes && boxes && boxes.length > 0) {
            boxes.forEach(item => {
                const pts = item.bbox || item;
                if (!pts || pts.length < 4) return;

                ctx.beginPath();
                ctx.moveTo(pts[0][0], pts[0][1]);
                ctx.lineTo(pts[1][0], pts[1][1]);
                ctx.lineTo(pts[2][0], pts[2][1]);
                ctx.lineTo(pts[3][0], pts[3][1]);
                ctx.closePath();

                // Translucent fill
                ctx.fillStyle = "rgba(6, 182, 212, 0.15)";
                ctx.fill();

                // High-visibility border
                ctx.lineWidth = Math.max(2, Math.round(canvas.width / 500));
                ctx.strokeStyle = "#06b6d4";
                ctx.stroke();
            });
        }
    };

    if (typeof imageSource === "string") {
        img.src = imageSource;
    } else if (imageSource instanceof File || imageSource instanceof Blob) {
        img.src = URL.createObjectURL(imageSource);
    }
}

// ------------------------------------------------------------------------------
// TAB 1: Document Scanner
// ------------------------------------------------------------------------------
function initDocumentScanner() {
    const fileInput = document.getElementById("fileDoc");
    const dropZone = document.getElementById("dropDoc");
    const btnLoadSample = document.getElementById("btnLoadDocSample");
    const btnExportPdf = document.getElementById("btnExportPdf");
    const btnCopy = document.getElementById("btnCopyText");
    const canvas = document.getElementById("canvasDoc");
    const chkDeskew = document.getElementById("chkDeskew");
    const chkShowBoxes = document.getElementById("chkShowBoxes");

    let currentFile = null;
    let currentResult = null;

    // Subtabs
    const subtabBtns = document.querySelectorAll(".btn-subtab");
    subtabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            subtabBtns.forEach(b => b.classList.remove("active"));
            document.querySelectorAll(".subtab-content").forEach(c => c.classList.remove("active"));
            btn.classList.add("active");
            const target = document.getElementById(btn.getAttribute("data-subtab"));
            if (target) target.classList.add("active");
        });
    });

    chkShowBoxes.addEventListener("change", () => {
        if (currentFile && currentResult) {
            drawImageAndBoxes(canvas, currentFile, currentResult.lines, chkShowBoxes.checked);
        }
    });

    async function processFile(file) {
        currentFile = file;
        spinner.show("Scanning document & running Recursive XY-Cut...");

        const formData = new FormData();
        formData.append("file", file);

        const deskew = chkDeskew.checked;
        const t0 = performance.now();

        try {
            const res = await fetch(`/api/v1/scan?deskew=${deskew}`, {
                method: "POST",
                body: formData
            });

            if (!res.ok) throw new Error(await res.text());
            const data = await res.json();
            currentResult = data;

            // Update stats
            document.getElementById("statsDoc").style.display = "flex";
            document.getElementById("statLines").textContent = data.total_lines;
            document.getElementById("statConfidence").textContent = `${Math.round(data.confidence * 100)}%`;
            document.getElementById("statSkew").textContent = `${data.skew_angle.toFixed(1)}°`;
            document.getElementById("statTime").textContent = `${Math.round(data.duration_ms)} ms`;

            // Outputs
            document.getElementById("outMarkdown").textContent = data.markdown || "No text detected";
            document.getElementById("outPlainText").textContent = data.plain_text || "No text detected";
            document.getElementById("outJson").textContent = JSON.stringify(data.lines, null, 2);

            // Enable PDF export
            btnExportPdf.disabled = false;

            // Draw bounding boxes on canvas
            drawImageAndBoxes(canvas, file, data.lines, chkShowBoxes.checked);
        } catch (err) {
            alert("Scan failed: " + err.message);
        } finally {
            spinner.hide();
        }
    }

    fileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files[0]) {
            processFile(e.target.files[0]);
        }
    });

    btnLoadSample.addEventListener("click", async () => {
        spinner.show("Loading sample document...");
        try {
            const res = await fetch("/api/v1/sample/document");
            if (!res.ok) throw new Error("Failed to load sample");
            const blob = await res.blob();
            const file = new File([blob], "sample_document.png", { type: "image/png" });
            processFile(file);
        } catch (err) {
            alert(err.message);
            spinner.hide();
        }
    });

    btnExportPdf.addEventListener("click", async () => {
        if (!currentFile) return;
        spinner.show("Generating Searchable PDF with invisible text layer...");
        try {
            const formData = new FormData();
            formData.append("file", currentFile);
            const res = await fetch(`/api/v1/scan?export_pdf=true`, {
                method: "POST",
                body: formData
            });
            if (!res.ok) throw new Error("Failed to export PDF");
            const blob = await res.blob();
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = `searchable_${currentFile.name.replace(/\.[^/.]+$/, "")}.pdf`;
            a.click();
            URL.revokeObjectURL(url);
        } catch (err) {
            alert(err.message);
        } finally {
            spinner.hide();
        }
    });

    btnCopy.addEventListener("click", () => {
        const activeSubtab = document.querySelector(".subtab-content.active");
        if (activeSubtab) {
            const text = activeSubtab.querySelector("pre").textContent;
            navigator.clipboard.writeText(text);
            const orig = btnCopy.textContent;
            btnCopy.textContent = "✓ Copied!";
            setTimeout(() => btnCopy.textContent = orig, 1500);
        }
    });
}

// ------------------------------------------------------------------------------
// TAB 2: PDF & Image Reader
// ------------------------------------------------------------------------------
function initPdfReader() {
    const fileInput = document.getElementById("filePdf");
    const btnLoadSample = document.getElementById("btnLoadPdfSample");
    const btnCopy = document.getElementById("btnCopyPdfText");
    const chkForceOcr = document.getElementById("chkForceOcr");
    const outText = document.getElementById("outPdfText");
    const pageList = document.getElementById("pdfPageList");

    async function processPdf(file) {
        spinner.show("Ingesting PDF (evaluating digital text fastpath)...");
        const formData = new FormData();
        formData.append("file", file);

        const forceOcr = chkForceOcr.checked;
        try {
            const res = await fetch(`/api/v1/read?force_ocr=${forceOcr}`, {
                method: "POST",
                body: formData
            });
            if (!res.ok) throw new Error(await res.text());
            const data = await res.json();

            outText.textContent = data.full_text || "No text found in document.";

            // Render page list items
            pageList.innerHTML = "";
            data.pages.forEach(p => {
                const item = document.createElement("div");
                item.className = "entity-card";
                item.style.marginBottom = "0.5rem";
                const badgeClass = p.extraction_method === "digital_fastpath" ? "badge-success" : "badge-info";
                const badgeLabel = p.extraction_method === "digital_fastpath" ? "⚡ Digital Fast-Path" : "📷 Raster OCR";

                item.innerHTML = `
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <strong>Page ${p.page_number}</strong>
                        <span class="badge ${badgeClass}">${badgeLabel}</span>
                    </div>
                    <div style="font-size: 0.8rem; color: var(--text-secondary); margin-top: 0.3rem;">
                        Lines: ${p.lines_count} | Confidence: ${Math.round((p.confidence || 1.0) * 100)}%
                    </div>
                `;
                pageList.appendChild(item);
            });
        } catch (err) {
            alert("Error reading document: " + err.message);
        } finally {
            spinner.hide();
        }
    }

    fileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files[0]) {
            processPdf(e.target.files[0]);
        }
    });

    btnLoadSample.addEventListener("click", async () => {
        spinner.show("Loading sample document...");
        try {
            const res = await fetch("/api/v1/sample/document");
            const blob = await res.blob();
            processPdf(new File([blob], "sample_doc.png", { type: "image/png" }));
        } catch (err) {
            alert(err.message);
            spinner.hide();
        }
    });

    btnCopy.addEventListener("click", () => {
        navigator.clipboard.writeText(outText.textContent);
        btnCopy.textContent = "✓ Copied!";
        setTimeout(() => btnCopy.textContent = "📋 Copy Full Text", 1500);
    });
}

// ------------------------------------------------------------------------------
// TAB 3: Structured Extractor (Invoices & ID Cards)
// ------------------------------------------------------------------------------
function initStructuredExtractor() {
    const modeInvoice = document.getElementById("modeInvoice");
    const modeID = document.getElementById("modeID");
    const fileInput = document.getElementById("fileStructured");
    const btnLoadSample = document.getElementById("btnLoadStructuredSample");
    const canvas = document.getElementById("canvasStructured");
    const inputTitle = document.getElementById("structuredInputTitle");
    const dropPrompt = document.getElementById("structuredDropPrompt");
    const mathBadge = document.getElementById("mathBadge");

    const viewInvoice = document.getElementById("viewInvoiceContainer");
    const viewId = document.getElementById("viewIdCardContainer");
    const outJson = document.getElementById("outStructuredJson");

    let currentMode = "invoice"; // 'invoice' or 'id'

    modeInvoice.addEventListener("click", () => {
        currentMode = "invoice";
        modeInvoice.classList.add("active");
        modeID.classList.remove("active");
        viewInvoice.style.display = "block";
        viewId.style.display = "none";
        inputTitle.textContent = "Invoice Image";
        dropPrompt.innerHTML = "<strong>Upload invoice image</strong>";
        btnLoadSample.textContent = "⚡ Load Sample Invoice";
    });

    modeID.addEventListener("click", () => {
        currentMode = "id";
        modeID.classList.add("active");
        modeInvoice.classList.remove("active");
        viewInvoice.style.display = "none";
        viewId.style.display = "block";
        inputTitle.textContent = "ID Card / Passport Image";
        dropPrompt.innerHTML = "<strong>Upload ID card or passport image</strong>";
        btnLoadSample.textContent = "⚡ Load Sample ID Card";
    });

    async function processStructured(file) {
        spinner.show(currentMode === "invoice" ? "Extracting invoice key-values and validating totals..." : "Extracting ID fields & parsing MRZ...");
        const formData = new FormData();
        formData.append("file", file);

        const endpoint = currentMode === "invoice" ? "/api/v1/extract-invoice" : "/api/v1/extract-id";

        try {
            const res = await fetch(endpoint, {
                method: "POST",
                body: formData
            });
            if (!res.ok) throw new Error(await res.text());
            const data = await res.json();

            outJson.textContent = JSON.stringify(data, null, 2);
            drawImageAndBoxes(canvas, file);

            if (currentMode === "invoice") {
                // Populate invoice cards
                document.getElementById("invNo").textContent = data.invoice_number || "N/A";
                document.getElementById("invDate").textContent = data.invoice_date || "N/A";
                document.getElementById("invVendor").textContent = data.vendor_name || "N/A";
                document.getElementById("invCustomer").textContent = data.customer_name || "N/A";

                const totals = data.totals || {};
                document.getElementById("invSubtotal").textContent = totals.subtotal ? `$${totals.subtotal.toFixed(2)}` : "--";
                document.getElementById("invTax").textContent = totals.tax ? `$${totals.tax.toFixed(2)}` : "--";
                document.getElementById("invTotal").textContent = totals.total ? `$${totals.total.toFixed(2)}` : "--";

                // Math badge
                mathBadge.style.display = "inline-flex";
                if (data.is_math_valid) {
                    mathBadge.className = "badge badge-success";
                    mathBadge.textContent = "✓ Math Balanced";
                } else {
                    mathBadge.className = "badge badge-warning";
                    mathBadge.textContent = "⚠️ Arithmetic Discrepancy";
                }

                // Table items
                const tbody = document.querySelector("#tableLineItems tbody");
                tbody.innerHTML = "";
                if (data.line_items && data.line_items.length > 0) {
                    data.line_items.forEach(itm => {
                        const tr = document.createElement("tr");
                        tr.innerHTML = `
                            <td>${itm.description}</td>
                            <td class="text-right">${itm.quantity || ""}</td>
                            <td class="text-right">${itm.unit_price ? `$${itm.unit_price.toFixed(2)}` : ""}</td>
                            <td class="text-right" style="color: var(--accent-emerald); font-weight: 600;">$${itm.amount ? itm.amount.toFixed(2) : ""}</td>
                        `;
                        tbody.appendChild(tr);
                    });
                } else {
                    tbody.innerHTML = `<tr><td colspan="4" class="text-center">No line items parsed</td></tr>`;
                }
            } else {
                // Populate ID card fields
                mathBadge.style.display = "none";
                document.getElementById("idType").textContent = (data.card_type || "UNKNOWN").toUpperCase();
                document.getElementById("idNumber").textContent = data.document_number || "N/A";
                document.getElementById("idName").textContent = data.full_name || "N/A";
                document.getElementById("idDob").textContent = data.date_of_birth || "N/A";
                document.getElementById("idBlood").textContent = data.blood_group || "N/A";

                const mrzValid = data.mrz_data ? (data.mrz_data.document_number_check ? "✓ Valid Checksum" : "⚠️ Invalid Checksum") : "No MRZ";
                document.getElementById("idMrz").textContent = mrzValid;
            }
        } catch (err) {
            alert("Extraction error: " + err.message);
        } finally {
            spinner.hide();
        }
    }

    fileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files[0]) {
            processStructured(e.target.files[0]);
        }
    });

    btnLoadSample.addEventListener("click", async () => {
        const sampleType = currentMode === "invoice" ? "invoice" : "id_card";
        spinner.show(`Loading sample ${currentMode}...`);
        try {
            const res = await fetch(`/api/v1/sample/${sampleType}`);
            const blob = await res.blob();
            processStructured(new File([blob], `sample_${sampleType}.png`, { type: "image/png" }));
        } catch (err) {
            alert(err.message);
            spinner.hide();
        }
    });
}

// ------------------------------------------------------------------------------
// TAB 4: CCTV ALPR
// ------------------------------------------------------------------------------
function initAlpr() {
    const fileInput = document.getElementById("fileAlpr");
    const btnLoadSample = document.getElementById("btnLoadAlprSample");
    const canvas = document.getElementById("canvasAlpr");
    const resultsList = document.getElementById("alprResultsList");
    const countBadge = document.getElementById("alprCountBadge");

    async function processAlpr(file) {
        spinner.show("Detecting vehicle license plates & applying CLAHE...");
        const formData = new FormData();
        formData.append("file", file);

        try {
            const res = await fetch("/api/v1/alpr", {
                method: "POST",
                body: formData
            });
            if (!res.ok) throw new Error(await res.text());
            const detections = await res.json();

            countBadge.textContent = `${detections.length} Plates`;

            // Draw plate bounding boxes
            const boxes = detections.map(d => {
                if (!d.bbox) return null;
                const b = d.bbox;
                return [
                    [b.x, b.y],
                    [b.x + b.width, b.y],
                    [b.x + b.width, b.y + b.height],
                    [b.x, b.y + b.height]
                ];
            }).filter(Boolean);

            drawImageAndBoxes(canvas, file, boxes, true);

            // Populate cards
            resultsList.innerHTML = "";
            if (detections.length === 0) {
                resultsList.innerHTML = `<div class="empty-state">No license plates recognized in this frame.</div>`;
                return;
            }

            detections.forEach(d => {
                const card = document.createElement("div");
                card.className = "plate-card";
                const formatLabel = d.format_name ? d.format_name.replace(/_/g, " ") : "Custom/Unknown";
                const validBadge = d.is_format_valid
                    ? `<span class="badge badge-success">✓ Valid Syntax</span>`
                    : `<span class="badge badge-info">Standard Match</span>`;

                card.innerHTML = `
                    <div class="plate-visual-tag">${d.normalized_text}</div>
                    <div class="plate-details">
                        <span style="font-size: 0.85rem; font-weight: 600; color: var(--accent-cyan);">${formatLabel}</span>
                        <div>${validBadge} <span class="badge" style="background: rgba(255,255,255,0.06);">${Math.round(d.confidence * 100)}% Conf</span></div>
                    </div>
                `;
                resultsList.appendChild(card);
            });
        } catch (err) {
            alert("ALPR Error: " + err.message);
        } finally {
            spinner.hide();
        }
    }

    fileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files[0]) {
            processAlpr(e.target.files[0]);
        }
    });

    btnLoadSample.addEventListener("click", async () => {
        spinner.show("Loading sample CCTV frame...");
        try {
            const res = await fetch("/api/v1/sample/cctv");
            const blob = await res.blob();
            processAlpr(new File([blob], "sample_cctv.png", { type: "image/png" }));
        } catch (err) {
            alert(err.message);
            spinner.hide();
        }
    });
}
