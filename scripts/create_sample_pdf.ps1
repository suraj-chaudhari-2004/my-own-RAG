$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$outputDir = Join-Path $projectRoot "sample_pdfs"
$outputPath = Join-Path $outputDir "rag_test_document.pdf"

New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

function Escape-PdfText {
    param([string]$Text)
    return $Text.Replace("\", "\\").Replace("(", "\(").Replace(")", "\)")
}

$lines = @(
    "Northwind AI Research Handbook",
    "Version: July 2026",
    "",
    "Project Phoenix is an internal retrieval augmented generation pilot.",
    "Its main goal is to answer employee questions from trusted PDF documents.",
    "The pilot owner is Mira Shah from the Knowledge Systems team.",
    "",
    "The application indexes PDF files by extracting text, splitting it into chunks,",
    "creating local embeddings, and retrieving the most relevant chunks for each question.",
    "",
    "OpenRouter is the approved cloud API provider for this pilot.",
    "The recommended OpenRouter model for initial testing is openai/gpt-4o-mini.",
    "The local model option should use Ollama with llama3.1 unless another model is configured.",
    "",
    "Security rules:",
    "1. Do not upload payroll records, medical records, or private customer data.",
    "2. Keep API keys in .env or .streamlit/secrets.toml, not in source code.",
    "3. Answers should cite the retrieved PDF source when possible.",
    "",
    "Evaluation checklist:",
    "Ask who owns Project Phoenix.",
    "Ask which API provider is approved.",
    "Ask which local model is recommended.",
    "Ask what data should not be uploaded.",
    "",
    "If a question asks about something not written here, the assistant should say",
    "that it could not find the answer in the PDFs."
)

$textOps = New-Object System.Collections.Generic.List[string]
$textOps.Add("BT")
$textOps.Add("/F1 12 Tf")
$textOps.Add("72 740 Td")
foreach ($line in $lines) {
    if ($line.Length -eq 0) {
        $textOps.Add("0 -18 Td")
    }
    else {
        $textOps.Add("(" + (Escape-PdfText $line) + ") Tj")
        $textOps.Add("0 -18 Td")
    }
}
$textOps.Add("ET")
$content = ($textOps -join "`n") + "`n"
$contentLength = [System.Text.Encoding]::ASCII.GetByteCount($content)

$objects = @(
    "1 0 obj`n<< /Type /Catalog /Pages 2 0 R >>`nendobj`n",
    "2 0 obj`n<< /Type /Pages /Kids [3 0 R] /Count 1 >>`nendobj`n",
    "3 0 obj`n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>`nendobj`n",
    "4 0 obj`n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>`nendobj`n",
    "5 0 obj`n<< /Length $contentLength >>`nstream`n$content`nendstream`nendobj`n"
)

$pdf = New-Object System.Text.StringBuilder
[void]$pdf.Append("%PDF-1.4`n")
$offsets = New-Object System.Collections.Generic.List[int]

foreach ($object in $objects) {
    $offsets.Add([System.Text.Encoding]::ASCII.GetByteCount($pdf.ToString()))
    [void]$pdf.Append($object)
}

$xrefOffset = [System.Text.Encoding]::ASCII.GetByteCount($pdf.ToString())
[void]$pdf.Append("xref`n")
[void]$pdf.Append("0 6`n")
[void]$pdf.Append("0000000000 65535 f `n")
foreach ($offset in $offsets) {
    [void]$pdf.Append(("{0:D10} 00000 n `n" -f $offset))
}
[void]$pdf.Append("trailer`n")
[void]$pdf.Append("<< /Size 6 /Root 1 0 R >>`n")
[void]$pdf.Append("startxref`n")
[void]$pdf.Append("$xrefOffset`n")
[void]$pdf.Append("%%EOF`n")

$bytes = [System.Text.Encoding]::ASCII.GetBytes($pdf.ToString())
[System.IO.File]::WriteAllBytes($outputPath, $bytes)

Write-Host "Created $outputPath"
