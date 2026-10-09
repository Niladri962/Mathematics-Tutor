"""Ingest NCERT textbook PDFs into page-referenced chunks.

Usage:
    1. Download chapter PDFs from https://ncert.nic.in/textbook.php into data/raw/ncert/
       Name them like  class8_ch02_linear_equations.pdf  so the reference reads well.
    2. python -m src.ingest_ncert

Output: data/processed/ncert_chunks.json, picked up automatically by the tutor.
Each chunk keeps its file and page, so citations point to an exact textbook page.
"""
from __future__ import annotations

import json
import re
import sys

from .data import PROCESSED, ROOT

RAW = ROOT / "data" / "raw" / "ncert"


def clean(text: str) -> str:
    text = re.sub(r"Reprint\s*\d{4}-\d{2}", " ", text)      # NCERT page footers
    text = re.sub(r"-\n(\w)", r"\1", text)                   # rejoin hyphenated words
    return re.sub(r"\s+", " ", text).strip()


def chunk_words(text: str, size: int = 180, overlap: int = 40):
    words = text.split()
    for i in range(0, len(words), size - overlap):
        piece = words[i:i + size]
        if len(piece) > 30:
            yield " ".join(piece)


def main():
    try:
        import fitz  # PyMuPDF
    except ImportError:
        sys.exit("Install PyMuPDF first: pip install pymupdf")
    pdfs = sorted(RAW.glob("*.pdf"))
    if not pdfs:
        sys.exit(f"No PDFs found in {RAW}. Download NCERT chapters there first.")
    chunks = []
    for pdf in pdfs:
        title = pdf.stem.replace("_", " ").title()
        with fitz.open(pdf) as doc:
            for page_no, page in enumerate(doc, start=1):
                for j, piece in enumerate(chunk_words(clean(page.get_text()))):
                    chunks.append({"id": f"{pdf.stem}_p{page_no}_{j}", "title": title,
                                   "text": piece, "ref": f"NCERT {title}, page {page_no}"})
    PROCESSED.mkdir(parents=True, exist_ok=True)
    out = PROCESSED / "ncert_chunks.json"
    out.write_text(json.dumps(chunks, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote {len(chunks)} chunks from {len(pdfs)} PDFs to {out}")


if __name__ == "__main__":
    main()
