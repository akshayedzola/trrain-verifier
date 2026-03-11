# TRRAIN Offer Letter Verifier — EdZola POC

AI-powered offer letter authenticity verification using a 6-layer forensic pipeline.

Built by [EdZola Technologies](https://edzolatechnologies.com) for TRRAIN's employment verification programme.

## What it does

Analyses offer letters submitted by job candidates and scores them for authenticity across 6 layers:

| Layer | Check |
|-------|-------|
| L1 | Text extraction (OCR fallback for scanned docs) |
| L2 | Field extraction + completeness via Claude AI |
| L3 | PDF metadata forensics (tool classification, modification history) |
| L4 | Visual forensics — Error Level Analysis (ELA) for manipulation detection |
| L5 | Historical comparison against known employer templates |
| L6 | Geotagged workplace photo verification (GPS + timestamp) |

Outputs an authenticity confidence score (0–100) with full breakdown and exportable PDF report.

## Running locally

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here
python trrain_offer_verifier.py
# → http://localhost:5001
```

## Deploying to Render

1. Push this repo to GitHub
2. Create a new Web Service on [render.com](https://render.com)
3. Connect the repo
4. Set environment variable: `ANTHROPIC_API_KEY=your_key`
5. Build command: `pip install -r requirements.txt`
6. Start command: `python trrain_offer_verifier.py`

## Stack

- Python / Flask
- PyMuPDF (text extraction + metadata)
- Anthropic Claude API (field extraction + completeness)
- Pillow + OpenCV (ELA + layout forensics)
- NumPy

---

*POC build — not for production use without additional hardening.*
