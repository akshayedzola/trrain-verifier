import os
import json
import base64
import hashlib
import re
import math
import tempfile
from datetime import datetime
from flask import Flask, request, jsonify, render_template_string
import fitz  # PyMuPDF
from PIL import Image, ImageChops, ImageEnhance, ImageFilter
import numpy as np
import cv2
import anthropic

app = Flask(__name__)
UPLOAD_FOLDER = "/tmp/trrain_uploads"
HISTORY_FILE = "/tmp/trrain_history.json"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ─────────────────────────────────────────────
# LAYER 1: OCR / Text Extraction via PyMuPDF
# ─────────────────────────────────────────────
def layer1_extract_text(pdf_path):
    result = {"text": "", "page_count": 0, "has_images": False, "extraction_method": "native"}
    try:
        doc = fitz.open(pdf_path)
        result["page_count"] = len(doc)
        full_text = ""
        image_count = 0
        for page in doc:
            text = page.get_text()
            full_text += text
            image_list = page.get_images()
            image_count += len(image_list)
        result["text"] = full_text.strip()
        result["has_images"] = image_count > 0
        result["image_count"] = image_count
        if len(full_text.strip()) < 50:
            result["extraction_method"] = "image_based_ocr_needed"
            # Try to extract from first page image for basic OCR via Claude
            page = doc[0]
            mat = fitz.Matrix(2, 2)
            clip = page.get_pixmap(matrix=mat)
            img_path = f"{UPLOAD_FOLDER}/page_render.png"
            clip.save(img_path)
            result["rendered_page"] = img_path
        doc.close()
    except Exception as e:
        result["error"] = str(e)
    return result

# ─────────────────────────────────────────────
# LAYER 2: Field Extraction + Completeness Check via Claude
# ─────────────────────────────────────────────
def layer2_field_extraction(text, image_path=None):
    client = anthropic.Anthropic()
    
    required_fields = [
        "candidate_name", "job_title", "salary", "joining_date",
        "employer_name", "employer_address", "employer_email",
        "employer_phone", "signature_block", "reporting_manager"
    ]
    
    prompt = f"""You are a document verification assistant for an NGO employment program.

Analyze this offer letter text and extract the following fields. Return ONLY valid JSON.

Required fields to extract:
- candidate_name
- job_title  
- salary (monthly/annual amount)
- joining_date
- employer_name
- employer_address
- employer_email
- employer_phone
- signature_block (name/designation of signatory)
- reporting_manager (optional)

Also assess:
- is_offer_letter: true or false — is this actually an employment offer letter? Be strict. Policy documents, HR handbooks, attendance sheets, appointment confirmations, NOCs, experience letters, salary slips are NOT offer letters.
- document_type: short label for what this document is (e.g. "offer letter", "attendance policy", "salary slip", "experience letter", "HR policy", "unknown")
- overall_completeness_score: 0-100 (how complete is the letter AS AN OFFER LETTER — if not an offer letter, set to 0)
- missing_fields: list of fields not found
- suspicious_text_patterns: list of any suspicious patterns you notice (e.g., placeholder text like "[Name]" or "[Date]", inconsistent formatting, copy-pasted boilerplate, generic templates with unfilled fields)
- has_placeholder_text: true or false — are there any unfilled template placeholders like [Candidate Name], [Date], [Salary], etc.?
- employer_email_domain: extract just the domain part of employer email if present (e.g. "relianceindustries.com") or null
- employer_email_is_free: true if employer email uses gmail/yahoo/hotmail/outlook.com/rediffmail/ymail — false otherwise — null if no email found
- joining_date_iso: the joining date reformatted as YYYY-MM-DD if you can parse it, otherwise null
- text_quality: "high", "medium", or "low"
- language_consistency: true/false (is the language consistent throughout)

Letter text:
\"\"\"
{text[:4000]}
\"\"\"

Return ONLY a JSON object, no markdown, no explanation."""

    messages = [{"role": "user", "content": prompt}]
    
    # If we have an image, use vision
    if image_path and os.path.exists(image_path):
        with open(image_path, "rb") as f:
            img_data = base64.b64encode(f.read()).decode()
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img_data}},
                {"type": "text", "text": prompt}
            ]
        }]

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        messages=messages
    )
    
    raw = response.content[0].text.strip()
    # Strip markdown if present
    raw = re.sub(r'^```json\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)
    
    try:
        return json.loads(raw)
    except:
        return {"error": "parse_failed", "raw": raw[:500], "overall_completeness_score": 0}

# ─────────────────────────────────────────────
# LAYER 3: PDF Metadata Forensics
# ─────────────────────────────────────────────
def layer3_metadata_forensics(pdf_path):
    result = {
        "anomalies": [],
        "metadata": {},
        "risk_score": 0,
        "signals": []
    }
    
    try:
        doc = fitz.open(pdf_path)
        meta = doc.metadata
        result["metadata"] = meta
        
        risk = 0
        
        # Check creation vs modification date
        created = meta.get("creationDate", "")
        modified = meta.get("modDate", "")
        
        if created and modified and created != modified:
            result["anomalies"].append("Document was modified after creation")
            result["signals"].append({"type": "warning", "msg": f"Modified after creation — Created: {created[:10]}, Modified: {modified[:10]}"})
            risk += 20
        
        # Check producer / creator tool — context-aware classification
        producer = meta.get("producer", "").lower()
        creator = meta.get("creator", "").lower()
        combined = producer + " " + creator

        # High risk: image editors — no legitimate offer letter workflow uses these
        image_editors = ["photoshop", "gimp", "inkscape", "illustrator", "paint.net", "affinity"]
        # Normal: cloud/office tools — expected for legitimate docs
        cloud_tools = ["google", "microsoft", "word", "libreoffice", "openoffice", "wps", "acrobat", "pdf24", "docx"]
        # Scanner apps — expected for physical letters, lower risk
        scanner_apps = ["camscanner", "adobe scan", "turboscan", "scanbot", "office lens", "genius scan"]
        # Post-processors — strip metadata by design, not a red flag
        post_processors = ["ilovepdf", "smallpdf", "compress", "ghostscript", "sejda", "pdfcandy", "pdf2go", "pdfescape"]

        tool_classified = False
        for t in image_editors:
            if t in combined:
                result["anomalies"].append(f"Document processed with image editor: {meta.get('producer','') or meta.get('creator','')}")
                result["signals"].append({"type": "danger", "msg": f"⚠ Image editing tool detected ({t}) — offer letters should not originate from image editors"})
                result["tool_category"] = "image_editor"
                risk += 40
                tool_classified = True
                break

        if not tool_classified:
            for t in scanner_apps:
                if t in combined:
                    result["signals"].append({"type": "info", "msg": f"Scanner app detected ({meta.get('creator','') or meta.get('producer','')}) — consistent with scanned physical letter"})
                    result["tool_category"] = "scanner"
                    tool_classified = True
                    break

        if not tool_classified:
            for t in post_processors:
                if t in combined:
                    result["signals"].append({"type": "info", "msg": f"PDF post-processor detected ({t}) — metadata may have been stripped by tool, not tampering"})
                    result["tool_category"] = "post_processor"
                    tool_classified = True
                    break

        if not tool_classified:
            for t in cloud_tools:
                if t in combined:
                    result["signals"].append({"type": "ok", "msg": f"Document origin: {meta.get('creator','') or meta.get('producer','')} — expected for legitimate digital offer letters"})
                    result["tool_category"] = "cloud_office"
                    tool_classified = True
                    break

        if not tool_classified:
            if combined.strip():
                result["signals"].append({"type": "info", "msg": f"Tool: {meta.get('creator','') or meta.get('producer','')} — unrecognised, manual review advised"})
                result["tool_category"] = "unknown"

        # Missing metadata — interpret in context
        tool_cat = result.get("tool_category", "")
        if not created:
            if tool_cat in ("post_processor", "scanner"):
                result["signals"].append({"type": "info", "msg": "No creation date — expected when PDF passed through post-processing tool"})
            else:
                result["signals"].append({"type": "warning", "msg": "No creation date in metadata — may indicate stripping after edits"})
                risk += 8

        if not meta.get("author") and not meta.get("creator"):
            if tool_cat == "post_processor":
                result["signals"].append({"type": "info", "msg": "Author/creator stripped by post-processor — not inherently suspicious"})
            elif tool_cat == "scanner":
                result["signals"].append({"type": "info", "msg": "No author metadata — normal for scanner-generated PDFs"})
            else:
                result["signals"].append({"type": "warning", "msg": "No author or creator metadata — document origin unclear"})
                risk += 8
        
        # Check revision count
        pdf_version = doc.pdf_version()
        if pdf_version:
            result["signals"].append({"type": "info", "msg": f"PDF version: {pdf_version}"})
        
        # Check for incremental updates (sign of post-edit)
        # Count xref count vs expected
        xref_count = doc.xref_length()
        result["signals"].append({"type": "info", "msg": f"PDF object count: {xref_count}"})
        
        # Check for embedded JavaScript or unusual actions
        for i in range(min(xref_count, 500)):
            try:
                xref_obj = doc.xref_object(i)
                if xref_obj and "/JS" in xref_obj:
                    result["anomalies"].append("Embedded JavaScript found — unusual for offer letters")
                    result["signals"].append({"type": "danger", "msg": "Embedded JavaScript detected in PDF"})
                    risk += 30
                    break
            except:
                pass
        
        result["risk_score"] = min(risk, 100)
        doc.close()
        
    except Exception as e:
        result["error"] = str(e)
    
    return result

# ─────────────────────────────────────────────
# LAYER 4: Visual / Image Forensics (ELA + Font consistency)
# ─────────────────────────────────────────────
def layer4_visual_forensics(pdf_path):
    result = {
        "ela_risk": 0,
        "anomalies": [],
        "signals": [],
        "ela_image_b64": None
    }
    
    try:
        doc = fitz.open(pdf_path)
        page = doc[0]
        
        # Render page at high res
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat)
        img_path = f"{UPLOAD_FOLDER}/ela_input.png"
        pix.save(img_path)
        doc.close()
        
        # ── ELA Analysis ──
        original = Image.open(img_path).convert("RGB")
        temp_jpg = f"{UPLOAD_FOLDER}/ela_temp.jpg"
        original.save(temp_jpg, "JPEG", quality=85)
        recompressed = Image.open(temp_jpg)
        
        ela_image = ImageChops.difference(original, recompressed)
        
        # Amplify differences for visibility
        ela_array = np.array(ela_image, dtype=np.float32)
        ela_amplified = np.clip(ela_array * 15, 0, 255).astype(np.uint8)
        ela_pil = Image.fromarray(ela_amplified)
        
        # Save ELA image as base64
        ela_display_path = f"{UPLOAD_FOLDER}/ela_display.png"
        ela_pil.save(ela_display_path)
        with open(ela_display_path, "rb") as f:
            result["ela_image_b64"] = base64.b64encode(f.read()).decode()
        
        # Analyze ELA stats
        ela_mean = ela_array.mean()
        ela_std = ela_array.std()
        ela_max = ela_array.max()
        
        result["ela_stats"] = {
            "mean": round(float(ela_mean), 2),
            "std": round(float(ela_std), 2),
            "max": round(float(ela_max), 2)
        }
        
        # High std deviation suggests pasted/manipulated regions
        if ela_std > 18:
            result["anomalies"].append("High ELA variance — possible image manipulation or pasting detected")
            result["signals"].append({"type": "danger", "msg": f"ELA std deviation {ela_std:.1f} — elevated (threshold: 18). Possible content manipulation."})
            result["ela_risk"] = 70
        elif ela_std > 10:
            result["signals"].append({"type": "warning", "msg": f"ELA std deviation {ela_std:.1f} — slightly elevated. May indicate mixed content sources."})
            result["ela_risk"] = 35
        else:
            result["signals"].append({"type": "ok", "msg": f"ELA std deviation {ela_std:.1f} — normal range. No obvious manipulation detected."})
            result["ela_risk"] = 10
        
        # ── OpenCV: Font/Layout Consistency ──
        img_cv = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # Check for inconsistent regions using edge density
        edges = cv2.Canny(img_cv, 50, 150)
        
        # Divide into horizontal slices and check edge density variance
        h, w = edges.shape
        slice_height = h // 8
        densities = []
        for i in range(8):
            slice_e = edges[i*slice_height:(i+1)*slice_height, :]
            density = slice_e.sum() / (slice_height * w)
            densities.append(float(density))
        
        density_std = float(np.std(densities))
        result["layout_density_std"] = round(density_std, 4)
        
        if density_std > 0.05:
            result["signals"].append({"type": "warning", "msg": f"Layout density variation {density_std:.3f} — inconsistent sections detected"})
        else:
            result["signals"].append({"type": "ok", "msg": "Layout density consistent across document"})
        
        # Clean up
        os.remove(temp_jpg)
        
    except Exception as e:
        result["error"] = str(e)
    
    return result

# ─────────────────────────────────────────────
# LAYER 5: Historical Document Comparison
# ─────────────────────────────────────────────
def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE) as f:
            return json.load(f)
    return {}

def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

def text_fingerprint(text):
    """Create a simple structural fingerprint of a document"""
    words = text.lower().split()
    word_count = len(words)
    
    # Character frequency distribution
    char_freq = {}
    for c in text.lower():
        if c.isalpha():
            char_freq[c] = char_freq.get(c, 0) + 1
    total_chars = sum(char_freq.values()) or 1
    char_vector = [char_freq.get(chr(ord('a')+i), 0)/total_chars for i in range(26)]
    
    # Common structural words
    structural_words = ["dear", "hereby", "joining", "salary", "designation", "regards", "sincerely", "offer", "employment", "position"]
    struct_vector = [1 if w in text.lower() else 0 for w in structural_words]
    
    return {
        "word_count": word_count,
        "char_vector": char_vector,
        "struct_vector": struct_vector,
        "hash": hashlib.md5(text[:500].encode()).hexdigest()
    }

def cosine_similarity(v1, v2):
    dot = sum(a*b for a,b in zip(v1,v2))
    mag1 = math.sqrt(sum(a*a for a in v1))
    mag2 = math.sqrt(sum(b*b for b in v2))
    if mag1 == 0 or mag2 == 0:
        return 0
    return dot / (mag1 * mag2)

def layer5_historical_comparison(text, employer_name, fields):
    result = {
        "employer": employer_name,
        "history_count": 0,
        "similarity_score": None,
        "anomalies": [],
        "signals": [],
        "is_new_employer": True
    }
    
    history = load_history()
    fingerprint = text_fingerprint(text)
    
    employer_key = (employer_name or "unknown").lower().strip()
    
    if employer_key in history:
        result["is_new_employer"] = False
        past_docs = history[employer_key]
        result["history_count"] = len(past_docs)
        
        # Compare against historical fingerprints
        similarities = []
        for past in past_docs:
            sim = cosine_similarity(fingerprint["char_vector"], past["char_vector"])
            similarities.append(sim)
        
        avg_sim = sum(similarities) / len(similarities)
        result["similarity_score"] = round(avg_sim * 100, 1)
        
        if avg_sim < 0.7:
            result["anomalies"].append(f"Low similarity ({avg_sim:.0%}) to historical letters from {employer_name}")
            result["signals"].append({"type": "danger", "msg": f"Only {avg_sim:.0%} similar to {len(past_docs)} known letters from this employer — template may differ significantly"})
        elif avg_sim < 0.85:
            result["signals"].append({"type": "warning", "msg": f"{avg_sim:.0%} similarity to historical letters — moderate deviation from known template"})
        else:
            result["signals"].append({"type": "ok", "msg": f"{avg_sim:.0%} similarity to {len(past_docs)} historical letters from this employer — consistent template"})
        
        # Check for exact duplicate (suspicious)
        for past in past_docs:
            if past["hash"] == fingerprint["hash"]:
                result["is_exact_duplicate"] = True
                result["duplicate_processed_at"] = past.get("processed_at", "unknown date")
                result["anomalies"].append("Exact duplicate of previously seen document")
                result["signals"].append({"type": "danger", "msg": f"EXACT DUPLICATE — this document was already processed on {past.get('processed_at','unknown')[:10]}. Possible reuse/recycling."})
    else:
        result["signals"].append({"type": "info", "msg": f"First time seeing letters from '{employer_name}' — no historical baseline yet"})
    
    # Store this document in history
    if employer_key not in history:
        history[employer_key] = []
    
    history[employer_key].append({
        "char_vector": fingerprint["char_vector"],
        "struct_vector": fingerprint["struct_vector"],
        "word_count": fingerprint["word_count"],
        "hash": fingerprint["hash"],
        "processed_at": datetime.now().isoformat()
    })
    save_history(history)
    
    return result

# ─────────────────────────────────────────────
# LAYER 6: Geotag Verification
# ─────────────────────────────────────────────
def layer6_geotag_verification(image_file=None, employer_address=None, joining_date=None):
    result = {
        "has_geotag": False,
        "gps_coords": None,
        "timestamp": None,
        "address_match": None,
        "anomalies": [],
        "signals": []
    }
    
    if not image_file:
        result["signals"].append({"type": "info", "msg": "No workplace photo uploaded — geotag verification skipped"})
        return result
    
    try:
        img = Image.open(image_file)
        exif_data = img._getexif() if hasattr(img, '_getexif') and img._getexif() else {}
        
        if not exif_data:
            result["signals"].append({"type": "warning", "msg": "Photo has no EXIF metadata — cannot verify location or timestamp"})
            return result
        
        # GPS tags: 34853 = GPSInfo, 36867 = DateTimeOriginal
        GPS_TAG = 34853
        DATETIME_TAG = 36867
        
        gps_info = exif_data.get(GPS_TAG)
        datetime_str = exif_data.get(DATETIME_TAG)
        
        if datetime_str:
            result["timestamp"] = datetime_str
            result["signals"].append({"type": "info", "msg": f"Photo timestamp: {datetime_str}"})
            
            # Check if timestamp is near joining date
            if joining_date:
                try:
                    photo_dt = datetime.strptime(datetime_str, "%Y:%m:%d %H:%M:%S")
                    # Simple check — would compare with joining_date in production
                    result["signals"].append({"type": "ok", "msg": "Timestamp extracted — manual review against joining date recommended"})
                except:
                    pass
        else:
            result["signals"].append({"type": "warning", "msg": "No timestamp in photo EXIF data"})
        
        if gps_info:
            result["has_geotag"] = True
            
            def convert_to_degrees(value):
                d, m, s = value
                if isinstance(d, tuple): d = d[0] / d[1]
                if isinstance(m, tuple): m = m[0] / m[1]
                if isinstance(s, tuple): s = s[0] / s[1]
                return d + (m / 60.0) + (s / 3600.0)
            
            try:
                lat = convert_to_degrees(gps_info[2])
                if gps_info[1] == 'S': lat = -lat
                lon = convert_to_degrees(gps_info[4])
                if gps_info[3] == 'W': lon = -lon
                
                result["gps_coords"] = {"lat": round(lat, 6), "lon": round(lon, 6)}
                result["signals"].append({"type": "ok", "msg": f"GPS coordinates extracted: {lat:.4f}, {lon:.4f}"})
                
                # In production: call Google Maps Geocoding API here
                # For POC: flag for manual verification
                if employer_address:
                    result["signals"].append({"type": "info", "msg": f"Address to verify against: {employer_address} — requires Geocoding API for automated match"})
                    result["address_match"] = "manual_review_required"
                
            except Exception as e:
                result["signals"].append({"type": "warning", "msg": f"Could not parse GPS coordinates: {str(e)}"})
        else:
            result["signals"].append({"type": "warning", "msg": "No GPS data in photo — location cannot be verified"})
    
    except Exception as e:
        result["error"] = str(e)
        result["signals"].append({"type": "warning", "msg": f"Could not read photo metadata: {str(e)}"})
    
    return result

# ─────────────────────────────────────────────
# SCORING ENGINE
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# INPUT VALIDATION GATE
# Abort before scoring if the document is unreadable/empty/corrupted
# ─────────────────────────────────────────────
def validate_inputs(l1, l2):
    """Returns (ok: bool, reason: str)"""
    text = l1.get("text", "")
    page_count = l1.get("page_count", 0)
    extraction_method = l1.get("extraction_method", "native")

    if l1.get("error"):
        return False, f"PDF could not be opened: {l1['error']}"
    if page_count == 0:
        return False, "PDF has 0 pages — file may be corrupted or empty"
    if len(text.strip()) < 80 and extraction_method != "image_based_ocr_needed":
        return False, f"Extracted text too short ({len(text.strip())} chars) — possibly a blank, encrypted, or image-only PDF with no OCR fallback"
    if l2.get("error") == "parse_failed":
        return False, "Claude could not parse document content — text may be garbled or non-English"
    return True, ""


# ─────────────────────────────────────────────
# TEMPORAL COHERENCE CHECK
# Checks joining date vs PDF creation date vs today
# ─────────────────────────────────────────────
def check_temporal_coherence(l2, l3):
    """Returns list of override signals with type and msg"""
    signals = []
    today = datetime.now()

    joining_iso = l2.get("joining_date_iso")
    meta = l3.get("metadata", {})
    pdf_created_raw = meta.get("creationDate", "")

    joining_dt = None
    if joining_iso:
        try:
            joining_dt = datetime.strptime(joining_iso, "%Y-%m-%d")
        except:
            pass

    if joining_dt:
        # Joining date in the past (>60 days ago)
        days_ago = (today - joining_dt).days
        if days_ago > 60:
            signals.append({
                "type": "danger",
                "msg": f"Joining date {joining_iso} is {days_ago} days in the past — offer letters are typically submitted before or near joining",
                "override": "cap_score",
                "cap": 50
            })
        # Joining date implausibly far in future (>18 months)
        elif days_ago < -540:
            signals.append({
                "type": "warning",
                "msg": f"Joining date {joining_iso} is more than 18 months away — unusually far future date",
                "override": None
            })

    # PDF creation date vs joining date
    if pdf_created_raw and joining_dt:
        try:
            # PyMuPDF format: D:20240115120000+05'30'
            clean = pdf_created_raw.replace("D:", "")[:8]
            pdf_created_dt = datetime.strptime(clean, "%Y%m%d")
            delta_days = (pdf_created_dt - joining_dt).days
            if delta_days > 30:
                signals.append({
                    "type": "danger",
                    "msg": f"PDF was created {delta_days} days AFTER the stated joining date — document may have been fabricated retrospectively",
                    "override": "cap_score",
                    "cap": 40
                })
            elif delta_days > 0:
                signals.append({
                    "type": "warning",
                    "msg": f"PDF creation date is {delta_days} days after joining date — borderline, may need review",
                    "override": None
                })
            else:
                signals.append({
                    "type": "ok",
                    "msg": f"PDF creation date ({pdf_created_dt.strftime('%Y-%m-%d')}) precedes joining date — temporally consistent",
                    "override": None
                })
        except:
            pass

    return signals


# ─────────────────────────────────────────────
# CROSS-LAYER OVERRIDE ENGINE
# Applies hard caps and floors based on combined signal logic
# This runs AFTER individual layer scores are computed
# ─────────────────────────────────────────────
def apply_cross_layer_overrides(score, breakdown, l2, l3, l4, l5, temporal_signals):
    """
    Applies post-scoring overrides based on cross-layer logic.
    Returns (final_score, overrides_applied: list of str)
    """
    overrides = []

    # Count danger signals across all layers
    all_signals = (
        l3.get("signals", []) +
        l4.get("signals", []) +
        l5.get("signals", []) +
        temporal_signals
    )
    danger_count = sum(1 for s in all_signals if s.get("type") == "danger")

    # ── Override 1: Exact duplicate — hard cap at 25 ──
    if l5.get("is_exact_duplicate"):
        if score > 25:
            score = 25
            overrides.append(f"EXACT DUPLICATE detected — score capped at 25 regardless of other layers")

    # ── Override 2: Too many danger signals — cap at 50 ──
    if danger_count >= 3 and score > 50:
        score = 50
        overrides.append(f"{danger_count} danger signals across layers — score capped at 50 (REVIEW RECOMMENDED floor)")

    # ── Override 3: Completeness too low — cap at 40 ──
    completeness = l2.get("overall_completeness_score", 100)
    missing = len(l2.get("missing_fields", []))
    if completeness < 30 or missing >= 6:
        if score > 40:
            score = 40
            overrides.append(f"Critical incompleteness ({completeness}% complete, {missing} missing fields) — score capped at 40")

    # ── Override 4: Placeholder text detected ──
    if l2.get("has_placeholder_text") is True:
        if score > 35:
            score = 35
            overrides.append("Unfilled template placeholders detected — document appears to be a blank template, not a real offer letter. Score capped at 35.")

    # ── Override 5: Free email used by employer ──
    if l2.get("employer_email_is_free") is True:
        email = l2.get("employer_email", "")
        overrides.append(f"Employer using free email provider — legitimate employers use domain email. Penalty applied.")
        score = max(0, score - 12)

    # ── Override 6: Temporal coherence caps ──
    for sig in temporal_signals:
        if sig.get("override") == "cap_score":
            cap = sig["cap"]
            if score > cap:
                score = cap
                overrides.append(f"Temporal anomaly: {sig['msg'][:80]}... — score capped at {cap}")

    return round(max(0, min(100, score)), 1), overrides


def compute_confidence_score(l2, l3, l4, l5):
    """Compute overall authenticity confidence score (higher = more authentic)"""

    # ── Hard gate: document type check ──
    # If Claude determined this is not an offer letter, skip scoring entirely.
    # A clean policy document should not score 60 just because its PDF metadata is fine.
    is_offer_letter = l2.get("is_offer_letter", True)  # default True for backward compat
    document_type = l2.get("document_type", "offer letter")
    if is_offer_letter is False or (isinstance(is_offer_letter, str) and is_offer_letter.lower() == "false"):
        return {
            "score": 0,
            "verdict": "NOT AN OFFER LETTER",
            "color": "red",
            "document_type": document_type,
            "breakdown": {
                "Document Type Gate": {
                    "max": 100,
                    "score": 0,
                    "detail": f"Document classified as '{document_type}' — not an employment offer letter. Authenticity scoring does not apply."
                }
            }
        }

    breakdown = {}

    # ── Layer 2: Field completeness (weight 30%) ──
    completeness = l2.get("overall_completeness_score", 50)
    missing = len(l2.get("missing_fields", []))
    suspicious_patterns = l2.get("suspicious_text_patterns", []) or []
    l2_base = completeness * 0.30
    l2_penalty = min(15, missing * 3) + min(24, len(suspicious_patterns) * 8)
    l2_contribution = max(0, l2_base - l2_penalty)
    breakdown["L2 Field Completeness"] = {
        "max": 30,
        "score": round(l2_contribution, 1),
        "detail": f"Completeness {completeness}% · {missing} missing fields · {len(suspicious_patterns)} suspicious pattern(s)"
    }

    # ── Layer 3: Metadata forensics (weight 25%) ──
    meta_risk = l3.get("risk_score", 0)
    tool_cat = l3.get("tool_category", "")
    # Post-processors and scanners get softer metadata penalty
    if tool_cat in ("post_processor", "scanner"):
        meta_risk = min(meta_risk, 20)  # cap at 20 for these expected tools
    l3_contribution = max(0, 25 - (meta_risk * 0.25))
    breakdown["L3 Metadata Forensics"] = {
        "max": 25,
        "score": round(l3_contribution, 1),
        "detail": f"Metadata risk score {meta_risk}/100 · Tool category: {tool_cat or 'unknown'}"
    }

    # ── Layer 4: Visual / ELA forensics (weight 25%) ──
    ela_risk = l4.get("ela_risk", 0)
    l4_contribution = max(0, 25 - (ela_risk * 0.25))
    ela_stats = l4.get("ela_stats", {})
    breakdown["L4 Visual Forensics"] = {
        "max": 25,
        "score": round(l4_contribution, 1),
        "detail": f"ELA risk {ela_risk}/100 · std={ela_stats.get('std','—')} · Layout density checked"
    }

    # ── Layer 5: Historical comparison (weight 20%) ──
    if l5.get("similarity_score") is not None:
        sim = l5["similarity_score"]
        if sim >= 85:
            l5_contribution = 20
        elif sim >= 70:
            l5_contribution = 14
        else:
            l5_contribution = 4
        detail = f"Template similarity {sim}% against {l5.get('history_count',0)} prior letters"
    else:
        l5_contribution = 12  # neutral — no history yet
        detail = "No historical baseline — new employer, neutral score"
    breakdown["L5 Historical Match"] = {
        "max": 20,
        "score": round(l5_contribution, 1),
        "detail": detail
    }

    total = sum(b["score"] for b in breakdown.values())
    total = max(0, min(100, total))

    # Temporal coherence — run here so it has access to l2 and l3
    temporal_signals = check_temporal_coherence(l2, l3)

    # Cross-layer override engine
    total, overrides = apply_cross_layer_overrides(total, breakdown, l2, l3, l4, l5, temporal_signals)

    if overrides:
        breakdown["⚡ Score Overrides Applied"] = {
            "max": 0,
            "score": 0,
            "detail": " | ".join(overrides)
        }

    if total >= 75:
        verdict = "LIKELY AUTHENTIC"
        color = "green"
    elif total >= 50:
        verdict = "REVIEW RECOMMENDED"
        color = "amber"
    else:
        verdict = "SUSPICIOUS — FLAG FOR MANUAL REVIEW"
        color = "red"

    return {
        "score": round(total, 1),
        "verdict": verdict,
        "color": color,
        "breakdown": breakdown,
        "overrides": overrides,
        "temporal_signals": temporal_signals
    }

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/analyze", methods=["POST"])
def analyze():
    if "pdf" not in request.files:
        return jsonify({"error": "No PDF uploaded"}), 400

    pdf_file = request.files["pdf"]
    photo_file = request.files.get("photo")

    # UUID session folder — prevents concurrent upload collisions
    import uuid
    session_id = str(uuid.uuid4())
    session_dir = os.path.join(UPLOAD_FOLDER, session_id)
    os.makedirs(session_dir, exist_ok=True)

    pdf_path = os.path.join(session_dir, "input.pdf")
    pdf_file.save(pdf_path)

    photo_path = None
    if photo_file and photo_file.filename:
        photo_path = os.path.join(session_dir, "workplace_photo.jpg")
        photo_file.save(photo_path)

    results = {}

    print("Running Layer 1: Text Extraction...")
    l1 = layer1_extract_text(pdf_path)
    results["layer1"] = l1

    print("Running Layer 2: Field Extraction via Claude...")
    image_path = l1.get("rendered_page")
    l2 = layer2_field_extraction(l1["text"], image_path)
    results["layer2"] = l2

    # ── Input validation gate — abort before scoring if doc is unreadable ──
    valid, reason = validate_inputs(l1, l2)
    if not valid:
        return jsonify({
            "error": "insufficient_data",
            "final_score": {
                "score": 0,
                "verdict": "CANNOT SCORE — INSUFFICIENT DATA",
                "color": "red",
                "breakdown": {
                    "Input Validation": {
                        "max": 100,
                        "score": 0,
                        "detail": reason
                    }
                },
                "overrides": [reason],
                "temporal_signals": []
            },
            "layer1": l1,
            "layer2": l2,
            "all_anomalies": [reason],
            "candidate_name": "Unknown",
            "employer_name": "Unknown",
            "analyzed_at": datetime.now().isoformat()
        })

    print("Running Layer 3: Metadata Forensics...")
    l3 = layer3_metadata_forensics(pdf_path)
    results["layer3"] = l3

    print("Running Layer 4: Visual Forensics (ELA)...")
    l4 = layer4_visual_forensics(pdf_path)
    results["layer4"] = l4

    print("Running Layer 5: Historical Comparison...")
    employer = l2.get("employer_name", "unknown")
    l5 = layer5_historical_comparison(l1["text"], employer, l2)
    results["layer5"] = l5

    print("Running Layer 6: Geotag Verification...")
    joining_date = l2.get("joining_date")
    employer_address = l2.get("employer_address")
    l6 = layer6_geotag_verification(photo_path, employer_address, joining_date)
    results["layer6"] = l6

    print("Computing confidence score...")
    score_result = compute_confidence_score(l2, l3, l4, l5)
    results["final_score"] = score_result

    all_anomalies = (
        l3.get("anomalies", []) +
        l4.get("anomalies", []) +
        l5.get("anomalies", []) +
        l6.get("anomalies", []) +
        (l2.get("suspicious_text_patterns", []) or [])
    )
    # Add override reasons as anomalies so they show in the anomaly box
    overrides = score_result.get("overrides", [])
    all_anomalies += overrides

    # Surface temporal signals as anomalies too if they are danger/warning
    for ts in score_result.get("temporal_signals", []):
        if ts.get("type") in ("danger", "warning"):
            all_anomalies.append(ts["msg"])

    results["all_anomalies"] = all_anomalies
    results["candidate_name"] = l2.get("candidate_name", "Unknown Candidate")
    results["employer_name"] = l2.get("employer_name", "Unknown Employer")
    results["analyzed_at"] = datetime.now().isoformat()

    return jsonify(results)

@app.route("/history")
def get_history():
    history = load_history()
    summary = {emp: len(docs) for emp, docs in history.items()}
    return jsonify(summary)

@app.route("/clear_history", methods=["POST"])
def clear_history():
    save_history({})
    return jsonify({"status": "cleared"})

# ─────────────────────────────────────────────
# HTML TEMPLATE
# ─────────────────────────────────────────────
HTML_TEMPLATE = """

<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TRRAIN · Offer Letter Verifier</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
  :root {
    --bg: #0a0e17; --surface: #111827; --surface2: #1a2235; --border: #1e2d45;
    --accent: #00d4aa; --accent2: #0088ff; --warn: #f59e0b; --danger: #ef4444;
    --ok: #10b981; --text: #e2e8f0; --muted: #64748b;
    --mono: 'IBM Plex Mono', monospace; --sans: 'IBM Plex Sans', sans-serif;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: var(--sans); min-height: 100vh; }
  .grid-bg {
    position: fixed; inset: 0; pointer-events: none; z-index: 0;
    background-image: linear-gradient(rgba(0,212,170,0.03) 1px, transparent 1px),
      linear-gradient(90deg, rgba(0,212,170,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
  }
  .container { max-width: 1100px; margin: 0 auto; padding: 2rem; position: relative; z-index: 1; }
  header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 1.5rem 2rem; border-bottom: 1px solid var(--border); position: relative; z-index: 1;
  }
  .logo { display: flex; align-items: center; gap: 12px; }
  .logo-mark {
    width: 36px; height: 36px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    border-radius: 8px; display: flex; align-items: center; justify-content: center;
    font-family: var(--mono); font-size: 14px; font-weight: 600; color: #000;
  }
  .logo-text { font-size: 0.9rem; font-weight: 500; color: var(--muted); letter-spacing: 0.1em; text-transform: uppercase; }
  .logo-title { font-size: 1.1rem; font-weight: 600; color: var(--text); }
  .status-bar { font-family: var(--mono); font-size: 0.7rem; color: var(--muted); display: flex; gap: 1rem; align-items: center; }
  .status-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--accent); animation: pulse 2s ease-in-out infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }

  /* ── Upload Zone with Drag & Drop ── */
  .upload-zone {
    background: var(--surface); border: 2px dashed var(--border);
    border-radius: 16px; padding: 3rem 2rem; text-align: center;
    cursor: pointer; transition: all 0.3s; position: relative; overflow: hidden; margin: 2rem 0;
  }
  .upload-zone:hover, .upload-zone.dragover {
    border-color: var(--accent); background: rgba(0,212,170,0.05);
    transform: scale(1.01);
  }
  .upload-zone.dragover { box-shadow: 0 0 40px rgba(0,212,170,0.15); }
  .upload-zone::before {
    content: ''; position: absolute; inset: 0;
    background: linear-gradient(135deg, rgba(0,212,170,0.05) 0%, transparent 60%);
    pointer-events: none;
  }
  .drop-hint {
    font-size: 0.8rem; color: var(--accent); font-family: var(--mono);
    background: rgba(0,212,170,0.08); border: 1px solid rgba(0,212,170,0.2);
    border-radius: 6px; padding: 0.35rem 0.75rem; display: inline-block; margin-bottom: 1rem;
  }
  .upload-icon { font-size: 3rem; margin-bottom: 1rem; }
  .upload-title { font-size: 1.2rem; font-weight: 600; margin-bottom: 0.5rem; }
  .upload-sub { font-size: 0.85rem; color: var(--muted); margin-bottom: 1.5rem; }
  .file-inputs { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem; }
  .file-input-group {
    background: var(--surface2); border: 1px solid var(--border);
    border-radius: 10px; padding: 1rem; text-align: left;
  }
  .file-input-group label.group-label {
    display: block; font-family: var(--mono); font-size: 0.7rem; color: var(--accent);
    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.5rem;
  }
  .file-input-group .desc { font-size: 0.75rem; color: var(--muted); margin-bottom: 0.75rem; }
  input[type="file"] { display: none; }
  .file-btn {
    display: inline-block; padding: 0.5rem 1rem;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 6px; font-size: 0.8rem; cursor: pointer; transition: all 0.2s; color: var(--text);
  }
  .file-btn:hover { border-color: var(--accent); color: var(--accent); }
  .file-name { font-family: var(--mono); font-size: 0.7rem; color: var(--muted); margin-top: 0.4rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .file-name.selected { color: var(--accent); }
  .btn-analyze {
    width: 100%; padding: 1rem 2rem;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    color: #000; border: none; border-radius: 10px; font-size: 1rem; font-weight: 600;
    cursor: pointer; transition: all 0.3s; font-family: var(--sans); letter-spacing: 0.02em;
  }
  .btn-analyze:hover { transform: translateY(-2px); box-shadow: 0 8px 30px rgba(0,212,170,0.3); }
  .btn-analyze:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }

  /* ── Progress ── */
  .layers-progress { display: none; margin: 2rem 0; }
  .layers-progress.active { display: block; }
  .progress-title { font-family: var(--mono); font-size: 0.75rem; color: var(--accent); text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 1rem; }
  .layer-steps { display: flex; flex-direction: column; gap: 0.5rem; }
  .layer-step {
    display: flex; align-items: center; gap: 1rem;
    padding: 0.75rem 1rem; background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; font-size: 0.85rem; transition: all 0.3s;
  }
  .layer-step.active { border-color: var(--accent); background: rgba(0,212,170,0.05); }
  .layer-step.done { border-color: var(--ok); opacity: 0.7; }
  .step-num { font-family: var(--mono); font-size: 0.7rem; color: var(--muted); min-width: 60px; }
  .step-label { flex: 1; }
  .step-status { font-family: var(--mono); font-size: 0.7rem; }

  /* ── Results ── */
  .results { display: none; }
  .results.active { display: block; }
  .score-banner {
    padding: 2rem; border-radius: 16px; margin: 2rem 0;
    display: flex; align-items: center; justify-content: space-between; gap: 2rem;
  }
  .score-banner.green { background: rgba(16,185,129,0.1); border: 1px solid rgba(16,185,129,0.3); }
  .score-banner.amber { background: rgba(245,158,11,0.1); border: 1px solid rgba(245,158,11,0.3); }
  .score-banner.red { background: rgba(239,68,68,0.1); border: 1px solid rgba(239,68,68,0.3); }
  .score-number { font-family: var(--mono); font-size: 4rem; font-weight: 600; line-height: 1; }
  .score-banner.green .score-number { color: var(--ok); }
  .score-banner.amber .score-number { color: var(--warn); }
  .score-banner.red .score-number { color: var(--danger); }
  .score-label { font-size: 0.75rem; color: var(--muted); font-family: var(--mono); text-transform: uppercase; }
  .score-verdict { font-size: 1.3rem; font-weight: 600; margin-top: 0.25rem; }
  .score-banner.green .score-verdict { color: var(--ok); }
  .score-banner.amber .score-verdict { color: var(--warn); }
  .score-banner.red .score-verdict { color: var(--danger); }

  /* ── Score Breakdown ── */
  .score-breakdown {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 1.5rem; margin: 1rem 0;
  }
  .breakdown-title {
    font-family: var(--mono); font-size: 0.7rem; color: var(--accent);
    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 1.25rem;
  }
  .breakdown-row {
    display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;
  }
  .breakdown-label { font-size: 0.8rem; min-width: 180px; }
  .breakdown-bar-wrap { flex: 1; height: 8px; background: var(--surface2); border-radius: 4px; overflow: hidden; }
  .breakdown-bar { height: 100%; border-radius: 4px; transition: width 1s ease; }
  .breakdown-score { font-family: var(--mono); font-size: 0.75rem; min-width: 60px; text-align: right; }
  .breakdown-detail { font-size: 0.7rem; color: var(--muted); margin-top: -0.5rem; margin-bottom: 0.5rem; padding-left: 180px; }

  /* ── Anomalies ── */
  .anomalies-box {
    background: rgba(239,68,68,0.07); border: 1px solid rgba(239,68,68,0.25);
    border-radius: 12px; padding: 1.5rem; margin: 1rem 0;
  }
  .anomalies-title { font-family: var(--mono); font-size: 0.75rem; color: var(--danger); text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 1rem; }
  .anomaly-item { display: flex; align-items: flex-start; gap: 0.75rem; padding: 0.5rem 0; border-bottom: 1px solid rgba(239,68,68,0.1); font-size: 0.85rem; }
  .anomaly-item:last-child { border-bottom: none; }

  /* ── Layer Cards ── */
  .layer-cards { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin: 1.5rem 0; }
  .layer-card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 1.25rem; }
  .layer-card-header { display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1rem; padding-bottom: 0.75rem; border-bottom: 1px solid var(--border); }
  .layer-num { font-family: var(--mono); font-size: 0.65rem; color: var(--accent); background: rgba(0,212,170,0.1); padding: 0.2rem 0.5rem; border-radius: 4px; text-transform: uppercase; letter-spacing: 0.1em; }
  .layer-card-title { font-size: 0.9rem; font-weight: 600; }
  .signal { display: flex; align-items: flex-start; gap: 0.5rem; font-size: 0.78rem; padding: 0.3rem 0; line-height: 1.4; }
  .signal-dot { width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; margin-top: 0.35rem; }
  .signal.ok .signal-dot { background: var(--ok); }
  .signal.warning .signal-dot { background: var(--warn); }
  .signal.danger .signal-dot { background: var(--danger); }
  .signal.info .signal-dot { background: var(--accent2); }
  .signal.ok .signal-text { color: #a7f3d0; }
  .signal.warning .signal-text { color: #fcd34d; }
  .signal.danger .signal-text { color: #fca5a5; }
  .signal.info .signal-text { color: #93c5fd; }

  /* ── Fields ── */
  .fields-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 0.75rem; margin: 1rem 0; }
  .field-item { background: var(--surface2); border: 1px solid var(--border); border-radius: 8px; padding: 0.75rem; }
  .field-key { font-family: var(--mono); font-size: 0.65rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.3rem; }
  .field-val { font-size: 0.8rem; color: var(--text); word-break: break-word; }
  .field-missing { color: var(--danger); font-style: italic; }
  .field-found { color: var(--ok); }

  /* ── ELA ── */
  .ela-label { font-family: var(--mono); font-size: 0.65rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.5rem; }
  .ela-img { width: 100%; border-radius: 6px; border: 1px solid var(--border); max-height: 200px; object-fit: contain; background: #000; }
  .ela-stats { display: flex; gap: 1rem; margin-top: 0.5rem; }
  .ela-stat { font-family: var(--mono); font-size: 0.7rem; color: var(--muted); }
  .ela-stat span { color: var(--text); }

  /* ── Metadata table ── */
  .meta-row { display: flex; gap: 0.5rem; padding: 0.3rem 0; border-bottom: 1px solid rgba(255,255,255,0.04); font-size: 0.75rem; }
  .meta-key { font-family: var(--mono); color: var(--muted); min-width: 100px; }
  .meta-val { color: var(--text); word-break: break-all; }

  /* ── Action buttons ── */
  .action-row { display: flex; gap: 1rem; margin-top: 1.5rem; flex-wrap: wrap; }
  .btn-secondary {
    padding: 0.75rem 1.5rem; background: transparent;
    border: 1px solid var(--border); border-radius: 8px; color: var(--muted);
    cursor: pointer; font-family: var(--sans); font-size: 0.85rem; transition: all 0.2s;
  }
  .btn-secondary:hover { border-color: var(--accent); color: var(--accent); }
  .btn-export {
    padding: 0.75rem 1.5rem;
    background: rgba(0,136,255,0.1); border: 1px solid rgba(0,136,255,0.3);
    border-radius: 8px; color: #60a5fa; cursor: pointer; font-family: var(--sans);
    font-size: 0.85rem; transition: all 0.2s;
  }
  .btn-export:hover { background: rgba(0,136,255,0.2); border-color: #60a5fa; }

  .section-title {
    font-family: var(--mono); font-size: 0.7rem; color: var(--accent);
    text-transform: uppercase; letter-spacing: 0.1em; margin: 2rem 0 1rem;
    display: flex; align-items: center; gap: 0.75rem;
  }
  .section-title::after { content: ''; flex: 1; height: 1px; background: var(--border); }

  @media (max-width: 640px) {
    .layer-cards, .file-inputs { grid-template-columns: 1fr; }
    .score-banner { flex-direction: column; }
    .breakdown-detail { padding-left: 0; }
  }
</style>
</head>
<body>
<div class="grid-bg"></div>
<header>
  <div class="logo">
    <div class="logo-mark">OL</div>
    <div>
      <div class="logo-text">TRRAIN · EdZola POC</div>
      <div class="logo-title">Offer Letter Verifier</div>
    </div>
  </div>
  <div class="status-bar">
    <div class="status-dot"></div>
    <span>6-Layer Analysis Engine</span>
    <span>·</span>
    <span id="history-count">Loading...</span>
  </div>
</header>

<div class="container">

  <!-- Upload Zone -->
  <div id="upload-section">
    <div class="upload-zone" id="drop-zone"
      ondragover="handleDragOver(event)"
      ondragleave="handleDragLeave(event)"
      ondrop="handleDrop(event)"
      onclick="document.getElementById('pdf-input').click()">
      <div class="drop-hint">↓ Drop PDF here or click to browse</div>
      <div class="upload-icon">📄</div>
      <div class="upload-title">Upload Offer Letter for Verification</div>
      <div class="upload-sub">PDF format · Scanned or digital · Drag & drop supported</div>

      <div class="file-inputs" onclick="event.stopPropagation()">
        <div class="file-input-group">
          <label class="group-label">Offer Letter (Required)</label>
          <div class="desc">PDF file — scanned or digital</div>
          <label class="file-btn" for="pdf-input" onclick="event.stopPropagation()">Choose PDF</label>
          <input type="file" id="pdf-input" accept=".pdf">
          <div class="file-name" id="pdf-name">No file selected</div>
        </div>
        <div class="file-input-group">
          <label class="group-label">Workplace Photo (Optional)</label>
          <div class="desc">Geotagged photo for Layer 6 verification</div>
          <label class="file-btn" for="photo-input" onclick="event.stopPropagation()">Choose Photo</label>
          <input type="file" id="photo-input" accept="image/*">
          <div class="file-name" id="photo-name">No file selected</div>
        </div>
      </div>
    </div>
    <button class="btn-analyze" id="analyze-btn" onclick="runAnalysis()" disabled>
      Run 6-Layer Analysis
    </button>
  </div>

  <!-- Progress -->
  <div class="layers-progress" id="progress">
    <div class="progress-title">▶ Analysis Pipeline Running</div>
    <div class="layer-steps" id="layer-steps">
      <div class="layer-step" id="step-1"><span class="step-num">LAYER 1</span><span class="step-label">Text Extraction (OCR)</span><span class="step-status" id="s1">⏳ Waiting</span></div>
      <div class="layer-step" id="step-2"><span class="step-num">LAYER 2</span><span class="step-label">Field Extraction + Completeness (Claude)</span><span class="step-status" id="s2">⏳ Waiting</span></div>
      <div class="layer-step" id="step-3"><span class="step-num">LAYER 3</span><span class="step-label">PDF Metadata Forensics</span><span class="step-status" id="s3">⏳ Waiting</span></div>
      <div class="layer-step" id="step-4"><span class="step-num">LAYER 4</span><span class="step-label">Visual Forensics (ELA + Layout)</span><span class="step-status" id="s4">⏳ Waiting</span></div>
      <div class="layer-step" id="step-5"><span class="step-num">LAYER 5</span><span class="step-label">Historical Document Comparison</span><span class="step-status" id="s5">⏳ Waiting</span></div>
      <div class="layer-step" id="step-6"><span class="step-num">LAYER 6</span><span class="step-label">Geotag + Timestamp Verification</span><span class="step-status" id="s6">⏳ Waiting</span></div>
    </div>
  </div>

  <!-- Results -->
  <div class="results" id="results">

    <div class="section-title">Final Verdict</div>
    <div class="score-banner" id="score-banner">
      <div>
        <div class="score-label">Authenticity Score</div>
        <div class="score-number" id="score-num">—</div>
        <div class="score-label" style="margin-top:0.25rem">out of 100</div>
      </div>
      <div style="flex:1">
        <div class="score-verdict" id="score-verdict">—</div>
        <div style="font-size:0.8rem;color:var(--muted);margin-top:0.5rem" id="score-sub"></div>
      </div>
    </div>

    <!-- Score Breakdown -->
    <div class="score-breakdown">
      <div class="breakdown-title">Score Breakdown — How this was calculated</div>
      <div id="score-breakdown-rows"></div>
    </div>

    <div id="anomalies-section" style="display:none">
      <div class="anomalies-box">
        <div class="anomalies-title">⚠ Anomalies Detected</div>
        <div id="anomalies-list"></div>
      </div>
    </div>

    <div class="section-title">Extracted Fields</div>
    <div id="fields-completeness"></div>
    <div class="fields-grid" id="fields-grid"></div>

    <div class="section-title">Layer-by-Layer Analysis</div>
    <div class="layer-cards">
      <div class="layer-card">
        <div class="layer-card-header"><span class="layer-num">L1</span><span class="layer-card-title">Text Extraction</span></div>
        <div id="l1-detail"></div>
      </div>
      <div class="layer-card">
        <div class="layer-card-header"><span class="layer-num">L3</span><span class="layer-card-title">Metadata Forensics</span></div>
        <div id="l3-detail"></div>
        <div id="l3-meta" style="margin-top:1rem"></div>
      </div>
      <div class="layer-card" style="grid-column: span 2">
        <div class="layer-card-header"><span class="layer-num">L4</span><span class="layer-card-title">Visual Forensics — Error Level Analysis</span></div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;align-items:start">
          <div id="l4-signals"></div>
          <div>
            <div class="ela-label">ELA Visualization (bright = possible manipulation)</div>
            <img id="ela-img" class="ela-img" src="" alt="ELA" style="display:none">
            <div class="ela-stats" id="ela-stats"></div>
          </div>
        </div>
      </div>
      <div class="layer-card">
        <div class="layer-card-header"><span class="layer-num">L5</span><span class="layer-card-title">Historical Comparison</span></div>
        <div id="l5-detail"></div>
      </div>
      <div class="layer-card">
        <div class="layer-card-header"><span class="layer-num">L6</span><span class="layer-card-title">Geotag Verification</span></div>
        <div id="l6-detail"></div>
      </div>
    </div>

    <div class="action-row">
      <button class="btn-secondary" onclick="resetForm()">↩ Analyze Another Letter</button>
      <button class="btn-export" onclick="exportReport()" id="export-btn">⬇ Export PDF Report</button>
    </div>
  </div>

</div>

<script>
let pdfFile = null, photoFile = null, lastResults = null;

// ── File selection ──
document.getElementById('pdf-input').addEventListener('change', function(e) {
  setPDF(e.target.files[0]);
});
document.getElementById('photo-input').addEventListener('change', function(e) {
  photoFile = e.target.files[0];
  if (photoFile) {
    document.getElementById('photo-name').textContent = photoFile.name;
    document.getElementById('photo-name').className = 'file-name selected';
  }
});

function setPDF(file) {
  if (!file || file.type !== 'application/pdf') {
    alert('Please upload a PDF file.');
    return;
  }
  pdfFile = file;
  document.getElementById('pdf-name').textContent = file.name;
  document.getElementById('pdf-name').className = 'file-name selected';
  document.getElementById('analyze-btn').disabled = false;
  // Update drop zone icon
  document.querySelector('.upload-icon').textContent = '✅';
  document.querySelector('.drop-hint').textContent = '✓ ' + file.name;
}

// ── Drag & Drop ──
function handleDragOver(e) {
  e.preventDefault(); e.stopPropagation();
  document.getElementById('drop-zone').classList.add('dragover');
}
function handleDragLeave(e) {
  e.preventDefault(); e.stopPropagation();
  document.getElementById('drop-zone').classList.remove('dragover');
}
function handleDrop(e) {
  e.preventDefault(); e.stopPropagation();
  document.getElementById('drop-zone').classList.remove('dragover');
  const files = e.dataTransfer.files;
  if (files.length > 0) {
    const pdf = Array.from(files).find(f => f.type === 'application/pdf' || f.name.endsWith('.pdf'));
    const photo = Array.from(files).find(f => f.type.startsWith('image/'));
    if (pdf) setPDF(pdf);
    if (photo) {
      photoFile = photo;
      document.getElementById('photo-name').textContent = photo.name;
      document.getElementById('photo-name').className = 'file-name selected';
    }
    if (!pdf && !photo) alert('Please drop a PDF file (and optionally a photo).');
  }
}

// History count
fetch('/history').then(r=>r.json()).then(h => {
  const total = Object.values(h).reduce((a,b)=>a+b,0);
  const employers = Object.keys(h).length;
  document.getElementById('history-count').textContent = `${total} letters indexed · ${employers} employers`;
});

function setStep(n, status, done=false) {
  document.getElementById(`step-${n}`).className = 'layer-step' + (done ? ' done' : ' active');
  document.getElementById(`s${n}`).textContent = status;
}

async function runAnalysis() {
  if (!pdfFile) return;
  const btn = document.getElementById('analyze-btn');
  btn.disabled = true; btn.textContent = 'Analyzing...';
  document.getElementById('upload-section').style.opacity = '0.5';
  document.getElementById('progress').className = 'layers-progress active';
  document.getElementById('results').className = 'results';
  // Reset all steps
  for(let i=1;i<=6;i++) setStep(i,'⏳ Waiting',false);

  setStep(1,'⚡ Running',false);
  const formData = new FormData();
  formData.append('pdf', pdfFile);
  if (photoFile) formData.append('photo', photoFile);

  let stepTimer = 1;
  const interval = setInterval(() => {
    if (stepTimer <= 6) {
      if (stepTimer > 1) setStep(stepTimer-1,'✓ Done',true);
      setStep(stepTimer,'⚡ Running',false);
      stepTimer++;
    }
  }, 3500);

  try {
    const response = await fetch('/analyze', { method: 'POST', body: formData });
    const data = await response.json();
    clearInterval(interval);
    for(let i=1;i<=6;i++) setStep(i,'✓ Done',true);
    lastResults = data;
    setTimeout(() => renderResults(data), 400);
  } catch(err) {
    clearInterval(interval);
    alert('Analysis failed: ' + err.message);
    btn.disabled = false; btn.textContent = 'Run 6-Layer Analysis';
    document.getElementById('upload-section').style.opacity = '1';
    document.getElementById('progress').className = 'layers-progress';
  }
}

function renderSignals(signals) {
  if (!signals || !signals.length) return '<span style="color:var(--muted);font-size:0.75rem">No signals</span>';
  return signals.map(s => `<div class="signal ${s.type}"><div class="signal-dot"></div><div class="signal-text">${s.msg}</div></div>`).join('');
}

function renderResults(data) {
  document.getElementById('progress').className = 'layers-progress';
  document.getElementById('results').className = 'results active';
  document.getElementById('upload-section').style.opacity = '1';
  document.getElementById('analyze-btn').disabled = false;
  document.getElementById('analyze-btn').textContent = 'Run 6-Layer Analysis';

  const score = data.final_score;
  const isNotOffer = score.verdict === 'NOT AN OFFER LETTER';
  const banner = document.getElementById('score-banner');
  banner.className = `score-banner ${score.color}`;
  document.getElementById('score-num').textContent = isNotOffer ? '—' : score.score;
  document.getElementById('score-verdict').textContent = score.verdict;
  document.getElementById('score-sub').textContent = isNotOffer
    ? `Document classified as: ${score.document_type || 'non-offer document'} · Authenticity scoring not applicable · ${new Date().toLocaleString()}`
    : `Based on 6-layer forensic analysis · ${new Date().toLocaleString()}`;

  // Score Breakdown
  const breakdown = score.breakdown || {};
  const colorMap = { 'L2 Field Completeness': '#00d4aa', 'L3 Metadata Forensics': '#0088ff', 'L4 Visual Forensics': '#f59e0b', 'L5 Historical Match': '#a78bfa' };
  document.getElementById('score-breakdown-rows').innerHTML = Object.entries(breakdown).map(([key, val]) => {
    // Special rendering for the override row
    if (key.includes('Override')) {
      return `
        <div class="breakdown-row" style="background:rgba(245,158,11,0.07);border-radius:6px;padding:0.5rem;margin:0.25rem 0">
          <div class="breakdown-label" style="color:var(--warn);font-size:0.75rem">⚡ ${key}</div>
        </div>
        <div class="breakdown-detail" style="color:var(--warn);padding-left:0">${val.detail}</div>
      `;
    }
    const pct = val.max > 0 ? (val.score / val.max) * 100 : 0;
    const barColor = pct >= 70 ? 'var(--ok)' : pct >= 40 ? 'var(--warn)' : 'var(--danger)';
    return `
      <div class="breakdown-row">
        <div class="breakdown-label">${key}</div>
        <div class="breakdown-bar-wrap"><div class="breakdown-bar" style="width:${pct}%;background:${barColor}"></div></div>
        <div class="breakdown-score" style="color:${barColor}">${val.score} / ${val.max}</div>
      </div>
      <div class="breakdown-detail">${val.detail}</div>
    `;
  }).join('') + (isNotOffer ? '' : `
    <div class="breakdown-row" style="border-top:1px solid var(--border);padding-top:0.75rem;margin-top:0.25rem">
      <div class="breakdown-label" style="font-weight:600">Total Score</div>
      <div class="breakdown-bar-wrap"><div class="breakdown-bar" style="width:${score.score}%;background:${score.color==='green'?'var(--ok)':score.color==='amber'?'var(--warn)':'var(--danger)'}"></div></div>
      <div class="breakdown-score" style="font-weight:600;color:${score.color==='green'?'var(--ok)':score.color==='amber'?'var(--warn)':'var(--danger)'}">${score.score} / 100</div>
    </div>
  `);

  // Anomalies
  const anomalies = data.all_anomalies || [];
  if (anomalies.length > 0) {
    document.getElementById('anomalies-section').style.display = 'block';
    document.getElementById('anomalies-list').innerHTML = anomalies.map(a => `<div class="anomaly-item"><span style="color:var(--danger)">⚠</span><span>${a}</span></div>`).join('');
  } else {
    document.getElementById('anomalies-section').style.display = 'none';
  }

  // Fields
  const l2 = data.layer2 || {};
  const fieldKeys = ['candidate_name','job_title','salary','joining_date','employer_name','employer_address','employer_email','employer_phone','signature_block','reporting_manager'];
  const missing = l2.missing_fields || [];
  const comp = l2.overall_completeness_score || 0;
  const compColor = comp >= 70 ? 'var(--ok)' : comp >= 50 ? 'var(--warn)' : 'var(--danger)';
  document.getElementById('fields-completeness').innerHTML = `
    <div style="display:flex;align-items:center;gap:1rem;margin-bottom:0.75rem">
      <div style="font-size:0.8rem;color:var(--muted)">Completeness</div>
      <div style="flex:1;background:var(--surface2);height:6px;border-radius:3px;overflow:hidden">
        <div style="width:${comp}%;height:100%;background:${compColor};border-radius:3px;transition:width 1s ease"></div>
      </div>
      <div style="font-family:var(--mono);font-size:0.8rem;color:${compColor}">${comp}%</div>
    </div>`;
  document.getElementById('fields-grid').innerHTML = fieldKeys.map(k => {
    const val = l2[k]; const isMissing = missing.includes(k) || !val || val==='null' || val===null;
    return `<div class="field-item"><div class="field-key">${k.replace(/_/g,' ')}</div><div class="field-val ${isMissing?'field-missing':'field-found'}">${isMissing?'✗ Not found':val}</div></div>`;
  }).join('');

  // L1
  const l1 = data.layer1 || {};
  document.getElementById('l1-detail').innerHTML = `
    <div class="signal info"><div class="signal-dot"></div><div class="signal-text">Pages: ${l1.page_count||0} · Images: ${l1.image_count||0}</div></div>
    <div class="signal ${(l1.text?.length||0)>100?'ok':'warning'}"><div class="signal-dot"></div><div class="signal-text">Text extracted: ${l1.text?.length||0} chars · Method: ${l1.extraction_method||'native'}</div></div>
    ${l1.has_images?'<div class="signal info"><div class="signal-dot"></div><div class="signal-text">Document contains embedded images</div></div>':''}`;

  // L3
  const l3 = data.layer3 || {};
  const riskColor = (l3.risk_score||0)>50?'var(--danger)':(l3.risk_score||0)>20?'var(--warn)':'var(--ok)';
  document.getElementById('l3-detail').innerHTML = `
    <div style="margin-bottom:0.75rem">
      <div style="font-family:var(--mono);font-size:0.65rem;color:var(--muted);margin-bottom:0.3rem">METADATA RISK</div>
      <div style="display:flex;align-items:center;gap:0.75rem">
        <div style="font-family:var(--mono);font-size:1.5rem;color:${riskColor}">${l3.risk_score||0}</div>
        <div style="font-size:0.75rem;color:var(--muted)">/100</div>
      </div>
    </div>
    ${renderSignals(l3.signals)}`;
  const meta = l3.metadata || {};
  const metaKeys = ['creator','producer','creationDate','modDate','author','subject'];
  document.getElementById('l3-meta').innerHTML = `
    <div style="font-family:var(--mono);font-size:0.65rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.5rem">Raw Metadata</div>
    ${metaKeys.filter(k=>meta[k]).map(k=>`<div class="meta-row"><span class="meta-key">${k}</span><span class="meta-val">${meta[k]}</span></div>`).join('') || '<div style="font-size:0.75rem;color:var(--muted)">No metadata present — see signals above for interpretation</div>'}`;

  // L4
  const l4 = data.layer4 || {};
  document.getElementById('l4-signals').innerHTML = renderSignals(l4.signals);
  if (l4.ela_image_b64) {
    const elaImg = document.getElementById('ela-img');
    elaImg.src = `data:image/png;base64,${l4.ela_image_b64}`; elaImg.style.display = 'block';
    const s = l4.ela_stats||{};
    document.getElementById('ela-stats').innerHTML = `<span class="ela-stat">Mean: <span>${s.mean}</span></span><span class="ela-stat">Std: <span>${s.std}</span></span><span class="ela-stat">Max: <span>${s.max}</span></span>`;
  }

  // L5
  const l5 = data.layer5 || {};
  document.getElementById('l5-detail').innerHTML = `
    <div class="signal info"><div class="signal-dot"></div><div class="signal-text">Employer: ${l5.employer||'Unknown'}</div></div>
    <div class="signal ${l5.is_new_employer?'info':'ok'}"><div class="signal-dot"></div><div class="signal-text">${l5.is_new_employer?'First document — baseline established':`${l5.history_count} historical letters on file`}</div></div>
    ${l5.similarity_score!=null?`<div class="signal ${l5.similarity_score>=85?'ok':l5.similarity_score>=70?'warning':'danger'}"><div class="signal-dot"></div><div class="signal-text">Template similarity: ${l5.similarity_score}%</div></div>`:''}
    ${renderSignals(l5.signals)}`;

  // L6
  const l6 = data.layer6 || {};
  document.getElementById('l6-detail').innerHTML = renderSignals(l6.signals) +
    (l6.gps_coords?`<div class="signal ok"><div class="signal-dot"></div><div class="signal-text">GPS: ${l6.gps_coords.lat}, ${l6.gps_coords.lon}</div></div>`:'');

  // Temporal coherence signals — injected into L3 card as they relate to dates
  const temporalSignals = score.temporal_signals || [];
  if (temporalSignals.length > 0) {
    document.getElementById('l3-detail').innerHTML += `
      <div style="margin-top:0.75rem;padding-top:0.75rem;border-top:1px solid var(--border)">
        <div style="font-family:var(--mono);font-size:0.65rem;color:var(--warn);text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.4rem">Temporal Coherence</div>
        ${renderSignals(temporalSignals)}
      </div>`;
  }

  // Employer email flag
  const emailFree = data.layer2?.employer_email_is_free;
  const emailDomain = data.layer2?.employer_email_domain;
  if (emailFree === true) {
    document.getElementById('l1-detail').innerHTML += `<div class="signal danger"><div class="signal-dot"></div><div class="signal-text">Employer using free email (${emailDomain || 'unknown domain'}) — legitimate employers use domain email</div></div>`;
  } else if (emailFree === false && emailDomain) {
    document.getElementById('l1-detail').innerHTML += `<div class="signal ok"><div class="signal-dot"></div><div class="signal-text">Employer domain email: ${emailDomain}</div></div>`;
  }

  // Refresh history
  fetch('/history').then(r=>r.json()).then(h => {
    const total = Object.values(h).reduce((a,b)=>a+b,0);
    document.getElementById('history-count').textContent = `${total} letters indexed · ${Object.keys(h).length} employers`;
  });

  document.getElementById('results').scrollIntoView({behavior:'smooth'});
}

// ── PDF Report Export ──
function exportReport() {
  if (!lastResults) return;
  const d = lastResults;
  const score = d.final_score;
  const l2 = d.layer2 || {};
  const l3 = d.layer3 || {};
  const l4 = d.layer4 || {};
  const l5 = d.layer5 || {};
  const l6 = d.layer6 || {};
  const breakdown = score.breakdown || {};
  const scoreColor = score.color === 'green' ? '#10b981' : score.color === 'amber' ? '#f59e0b' : '#ef4444';
  const fieldKeys = ['candidate_name','job_title','salary','joining_date','employer_name','employer_address','employer_email','employer_phone','signature_block','reporting_manager'];
  const missing = l2.missing_fields || [];

  const anomalyRows = (d.all_anomalies||[]).map(a => `<tr><td style="padding:6px 10px;border-bottom:1px solid #fee2e2">⚠ ${a}</td></tr>`).join('');
  const fieldRows = fieldKeys.map(k => {
    const val = l2[k]; const isMiss = missing.includes(k)||!val||val==='null'||val===null;
    return `<tr><td style="padding:5px 10px;border-bottom:1px solid #f1f5f9;color:#64748b;font-size:12px">${k.replace(/_/g,' ').toUpperCase()}</td><td style="padding:5px 10px;border-bottom:1px solid #f1f5f9;color:${isMiss?'#ef4444':'#0f172a'};font-size:12px">${isMiss?'Not found':val}</td></tr>`;
  }).join('');
  const breakdownRows = Object.entries(breakdown).map(([key, val]) => {
    const pct = Math.round((val.score/val.max)*100);
    const barColor = pct>=70?'#10b981':pct>=40?'#f59e0b':'#ef4444';
    return `
      <tr>
        <td style="padding:8px 10px;font-size:12px;color:#374151">${key}</td>
        <td style="padding:8px 10px">
          <div style="background:#f1f5f9;border-radius:3px;height:8px;overflow:hidden">
            <div style="width:${pct}%;height:100%;background:${barColor};border-radius:3px"></div>
          </div>
        </td>
        <td style="padding:8px 10px;font-family:monospace;font-size:12px;color:${barColor};white-space:nowrap">${val.score} / ${val.max}</td>
      </tr>
      <tr><td colspan="3" style="padding:0 10px 8px;font-size:11px;color:#94a3b8">${val.detail}</td></tr>`;
  }).join('');

  const allSignals = [
    ...([...(l3.signals||[])].map(s => ({...s, layer:'L3 Metadata'}))),
    ...([...(l4.signals||[])].map(s => ({...s, layer:'L4 Visual'}))),
    ...([...(l5.signals||[])].map(s => ({...s, layer:'L5 Historical'}))),
    ...([...(l6.signals||[])].map(s => ({...s, layer:'L6 Geotag'}))),
  ];
  const sigColor = {'ok':'#10b981','warning':'#f59e0b','danger':'#ef4444','info':'#3b82f6'};
  const signalRows = allSignals.map(s => `
    <tr>
      <td style="padding:5px 10px;font-size:11px;color:#64748b">${s.layer}</td>
      <td style="padding:5px 10px"><span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${sigColor[s.type]||'#94a3b8'}"></span></td>
      <td style="padding:5px 10px;font-size:11px;color:#374151">${s.msg}</td>
    </tr>`).join('');

  const html = `<!DOCTYPE html><html><head><meta charset="UTF-8">
<title>Offer Letter Verification Report</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
  body { font-family: Inter, sans-serif; color: #0f172a; margin: 0; padding: 40px; background: #fff; max-width: 900px; margin: 0 auto; }
  .header { display: flex; align-items: flex-start; justify-content: space-between; border-bottom: 2px solid #e2e8f0; padding-bottom: 20px; margin-bottom: 24px; }
  .logo-area h1 { font-size: 20px; font-weight: 700; color: #0f172a; margin: 0 0 4px; }
  .logo-area p { font-size: 12px; color: #64748b; margin: 0; }
  .score-box { text-align: center; padding: 16px 24px; border-radius: 12px; background: ${score.color==='green'?'#f0fdf4':score.color==='amber'?'#fffbeb':'#fef2f2'}; border: 1px solid ${score.color==='green'?'#bbf7d0':score.color==='amber'?'#fed7aa':'#fecaca'}; }
  .score-num { font-size: 48px; font-weight: 700; color: ${scoreColor}; line-height: 1; }
  .score-verdict { font-size: 11px; font-weight: 600; color: ${scoreColor}; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 4px; }
  .section { margin-bottom: 28px; }
  .section h2 { font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: #64748b; border-bottom: 1px solid #e2e8f0; padding-bottom: 8px; margin-bottom: 12px; }
  table { width: 100%; border-collapse: collapse; }
  .footer { margin-top: 40px; border-top: 1px solid #e2e8f0; padding-top: 16px; display: flex; justify-content: space-between; font-size: 10px; color: #94a3b8; }
</style>
</head><body>
<div class="header">
  <div class="logo-area">
    <h1>Offer Letter Verification Report</h1>
    <p>TRRAIN · EdZola AI Verification Engine (POC) · Generated ${new Date().toLocaleString()}</p>
    <p style="margin-top:6px;font-size:13px"><strong>Candidate:</strong> ${d.candidate_name||'—'} &nbsp;|&nbsp; <strong>Employer:</strong> ${d.employer_name||'—'}</p>
  </div>
  <div class="score-box">
    <div class="score-num">${score.score}</div>
    <div style="font-size:11px;color:#94a3b8;margin:2px 0">/ 100</div>
    <div class="score-verdict">${score.verdict}</div>
  </div>
</div>

<div class="section">
  <h2>Score Breakdown</h2>
  <table><tbody>${breakdownRows}</tbody></table>
</div>

${(d.all_anomalies||[]).length > 0 ? `
<div class="section">
  <h2>⚠ Anomalies Detected (${(d.all_anomalies||[]).length})</h2>
  <table style="background:#fef2f2;border-radius:8px;overflow:hidden"><tbody>${anomalyRows}</tbody></table>
</div>` : '<div class="section"><h2>Anomalies</h2><p style="color:#10b981;font-size:13px">✓ No anomalies detected</p></div>'}

<div class="section">
  <h2>Extracted Fields (Completeness: ${l2.overall_completeness_score||0}%)</h2>
  <table><tbody>${fieldRows}</tbody></table>
</div>

<div class="section">
  <h2>Layer Signals</h2>
  <table>
    <thead><tr>
      <th style="text-align:left;padding:5px 10px;font-size:11px;color:#64748b;background:#f8fafc">Layer</th>
      <th style="padding:5px 10px;background:#f8fafc"></th>
      <th style="text-align:left;padding:5px 10px;font-size:11px;color:#64748b;background:#f8fafc">Signal</th>
    </tr></thead>
    <tbody>${signalRows}</tbody>
  </table>
</div>

<div class="section">
  <h2>Analysis Summary</h2>
  <table>
    <tr><td style="padding:5px 10px;font-size:12px;color:#64748b">Text extracted</td><td style="padding:5px 10px;font-size:12px">${(d.layer1?.text?.length||0).toLocaleString()} characters</td></tr>
    <tr><td style="padding:5px 10px;font-size:12px;color:#64748b">Document pages</td><td style="padding:5px 10px;font-size:12px">${d.layer1?.page_count||0}</td></tr>
    <tr><td style="padding:5px 10px;font-size:12px;color:#64748b">Metadata risk score</td><td style="padding:5px 10px;font-size:12px">${l3.risk_score||0}/100</td></tr>
    <tr><td style="padding:5px 10px;font-size:12px;color:#64748b">ELA risk score</td><td style="padding:5px 10px;font-size:12px">${l4.ela_risk||0}/100</td></tr>
    <tr><td style="padding:5px 10px;font-size:12px;color:#64748b">Historical similarity</td><td style="padding:5px 10px;font-size:12px">${l5.similarity_score!=null?l5.similarity_score+'%':'No prior baseline'}</td></tr>
    <tr><td style="padding:5px 10px;font-size:12px;color:#64748b">Geotagged photo</td><td style="padding:5px 10px;font-size:12px">${l6.has_geotag?'Yes — GPS extracted':'Not provided or no GPS data'}</td></tr>
  </table>
</div>

<div class="footer">
  <span>TRRAIN · EdZola Technologies · Offer Letter Verification POC</span>
  <span>This report is for internal review purposes only — not a legal determination of document authenticity.</span>
</div>
</body></html>`;

  const win = window.open('', '_blank');
  win.document.write(html);
  win.document.close();
  setTimeout(() => win.print(), 800);
}

// ── Reset ──
function resetForm() {
  pdfFile = null; photoFile = null; lastResults = null;
  document.getElementById('pdf-name').textContent = 'No file selected';
  document.getElementById('pdf-name').className = 'file-name';
  document.getElementById('photo-name').textContent = 'No file selected';
  document.getElementById('photo-name').className = 'file-name';
  document.getElementById('pdf-input').value = '';
  document.getElementById('photo-input').value = '';
  document.getElementById('analyze-btn').disabled = true;
  document.getElementById('analyze-btn').textContent = 'Run 6-Layer Analysis';
  document.getElementById('upload-section').style.opacity = '1';
  document.getElementById('progress').className = 'layers-progress';
  document.getElementById('results').className = 'results';
  document.getElementById('anomalies-section').style.display = 'none';
  document.querySelector('.upload-icon').textContent = '📄';
  document.querySelector('.drop-hint').textContent = '↓ Drop PDF here or click to browse';
  for(let i=1;i<=6;i++) setStep(i,'⏳ Waiting',false);
  window.scrollTo({top:0,behavior:'smooth'});
}
</script>
</body>
</html>

"""

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(debug=False, host="0.0.0.0", port=port)
