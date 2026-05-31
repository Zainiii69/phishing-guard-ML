from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import func
from pydantic import BaseModel, field_validator
import pickle
import json
import os
import uvicorn
import pandas as pd
from typing import List, Dict, Optional

# Import local modules
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import ScanResult, get_db, init_db
from feature_extractor import FeatureExtractor
from whitelist import is_safe_domain
from score_adjuster import adjust_score
from url_checker import check_website_exists

# Load URL Model & Features
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../ml/phishing_model.pkl')
FEATURES_PATH = os.path.join(os.path.dirname(__file__), '../ml/features.json')

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(FEATURES_PATH, 'r') as f:
        feature_names = json.load(f)
    print("URL model loaded successfully.")
except Exception as e:
    print(f"Error loading URL model: {e}")
    model = None
    feature_names = []

# Load Email Model & Vectorizer
EMAIL_MODEL_PATH = os.path.join(os.path.dirname(__file__), '../ml/email_phishing_model.pkl')
EMAIL_VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), '../ml/email_vectorizer.pkl')

try:
    with open(EMAIL_MODEL_PATH, 'rb') as f:
        email_model = pickle.load(f)
    with open(EMAIL_VECTORIZER_PATH, 'rb') as f:
        email_vectorizer = pickle.load(f)
    print("Email model loaded successfully.")
except Exception as e:
    print(f"Error loading email model: {e}")
    email_model = None
    email_vectorizer = None

extractor = FeatureExtractor()

# Modern lifespan handler (replaces deprecated @app.on_event)
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app = FastAPI(title="Phishing Detection API", lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Models
class ScanRequest(BaseModel):
    url: str

    @field_validator('url')
    @classmethod
    def url_must_not_be_empty(cls, v):
        v = v.strip()
        if not v:
            raise ValueError('URL cannot be empty')
        if len(v) < 4:
            raise ValueError('URL is too short')
        return v

class ScanResponse(BaseModel):
    url: str
    is_phishing: bool
    confidence: float
    whitelisted: bool
    features: Dict[str, float]
    security_details: Optional[Dict] = None
    website_exists: bool = True
    existence_details: Optional[Dict] = None

class WebsiteExistsResponse(BaseModel):
    url: str
    exists: bool
    dns_resolves: bool
    is_reachable: bool
    status_code: int
    response_time_ms: float
    ip_address: Optional[str] = None
    error: Optional[str] = None

class HistoryItem(BaseModel):
    id: int
    url: str
    is_phishing: bool
    confidence: float
    whitelisted: bool
    timestamp: str
    scan_type: str = "url"

    class Config:
        from_attributes = True

class ScanEmailRequest(BaseModel):
    email_text: str

    @field_validator('email_text')
    @classmethod
    def email_text_must_not_be_empty(cls, v):
        v = v.strip()
        if not v:
            raise ValueError('Email content cannot be empty')
        return v

class ScanEmailResponse(BaseModel):
    email_preview: str
    is_phishing: bool
    confidence: float

class StatsResponse(BaseModel):
    total_scans: int
    phishing_detected: int
    safe_urls: int
    whitelisted_count: int
    total_urls: int
    total_emails: int
    url_phishing_detected: int
    email_phishing_detected: int


@app.post("/check-exists", response_model=WebsiteExistsResponse)
def check_exists(request: ScanRequest):
    """Check if a website exists and is reachable (standalone endpoint)."""
    result = check_website_exists(request.url)
    return WebsiteExistsResponse(
        url=request.url,
        exists=result['exists'],
        dns_resolves=result['dns_resolves'],
        is_reachable=result['is_reachable'],
        status_code=result['status_code'],
        response_time_ms=result['response_time_ms'],
        ip_address=result.get('ip_address'),
        error=result.get('error'),
    )


@app.post("/scan", response_model=ScanResponse)
def scan_url(request: ScanRequest, db: Session = Depends(get_db)):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # --- Step 0: Check if website exists ---
    existence = check_website_exists(request.url)
    existence_details = {
        'exists': existence['exists'],
        'dns_resolves': existence['dns_resolves'],
        'is_reachable': existence['is_reachable'],
        'status_code': existence['status_code'],
        'response_time_ms': existence['response_time_ms'],
        'ip_address': existence.get('ip_address'),
        'error': existence.get('error'),
    }

    # If website doesn't exist (DNS fails), return early with minimal analysis
    if not existence['dns_resolves']:
        db_result = ScanResult(
            url=request.url,
            is_phishing=True,
            confidence=0.85,
            whitelisted=False,
            features=json.dumps({}),
            security_details=json.dumps({}),
            scan_type="url"
        )
        db.add(db_result)
        db.commit()
        db.refresh(db_result)
        return {
            "url": request.url,
            "is_phishing": True,
            "confidence": 0.85,
            "whitelisted": False,
            "features": {},
            "security_details": {},
            "website_exists": False,
            "existence_details": existence_details,
        }

    # Extract Features (now includes SSL/TLS and protocol security checks)
    raw_features = extractor.extract(request.url)
    
    # Check Whitelist using the new subdomain-aware matching
    domain = extractor.extract_domain(request.url)
    whitelisted = is_safe_domain(domain)
    
    if whitelisted:
        # Bypass Model — whitelisted domains are treated as safe
        is_phishing = False
        phishing_prob = 0.0
    else:
        # Align features with model input
        input_df = pd.DataFrame([raw_features])
        
        # Ensure all model features exist, fill missing with 0
        try:
            input_data = input_df[feature_names]
        except KeyError:
            missing_cols = set(feature_names) - set(input_df.columns)
            for c in missing_cols:
                input_df[c] = 0
            input_data = input_df[feature_names]

        # Predict
        prob = model.predict_proba(input_data)[0]
        
        # Class 1 is Phishing
        phishing_prob = float(prob[1])
    
    # --- Score Adjustment Based on Security Signals ---
    ssl_info = getattr(extractor, 'last_ssl_info', {})
    security_headers = getattr(extractor, 'last_security_headers', {})
    redirect_info = getattr(extractor, 'last_redirect_info', {})
    mixed_content_info = getattr(extractor, 'last_mixed_content_info', {})

    score_result = adjust_score(
        ml_score=phishing_prob,
        ssl_info=ssl_info,
        security_headers=security_headers,
        redirect_info=redirect_info,
        mixed_content_info=mixed_content_info,
        is_whitelisted=whitelisted,
    )

    adjusted_prob = score_result['adjusted_score']
    is_phishing = adjusted_prob > 0.5
    
    # Build security details for response
    security_details = {
        'ssl_valid': ssl_info.get('is_valid', False),
        'ssl_issuer': ssl_info.get('issuer', 'N/A'),
        'ssl_expires_in_days': ssl_info.get('expires_in_days', -1),
        'ssl_self_signed': ssl_info.get('is_self_signed', False),
        'ssl_domain_match': ssl_info.get('domain_match', False),
        'tls_version': ssl_info.get('protocol_version', 'None'),
        'has_hsts': security_headers.get('has_hsts', False),
        'has_csp': security_headers.get('has_csp', False),
        'has_x_frame_options': security_headers.get('has_x_frame_options', False),
        'http_to_https_redirect': redirect_info.get('redirects_to_https', False),
        'mixed_content': mixed_content_info.get('has_mixed_content', False),
        'security_headers_score': security_headers.get('security_headers_score', 0),
        'score_adjustment': score_result['total_adjustment'],
        'original_ml_score': score_result['original_ml_score'],
        'adjusted_score': score_result['adjusted_score'],
        'penalties': score_result['penalties_applied'],
        'bonuses': score_result['bonuses_applied'],
    }

    # Convert features to native types for JSON serialization
    safe_features = {k: float(v) for k, v in raw_features.items()}
    
    # Save to History
    db_result = ScanResult(
        url=request.url,
        is_phishing=is_phishing,
        confidence=adjusted_prob,
        whitelisted=whitelisted,
        features=json.dumps(safe_features),
        security_details=json.dumps(security_details),
        scan_type="url"
    )
    db.add(db_result)
    db.commit()
    db.refresh(db_result)
    
    return {
        "url": request.url,
        "is_phishing": is_phishing,
        "confidence": adjusted_prob,
        "whitelisted": whitelisted,
        "features": {k: float(v) for k, v in raw_features.items()},
        "security_details": security_details,
        "website_exists": existence['exists'],
        "existence_details": existence_details,
    }

@app.post("/scan-email", response_model=ScanEmailResponse)
def scan_email(request: ScanEmailRequest, db: Session = Depends(get_db)):
    if not email_model or not email_vectorizer:
        raise HTTPException(status_code=500, detail="Email phishing model not loaded")

    email_text = request.email_text
    
    try:
        # Vectorize email text
        vec_text = email_vectorizer.transform([email_text])
        
        # Predict probability
        prob = email_model.predict_proba(vec_text)[0]
        
        # Class 1 is phishing, Class 0 is safe
        confidence = float(prob[1])
        is_phishing = confidence > 0.5
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    # Generate a preview snippet (max 100 chars) to display in lists
    preview = email_text[:100] + "..." if len(email_text) > 100 else email_text

    # Log to ScanResult history table
    db_result = ScanResult(
        url=preview,
        is_phishing=is_phishing,
        confidence=confidence,
        whitelisted=False,
        features=json.dumps({}),
        security_details=json.dumps({}),
        scan_type="email"
    )
    db.add(db_result)
    db.commit()
    db.refresh(db_result)

    return ScanEmailResponse(
        email_preview=preview,
        is_phishing=is_phishing,
        confidence=confidence
    )

@app.get("/history", response_model=List[HistoryItem])
def get_history(db: Session = Depends(get_db)):
    results = db.query(ScanResult).order_by(ScanResult.timestamp.desc()).limit(50).all()
    return [
        HistoryItem(
            id=r.id,
            url=r.url,
            is_phishing=r.is_phishing,
            confidence=r.confidence,
            whitelisted=r.whitelisted if r.whitelisted is not None else False,
            timestamp=r.timestamp.isoformat(),
            scan_type=r.scan_type if r.scan_type is not None else "url"
        )
        for r in results
    ]

@app.get("/stats", response_model=StatsResponse)
def get_stats(db: Session = Depends(get_db)):
    """Returns aggregate statistics from the entire scan history."""
    total = db.query(func.count(ScanResult.id)).scalar() or 0
    phishing = db.query(func.count(ScanResult.id)).filter(ScanResult.is_phishing == True).scalar() or 0
    whitelisted = db.query(func.count(ScanResult.id)).filter(ScanResult.whitelisted == True).scalar() or 0
    safe = total - phishing
    
    # URL vs Email stats breakdown
    total_urls = db.query(func.count(ScanResult.id)).filter(ScanResult.scan_type == "url").scalar() or 0
    total_emails = db.query(func.count(ScanResult.id)).filter(ScanResult.scan_type == "email").scalar() or 0
    
    url_phishing = db.query(func.count(ScanResult.id)).filter(ScanResult.scan_type == "url", ScanResult.is_phishing == True).scalar() or 0
    email_phishing = db.query(func.count(ScanResult.id)).filter(ScanResult.scan_type == "email", ScanResult.is_phishing == True).scalar() or 0
    
    return StatsResponse(
        total_scans=total,
        phishing_detected=phishing,
        safe_urls=safe,
        whitelisted_count=whitelisted,
        total_urls=total_urls,
        total_emails=total_emails,
        url_phishing_detected=url_phishing,
        email_phishing_detected=email_phishing
    )

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
