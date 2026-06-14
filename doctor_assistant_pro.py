"""
NFÖ Doctor's Assistant Pro
Medical-Grade Clinical Decision Support System
Band of Agents Hackathon - Track 3: Regulated & High-Stakes Workflows
6-Agent System with Clinical Reviewer
"""

import os
import time
import json
import re
import hashlib
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq
import streamlit as st
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import io
import sys
import os as _os

# Ensure clinical_scores.py is importable from the same directory
_dir = _os.path.dirname(_os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

try:
    from clinical_scores import (
        ClinicalData, extract_clinical_data,
        run_all_scores, compute_anchored_probabilities,
        ScoreResult,
    )
    SCORES_AVAILABLE = True
except ImportError:
    SCORES_AVAILABLE = False

# ============================================================
# BAND SDK INTEGRATION - HACKATHON
# ============================================================
try:
    from band_sdk import BandClient
    BAND_SDK_AVAILABLE = True
except ImportError:
    BAND_SDK_AVAILABLE = False

# ============================================================
# PAGE CONFIG — must be first Streamlit call
# ============================================================
st.set_page_config(
    page_title="NFÖ Clinical AI — Doctor's Assistant Pro | 6-Agent System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# GLOBAL CSS — Medical-grade dark clinical theme
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root palette ── */
:root {
  --bg-primary:    #0d1117;
  --bg-surface:    #161b22;
  --bg-elevated:   #1f2937;
  --bg-card:       #111827;
  --border:        #30363d;
  --border-accent: #21262d;
  --text-primary:  #f0f6fc;
  --text-secondary:#8b949e;
  --text-muted:    #6e7681;
  --accent-blue:   #58a6ff;
  --accent-green:  #3fb950;
  --accent-orange: #d29922;
  --accent-red:    #f85149;
  --accent-purple: #bc8cff;
  --accent-teal:   #39d353;
  --emergency-bg:  #2d0b0b;
  --emergency-border: #f85149;
}

/* ── Base ── */
html, body, [class*="css"] {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.main .block-container {
  padding-top: 1rem;
  padding-bottom: 2rem;
  max-width: 1200px;
}

/* ── App background ── */
.stApp { background: var(--bg-primary); }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: var(--bg-surface) !important;
  border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

/* ── Header bar ── */
.nfo-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  background: var(--bg-surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1rem 1.5rem;
  margin-bottom: 1.5rem;
}
.nfo-logo { font-size: 1.5rem; font-weight: 700; color: var(--accent-blue); letter-spacing: -0.5px; }
.nfo-logo span { color: var(--text-secondary); font-weight: 400; font-size: 0.85rem; margin-left: 0.5rem; }
.nfo-badge {
  background: rgba(63,185,80,0.12);
  border: 1px solid rgba(63,185,80,0.3);
  color: var(--accent-green);
  font-size: 0.7rem;
  font-weight: 600;
  padding: 3px 10px;
  border-radius: 20px;
  letter-spacing: 0.5px;
}
.band-badge {
  background: rgba(88,166,255,0.12);
  border: 1px solid rgba(88,166,255,0.3);
  color: var(--accent-blue);
  font-size: 0.65rem;
  font-weight: 600;
  padding: 3px 10px;
  border-radius: 20px;
}

/* ── Critical safety banner ── */
.safety-banner {
  background: rgba(248,81,73,0.08);
  border: 1px solid rgba(248,81,73,0.3);
  border-left: 4px solid var(--accent-red);
  border-radius: 8px;
  padding: 0.75rem 1rem;
  margin-bottom: 1.5rem;
  font-size: 0.8rem;
  color: #ffa198;
}
.safety-banner strong { color: var(--accent-red); }

/* ── Emergency alert ── */
.emergency-alert {
  background: var(--emergency-bg);
  border: 2px solid var(--emergency-border);
  border-radius: 10px;
  padding: 1rem 1.25rem;
  margin: 1rem 0;
  animation: pulse-border 2s infinite;
}
@keyframes pulse-border {
  0%, 100% { box-shadow: 0 0 0 0 rgba(248,81,73,0.4); }
  50%       { box-shadow: 0 0 0 6px rgba(248,81,73,0); }
}
.emergency-alert h3 { color: var(--accent-red); margin: 0 0 0.25rem; font-size: 1rem; }
.emergency-alert p  { color: #ffa198; margin: 0; font-size: 0.85rem; }

/* ── Section headers ── */
.section-header {
  font-size: 0.7rem;
  font-weight: 700;
  letter-spacing: 1.5px;
  text-transform: uppercase;
  color: var(--text-muted);
  margin: 1.5rem 0 0.75rem;
  padding-bottom: 0.4rem;
  border-bottom: 1px solid var(--border);
}

/* ── Input fields ── */
.stTextInput input, .stTextArea textarea, .stSelectbox > div > div {
  background: var(--bg-elevated) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  color: var(--text-primary) !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 0.875rem !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
  border-color: var(--accent-blue) !important;
  box-shadow: 0 0 0 3px rgba(88,166,255,0.1) !important;
}
label, .stSelectbox label, .stTextInput label, .stTextArea label {
  color: var(--text-secondary) !important;
  font-size: 0.78rem !important;
  font-weight: 500 !important;
  letter-spacing: 0.3px !important;
}

/* ── Primary analyze button ── */
.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
  border: none !important;
  border-radius: 10px !important;
  color: white !important;
  font-weight: 600 !important;
  font-size: 0.9rem !important;
  padding: 0.65rem 2rem !important;
  letter-spacing: 0.3px !important;
  transition: all 0.2s !important;
  box-shadow: 0 4px 12px rgba(37,99,235,0.35) !important;
}
.stButton > button[kind="primary"]:hover {
  background: linear-gradient(135deg, #1e40af, #1d4ed8) !important;
  box-shadow: 0 6px 20px rgba(37,99,235,0.5) !important;
  transform: translateY(-1px) !important;
}

/* ── Secondary buttons ── */
.stButton > button:not([kind="primary"]) {
  background: var(--bg-elevated) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  color: var(--text-secondary) !important;
  font-size: 0.8rem !important;
}
.stButton > button:not([kind="primary"]):hover {
  border-color: var(--accent-blue) !important;
  color: var(--accent-blue) !important;
}

/* ── Confidence pills ── */
.conf-pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 14px;
  border-radius: 20px;
  font-size: 0.75rem;
  font-weight: 700;
  letter-spacing: 0.5px;
}
.conf-high   { background: rgba(63,185,80,0.12);  border: 1px solid rgba(63,185,80,0.35);  color: #3fb950; }
.conf-medium { background: rgba(210,153,34,0.12); border: 1px solid rgba(210,153,34,0.35); color: #d29922; }
.conf-low    { background: rgba(248,81,73,0.12);  border: 1px solid rgba(248,81,73,0.35);  color: #f85149; }

/* ── Diagnosis cards ── */
.diag-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 1rem 1.25rem;
  margin: 0.6rem 0;
  position: relative;
  overflow: hidden;
  transition: border-color 0.2s;
}
.diag-card:hover { border-color: var(--accent-blue); }
.diag-card::before {
  content: '';
  position: absolute;
  left: 0; top: 0; bottom: 0;
  width: 4px;
}
.diag-high   .diag-card::before { background: var(--accent-red); }
.diag-medium .diag-card::before { background: var(--accent-orange); }
.diag-low    .diag-card::before { background: var(--accent-blue); }

.diag-name { font-size: 0.95rem; font-weight: 600; color: var(--text-primary); }
.diag-prob { font-size: 0.8rem; font-weight: 700; color: var(--accent-blue); float: right; }
.diag-rationale { font-size: 0.8rem; color: var(--text-secondary); margin-top: 0.4rem; }
.diag-missing { font-size: 0.75rem; color: var(--accent-orange); margin-top: 0.3rem; }

/* ── Probability bar ── */
.prob-bar-wrap { margin: 0.5rem 0 0.25rem; }
.prob-bar-track {
  background: var(--border);
  border-radius: 4px;
  height: 5px;
  overflow: hidden;
}
.prob-bar-fill {
  height: 5px;
  border-radius: 4px;
  transition: width 0.6s ease;
}

/* ── Fact trace items ── */
.fact-item {
  display: flex;
  align-items: flex-start;
  gap: 0.75rem;
  padding: 0.5rem 0.75rem;
  border-radius: 8px;
  margin: 0.3rem 0;
  background: var(--bg-elevated);
  border: 1px solid var(--border-accent);
  font-size: 0.8rem;
}
.fact-id {
  background: rgba(88,166,255,0.15);
  border: 1px solid rgba(88,166,255,0.25);
  color: var(--accent-blue);
  border-radius: 4px;
  padding: 1px 6px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.7rem;
  white-space: nowrap;
  flex-shrink: 0;
}
.fact-text  { color: var(--text-primary); font-weight: 500; }
.fact-src   { color: var(--text-muted); font-size: 0.72rem; }
.fact-cat {
  font-size: 0.65rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  padding: 1px 6px;
  border-radius: 4px;
  margin-left: auto;
  flex-shrink: 0;
}
.cat-symptoms { background: rgba(63,185,80,0.12);  color: #3fb950; }
.cat-history  { background: rgba(88,166,255,0.12); color: var(--accent-blue); }
.cat-vitals   { background: rgba(188,140,255,0.12);color: var(--accent-purple);}
.cat-labs     { background: rgba(210,153,34,0.12); color: var(--accent-orange);}
.cat-safety   { background: rgba(248,81,73,0.12);  color: var(--accent-red);  }
.cat-process  { background: rgba(110,118,129,0.12);color: var(--text-muted);  }

/* ── SOAP note ── */
.soap-block {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 1.25rem;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.8rem;
  color: var(--text-primary);
  line-height: 1.6;
  white-space: pre-wrap;
}

/* ── Agent progress row ── */
.agent-row {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 0.5rem 0.75rem;
  border-radius: 8px;
  background: var(--bg-elevated);
  margin: 0.3rem 0;
  font-size: 0.8rem;
  border: 1px solid var(--border-accent);
  transition: all 0.3s;
}
.agent-row.done   { border-color: rgba(63,185,80,0.3);  background: rgba(63,185,80,0.05);  }
.agent-row.active { border-color: rgba(88,166,255,0.3); background: rgba(88,166,255,0.05); }
.agent-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.dot-done   { background: var(--accent-green); }
.dot-active { background: var(--accent-blue); animation: blink 1s infinite; }
.dot-wait   { background: var(--text-muted); }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }

/* ── Vitals grid ── */
.vital-chip {
  background: var(--bg-elevated);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 0.5rem 0.75rem;
  text-align: center;
  font-size: 0.75rem;
}
.vital-val  { font-size: 1rem; font-weight: 700; color: var(--text-primary); font-family: 'JetBrains Mono', monospace; }
.vital-lbl  { color: var(--text-muted); font-size: 0.65rem; letter-spacing: 0.5px; }

/* ── Missing info pills ── */
.missing-badge {
  display: inline-block;
  background: rgba(210,153,34,0.1);
  border: 1px solid rgba(210,153,34,0.3);
  color: var(--accent-orange);
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 0.72rem;
  margin: 2px;
}

/* ── Tabs ── */
.stTabs [role="tablist"] { border-bottom: 1px solid var(--border); gap: 0; }
.stTabs [role="tab"] {
  background: transparent !important;
  color: var(--text-secondary) !important;
  border: none !important;
  border-bottom: 2px solid transparent !important;
  border-radius: 0 !important;
  font-size: 0.8rem !important;
  font-weight: 500 !important;
  padding: 0.5rem 1rem !important;
}
.stTabs [role="tab"][aria-selected="true"] {
  color: var(--accent-blue) !important;
  border-bottom-color: var(--accent-blue) !important;
}

/* ── Metric boxes ── */
.stMetric { background: var(--bg-elevated); border-radius: 8px; border: 1px solid var(--border); padding: 0.75rem; }
.stMetric [data-testid="stMetricValue"] { color: var(--text-primary) !important; font-size: 1.5rem !important; }
.stMetric [data-testid="stMetricLabel"] { color: var(--text-secondary) !important; font-size: 0.75rem !important; }

/* ── Progress bar ── */
.stProgress > div > div { background: var(--accent-blue) !important; border-radius: 4px; }
.stProgress > div { background: var(--border) !important; border-radius: 4px; }

/* ── Expander ── */
.streamlit-expanderHeader {
  background: var(--bg-elevated) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  color: var(--text-secondary) !important;
  font-size: 0.8rem !important;
}
.streamlit-expanderContent {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-top: none !important;
}

/* ── Hide Streamlit branding ── */
#MainMenu, footer, .stDeployButton { visibility: hidden; }
header { background: transparent !important; }

/* ── Download button ── */
.stDownloadButton button {
  background: var(--bg-elevated) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  color: var(--text-primary) !important;
  font-size: 0.8rem !important;
}
.stDownloadButton button:hover {
  border-color: var(--accent-blue) !important;
  color: var(--accent-blue) !important;
}

/* ── Workflow visualization ── */
.workflow-node {
  text-align: center;
  padding: 0.5rem;
  border-radius: 8px;
  background: var(--bg-elevated);
  border: 1px solid var(--border);
  font-size: 0.7rem;
}
.workflow-node.done { border-color: var(--accent-green); background: rgba(63,185,80,0.1); }
.workflow-node.active { border-color: var(--accent-blue); background: rgba(88,166,255,0.1); animation: pulse 1s infinite; }
.workflow-arrow { text-align: center; font-size: 1.2rem; color: var(--text-muted); }

/* ── Info/warning/error override ── */
.stAlert {
  border-radius: 8px !important;
  font-size: 0.82rem !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }
</style>
""", unsafe_allow_html=True)


# ============================================================
# TESSERACT CHECK (optional OCR)
# ============================================================
def check_tesseract():
    try:
        from subprocess import run
        result = run(['tesseract', '--version'], capture_output=True, text=True)
        return result.returncode == 0, result.stdout.split('\n')[0] if result.returncode == 0 else "Not found"
    except Exception:
        return False, "Tesseract not installed"

TESSERACT_OK, TESSERACT_MSG = check_tesseract()

# ============================================================
# LOAD API KEY
# ============================================================
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("❌ **GROQ_API_KEY** not found. Create a `.env` file: `GROQ_API_KEY=gsk_your_key`")
    st.stop()

try:
    groq_client = Groq(api_key=api_key)
except Exception as e:
    st.error(f"❌ Groq init failed: {e}")
    st.stop()

# ============================================================
# BAND CLIENT INITIALIZATION - HACKATHON
# ============================================================
band_client = None
BAND_AVAILABLE = False
band_api_key = os.getenv("BAND_API_KEY")

if BAND_SDK_AVAILABLE and band_api_key:
    try:
        band_client = BandClient(api_key=band_api_key)
        BAND_AVAILABLE = True
    except Exception as e:
        st.warning(f"⚠️ Band client init failed: {e}")
elif band_api_key:
    BAND_AVAILABLE = True  # Will use REST API fallback

# Optional Google Gemini for OCR
google_client = None
google_api_key = os.getenv("GEMINI_API_KEY")
if google_api_key:
    try:
        from google import genai as google_genai
        google_client = google_genai.Client(api_key=google_api_key)
    except Exception:
        pass

# ============================================================
# SESSION STATE
# ============================================================
for key, default in [
    ('cases', []),
    ('current_results', None),
    ('ocr_text', ""),
    ('analysis_running', False),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ============================================================
# UTILITIES
# ============================================================
def safe_text(text, default=""):
    return text if text is not None else default

def safe_html(text):
    if text is None:
        return ""
    return str(text).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


# ============================================================
# BAND HELPER FUNCTIONS - HACKATHON
# ============================================================
def send_to_band(room_name, agent_name, action, data, case_id=None):
    """Send agent collaboration data to Band"""
    if not BAND_AVAILABLE:
        return False
    
    try:
        if band_client and BAND_SDK_AVAILABLE:
            # Use Band SDK
            room = band_client.get_or_create_room(room_name)
            room.post_message({
                "agent": agent_name,
                "action": action,
                "data": data,
                "case_id": case_id,
                "timestamp": datetime.now().isoformat()
            })
            return True
        else:
            return False
    except Exception as e:
        return False

# ============================================================
# FACT TRACE  — individual provenance per clinical fact
# ============================================================
class FactTrace:
    def __init__(self, data=None):
        self.facts = data if isinstance(data, list) else []
        ids = [f.get("fact_id", 0) for f in self.facts if isinstance(f.get("fact_id"), int)]
        self._next_id = max(ids, default=0) + 1

    def add_fact(self, fact, source, confidence="HIGH", category="general"):
        fid = self._next_id
        self._next_id += 1
        self.facts.append({
            "fact_id": fid,
            "fact": safe_text(fact),
            "source": safe_text(source),
            "timestamp": datetime.now().isoformat(),
            "confidence": confidence,
            "category": category,
        })
        return fid

    def get_clinical_facts(self):
        return [f for f in self.facts if f.get("category") in ("symptoms", "history", "vitals", "labs")]

    def get_all(self):
        return self.facts

    def to_list(self):
        return self.facts

    def to_json(self):
        return json.dumps(self.facts, indent=2, default=str)


# ============================================================
# EXTRACT CLINICAL ENTITIES from free text → add to FactTrace
# ============================================================
CONDITION_PATTERNS = {
    "Hypertension":  [r'\bhtn\b', r'hypertension', r'high blood pressure'],
    "Diabetes":      [r'\bdiabetes\b', r'diabetic', r't2dm', r'type 2 diabetes'],
    "Coronary Artery Disease": [r'\bcad\b', r'coronary artery disease', r'heart disease'],
    "Myocardial Infarction (prior)": [r'prior mi\b', r'old mi\b', r'previous heart attack'],
    "COPD":          [r'\bcopd\b', r'chronic obstructive pulmonary'],
    "Asthma":        [r'\basthma\b'],
    "Atrial Fibrillation": [r'\bafib\b', r'atrial fibrillation'],
    "Smoking":       [r'smok', r'pack.year', r'cigarette'],
    "Hyperlipidemia":[r'hyperlipidemia', r'high cholesterol', r'dyslipidemia', r'statin'],
}

SYMPTOM_PATTERNS = {
    "Chest pain":               [r'chest pain', r'chest discomfort', r'crushing chest', r'pressure in chest'],
    "Left arm radiation":       [r'radiates? to left arm', r'left arm pain', r'arm radiation'],
    "Jaw pain/radiation":       [r'jaw pain', r'radiates? to jaw'],
    "Diaphoresis":              [r'\bsweating\b', r'diaphoresis', r'drenching sweat'],
    "Nausea/vomiting":          [r'\bnausea\b', r'nausea', r'vomiting'],
    "Shortness of breath":      [r'shortness of breath', r'\bsob\b', r'dyspnea', r"can't breathe", r'difficulty breathing'],
    "Facial droop":             [r'facial droop', r'face drooping', r'drooping face'],
    "Slurred speech":           [r'slurred speech', r'difficulty speaking', r"can't speak", r'speech difficulty'],
    "Arm weakness":             [r'arm weakness', r'weak arm', r'one.sided weakness', r'sudden weakness'],
    "Leg weakness":             [r'leg weakness', r'weak leg'],
    "Headache":                 [r'\bheadache\b', r'head pain', r'cephalgia'],
    "Fever":                    [r'\bfever\b', r'febrile', r'temperature [\d]'],
    "Confusion":                [r'\bconfusion\b', r'altered mental', r'disoriented'],
    "Abdominal pain":           [r'abdominal pain', r'belly pain', r'stomach pain', r'epigastric'],
    "Syncope":                  [r'\bsyncope\b', r'passed out', r'lost consciousness', r'fainted'],
}

def extract_clinical_entities(text, source, fact_trace, category):
    if not text:
        return
    tl = text.lower()
    patterns = CONDITION_PATTERNS if category == "history" else SYMPTOM_PATTERNS
    for label, rxs in patterns.items():
        for rx in rxs:
            if re.search(rx, tl):
                fact_trace.add_fact(label, source, "HIGH", category)
                break


# ============================================================
# CONFIDENCE CALCULATION
# ============================================================
DURATION_WORDS  = ["hour", "minute", "day", "week", "month", "year", "ago", "since", "onset", "started", "began"]
SEVERITY_WORDS  = ["mild", "moderate", "severe", "crushing", "sharp", "dull", "burning", "stabbing", "10/10", "9/10", "8/10", "7/10", "/10"]

def calculate_confidence(symptoms, medical_history, vitals, ocr_text=""):
    score = 0
    if symptoms and len(symptoms) > 20:   score += 25
    if medical_history and len(medical_history) > 10: score += 20
    if vitals and any(v for v in vitals.values() if v): score += 15
    if ocr_text and len(ocr_text) > 50:   score += 20
    tl = (symptoms or "").lower()
    if any(w in tl for w in DURATION_WORDS): score += 10
    if any(w in tl for w in SEVERITY_WORDS): score += 10

    if score >= 70:  return "HIGH", score
    if score >= 40:  return "MEDIUM", score
    return "LOW", score

def get_missing_fields(symptoms, medical_history, vitals):
    missing = []
    tl = (symptoms or "").lower()
    if not symptoms or len(symptoms) < 20:            missing.append("Detailed symptom description")
    if not any(w in tl for w in DURATION_WORDS):      missing.append("Duration / onset time")
    if not any(w in tl for w in SEVERITY_WORDS):      missing.append("Severity (e.g. 8/10, severe)")
    if not medical_history or len(medical_history) < 10: missing.append("Medical history / past conditions")
    return missing


# ============================================================
# EMERGENCY DETECTION
# ============================================================
EMERGENCY_PHRASES = [
    "chest pain", "crushing chest", "heart attack", "cardiac arrest",
    "facial droop", "face drooping", "slurred speech", "arm weakness",
    "leg weakness", "sudden weakness", "sudden numbness", "one-sided weakness",
    "difficulty speaking", "can't speak", "stroke", "cva",
    "sudden confusion", "sudden vision loss", "sudden severe headache",
    "shortness of breath", "can't breathe", "difficulty breathing",
    "severe bleeding", "unconscious", "unresponsive", "not breathing",
    "seizure", "anaphylaxis", "severe allergic", "choking",
    "passed out", "syncope", "head injury",
]

def check_emergency(text, fact_trace):
    if not text:
        return False, None
    tl = text.lower()
    for phrase in EMERGENCY_PHRASES:
        if phrase in tl:
            fact_trace.add_fact(f"Emergency keyword detected: {phrase}", "Emergency detector", "HIGH", "safety")
            return True, phrase
    return False, None


# ============================================================
# MEDICAL GATEWAY — ensure input is clinical
# ============================================================
MEDICAL_KEYWORDS = [
    "pain", "ache", "fever", "cough", "headache", "chest", "breath",
    "nausea", "vomit", "diarrhea", "rash", "swelling", "bleeding",
    "fatigue", "dizzy", "weakness", "numb", "blood", "pressure",
    "heart", "lung", "stomach", "infection", "diabetes", "asthma",
    "stroke", "droop", "slurred", "vision", "confusion", "seizure",
    "paralysis", "syncope", "dyspnea", "dysphagia", "palpitation",
    "edema", "pallor", "jaundice", "diaphoresis", "tachycardia",
]

def is_medical_input(text):
    if not text:
        return False
    tl = text.lower()
    return any(kw in tl for kw in MEDICAL_KEYWORDS)


# ============================================================
# GROQ CALL  — retry wrapper
# ============================================================
def call_groq(prompt, system_msg="You are a clinical AI assistant.", max_tokens=900, temperature=0.0):
    if not prompt:
        return "Error: empty prompt"
    for attempt in range(3):
        try:
            resp = groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": safe_text(system_msg)},
                    {"role": "user",   "content": prompt},
                ],
                model="llama-3.1-8b-instant",
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return safe_text(resp.choices[0].message.content, "No response")
        except Exception as e:
            if attempt == 2:
                return f"API Error: {str(e)[:120]}"
            time.sleep(2 ** attempt)
    return "Error: max retries"


# ============================================================
# JSON PARSER with fallback
# ============================================================
def parse_diagnoses_json(response):
    if not response:
        return []
    for pattern in [r'\{[\s\S]*\}', r'\[[\s\S]*\]']:
        m = re.search(pattern, response)
        if m:
            try:
                data = json.loads(m.group())
                if isinstance(data, dict):
                    return data.get("diagnoses", [])
                if isinstance(data, list):
                    return data
            except json.JSONDecodeError:
                continue
    return []


# ============================================================
# AGENT 1 — Symptom Extractor
# ============================================================
def agent_extract_symptoms(symptoms, medical_history, ocr_text, fact_trace):
    extract_clinical_entities(symptoms, "Symptom entry", fact_trace, "symptoms")
    extract_clinical_entities(medical_history, "Medical history field", fact_trace, "history")
    if ocr_text:
        extract_clinical_entities(ocr_text, "Uploaded document", fact_trace, "labs")

    prompt = f"""
You are a clinical information extractor. Your ONLY job is to extract what is explicitly stated.
NEVER add, invent, or infer anything not written below.
If a field is missing, write exactly: "Not documented".

PATIENT SYMPTOMS: \"\"\"{safe_text(symptoms)}\"\"\"
MEDICAL HISTORY: \"\"\"{safe_text(medical_history)}\"\"\"
DOCUMENT TEXT: \"\"\"{safe_text(ocr_text[:500] if ocr_text else '')}\"\"\"

Extract and format:
Chief complaint: 
Onset / duration: 
Severity: 
Associated symptoms: 
Relevant medical history: 
Current medications: 
Risk factors: 
"""
    result = call_groq(prompt, system_msg="Clinical extractor. Never invent. Only extract stated facts.")
    fact_trace.add_fact("Clinical extraction completed", "Agent 1 — Extractor", "HIGH", "process")
    return safe_text(result, "Extraction failed")


# ============================================================
# AGENT 2 — Diagnostic Reasoner (score-anchored)
# ============================================================
_NONSENSE = {"psychological", "cognitive", "technical", "system", "agent", "process",
             "metadata", "session", "initialized", "assistant", "llm"}

def agent_diagnoses(analysis, fact_trace, confidence_level, is_emergency,
                    symptoms_text="", history_text="", vitals=None,
                    age=None, gender=""):
    vitals = vitals or {}
    cd = None
    scores = []
    anchored = {}

    if SCORES_AVAILABLE:
        try:
            cd = extract_clinical_data(symptoms_text, history_text, vitals, age, gender)
            scores = run_all_scores(cd)
            anchored = compute_anchored_probabilities(scores, cd)
        except Exception as e:
            pass

    clinical = fact_trace.get_clinical_facts()
    facts_block = "\n".join(
        f"  [{f['fact_id']}] {f['fact']}  (source: {f['source']})"
        for f in clinical[:20]
    ) if clinical else "(No specific clinical facts extracted — use analysis text only)"

    anchored_block = ""
    if anchored:
        lines = []
        for cond, info in anchored.items():
            lines.append(f"  • {cond}: {info['prob']} (anchored by {info['basis']})")
        anchored_block = "EVIDENCE-BASED PROBABILITY ANCHORS (DO NOT change these values):\n" + "\n".join(lines)

    prompt = f"""
You are a senior clinical diagnostician. Generate differential diagnoses.

ABSOLUTE RULES:
1. Use ONLY the clinical facts and analysis below.
2. For conditions listed in ANCHORED PROBABILITIES, use EXACTLY those probability ranges — do not change them.
3. Do NOT diagnose system events, metadata, or internal processes.
4. Every diagnosis must be a real medical condition.
5. For minimal information (e.g. "headache" only), keep probabilities <40% and add "Insufficient Information" entry.
6. Probabilities should sum to approximately 100%.

{anchored_block}

CLINICAL FACTS:
{facts_block}

CLINICAL ANALYSIS:
{analysis[:800]}

CONFIDENCE: {confidence_level}
EMERGENCY FLAGGED: {is_emergency}

Return ONLY valid JSON:
{{
  "diagnoses": [
    {{
      "name": "Exact condition name",
      "probability": "XX%",
      "rationale": "1-2 sentence evidence-based rationale citing specific symptoms/history",
      "supporting_fact_ids": [],
      "missing_evidence": ["specific test or data that would confirm or rule out"]
    }}
  ]
}}
Generate 3–5 differential diagnoses ordered by probability.
"""
    raw = call_groq(prompt, system_msg=(
        "Clinical diagnostic engine. Return ONLY valid JSON. "
        "Never fabricate probabilities for score-anchored conditions. "
        "Never diagnose system events."
    ))
    diagnoses = parse_diagnoses_json(raw)

    valid = []
    for d in diagnoses:
        name_lower = d.get("name", "").lower()
        if any(ns in name_lower for ns in _NONSENSE):
            continue
        d.setdefault("name", "Unknown condition")
        d.setdefault("probability", "Uncertain")
        d.setdefault("rationale", "Based on available clinical information")
        d.setdefault("supporting_fact_ids", [])
        d.setdefault("missing_evidence", [])
        valid.append(d)

    for d in valid:
        name = d.get("name", "")
        for anchored_name, info in anchored.items():
            if (anchored_name.lower() in name.lower()) or (name.lower() in anchored_name.lower()):
                d["probability"] = info["prob"]
                d["score_anchor"] = info["basis"]
                d.setdefault("missing_evidence", [])
                d["rationale"] = f"[{info['score_name']} {info['score_val']}] " + d["rationale"]
                break

    existing_names = {d.get("name","").lower() for d in valid}
    for anchored_name, info in anchored.items():
        already = any(anchored_name.lower() in n or n in anchored_name.lower() for n in existing_names)
        if not already:
            valid.insert(0, {
                "name": anchored_name,
                "probability": info["prob"],
                "rationale": f"[{info['score_name']} {info['score_val']}] Score-derived probability based on clinical criteria.",
                "supporting_fact_ids": [],
                "missing_evidence": [],
                "score_anchor": info["basis"],
            })

    if not valid and "headache" in analysis.lower():
        valid = [
            {"name": "Tension Headache", "probability": "35%", "rationale": "Most common headache. Bilateral, pressure-like.", "supporting_fact_ids": [], "missing_evidence": ["Duration", "Severity", "Location"]},
            {"name": "Migraine", "probability": "25%", "rationale": "Unilateral, throbbing; nausea, photophobia.", "supporting_fact_ids": [], "missing_evidence": ["Aura", "Unilateral location", "Photophobia"]},
            {"name": "Cluster Headache", "probability": "10%", "rationale": "Severe periorbital pain; autonomic features.", "supporting_fact_ids": [], "missing_evidence": ["Lacrimation", "Restlessness"]},
            {"name": "Secondary Headache", "probability": "10%", "rationale": "Red flags should be excluded.", "supporting_fact_ids": [], "missing_evidence": ["Fever", "Neck stiffness", "Neurological signs"]},
            {"name": "Insufficient Information", "probability": "20%", "rationale": "More clinical data required.", "supporting_fact_ids": [], "missing_evidence": ["Duration", "Severity", "Associated symptoms"]},
        ]

    def sort_key(d):
        m = re.search(r'(\d+)', str(d.get("probability", "0")))
        return int(m.group(1)) if m else 0

    valid.sort(key=sort_key, reverse=True)
    return valid, scores


# ============================================================
# AGENT 3 — SOAP Writer (strict non-invention)
# ============================================================
def agent_soap(analysis, vitals, medical_history, confidence_level, diagnoses, fact_trace):
    vparts = []
    if vitals.get("bp"):   vparts.append(f"BP: {vitals['bp']}")
    if vitals.get("hr"):   vparts.append(f"HR: {vitals['hr']}")
    if vitals.get("temp"): vparts.append(f"Temp: {vitals['temp']}")
    if vitals.get("o2"):   vparts.append(f"O₂ Sat: {vitals['o2']}")
    vitals_str = " | ".join(vparts) if vparts else "Not documented"

    for k, label in [("bp","BP"), ("hr","HR"), ("temp","Temperature"), ("o2","O₂ Sat")]:
        if vitals.get(k):
            fact_trace.add_fact(f"{label}: {vitals[k]}", "Vital signs entry", "HIGH", "vitals")

    dx_list = "\n".join(
        f"  {i+1}. {d.get('name','?')} ({d.get('probability','?')})"
        for i, d in enumerate(diagnoses[:4])
    ) if diagnoses else "  Unable to generate without sufficient data"

    prompt = f"""
You are a clinical documentation specialist writing a SOAP note.

ABSOLUTE RULES:
- COPY ONLY facts from ANALYSIS and HISTORY below. Do NOT add exam findings.
- Physical exam section: write "Not performed / not documented"
- Vital signs: Use ONLY: {vitals_str}
- Assessment: List the differentials provided
- Plan: Write [PHYSICIAN TO COMPLETE]
- NEVER invent findings, vitals, or medications.

CLINICAL ANALYSIS:
{analysis}

MEDICAL HISTORY:
{safe_text(medical_history, "Not provided")}

VITAL SIGNS:
{vitals_str}

DIFFERENTIAL DIAGNOSES:
{dx_list}

CONFIDENCE LEVEL: {confidence_level}

Write the SOAP note now:

S (Subjective):
[Patient's complaint and symptoms as extracted]

O (Objective):
Vital signs: {vitals_str}
Physical exam: Not performed / not documented
Labs / imaging: Not available

A (Assessment):
{dx_list}
[PHYSICIAN TO COMPLETE FINAL ASSESSMENT]

P (Plan):
Recommended investigations: [suggest tests only]
[PHYSICIAN TO COMPLETE TREATMENT PLAN]

⚠ AI-GENERATED DRAFT — Physician review required.
"""
    return call_groq(prompt, system_msg="Clinical documentation AI. Never invent findings, vitals, or medications.")


# ============================================================
# AGENT 4 — Treatment / Management (safe, no specific meds)
# ============================================================
def agent_treatment(soap, is_emergency, confidence_level, diagnoses):
    top_dx = diagnoses[0].get("name", "Unknown") if diagnoses else "Unknown"

    emergency_block = ""
    if is_emergency:
        emergency_block = """
🚨 EMERGENCY PROTOCOL:
- Activate emergency response system immediately
- Do NOT administer any medication before physician assessment
- For stroke: STAT CT head required before ANY treatment
- For chest pain: 12-lead ECG + troponin BEFORE any medications
- Continuous cardiac/O₂ monitoring
- Establish IV access
"""

    prompt = f"""
You are a conservative clinical management advisor.

TOP WORKING DIAGNOSIS: {top_dx}
EMERGENCY: {is_emergency}
CONFIDENCE: {confidence_level}

{emergency_block}

Provide management considerations in this structure:

IMMEDIATE ACTIONS:
- [Triage steps — no specific medications]

RECOMMENDED INVESTIGATIONS:
- [Lab tests, imaging, ECG]

SPECIALIST REFERRAL:
- [Which specialty]

MONITORING PARAMETERS:
- [What to watch for]

SAFETY NOTES:
- [Red flags]

ABSOLUTE RESTRICTIONS:
- No specific drug names or doses
- For stroke: "Brain imaging required before any pharmacological treatment"
- For chest pain: "ECG and serial troponin required"

⚠ Educational framework only — all treatment decisions require physician assessment.
"""
    return call_groq(prompt, system_msg="Conservative clinical management advisor. Never recommend specific medications.")


# ============================================================
# AGENT 5 — Patient Instructions
# ============================================================
def agent_instructions(soap, is_emergency):
    if is_emergency:
        return """⚠️ EMERGENCY — Seek immediate medical care.

DO NOT:
- Drive yourself to hospital
- Take any medications before physician evaluation
- Wait to see if symptoms improve

DO:
- Call emergency services immediately
- Note the exact time symptoms started
- Have someone stay with you
- Bring a list of your current medications

These instructions are for informational use only.
All medical decisions must be made by your treating physician."""

    prompt = f"""
Based on this SOAP note: {soap[:600]}

Write simple patient instructions:

WHAT TO DO:
- [Simple, safe actions]

WHAT TO WATCH FOR (seek care immediately if):
- [Danger signs]

WHEN TO CALL YOUR DOCTOR:
- [Specific triggers]

⚠ These are general educational guidelines only.
Always follow your physician's specific instructions.
"""
    return call_groq(prompt, system_msg="Patient educator. Plain language. Safe general guidance only.")


# ============================================================
# AGENT 6 — Clinical Reviewer (NEW!)
# ============================================================
def agent_reviewer(diagnoses, soap, treatment, fact_trace, is_emergency):
    """
    Reviews outputs from Agents 2-4 for:
    - Contradictions
    - Missing evidence
    - Safety concerns
    - Inconsistent recommendations
    """
    
    dx_summary = "\n".join([f"- {d['name']}: {d['probability']}" for d in diagnoses[:5]])
    soap_preview = soap[:500]
    treatment_preview = treatment[:300]
    
    clinical_facts = fact_trace.get_clinical_facts()
    facts_summary = "\n".join([f"- {f['fact']}" for f in clinical_facts[:10]])
    
    prompt = f"""
You are a Clinical Reviewer Agent. Your job is to review outputs from other agents and flag issues.

DIFFERENTIAL DIAGNOSES:
{dx_summary}

SOAP NOTE PREVIEW:
{soap_preview}

TREATMENT PLAN PREVIEW:
{treatment_preview}

CLINICAL FACTS AVAILABLE:
{facts_summary}

EMERGENCY FLAG: {is_emergency}

Review for:
1. CONTRADICTIONS: Do any diagnoses conflict with each other?
2. MISSING EVIDENCE: What key clinical data is missing?
3. SAFETY CONCERNS: Are there dangerous recommendations or missing red flags?
4. INCONSISTENT RECOMMENDATIONS: Does the treatment plan match the top diagnosis?

Return ONLY valid JSON:
{{
  "contradictions": ["list any contradictions", "or empty array"],
  "missing_evidence": ["specific tests or data needed"],
  "safety_concerns": ["any red flags or dangerous omissions"],
  "inconsistent_recommendations": ["any mismatches between diagnosis and plan"],
  "overall_assessment": "PASS / FLAG / REVISION NEEDED",
  "reviewer_notes": "brief summary for physicians"
}}
"""
    
    raw = call_groq(prompt, system_msg="Clinical reviewer. Identify contradictions, missing evidence, safety issues.")
    
    try:
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if json_match:
            review = json.loads(json_match.group())
        else:
            review = {
                "contradictions": [],
                "missing_evidence": [],
                "safety_concerns": [],
                "inconsistent_recommendations": [],
                "overall_assessment": "PASS",
                "reviewer_notes": "Unable to parse full review"
            }
    except:
        review = {
            "contradictions": [],
            "missing_evidence": ["Review parsing failed"],
            "safety_concerns": [],
            "inconsistent_recommendations": [],
            "overall_assessment": "FLAG",
            "reviewer_notes": "Review agent encountered error"
        }
    
    fact_trace.add_fact(f"Reviewer assessment: {review['overall_assessment']}", "Agent 6 — Reviewer", "HIGH", "safety")
    if review.get("safety_concerns"):
        for concern in review["safety_concerns"]:
            fact_trace.add_fact(f"Safety concern: {concern}", "Agent 6 — Reviewer", "HIGH", "safety")
    
    return review


# ============================================================
# OCR — extract text from uploaded file
# ============================================================
def extract_text_from_file(uploaded_file):
    if not uploaded_file:
        return ""
    try:
        file_bytes = uploaded_file.read()
        ftype = uploaded_file.type

        if ftype == "application/pdf":
            try:
                import fitz
                doc = fitz.open(stream=file_bytes, filetype="pdf")
                text = ""
                for i in range(min(doc.page_count, 5)):
                    page = doc[i]
                    pt = page.get_text()
                    if pt.strip():
                        text += f"\n--- Page {i+1} ---\n{pt}"
                    elif TESSERACT_OK:
                        from PIL import Image
                        import pytesseract
                        pix = page.get_pixmap()
                        img = Image.open(io.BytesIO(pix.tobytes("png")))
                        text += f"\n--- Page {i+1} (OCR) ---\n{pytesseract.image_to_string(img)}"
                doc.close()
                return text[:3000].strip()
            except Exception as e:
                return f"PDF error: {e}"

        elif ftype in ("image/jpeg", "image/jpg", "image/png"):
            from PIL import Image
            image = Image.open(io.BytesIO(file_bytes))
            if google_client:
                try:
                    resp = google_client.models.generate_content(
                        model="gemini-1.5-flash",
                        contents=["Extract all text from this medical document. Return only extracted text.", image]
                    )
                    return safe_text(resp.text, "")[:3000]
                except Exception:
                    pass
            if TESSERACT_OK:
                import pytesseract
                return pytesseract.image_to_string(image)[:3000]
            return "OCR unavailable — Tesseract not installed"
    except Exception as e:
        return f"OCR Error: {e}"
    return ""


# ============================================================
# PDF REPORT GENERATION
# ============================================================
def generate_pdf_report(patient_data, results, vitals, medical_history, confidence_level, fact_trace, diagnoses, reviewer_output):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            topMargin=0.75*inch, bottomMargin=0.75*inch,
                            leftMargin=0.9*inch, rightMargin=0.9*inch)
    styles = getSampleStyleSheet()
    story = []

    conf_color = colors.HexColor("#3fb950") if confidence_level == "HIGH" else \
                 colors.HexColor("#d29922") if confidence_level == "MEDIUM" else \
                 colors.HexColor("#f85149")

    title_style = ParagraphStyle("Title", parent=styles["Heading1"], fontSize=18,
                                  textColor=colors.HexColor("#1a202c"), spaceAfter=4)
    h2_style = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=12,
                                  textColor=colors.HexColor("#2d3748"), spaceBefore=16, spaceAfter=6)
    body_style = ParagraphStyle("Body", parent=styles["Normal"], fontSize=9,
                                  textColor=colors.HexColor("#4a5568"), leading=14)
    mono_style = ParagraphStyle("Mono", parent=styles["Code"], fontSize=8,
                                  textColor=colors.HexColor("#2d3748"), leading=13, fontName="Courier")
    disc_style = ParagraphStyle("Disc", parent=styles["Normal"], fontSize=8,
                                  textColor=colors.HexColor("#e53e3e"), fontName="Helvetica-Bold")

    def esc(t):
        return safe_html(t)[:2000].replace('\n', '<br/>')

    story.append(Paragraph("NFÖ Clinical AI — Medical Consultation Report", title_style))
    story.append(HRFlowable(width="100%", thickness=1, color=conf_color, spaceAfter=8))

    meta = [
        ["Patient", patient_data.get("name", "Unknown"), "Date", datetime.now().strftime("%Y-%m-%d %H:%M")],
        ["Age", patient_data.get("age", "N/A"), "Gender", patient_data.get("gender", "N/A")],
        ["Confidence", confidence_level, "Generated by", "NFÖ Doctor's Assistant Pro (6-Agent System)"],
    ]
    tbl = Table(meta, colWidths=[1.0*inch, 2.5*inch, 1.0*inch, 2.5*inch])
    tbl.setStyle(TableStyle([
        ("FONTSIZE", (0,0), (-1,-1), 8),
        ("TEXTCOLOR", (0,0), (0,-1), colors.HexColor("#718096")),
        ("TEXTCOLOR", (2,0), (2,-1), colors.HexColor("#718096")),
        ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTNAME", (2,0), (2,-1), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 12))

    if diagnoses:
        story.append(Paragraph("Differential Diagnoses", h2_style))
        for i, d in enumerate(diagnoses[:5]):
            prob = d.get("probability", "?")
            story.append(Paragraph(
                f"<b>{i+1}. {esc(d.get('name','?'))}</b>  —  {esc(prob)}",
                ParagraphStyle("DxHead", parent=styles["Normal"], fontSize=10,
                               textColor=colors.HexColor("#2b6cb0"), spaceBefore=6)
            ))
            story.append(Paragraph(f"Rationale: {esc(d.get('rationale',''))}", body_style))
            if d.get("missing_evidence"):
                me = ", ".join(d["missing_evidence"][:4])
                story.append(Paragraph(f"Missing evidence: {esc(me)}", body_style))
        story.append(Spacer(1, 8))

    # Reviewer section
    if reviewer_output:
        story.append(Paragraph("Clinical Reviewer Assessment", h2_style))
        story.append(Paragraph(f"<b>Overall Assessment:</b> {reviewer_output.get('overall_assessment', 'N/A')}", body_style))
        if reviewer_output.get("safety_concerns"):
            story.append(Paragraph("<b>Safety Concerns:</b>", body_style))
            for concern in reviewer_output["safety_concerns"]:
                story.append(Paragraph(f"• {esc(concern)}", body_style))
        if reviewer_output.get("missing_evidence"):
            story.append(Paragraph("<b>Missing Evidence:</b>", body_style))
            for evidence in reviewer_output["missing_evidence"]:
                story.append(Paragraph(f"• {esc(evidence)}", body_style))
        story.append(Spacer(1, 8))

    story.append(Paragraph("SOAP Note", h2_style))
    story.append(Paragraph(esc(results.get("soap", "Not generated")), mono_style))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Management Considerations", h2_style))
    story.append(Paragraph(esc(results.get("treatment", "")), body_style))
    story.append(Spacer(1, 8))

    if fact_trace:
        clinical = fact_trace.get_clinical_facts()
        if clinical:
            story.append(Paragraph("Clinical Fact Trace", h2_style))
            for f in clinical[:15]:
                story.append(Paragraph(
                    f"[{f.get('fact_id')}] <b>{esc(f.get('fact',''))}</b> — {esc(f.get('source',''))}",
                    body_style
                ))

    story.append(Spacer(1, 16))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e2e8f0")))
    story.append(Paragraph(
        "⚠ AI-GENERATED DRAFT — Physician review required before any clinical use. "
        "Built for Band of Agents Hackathon - Track 3: Regulated & High-Stakes Workflows",
        disc_style
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer


# ============================================================
# CASE SAVE / LOAD
# ============================================================
CASES_FILE = "nfo_medical_cases.json"

def save_case(patient_data, symptoms, medical_history, results, vitals,
              confidence_level, fact_trace, diagnoses, reviewer_output):
    case = {
        "id": len(st.session_state.cases) + 1,
        "timestamp": datetime.now().isoformat(),
        "patient": patient_data,
        "symptoms": safe_text(symptoms),
        "medical_history": safe_text(medical_history),
        "vitals": vitals or {},
        "confidence_level": safe_text(confidence_level),
        "analysis": safe_text(results.get("analysis", "")),
        "soap": safe_text(results.get("soap", "")),
        "treatment": safe_text(results.get("treatment", "")),
        "instructions": safe_text(results.get("instructions", "")),
        "facts": fact_trace.to_list() if fact_trace else [],
        "diagnoses": diagnoses or [],
        "reviewer": reviewer_output or {},
    }
    st.session_state.cases.insert(0, case)
    try:
        with open(CASES_FILE, "w") as f:
            json.dump(st.session_state.cases, f, indent=2, default=str)
    except Exception as e:
        st.warning(f"Case save warning: {e}")
    return case["id"]

def load_cases():
    if os.path.exists(CASES_FILE):
        try:
            with open(CASES_FILE, "r") as f:
                loaded = json.load(f)
                for c in loaded:
                    c.setdefault("facts", [])
                    c.setdefault("diagnoses", [])
                    c.setdefault("reviewer", {})
                st.session_state.cases = loaded
        except Exception:
            st.session_state.cases = []

load_cases()


# ============================================================
# HELPERS — rendering
# ============================================================
def prob_to_float(prob_str):
    m = re.search(r'(\d+)', str(prob_str))
    return int(m.group(1)) / 100 if m else 0.0

def prob_color(p):
    if p >= 0.6: return "#f85149"
    if p >= 0.35: return "#d29922"
    return "#58a6ff"

def render_diagnoses(diagnoses):
    if not diagnoses:
        st.info("No diagnoses generated — provide more clinical information.")
        return
    for d in diagnoses:
        name = safe_html(d.get("name", "Unknown"))
        prob = safe_html(d.get("probability", "?"))
        ratio = safe_html(d.get("rationale", ""))
        miss = d.get("missing_evidence", [])
        pf = prob_to_float(prob)
        bar_c = prob_color(pf)
        bar_w = int(pf * 100)

        st.markdown(f"""
<div class="diag-card">
  <span class="diag-name">{name}</span>
  <span class="diag-prob">{prob}</span>
  <div class="prob-bar-wrap">
    <div class="prob-bar-track">
      <div class="prob-bar-fill" style="width:{bar_w}%;background:{bar_c}"></div>
    </div>
  </div>
  <div class="diag-rationale">{ratio}</div>
  {"<div class='diag-missing'>⚠ Missing: " + ", ".join(safe_html(m) for m in miss[:4]) + "</div>" if miss else ""}
</div>
""", unsafe_allow_html=True)

def render_fact_trace(fact_trace):
    if not fact_trace:
        st.info("No fact trace available.")
        return
    facts = fact_trace.get_all()
    clinical = fact_trace.get_clinical_facts()

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Facts", len(facts))
    c2.metric("Clinical Facts", len(clinical))
    c3.metric("Process Events", len(facts) - len(clinical))

    st.markdown('<div class="section-header">Clinical Fact Trace</div>', unsafe_allow_html=True)
    for f in facts[:40]:
        cat = f.get("category", "general")
        cat_class = f"cat-{cat}" if cat in ("symptoms","history","vitals","labs","safety","process") else ""
        st.markdown(f"""
<div class="fact-item">
  <span class="fact-id">#{f.get('fact_id','?')}</span>
  <div>
    <div class="fact-text">{safe_html(f.get('fact',''))}</div>
    <div class="fact-src">{safe_html(f.get('source',''))}</div>
  </div>
  <span class="fact-cat {cat_class}">{safe_html(cat)}</span>
</div>
""", unsafe_allow_html=True)

    st.download_button(
        "📥 Download Fact Trace JSON",
        data=fact_trace.to_json(),
        file_name=f"fact_trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )


def render_agent_workflow(current_agent_idx):
    """Render the 6-agent workflow visualization"""
    agents = [
        {"id": 0, "name": "Agent 1", "role": "Symptom Extractor"},
        {"id": 1, "name": "Agent 2", "role": "Diagnostic Reasoner"},
        {"id": 2, "name": "Agent 3", "role": "SOAP Writer"},
        {"id": 3, "name": "Agent 4", "role": "Management Planner"},
        {"id": 4, "name": "Agent 5", "role": "Patient Instructions"},
        {"id": 5, "name": "Agent 6", "role": "Clinical Reviewer"},
    ]
    
    cols = st.columns(6)
    for i, agent in enumerate(agents):
        with cols[i]:
            if i < current_agent_idx:
                st.markdown(f'<div class="workflow-node done">✅ {agent["name"]}<br><span style="font-size:0.6rem;">{agent["role"]}</span></div>', unsafe_allow_html=True)
            elif i == current_agent_idx:
                st.markdown(f'<div class="workflow-node active">⚡ {agent["name"]}<br><span style="font-size:0.6rem;">{agent["role"]}</span></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="workflow-node">⏳ {agent["name"]}<br><span style="font-size:0.6rem;">{agent["role"]}</span></div>', unsafe_allow_html=True)
    
    # Arrows between columns
    st.markdown("---")


# ============================================================
# MAIN APP
# ============================================================
def main():
    # Header
    st.markdown("""
<div class="nfo-header">
  <div>
    <div class="nfo-logo">NFÖ Clinical AI <span>Doctor's Assistant Pro</span></div>
    <div style="font-size:0.72rem;color:#8b949e;margin-top:2px;">
      6-Agent Clinical Decision Support · Evidence-Traced · Band Collaboration
    </div>
  </div>
  <div style="display:flex;gap:8px;">
    <div class="band-badge">🎭 BAND OF AGENTS</div>
    <div class="nfo-badge">TRACK 3: REGULATED WORKFLOWS</div>
  </div>
</div>
""", unsafe_allow_html=True)

    # Safety banner
    st.markdown("""
<div class="safety-banner">
  <strong>⚠ CLINICAL DECISION SUPPORT — 6-AGENT COLLABORATIVE SYSTEM</strong><br>
  Agents: Symptom Extractor → Diagnostic Reasoner → SOAP Writer → Management Planner → Patient Instructions → Clinical Reviewer<br>
  All outputs require physician review. Band-powered agent collaboration.
</div>
""", unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown('<div class="section-header">Case History</div>', unsafe_allow_html=True)
        if st.session_state.cases:
            for case in st.session_state.cases[:15]:
                date = case["timestamp"][:10]
                name = case["patient"].get("name", "Unknown")
                conf = case.get("confidence_level", "?")
                label = f"#{case['id']} · {name} · {date}"
                conf_emoji = "🟢" if conf=="HIGH" else "🟡" if conf=="MEDIUM" else "🔴"
                if st.button(f"{conf_emoji} {label}", key=f"case_{case['id']}", use_container_width=True):
                    ft = FactTrace(data=case.get("facts", []))
                    st.session_state.current_results = {
                        "symptoms": case.get("symptoms", ""),
                        "analysis": case.get("analysis", ""),
                        "soap": case.get("soap", ""),
                        "treatment": case.get("treatment", ""),
                        "instructions": case.get("instructions", ""),
                        "patient_data": case.get("patient", {}),
                        "vitals": case.get("vitals", {}),
                        "medical_history": case.get("medical_history", ""),
                        "confidence_level": case.get("confidence_level", "?"),
                        "fact_trace": ft,
                        "diagnoses": case.get("diagnoses", []),
                        "reviewer_output": case.get("reviewer", {}),
                        "missing_fields": [],
                    }
                    st.rerun()
        else:
            st.caption("No cases saved yet.")

        st.markdown("---")
        if st.button("🗑 Clear All Cases", use_container_width=True):
            st.session_state.cases = []
            st.session_state.current_results = None
            if os.path.exists(CASES_FILE):
                try: os.remove(CASES_FILE)
                except: pass
            st.rerun()

        st.markdown("---")
        st.markdown('<div class="section-header">Band of Agents</div>', unsafe_allow_html=True)
        if BAND_AVAILABLE:
            st.success("✅ Band Connected - 6 Agents Collaborating")
            st.caption("Agent pipeline: 1→2→3→4→5→6")
        else:
            st.warning("⚠️ Band not connected")
            st.caption("Add BAND_API_KEY to .env file")

        st.markdown("---")
        st.markdown('<div class="section-header">System Status</div>', unsafe_allow_html=True)
        if TESSERACT_OK:
            st.markdown("🟢 **OCR** Active")
        else:
            st.markdown("🔴 **OCR** Disabled")
        if SCORES_AVAILABLE:
            st.markdown("🟢 **Clinical Scores** Active")
        if google_client:
            st.markdown("🟢 **Gemini OCR** Active")

    # Input Form
    st.markdown('<div class="section-header">Patient Information</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([2,1,1])
    with c1: patient_name = st.text_input("Patient Name", placeholder="e.g. John Smith")
    with c2: patient_age = st.text_input("Age", placeholder="e.g. 58")
    with c3: patient_gender = st.selectbox("Gender", ["Select", "Male", "Female", "Other"])

    st.markdown('<div class="section-header">Medical History & Risk Factors</div>', unsafe_allow_html=True)
    medical_history = st.text_area(
        "Past medical history · Medications · Allergies · Family history · Risk factors",
        height=90,
        placeholder="e.g. Hypertension, Type 2 Diabetes, Smoker 20 pack-years"
    )

    st.markdown('<div class="section-header">Vital Signs</div>', unsafe_allow_html=True)
    v1, v2, v3, v4 = st.columns(4)
    with v1: bp = st.text_input("BP (mmHg)", placeholder="165/100")
    with v2: hr = st.text_input("HR (bpm)", placeholder="104")
    with v3: temp = st.text_input("Temp", placeholder="38.2 °C")
    with v4: o2 = st.text_input("O₂ Sat (%)", placeholder="95")

    st.markdown('<div class="section-header">Current Symptoms (Chief Complaint)</div>', unsafe_allow_html=True)
    symptoms_input = st.text_area(
        "Describe symptoms in clinical detail",
        height=110,
        placeholder="e.g. 58M with crushing chest pain radiating to left arm, diaphoresis, nausea"
    )

    st.markdown('<div class="section-header">Upload Document (optional)</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload PDF, JPG, or PNG", type=["jpg", "jpeg", "png", "pdf"], label_visibility="collapsed")
    if uploaded_file:
        if TESSERACT_OK or google_client:
            with st.spinner("Extracting text from document…"):
                ocr_text = extract_text_from_file(uploaded_file)
                if ocr_text and not ocr_text.startswith("Error"):
                    st.session_state.ocr_text = ocr_text
                    st.success(f"✅ Extracted {len(ocr_text)} characters")
                    with st.expander("Preview extracted text"):
                        st.text(ocr_text[:800])
                    if st.button("➕ Append to symptoms"):
                        symptoms_input += f"\n\n[Document]: {ocr_text[:600]}"
                        st.rerun()
                else:
                    st.error(f"Extraction failed: {ocr_text}")
        else:
            st.info("ℹ️ OCR unavailable. Install Tesseract.")

    st.markdown("---")

    bc1, bc2, bc3 = st.columns([1,2,1])
    with bc2:
        analyze_btn = st.button("🚀 RUN 6-AGENT CLINICAL ANALYSIS", type="primary", use_container_width=True)

    # Analysis Pipeline
    if analyze_btn:
        symptoms_stripped = symptoms_input.strip()
        if not symptoms_stripped:
            st.error("❌ Please enter the patient's symptoms.")
        elif not is_medical_input(symptoms_stripped):
            st.warning("⚠️ Input does not appear to contain medical content.")
        else:
            vitals = {}
            if bp: vitals["bp"] = bp
            if hr: vitals["hr"] = hr
            if temp: vitals["temp"] = temp
            if o2: vitals["o2"] = o2

            ocr_text = st.session_state.get("ocr_text", "")
            fact_trace = FactTrace()
            fact_trace.add_fact("6-Agent Analysis Session Initiated", "System", "HIGH", "process")

            case_id = hashlib.md5(f"{datetime.now()}{patient_name}{symptoms_stripped}".encode()).hexdigest()[:8]
            band_room_name = f"nfo_clinical_6agent_{case_id}"

            send_to_band(band_room_name, "orchestrator", "session_start", {
                "case_id": case_id, "patient_age": patient_age, "symptoms_preview": symptoms_stripped[:200]
            }, case_id)

            confidence_level, confidence_score = calculate_confidence(symptoms_stripped, medical_history, vitals, ocr_text)
            missing_fields = get_missing_fields(symptoms_stripped, medical_history, vitals)

            conf_class = {"HIGH":"conf-high","MEDIUM":"conf-medium","LOW":"conf-low"}[confidence_level]
            st.markdown(f'<div style="display:flex;gap:1rem;margin:1rem 0;"><span class="conf-pill {conf_class}">● CONFIDENCE: {confidence_level} ({confidence_score}%)</span></div>', unsafe_allow_html=True)

            if missing_fields:
                pills = "".join(f'<span class="missing-badge">⚠ {safe_html(f)}</span>' for f in missing_fields)
                st.markdown(f'<div>{pills}</div>', unsafe_allow_html=True)

            is_emergency, emergency_phrase = check_emergency(symptoms_stripped, fact_trace)
            if is_emergency:
                st.markdown(f"""
<div class="emergency-alert">
  <h3>🚨 EMERGENCY DETECTED</h3>
  <p>Trigger: {safe_html(emergency_phrase)}<br>Immediate physician evaluation required.</p>
</div>
""", unsafe_allow_html=True)
                send_to_band(band_room_name, "orchestrator", "emergency_detected", {"trigger": emergency_phrase}, case_id)

            patient_data = {"name": patient_name or "Unknown", "age": patient_age or "N/A", "gender": patient_gender if patient_gender != "Select" else "N/A"}

            st.markdown('<div class="section-header">6-Agent Collaboration Pipeline</div>', unsafe_allow_html=True)
            
            # Workflow visualization
            workflow_placeholder = st.empty()
            
            prog = st.progress(0)
            results = {}

            # Agent 1
            with workflow_placeholder.container():
                render_agent_workflow(0)
            results["analysis"] = agent_extract_symptoms(symptoms_stripped, medical_history, ocr_text, fact_trace)
            prog.progress(15)
            send_to_band(band_room_name, "agent_1_symptom_extractor", "completed", {"output_preview": results["analysis"][:300]}, case_id)

            # Agent 2
            with workflow_placeholder.container():
                render_agent_workflow(1)
            diagnoses, scores = agent_diagnoses(
                results["analysis"], fact_trace, confidence_level, is_emergency,
                symptoms_text=symptoms_stripped, history_text=medical_history,
                vitals=vitals, age=patient_age if patient_age and patient_age != "N/A" else None,
                gender=patient_gender if patient_gender != "Select" else ""
            )
            prog.progress(35)
            
            scores_serializable = []
            for s in scores:
                if hasattr(s, '__dict__'):
                    scores_serializable.append({"name": s.name, "score": s.score, "risk_level": s.risk_level})
            send_to_band(band_room_name, "agent_2_diagnostic_reasoner", "completed", {
                "diagnoses": [{"name": d["name"], "probability": d["probability"]} for d in diagnoses[:3]],
                "scores": scores_serializable
            }, case_id)

            # Agent 3
            with workflow_placeholder.container():
                render_agent_workflow(2)
            results["soap"] = agent_soap(results["analysis"], vitals, medical_history, confidence_level, diagnoses, fact_trace)
            prog.progress(55)
            send_to_band(band_room_name, "agent_3_soap_writer", "completed", {"soap_preview": results["soap"][:300]}, case_id)

            # Agent 4
            with workflow_placeholder.container():
                render_agent_workflow(3)
            results["treatment"] = agent_treatment(results["soap"], is_emergency, confidence_level, diagnoses)
            prog.progress(70)
            send_to_band(band_room_name, "agent_4_management_planner", "completed", {"treatment_preview": results["treatment"][:200]}, case_id)

            # Agent 5
            with workflow_placeholder.container():
                render_agent_workflow(4)
            results["instructions"] = agent_instructions(results["soap"], is_emergency)
            prog.progress(85)
            send_to_band(band_room_name, "agent_5_patient_instructions", "completed", {"is_emergency": is_emergency}, case_id)

            # Agent 6 - Clinical Reviewer
            with workflow_placeholder.container():
                render_agent_workflow(5)
            reviewer_output = agent_reviewer(diagnoses, results["soap"], results["treatment"], fact_trace, is_emergency)
            prog.progress(100)
            send_to_band(band_room_name, "agent_6_clinical_reviewer", "completed", {
                "overall_assessment": reviewer_output.get("overall_assessment"),
                "safety_concerns": reviewer_output.get("safety_concerns", []),
                "missing_evidence": reviewer_output.get("missing_evidence", [])
            }, case_id)

            workflow_placeholder.empty()
            st.success("✅ All 6 agents completed - Clinical review available")

            send_to_band(band_room_name, "orchestrator", "session_complete", {
                "case_id": case_id, "total_agents": 6, "reviewer_assessment": reviewer_output.get("overall_assessment")
            }, case_id)

            results["scores"] = scores
            results["symptoms"] = symptoms_stripped
            
            # Display reviewer summary
            if reviewer_output.get("overall_assessment") != "PASS":
                st.warning(f"⚠️ Clinical Reviewer: {reviewer_output['overall_assessment']}")
                if reviewer_output.get("safety_concerns"):
                    for concern in reviewer_output["safety_concerns"]:
                        st.error(f"🚨 {concern}")
                if reviewer_output.get("missing_evidence"):
                    st.info(f"📋 Missing evidence: {', '.join(reviewer_output['missing_evidence'][:3])}")
            else:
                st.success("✅ Clinical Reviewer: PASS - No major concerns identified")
            
            case_id_saved = save_case(
                patient_data, symptoms_stripped, medical_history,
                results, vitals, confidence_level, fact_trace, diagnoses, reviewer_output
            )
            st.caption(f"Case #{case_id_saved} saved · {len(fact_trace.get_all())} facts traced · {len(diagnoses)} diagnoses · 6 agents")

            st.session_state.current_results = {
                **results,
                "patient_data": patient_data,
                "vitals": vitals,
                "medical_history": medical_history,
                "confidence_level": confidence_level,
                "missing_fields": missing_fields,
                "fact_trace": fact_trace,
                "diagnoses": diagnoses,
                "is_emergency": is_emergency,
                "scores": scores,
                "reviewer_output": reviewer_output,
                "band_room": band_room_name if BAND_AVAILABLE else None,
            }

    # Results Display
    if st.session_state.current_results:
        res = st.session_state.current_results
        patient = res.get("patient_data", {})
        conf = res.get("confidence_level", "?")
        ft = res.get("fact_trace")
        diags = res.get("diagnoses", [])
        vitals = res.get("vitals", {})
        reviewer = res.get("reviewer_output", {})
        band_room = res.get("band_room")

        st.markdown("---")

        band_info = f'<span style="background:#1a2d4a;padding:2px 10px;border-radius:15px;font-size:0.7rem;">🎭 Band Room: {band_room}</span>' if band_room else ''
        st.markdown(f"""
<div style="background:var(--bg-surface);border:1px solid var(--border);border-radius:10px;padding:0.75rem 1.25rem;
            display:flex;align-items:center;justify-content:space-between;margin-bottom:1rem;">
  <div>
    <span style="color:var(--text-primary);font-weight:600;">{safe_html(patient.get('name','Unknown'))}</span>
    <span style="color:var(--text-muted);font-size:0.8rem;margin-left:0.75rem;">
      {safe_html(patient.get('age','?'))} · {safe_html(patient.get('gender','?'))}
    </span>
    {band_info}
  </div>
  <span class="conf-pill {'conf-high' if conf=='HIGH' else 'conf-medium' if conf=='MEDIUM' else 'conf-low'}">● {safe_html(conf)} CONFIDENCE</span>
</div>
""", unsafe_allow_html=True)

        if reviewer.get("overall_assessment"):
            if reviewer["overall_assessment"] == "PASS":
                st.success(f"✅ Clinical Reviewer: {reviewer['overall_assessment']}")
            elif reviewer["overall_assessment"] == "FLAG":
                st.warning(f"⚠️ Clinical Reviewer: {reviewer['overall_assessment']}")
            else:
                st.info(f"ℹ️ Clinical Reviewer: {reviewer['overall_assessment']}")

        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "🔬 Diagnoses", "📝 SOAP Note", "💊 Management", "👤 Instructions",
            "🔍 Reviewer", "📄 Export", "📋 Fact Trace", "📊 Clinical Scores"
        ])

        with tab1:
            render_diagnoses(diags)

        with tab2:
            st.markdown(res.get("soap", "Not generated"))

        with tab3:
            st.markdown(res.get("treatment", "Not generated"))

        with tab4:
            st.markdown(res.get("instructions", "Not generated"))

        with tab5:
            st.markdown('<div class="section-header">Agent 6 — Clinical Reviewer Assessment</div>', unsafe_allow_html=True)
            if reviewer:
                st.markdown(f"**Overall Assessment:** {reviewer.get('overall_assessment', 'N/A')}")
                if reviewer.get("contradictions"):
                    st.markdown("**Contradictions Found:**")
                    for c in reviewer["contradictions"]:
                        st.error(f"• {c}")
                if reviewer.get("safety_concerns"):
                    st.markdown("**Safety Concerns:**")
                    for s in reviewer["safety_concerns"]:
                        st.error(f"🚨 {s}")
                if reviewer.get("missing_evidence"):
                    st.markdown("**Missing Evidence:**")
                    for m in reviewer["missing_evidence"]:
                        st.info(f"📋 {m}")
                if reviewer.get("inconsistent_recommendations"):
                    st.markdown("**Inconsistent Recommendations:**")
                    for i in reviewer["inconsistent_recommendations"]:
                        st.warning(f"• {i}")
                st.markdown(f"**Reviewer Notes:** {reviewer.get('reviewer_notes', 'N/A')}")
            else:
                st.info("No reviewer output available.")

        with tab6:
            if st.button("📄 Generate PDF Report"):
                pdf_buf = generate_pdf_report(
                    patient, res, vitals, res.get("medical_history",""),
                    conf, ft, diags, reviewer
                )
                st.download_button("⬇️ Download PDF", data=pdf_buf,
                    file_name=f"nfo_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf")

        with tab7:
            render_fact_trace(ft)

        with tab8:
            scores_data = res.get("scores", [])
            if scores_data:
                for score in scores_data:
                    with st.expander(f"📊 {score.name} — Score: {score.score}/{score.max_score} ({score.risk_level} Risk)"):
                        st.markdown(f"**Risk Level:** {score.risk_label}")
                        st.markdown(f"**Probability Range:** {score.probability_range}")
                        st.markdown(f"**Recommendation:** {score.recommendation}")
            else:
                st.info("No clinical scores applicable.")

    st.markdown("---")
    st.markdown(
        '<div style="text-align:center;color:var(--text-muted);font-size:0.7rem;">'
        "NFÖ Clinical AI · 6-Agent System · Band of Agents Hackathon · Track 3: Regulated & High-Stakes Workflows<br>"
        "⚠ AI-generated drafts require physician review · For medical professionals only"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()