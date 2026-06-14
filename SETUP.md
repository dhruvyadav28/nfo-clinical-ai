# NFÖ Doctor's Assistant Pro — Setup Guide

## Project folder structure

```
nfo_medical_ai/
├── doctor_assistant_pro.py   ← main app
├── requirements.txt          ← Python packages
├── .env                      ← API keys (create this yourself)
├── nfo_medical_cases.json    ← auto-created when you save cases
└── SETUP.md                  ← this file
```

---

## Step 1 — Create your project folder

```
mkdir C:\Users\DHRUV\Desktop\nfo_medical_ai
cd    C:\Users\DHRUV\Desktop\nfo_medical_ai
```

Copy `doctor_assistant_pro.py` and `requirements.txt` into that folder.

---

## Step 2 — Create .env file

Create a file called exactly `.env` (no .txt extension) inside the folder:

```
GROQ_API_KEY=gsk_your_actual_key_here
# Optional:
# GEMINI_API_KEY=your_google_gemini_key
```

Get your Groq key free at: https://console.groq.com

---

## Step 3 — Install Python packages

```bash
C:\Users\DHRUV\AppData\Local\Microsoft\WindowsApps\python.exe -m pip install -r requirements.txt
```

Or individually:

```bash
python -m pip install streamlit groq python-dotenv reportlab PyMuPDF Pillow pytesseract
```

---

## Step 4 — Install Tesseract OCR (for document upload)

**Optional** — the app works without it; only OCR upload is disabled.

1. Download: https://github.com/UB-Mannheim/tesseract/wiki
2. Install to default path: `C:\Program Files\Tesseract-OCR\`
3. During install, check "Add to PATH"
4. Verify: open a new terminal → `tesseract --version`

---

## Step 5 — Run the app

```bash
cd C:\Users\DHRUV\Desktop\nfo_medical_ai
C:\Users\DHRUV\AppData\Local\Microsoft\WindowsApps\python.exe -m streamlit run doctor_assistant_pro.py
```

App opens at: http://localhost:8501

---

## Step 6 — Test cases (run in order)

### Test 1 — Headache (minimal info)
**Symptoms:** `Headache`  
**Expected:** LOW confidence, conservative differentials, no invented vitals

### Test 2 — Classic Heart Attack
**Age:** 58, **Gender:** Male  
**History:** Hypertension, Type 2 diabetes, smoker 20 pack-years  
**Vitals:** BP 165/100, HR 104  
**Symptoms:** Crushing chest pain started 45 minutes ago. Radiates to left arm and jaw. Diaphoresis and nausea. 9/10 severity.  
**Expected:** 🚨 EMERGENCY detected, ACS/MI at top, no invented exam findings

### Test 3 — Stroke
**Age:** 71, **Gender:** Female  
**History:** Hypertension, atrial fibrillation  
**Symptoms:** Sudden facial droop. Slurred speech. Right arm weakness. Started 30 minutes ago.  
**Expected:** 🚨 EMERGENCY, Stroke ≥80%, brain imaging before treatment

### Test 4 — Pulmonary Embolism
**Age:** 34, **Gender:** Female  
**History:** Oral contraceptive use, recent 12-hour flight  
**Vitals:** HR 125  
**Symptoms:** Sudden sharp chest pain and shortness of breath starting 2 hours ago. No fever.  
**Expected:** PE near top, CT pulmonary angiography suggested

### Test 5 — Anti-hallucination check
**Symptoms:** `I have a headache` (nothing else)  
**Expected:** No invented vitals, no invented exam, no system-process diagnoses

---

## What's different from old versions

| Feature | Old versions | This version |
|---|---|---|
| File name | `doctor_assistant_upgraded.py` etc. | `doctor_assistant_pro.py` |
| Case save file | `cases.json` (shared) | `nfo_medical_cases.json` (isolated) |
| Theme | Default Streamlit | Dark clinical UI (custom CSS) |
| Fact trace | Mixed clinical+process | Clinical facts only for diagnosis |
| Emergency detection | Partial | Full phrase matching |
| Treatment safety | Recommended medications | No specific drug names |
| Confidence display | Text only | Visual pills + score |
| Vitals display | In SOAP only | Visual chip strip |
| OCR | Required Tesseract | Gracefully disabled if missing |

---

## Troubleshooting

**App won't start:**  
→ Check `.env` file exists with valid `GROQ_API_KEY`

**OCR warning on startup:**  
→ Normal if Tesseract not installed. App still works fully — only document upload OCR is disabled.

**Empty diagnoses:**  
→ Add more symptom detail — duration, severity, associated symptoms

**PDF export fails:**  
→ Run: `pip install reportlab`

**Rate limit error from Groq:**  
→ Free tier has limits. Wait 30 seconds and retry.
