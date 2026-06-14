# NFÖ Clinical AI
### 6-Agent Clinical Decision Support System
*Band of Agents Hackathon — Track 3: Regulated & High-Stakes Workflows*

---

> **AI assists physicians. AI does not replace them.**  
> Every clinical fact is traced to its source. Every AI output is reviewable by a physician.

---

## What It Does

NFÖ Clinical AI processes a patient presentation through a 6-agent pipeline — each agent with a narrow, auditable role.

---

## 6-Agent Pipeline (Band SDK)
Patient Input
│
▼
[A1] Symptom Extractor
│
▼
[A2] Diagnostic Reasoner ←──── ↩ feedback loop from A6
│
├──────────────────────┬
▼ ▼
[A3] SOAP Writer [A4] Management Planner
│ │
│ ├──────────────────────┐
│ ▼ ▼
│ [A5] Patient Instructions [A6] Clinical Reviewer
│ │ │
└──────────────────────┴──────────────────────┘
│
↩ feedback → A2

| Agent | Role |
|-------|------|
| 🔍 A1 | Symptom Extractor |
| 🧠 A2 | Diagnostic Reasoner |
| 📋 A3 | SOAP Writer |
| 💊 A4 | Management Planner |
| 🗣️ A5 | Patient Instructions |
| ✅ A6 | Clinical Reviewer |

---

## Clinical Scoring Engine

| Score | Condition |
|-------|-----------|
| HEART | ACS Risk |
| Wells | Pulmonary Embolism |
| qSOFA | Sepsis |
| BE-FAST | Stroke |
| CHA₂DS₂-VASc | AFib |
| Alvarado | Appendicitis |
| CURB-65 | Pneumonia |

---

## Features

- ✅ FactTrace audit trail
- ✅ Emergency detection
- ✅ Reviewer feedback loop (A6 → A2)
- ✅ PDF report generation
- ✅ Band collaboration timeline

---

## Run Locally

```bash
pip install -r requirements.txt
cp .env.example .env
# Add GROQ_API_KEY to .env
streamlit run doctor_assistant_pro.py
