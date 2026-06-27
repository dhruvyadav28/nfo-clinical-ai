# NFÖ Clinical AI
### 6-Agent Clinical Decision Support System
> Built for Band of Agents Hackathon — Track 3: Regulated & High-Stakes Workflows

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)
![Groq](https://img.shields.io/badge/LLM-Groq_API-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🏥 What It Does

NFÖ Clinical AI processes a patient presentation through a 
6-agent pipeline — each agent with a narrow, auditable role — 
to assist physicians with clinical decision support.

> **AI assists physicians. AI does not replace them.**
> Every clinical fact is traced to its source.
> Every AI output is reviewable by a physician.

---

## 🤖 6-Agent Pipeline

| Agent | Role |
|-------|------|
| 🔍 A1 | Symptom Extractor |
| 🧠 A2 | Diagnostic Reasoner |
| 📋 A3 | SOAP Writer |
| 💊 A4 | Management Planner |
| 🗣️ A5 | Patient Instructions Generator |
| ✅ A6 | Clinical Reviewer (feedback → A2) |

A6 closes a feedback loop back to A2, 
ensuring the system self-reviews before 
outputting any clinical recommendation.

---

## 🧮 Clinical Scoring Engine

| Score | Condition |
|-------|-----------|
| HEART | ACS / Cardiac Risk |
| Wells | Pulmonary Embolism |
| qSOFA | Sepsis |
| BE-FAST | Stroke |
| CHA₂DS₂-VASc | Atrial Fibrillation |
| Alvarado | Appendicitis |
| CURB-65 | Pneumonia |

---

## ✨ Key Features

- ✅ FactTrace audit trail — every output traceable to source
- ✅ Emergency detection with escalation logic
- ✅ Reviewer feedback loop (A6 → A2)
- ✅ PDF clinical report generation
- ✅ Band collaboration timeline visualization

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **LLM Inference:** Groq API
- **Agent Framework:** Band SDK
- **Report Generation:** PDF export
- **Language:** Python 3.10+

---

## 🚀 Run Locally

git clone https://github.com/dhruvyadav28/nfo-clinical-ai.git
cd nfo-clinical-ai
pip install -r requirements.txt
cp .env.example .env
# Add your GROQ_API_KEY to .env
streamlit run doctor_assistant_pro.py

---

## ⚠️ Disclaimer

This system is a research/hackathon prototype.
It is NOT a certified medical device and must NOT
be used for real clinical decisions without 
physician oversight and regulatory approval.

---

## 👨‍💻 Author

**Dhruv Yadav**
Full-Stack Developer & AI Systems Builder
📍 Miskolc, Hungary
📧 dkash2811@gmail.com
