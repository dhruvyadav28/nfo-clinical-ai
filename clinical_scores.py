"""
NFÖ Clinical Scoring Engine
Evidence-based clinical scoring systems for decision support.

Scores implemented:
  - HEART Score       (ACS / Chest pain)
  - Wells PE Score    (Pulmonary Embolism)
  - Wells DVT Score   (Deep Vein Thrombosis)
  - qSOFA             (Sepsis)
  - NIHSS / FAST      (Stroke)
  - CHA₂DS₂-VASc     (Atrial Fibrillation stroke risk)
  - Alvarado Score    (Appendicitis)
  - CURB-65           (Pneumonia severity)

All scores operate on a structured clinical_data dict extracted from patient input.
The LLM is used ONLY for explanation text — never to decide probability.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import re


# ──────────────────────────────────────────────────────────────
# Data structure passed to all scoring functions
# ──────────────────────────────────────────────────────────────
@dataclass
class ClinicalData:
    # Demographics
    age: Optional[int]           = None
    gender: Optional[str]        = None   # "Male" / "Female"

    # Symptoms (booleans — set by text analysis)
    has_chest_pain: bool         = False
    chest_pain_type: str         = ""     # "typical" / "atypical" / "non_cardiac"
    has_diaphoresis: bool        = False
    has_dyspnea: bool            = False
    has_nausea: bool             = False
    has_syncope: bool            = False
    has_palpitations: bool       = False
    has_fever: bool              = False
    has_cough: bool              = False
    has_facial_droop: bool       = False
    has_arm_weakness: bool       = False
    has_leg_weakness: bool       = False
    has_speech_difficulty: bool  = False
    has_vision_changes: bool     = False
    has_sudden_onset: bool       = False
    has_headache: bool           = False
    has_severe_headache: bool    = False
    has_neck_stiffness: bool     = False
    has_confusion: bool          = False
    has_abdominal_pain: bool     = False
    has_rlq_pain: bool           = False   # right lower quadrant
    has_nausea_vomiting: bool    = False
    has_anorexia: bool           = False
    has_rebound_tenderness: bool = False
    has_leg_swelling: bool       = False
    has_leg_pain: bool           = False
    has_hemoptysis: bool         = False
    has_tachycardia: bool        = False
    has_hypotension: bool        = False
    has_altered_mental_status: bool = False
    has_productive_cough: bool   = False
    has_pleuritic_pain: bool     = False
    has_radiation_left_arm: bool = False
    has_radiation_jaw: bool      = False
    symptom_onset_minutes: Optional[int] = None  # for stroke time window

    # Vital signs
    hr: Optional[int]            = None
    sbp: Optional[int]           = None
    dbp: Optional[int]           = None
    rr: Optional[int]            = None
    temp_c: Optional[float]      = None
    o2_sat: Optional[int]        = None

    # Medical history
    history_cad: bool            = False
    history_htn: bool            = False
    history_diabetes: bool       = False
    history_hyperlipidemia: bool = False
    history_smoking: bool        = False
    history_stroke_tia: bool     = False
    history_chf: bool            = False
    history_afib: bool           = False
    history_dvt_pe: bool         = False
    history_cancer: bool         = False
    history_immobility: bool     = False
    history_recent_surgery: bool = False
    history_ocp_use: bool        = False   # oral contraceptive pill
    history_vascular_disease: bool = False

    # ECG / labs (if available from document OCR)
    ecg_lbbb: bool               = False
    ecg_st_elevation: bool       = False
    ecg_new_changes: bool        = False
    troponin_elevated: bool      = False
    troponin_value: Optional[float] = None
    wbc_elevated: bool           = False
    crp_elevated: bool           = False
    ddimer_elevated: bool        = False

    # Raw text (for fallback)
    symptoms_text: str           = ""
    history_text: str            = ""


# ──────────────────────────────────────────────────────────────
# Score result
# ──────────────────────────────────────────────────────────────
@dataclass
class ScoreResult:
    name: str
    score: int
    max_score: int
    risk_level: str          # "Low" / "Moderate" / "High"
    risk_label: str          # e.g. "High — refer to cardiology"
    probability_range: str   # e.g. "65–85% MACE risk"
    components: list[dict]   = field(default_factory=list)
    recommendation: str      = ""
    applicable: bool         = True   # False if score not relevant for this presentation


# ──────────────────────────────────────────────────────────────
# TEXT → ClinicalData extractor
# ──────────────────────────────────────────────────────────────
def extract_clinical_data(symptoms_text: str, history_text: str,
                          vitals: dict, age: Optional[int],
                          gender: str) -> ClinicalData:
    """
    Parse free text and vitals dict into a structured ClinicalData object.
    Uses regex only — deterministic, no LLM.
    """
    cd = ClinicalData()
    cd.symptoms_text = symptoms_text or ""
    cd.history_text  = history_text  or ""
    cd.age    = age
    cd.gender = gender

    st = (symptoms_text or "").lower()
    ht = (history_text  or "").lower()
    both = st + " " + ht

    # ── Vitals ──
    if vitals:
        if vitals.get("hr"):
            m = re.search(r'(\d+)', str(vitals["hr"]))
            if m: cd.hr = int(m.group(1))
        if vitals.get("bp"):
            m = re.search(r'(\d{2,3})/(\d{2,3})', str(vitals["bp"]))
            if m:
                cd.sbp = int(m.group(1))
                cd.dbp = int(m.group(2))
        if vitals.get("temp"):
            m = re.search(r'(\d{2,3}\.?\d*)', str(vitals["temp"]))
            if m:
                t = float(m.group(1))
                cd.temp_c = t if t < 50 else (t - 32) * 5/9  # convert F→C if needed
        if vitals.get("o2"):
            m = re.search(r'(\d+)', str(vitals["o2"]))
            if m: cd.o2_sat = int(m.group(1))

    # Derive tachycardia / hypotension from vitals
    if cd.hr and cd.hr > 100:  cd.has_tachycardia  = True
    if cd.sbp and cd.sbp < 90: cd.has_hypotension  = True
    if cd.temp_c and cd.temp_c >= 38.0: cd.has_fever = True
    if cd.rr and cd.rr >= 22:  pass  # used in qSOFA

    # ── Symptoms ──
    cd.has_chest_pain       = bool(re.search(r'chest (pain|discomfort|pressure|tightness|heaviness)', st))
    cd.has_diaphoresis      = bool(re.search(r'sweat|diaphor', st))
    cd.has_dyspnea          = bool(re.search(r'short.?ness of breath|dyspnea|\bsob\b|difficulty breath|can.?t breathe', st))
    cd.has_nausea           = bool(re.search(r'\bnausea\b|vomit', st))
    cd.has_syncope          = bool(re.search(r'syncope|faint|passed out|lost consciousness', st))
    cd.has_palpitations     = bool(re.search(r'palpitat|racing heart|irregular heart', st))
    cd.has_fever            = cd.has_fever or bool(re.search(r'\bfever\b|febrile|pyrexia', both))
    cd.has_cough            = bool(re.search(r'\bcough\b', st))
    cd.has_productive_cough = bool(re.search(r'productive cough|coughing up|sputum|phlegm', st))
    cd.has_facial_droop     = bool(re.search(r'facial droop|face droop|drooping face', st))
    cd.has_arm_weakness     = bool(re.search(r'arm weakness|weak arm|one.?sided weakness|hemiparesis', st))
    cd.has_leg_weakness     = bool(re.search(r'leg weakness|weak leg', st))
    cd.has_speech_difficulty= bool(re.search(r'slurred speech|speech difficulty|difficulty speaking|can.?t speak|aphasia|dysarthria', st))
    cd.has_vision_changes   = bool(re.search(r'vision (loss|change|blurr|double)|diplopia|amaurosis', st))
    cd.has_sudden_onset     = bool(re.search(r'sudden|abrupt|acute onset|instant', st))
    cd.has_headache         = bool(re.search(r'\bheadache\b|head pain|cephalgia', st))
    cd.has_severe_headache  = bool(re.search(r'severe headache|worst headache|thunderclap|10/10 head', st))
    cd.has_neck_stiffness   = bool(re.search(r'neck stiff|nuchal rigid|meningism', st))
    cd.has_confusion        = bool(re.search(r'\bconfusion\b|confus|disoriented|altered mental|agitat', st))
    cd.has_altered_mental_status = cd.has_confusion
    cd.has_abdominal_pain   = bool(re.search(r'abdom|belly|stomach|epigast|periumb', st))
    cd.has_rlq_pain         = bool(re.search(r'right lower|rlq|mcburney|right iliac fossa', st))
    cd.has_nausea_vomiting  = bool(re.search(r'nausea|vomit', st))
    cd.has_anorexia         = bool(re.search(r'anorexia|loss of appetite|not eating|no appetite', st))
    cd.has_rebound_tenderness = bool(re.search(r'rebound|guarding|periton', st))
    cd.has_leg_swelling     = bool(re.search(r'leg swelling|calf swelling|leg edema', st))
    cd.has_leg_pain         = bool(re.search(r'leg pain|calf pain|calf tenderness', st))
    cd.has_hemoptysis       = bool(re.search(r'hemoptysis|coughing blood|blood in sputum', st))
    cd.has_pleuritic_pain   = bool(re.search(r'pleuritic|worse on breath|pain on inspiration', st))
    cd.has_radiation_left_arm = bool(re.search(r'radiat.*(left arm|arm|shoulder)|left arm.*pain', st))
    cd.has_radiation_jaw    = bool(re.search(r'radiat.*(jaw|neck)|jaw.*pain', st))

    # ── Symptom onset time (stroke) ──
    m = re.search(r'(\d+)\s*(min|hour|hr)', st)
    if m:
        val = int(m.group(1))
        unit = m.group(2)
        cd.symptom_onset_minutes = val if "min" in unit else val * 60

    # ── Chest pain type classification ──
    if cd.has_chest_pain:
        typical = (cd.has_radiation_left_arm or cd.has_radiation_jaw) and cd.has_diaphoresis
        if typical:
            cd.chest_pain_type = "typical"
        elif cd.has_radiation_left_arm or cd.has_radiation_jaw or cd.has_diaphoresis:
            cd.chest_pain_type = "atypical"
        else:
            cd.chest_pain_type = "non_cardiac"

    # ── History ──
    cd.history_cad          = bool(re.search(r'\bcad\b|coronary artery disease|heart disease|angina|prior mi\b|previous (heart attack|mi)', ht))
    cd.history_htn          = bool(re.search(r'\bhtn\b|hypertension|high blood pressure', ht))
    cd.history_diabetes     = bool(re.search(r'\bdiabetes\b|diabetic|t2dm|type 2 diabetes', ht))
    cd.history_hyperlipidemia = bool(re.search(r'hyperlipidemia|high cholesterol|dyslipidemia|statin', ht))
    cd.history_smoking      = bool(re.search(r'smok|pack.?year|cigarette|tobacco', ht))
    cd.history_stroke_tia   = bool(re.search(r'\bstroke\b|\btia\b|transient ischemic', ht))
    cd.history_chf          = bool(re.search(r'\bchf\b|heart failure|cardiac failure', ht))
    cd.history_afib         = bool(re.search(r'\bafib\b|atrial fibrillation|a.?fib', ht))
    cd.history_dvt_pe       = bool(re.search(r'\bdvt\b|\bpe\b|deep vein|pulmonary embolism|clot', ht))
    cd.history_cancer       = bool(re.search(r'cancer|malignancy|tumor|oncology|chemotherapy', ht))
    cd.history_immobility   = bool(re.search(r'immob|bed rest|bedbound|wheelchair', ht))
    cd.history_recent_surgery = bool(re.search(r'recent surgery|post.?op|operation|surgical', ht))
    cd.history_ocp_use      = bool(re.search(r'oral contraceptive|birth control pill|\bocp\b', ht))
    cd.history_vascular_disease = bool(re.search(r'vascular disease|pad\b|peripheral artery|aortic', ht))

    return cd


# ──────────────────────────────────────────────────────────────
# HEART Score  (ACS risk for chest pain presentations)
# ──────────────────────────────────────────────────────────────
def score_heart(cd: ClinicalData) -> ScoreResult:
    """
    HEART Score: History, ECG, Age, Risk factors, Troponin
    Range: 0–10  |  Low ≤3, Moderate 4–6, High ≥7
    Reference: Six et al., 2008 / 2010
    """
    if not cd.has_chest_pain:
        return ScoreResult("HEART Score", 0, 10, "N/A", "Not applicable",
                           "N/A", applicable=False,
                           recommendation="HEART Score requires chest pain as chief complaint.")

    components = []
    total = 0

    # H — History
    if cd.chest_pain_type == "typical":
        h = 2; h_label = "Highly suspicious (typical ACS pattern)"
    elif cd.chest_pain_type == "atypical":
        h = 1; h_label = "Moderately suspicious (atypical pattern)"
    else:
        h = 0; h_label = "Slightly suspicious / non-cardiac"
    total += h
    components.append({"item": "History", "score": h, "max": 2, "detail": h_label})

    # E — ECG
    if cd.ecg_st_elevation or cd.ecg_lbbb:
        e = 2; e_label = "ST elevation / LBBB"
    elif cd.ecg_new_changes:
        e = 1; e_label = "Non-specific repolarisation changes"
    else:
        e = 0; e_label = "Normal / not available"
    total += e
    components.append({"item": "ECG", "score": e, "max": 2, "detail": e_label})

    # A — Age
    age = cd.age or 0
    if age >= 65:
        a = 2; a_label = f"Age ≥65 ({age}y)"
    elif age >= 45:
        a = 1; a_label = f"Age 45–64 ({age}y)"
    else:
        a = 0; a_label = f"Age <45 ({age}y)"
    total += a
    components.append({"item": "Age", "score": a, "max": 2, "detail": a_label})

    # R — Risk factors (≥3 risk factors OR known atherosclerosis)
    risk_factors = sum([
        cd.history_htn, cd.history_diabetes,
        cd.history_hyperlipidemia, cd.history_smoking,
        cd.history_cad, cd.history_vascular_disease,
    ])
    if cd.history_cad:
        r = 2; r_label = "Known CAD / atherosclerotic disease"
    elif risk_factors >= 3:
        r = 2; r_label = f"≥3 risk factors ({risk_factors} identified)"
    elif risk_factors >= 1:
        r = 1; r_label = f"1–2 risk factors ({risk_factors} identified)"
    else:
        r = 0; r_label = "No known risk factors"
    total += r
    components.append({"item": "Risk Factors", "score": r, "max": 2, "detail": r_label})

    # T — Troponin
    if cd.troponin_elevated:
        t = 2; t_label = "Troponin ≥3× normal"
    elif cd.troponin_value is not None and cd.troponin_value > 0:
        t = 1; t_label = "Troponin 1–3× normal"
    else:
        t = 0; t_label = "Troponin normal / not available"
    total += t
    components.append({"item": "Troponin", "score": t, "max": 2, "detail": t_label})

    # Risk stratification
    if total <= 3:
        risk = "Low"
        label = "Low risk — 1.7% MACE rate. Consider early discharge + outpatient follow-up."
        prob  = "<2% 30-day MACE"
        rec   = "Low risk. Serial ECG + troponin. Consider early discharge protocol if negative."
    elif total <= 6:
        risk = "Moderate"
        label = "Moderate risk — 12% MACE rate. Admit for monitoring."
        prob  = "12–16% 30-day MACE"
        rec   = "Moderate risk. Admit for serial troponins, cardiology review."
    else:
        risk = "High"
        label = "High risk — 50%+ MACE rate. Early invasive strategy."
        prob  = ">50% 30-day MACE"
        rec   = "High risk. Urgent cardiology review. Early invasive strategy indicated."

    return ScoreResult(
        name="HEART Score (ACS)",
        score=total, max_score=10,
        risk_level=risk, risk_label=label,
        probability_range=prob,
        components=components,
        recommendation=rec,
    )


# ──────────────────────────────────────────────────────────────
# Wells PE Score
# ──────────────────────────────────────────────────────────────
def score_wells_pe(cd: ClinicalData) -> ScoreResult:
    """
    Wells Score for Pulmonary Embolism
    Range: 0–12.5  |  Low <2, Moderate 2–6, High >6
    """
    # Only relevant if dyspnea, chest pain, or tachycardia present
    if not (cd.has_dyspnea or cd.has_chest_pain or cd.has_tachycardia or cd.has_hemoptysis):
        return ScoreResult("Wells PE Score", 0, 12, "N/A", "Not applicable",
                           "N/A", applicable=False,
                           recommendation="Wells PE Score not indicated for this presentation.")

    components = []
    total = 0.0

    def add(item, val, detail):
        nonlocal total
        total += val
        components.append({"item": item, "score": val, "max": val, "detail": detail})

    if cd.history_dvt_pe:
        add("Prior DVT/PE", 1.5, "Previous DVT or PE documented")
    if cd.hr and cd.hr > 100:
        add("HR >100", 1.5, f"Current HR {cd.hr} bpm")
    if cd.history_recent_surgery or cd.history_immobility:
        add("Recent surgery/immobility", 1.5, "Within last 4 weeks")
    if cd.has_leg_swelling or cd.has_leg_pain:
        add("Clinical DVT signs", 3.0, "Leg swelling/pain suggesting DVT")
    if cd.has_hemoptysis:
        add("Haemoptysis", 1.0, "Blood in sputum")
    if cd.history_cancer:
        add("Active cancer", 1.0, "Malignancy in last 6 months")
    # PE as likely or more likely than alternative — if dyspnea + pleuritic + no fever
    if cd.has_dyspnea and cd.has_pleuritic_pain and not cd.has_fever:
        add("PE most likely diagnosis", 3.0, "Clinical presentation consistent with PE")
    elif cd.history_ocp_use and cd.has_dyspnea and cd.has_tachycardia:
        add("PE most likely diagnosis", 3.0, "OCP use + dyspnea + tachycardia — PE likely")

    total_int = int(total)

    if total < 2:
        risk = "Low"
        label = "Low probability — 2% PE prevalence"
        prob  = "~2% PE"
        rec   = "Low probability. D-dimer recommended. If negative, PE excluded."
    elif total <= 6:
        risk = "Moderate"
        label = "Moderate probability — 17% PE prevalence"
        prob  = "~17% PE"
        rec   = "Moderate probability. CT pulmonary angiography (CTPA) recommended."
    else:
        risk = "High"
        label = "High probability — 40–67% PE prevalence"
        prob  = "40–67% PE"
        rec   = "High probability. CTPA urgently. Consider empirical anticoagulation pending imaging."

    return ScoreResult(
        name="Wells PE Score",
        score=total_int, max_score=12,
        risk_level=risk, risk_label=label,
        probability_range=prob,
        components=components,
        recommendation=rec,
    )


# ──────────────────────────────────────────────────────────────
# qSOFA (Sepsis)
# ──────────────────────────────────────────────────────────────
def score_qsofa(cd: ClinicalData) -> ScoreResult:
    """
    quick SOFA — bedside sepsis screening
    Range: 0–3  |  ≥2 = high risk for poor outcome
    Reference: Singer et al., JAMA 2016
    """
    # Only relevant if infection/fever suspected
    if not (cd.has_fever or cd.has_cough or cd.has_confusion or cd.has_tachycardia):
        return ScoreResult("qSOFA", 0, 3, "N/A", "Not applicable",
                           "N/A", applicable=False,
                           recommendation="qSOFA not indicated without suspected infection.")

    components = []
    total = 0

    # Altered mental status
    ams = 1 if cd.has_altered_mental_status or cd.has_confusion else 0
    total += ams
    components.append({"item": "Altered mental status (GCS <15)", "score": ams, "max": 1,
                        "detail": "Confusion / altered mentation present" if ams else "No AMS documented"})

    # RR ≥22
    rr_score = 0
    if cd.rr and cd.rr >= 22:
        rr_score = 1
        rr_detail = f"RR {cd.rr} ≥22 /min"
    else:
        rr_detail = "RR not documented or <22"
    total += rr_score
    components.append({"item": "Respiratory rate ≥22", "score": rr_score, "max": 1, "detail": rr_detail})

    # SBP ≤100
    sbp_score = 0
    if cd.sbp and cd.sbp <= 100:
        sbp_score = 1
        sbp_detail = f"SBP {cd.sbp} mmHg ≤100"
    elif cd.has_hypotension:
        sbp_score = 1
        sbp_detail = "Hypotension documented"
    else:
        sbp_detail = "SBP not documented or >100"
    total += sbp_score
    components.append({"item": "SBP ≤100 mmHg", "score": sbp_score, "max": 1, "detail": sbp_detail})

    if total >= 2:
        risk = "High"
        label = "High risk — mortality risk >10%. Sepsis likely."
        prob  = ">10% in-hospital mortality"
        rec   = "High risk for sepsis. Blood cultures × 2, CBC, CMP, lactate, urine culture. IV access. Consider ICU."
    elif total == 1:
        risk = "Moderate"
        label = "Moderate concern — monitor closely."
        prob  = "Elevated risk if infection present"
        rec   = "1 qSOFA criterion. Monitor, investigate infection source."
    else:
        risk = "Low"
        label = "Low concern — sepsis less likely."
        prob  = "<5% mortality in isolation"
        rec   = "Low qSOFA. Sepsis less likely but reassess if deteriorating."

    return ScoreResult(
        name="qSOFA (Sepsis)",
        score=total, max_score=3,
        risk_level=risk, risk_label=label,
        probability_range=prob,
        components=components,
        recommendation=rec,
    )


# ──────────────────────────────────────────────────────────────
# FAST / Stroke Score  (NIHSS proxy for rapid triage)
# ──────────────────────────────────────────────────────────────
def score_fast_stroke(cd: ClinicalData) -> ScoreResult:
    """
    FAST (Face Arm Speech Time) stroke screening.
    Extended to BEFAST with Balance + Eyes.
    Combined with time-window assessment.
    """
    # Only relevant if neurological deficit or sudden onset symptoms
    neuro_sx = (cd.has_facial_droop or cd.has_arm_weakness or cd.has_speech_difficulty
                or cd.has_vision_changes or cd.has_leg_weakness or cd.has_severe_headache)
    if not (neuro_sx or cd.has_sudden_onset):
        return ScoreResult("BE-FAST Stroke Screen", 0, 6, "N/A", "Not applicable",
                           "N/A", applicable=False,
                           recommendation="BE-FAST not indicated without focal neurological symptoms.")

    components = []
    total = 0

    def add(item, val, detail):
        nonlocal total
        total += val
        components.append({"item": item, "score": val, "max": 1, "detail": detail})

    if cd.has_sudden_onset and (cd.has_leg_weakness or cd.has_confusion):
        add("Balance — sudden instability", 1, "Sudden balance/coordination loss")
    if cd.has_vision_changes:
        add("Eyes — vision change", 1, "Sudden vision loss or diplopia")
    if cd.has_facial_droop:
        add("Face — facial droop", 1, "Facial asymmetry / droop")
    if cd.has_arm_weakness:
        add("Arm — arm weakness", 1, "Unilateral arm weakness")
    if cd.has_speech_difficulty:
        add("Speech — difficulty", 1, "Slurred / absent speech")

    # Time is the most critical factor
    time_str = "Time — onset unknown"
    time_score = 0
    if cd.symptom_onset_minutes is not None:
        if cd.symptom_onset_minutes <= 270:   # ≤4.5 hours — tPA window
            time_score = 1
            time_str = f"Time — onset {cd.symptom_onset_minutes} min ago (WITHIN tPA window ≤4.5 hrs)"
        else:
            time_str = f"Time — onset {cd.symptom_onset_minutes} min ago (BEYOND 4.5 hr tPA window)"
    else:
        time_str = "Time — onset not documented (CRITICAL: document immediately)"
    add("Time — document onset", time_score if cd.symptom_onset_minutes else 0, time_str)

    if total >= 3:
        risk = "High"
        label = "High probability stroke. Immediate activation required."
        prob  = ">80% probability of stroke/TIA"
        rec   = (
            "IMMEDIATE: Activate stroke team. STAT non-contrast CT head. "
            "NPO. IV access × 2. 12-lead ECG. "
            + ("tPA eligibility assessment — within window." if
               cd.symptom_onset_minutes and cd.symptom_onset_minutes <= 270
               else "Document exact onset time urgently.")
        )
    elif total >= 1:
        risk = "Moderate"
        label = "Focal neurological symptoms — stroke possible."
        prob  = "Stroke/TIA possible — urgent imaging needed"
        rec   = "Urgent neurology review. CT head. Document onset time."
    else:
        risk = "Low"
        label = "Insufficient neurological features for stroke screen."
        prob  = "Low stroke probability from FAST criteria"
        rec   = "Consider other causes of neurological symptoms."

    return ScoreResult(
        name="BE-FAST Stroke Screen",
        score=total, max_score=6,
        risk_level=risk, risk_label=label,
        probability_range=prob,
        components=components,
        recommendation=rec,
    )


# ──────────────────────────────────────────────────────────────
# CHA₂DS₂-VASc  (AFib stroke risk)
# ──────────────────────────────────────────────────────────────
def score_cha2ds2_vasc(cd: ClinicalData) -> ScoreResult:
    """
    CHA₂DS₂-VASc — stroke risk in non-valvular AFib
    """
    if not cd.history_afib:
        return ScoreResult("CHA₂DS₂-VASc", 0, 9, "N/A", "Not applicable",
                           "N/A", applicable=False,
                           recommendation="CHA₂DS₂-VASc only applies to atrial fibrillation.")

    components = []
    total = 0

    def add(item, pts, detail):
        nonlocal total
        total += pts
        components.append({"item": item, "score": pts, "max": pts, "detail": detail})

    if cd.history_chf:                            add("Congestive Heart Failure",    1, "History of CHF")
    if cd.history_htn:                            add("Hypertension",                1, "History of HTN")
    age = cd.age or 0
    if age >= 75:                                 add("Age ≥75",                     2, f"Age {age}y")
    elif age >= 65:                               add("Age 65–74",                   1, f"Age {age}y")
    if cd.history_diabetes:                       add("Diabetes Mellitus",           1, "Diabetes documented")
    if cd.history_stroke_tia:                     add("Prior Stroke/TIA",            2, "Prior stroke or TIA")
    if cd.history_vascular_disease or cd.history_cad: add("Vascular Disease",        1, "Prior MI / PAD / aortic plaque")
    if cd.gender and "female" in cd.gender.lower(): add("Female Sex",                1, "Female sex category")

    if total == 0:
        risk = "Low"
        label = "Score 0 — Low risk"
        prob  = "~0% annual stroke risk"
        rec   = "No anticoagulation needed (male). Reassess."
    elif total == 1:
        risk = "Low-Moderate"
        label = "Score 1 — Low-Moderate risk"
        prob  = "~1.3% annual stroke risk"
        rec   = "Consider anticoagulation — physician decision."
    elif total <= 3:
        risk = "Moderate"
        label = f"Score {total} — Moderate risk"
        prob  = f"~{total * 1.5:.1f}% annual stroke risk"
        rec   = "Anticoagulation generally recommended — physician to assess bleeding risk."
    else:
        risk = "High"
        label = f"Score {total} — High risk"
        prob  = f"~{min(total * 2, 15):.0f}% annual stroke risk"
        rec   = "Anticoagulation strongly recommended. Physician to weigh risks/benefits."

    return ScoreResult(
        name="CHA₂DS₂-VASc (AFib)",
        score=total, max_score=9,
        risk_level=risk, risk_label=label,
        probability_range=prob,
        components=components,
        recommendation=rec,
    )


# ──────────────────────────────────────────────────────────────
# Alvarado Score  (Appendicitis)
# ──────────────────────────────────────────────────────────────
def score_alvarado(cd: ClinicalData) -> ScoreResult:
    """
    Alvarado (MANTRELS) score for acute appendicitis
    Range: 0–10  |  Low ≤4, Moderate 5–6, High ≥7
    """
    abdominal = cd.has_abdominal_pain or cd.has_rlq_pain
    if not abdominal:
        return ScoreResult("Alvarado Score", 0, 10, "N/A", "Not applicable",
                           "N/A", applicable=False,
                           recommendation="Alvarado Score requires abdominal pain.")

    components = []
    total = 0

    def add(item, pts, detail):
        nonlocal total
        total += pts
        components.append({"item": item, "score": pts, "max": pts, "detail": detail})

    if cd.has_abdominal_pain:         add("Migration to RLQ", 1, "Pain migrated to right lower quadrant")
    if cd.has_anorexia:               add("Anorexia", 1, "Loss of appetite")
    if cd.has_nausea_vomiting:        add("Nausea/Vomiting", 1, "Present")
    if cd.has_rlq_pain:               add("RLQ tenderness", 2, "Tenderness in right lower quadrant")
    if cd.has_rebound_tenderness:     add("Rebound tenderness", 1, "Peritoneal irritation signs")
    if cd.has_fever or (cd.temp_c and cd.temp_c >= 37.3): add("Elevated temperature", 1, "Temp ≥37.3°C")
    if cd.wbc_elevated:               add("Leukocytosis", 2, "WBC elevated")

    if total <= 4:
        risk = "Low"
        label = "Low probability appendicitis"
        prob  = "<31% appendicitis"
        rec   = "Low probability. Observe, serial exams. Discharge with return precautions."
    elif total <= 6:
        risk = "Moderate"
        label = "Moderate probability — consider imaging"
        prob  = "~47% appendicitis"
        rec   = "Moderate risk. CT abdomen/pelvis or ultrasound. Surgical consult."
    else:
        risk = "High"
        label = "High probability appendicitis — surgical consult"
        prob  = ">82% appendicitis"
        rec   = "High probability. Urgent surgical consult. NPO. IV access. Consider CT."

    return ScoreResult(
        name="Alvarado Score (Appendicitis)",
        score=total, max_score=10,
        risk_level=risk, risk_label=label,
        probability_range=prob,
        components=components,
        recommendation=rec,
    )


# ──────────────────────────────────────────────────────────────
# CURB-65  (Pneumonia severity)
# ──────────────────────────────────────────────────────────────
def score_curb65(cd: ClinicalData) -> ScoreResult:
    """
    CURB-65 — Community-Acquired Pneumonia severity
    Range: 0–5
    """
    pneumonia_sx = cd.has_cough or cd.has_productive_cough or cd.has_fever or cd.has_dyspnea
    if not pneumonia_sx:
        return ScoreResult("CURB-65", 0, 5, "N/A", "Not applicable",
                           "N/A", applicable=False,
                           recommendation="CURB-65 requires respiratory symptoms suggesting pneumonia.")

    components = []
    total = 0

    def add(item, pts, detail):
        nonlocal total
        total += pts
        components.append({"item": item, "score": pts, "max": 1, "detail": detail})

    # C — Confusion
    if cd.has_confusion:          add("Confusion (new onset)", 1, "New mental status change")
    else:                         add("Confusion", 0, "No confusion documented")

    # U — Urea >7 mmol/L — not easily extracted from text, skip if absent
    add("Urea >7 mmol/L", 0, "Lab value not available — assume 0")

    # R — RR ≥30
    if cd.rr and cd.rr >= 30:    add("RR ≥30 /min", 1, f"RR {cd.rr}")
    else:                         add("RR <30 /min", 0, "RR not documented or <30")

    # B — BP: SBP <90 or DBP ≤60
    if (cd.sbp and cd.sbp < 90) or (cd.dbp and cd.dbp <= 60) or cd.has_hypotension:
        add("Low BP (SBP<90 or DBP≤60)", 1, f"BP {cd.sbp}/{cd.dbp}")
    else:
        add("Blood Pressure", 0, "BP not critically low")

    # 65 — Age ≥65
    if cd.age and cd.age >= 65:   add("Age ≥65", 1, f"Age {cd.age}")
    else:                         add("Age <65", 0, f"Age {cd.age}")

    if total <= 1:
        risk = "Low"
        label = "Low severity — outpatient treatment"
        prob  = "<3% mortality"
        rec   = "Low severity CAP. Oral antibiotics. Outpatient management appropriate."
    elif total <= 2:
        risk = "Moderate"
        label = "Moderate severity — consider admission"
        prob  = "3–15% mortality"
        rec   = "Moderate severity. Consider hospital admission. Bloods + CXR."
    else:
        risk = "High"
        label = "High severity — hospital admission required"
        prob  = ">15% mortality"
        rec   = "Severe CAP. Hospital admission. Consider ICU if score ≥4."

    return ScoreResult(
        name="CURB-65 (Pneumonia)",
        score=total, max_score=5,
        risk_level=risk, risk_label=label,
        probability_range=prob,
        components=components,
        recommendation=rec,
    )


# ──────────────────────────────────────────────────────────────
# MASTER RUNNER — runs all applicable scores
# ──────────────────────────────────────────────────────────────
def run_all_scores(cd: ClinicalData) -> list[ScoreResult]:
    """Run all scoring systems and return applicable ones sorted by relevance."""
    all_scores = [
        score_heart(cd),
        score_wells_pe(cd),
        score_fast_stroke(cd),
        score_qsofa(cd),
        score_cha2ds2_vasc(cd),
        score_alvarado(cd),
        score_curb65(cd),
    ]
    return [s for s in all_scores if s.applicable]


# ──────────────────────────────────────────────────────────────
# PROBABILITY OVERRIDE — anchor LLM probabilities to scores
# ──────────────────────────────────────────────────────────────
def compute_anchored_probabilities(scores: list[ScoreResult], cd: ClinicalData) -> dict[str, dict]:
    """
    Return evidence-anchored probability ranges for key conditions.
    These override LLM guesses. Format: { "ACS": {"prob": "65–85%", "basis": "HEART 7/10"} }
    """
    anchored = {}

    for s in scores:
        if s.name == "HEART Score (ACS)" and s.applicable:
            risk_map = {"Low": "2–5%", "Moderate": "12–20%", "High": ">50%"}
            anchored["Acute Coronary Syndrome (ACS)"] = {
                "prob": risk_map.get(s.risk_level, "?"),
                "basis": f"HEART Score {s.score}/10 — {s.risk_level} risk",
                "score_name": "HEART",
                "score_val": f"{s.score}/10",
            }
        elif s.name == "Wells PE Score" and s.applicable:
            risk_map = {"Low": "2–5%", "Moderate": "17–25%", "High": "40–67%"}
            anchored["Pulmonary Embolism"] = {
                "prob": risk_map.get(s.risk_level, "?"),
                "basis": f"Wells PE {s.score}/12 — {s.risk_level} risk",
                "score_name": "Wells PE",
                "score_val": f"{s.score}/12",
            }
        elif s.name == "BE-FAST Stroke Screen" and s.applicable:
            risk_map = {"Low": "<10%", "Moderate": "30–50%", "High": ">80%"}
            anchored["Ischaemic Stroke / TIA"] = {
                "prob": risk_map.get(s.risk_level, "?"),
                "basis": f"BE-FAST {s.score}/6 — {s.risk_level} probability",
                "score_name": "BE-FAST",
                "score_val": f"{s.score}/6",
            }
        elif s.name == "qSOFA (Sepsis)" and s.applicable:
            risk_map = {"Low": "<5%", "Moderate": "5–10%", "High": ">10%"}
            anchored["Sepsis"] = {
                "prob": risk_map.get(s.risk_level, "?"),
                "basis": f"qSOFA {s.score}/3 — {s.risk_level} risk",
                "score_name": "qSOFA",
                "score_val": f"{s.score}/3",
            }
        elif s.name == "Alvarado Score (Appendicitis)" and s.applicable:
            risk_map = {"Low": "<31%", "Moderate": "~47%", "High": ">82%"}
            anchored["Acute Appendicitis"] = {
                "prob": risk_map.get(s.risk_level, "?"),
                "basis": f"Alvarado {s.score}/10 — {s.risk_level} risk",
                "score_name": "Alvarado",
                "score_val": f"{s.score}/10",
            }

    return anchored
