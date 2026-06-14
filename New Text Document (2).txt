"""
NFÖ Clinical AI — Band Timeline Visualization Component
"""

import streamlit as st

_GREEN   = "#3fb950"
_BLUE    = "#58a6ff"
_ORANGE  = "#d29922"
_RED     = "#f85149"
_PURPLE  = "#bc8cff"
_MUTED   = "#6e7681"
_SURFACE = "#161b22"
_BORDER  = "#30363d"
_CARD    = "#111827"

_AGENTS = [
    {"id": "A1", "icon": "🔍", "name": "Symptom Extractor", "role": "Extracts stated clinical facts only", "output_key": "analysis", "color": _BLUE, "sends_to": ["A2"]},
    {"id": "A2", "icon": "🧠", "name": "Diagnostic Reasoner", "role": "Score-anchored differential diagnosis", "output_key": "diagnoses", "color": _GREEN, "sends_to": ["A3", "A4"]},
    {"id": "A3", "icon": "📋", "name": "SOAP Writer", "role": "Structured note — no hallucination", "output_key": "soap", "color": _PURPLE, "sends_to": ["A6"]},
    {"id": "A4", "icon": "💊", "name": "Management Planner", "role": "Safe recommendations (no drug names)", "output_key": "treatment", "color": _ORANGE, "sends_to": ["A5", "A6"]},
    {"id": "A5", "icon": "🗣️", "name": "Patient Instructions", "role": "Plain-language guidance for patient", "output_key": "instructions", "color": _BLUE, "sends_to": ["A6"]},
    {"id": "A6", "icon": "✅", "name": "Clinical Reviewer", "role": "Reviews all outputs → feedback loop to A2", "output_key": "reviewer", "color": _RED, "sends_to": ["A2"]},
]

def _agent_by_id(aid):
    return next((a for a in _AGENTS if a["id"] == aid), None)

def render_band_timeline(results: dict, case_id: str = "CASE-001"):
    ran = set()
    for ag in _AGENTS:
        if results.get(ag["output_key"]):
            ran.add(ag["id"])
    
    st.markdown(f"""
    <div style="background:{_SURFACE};border:1px solid {_BORDER};border-radius:12px;padding:1rem 1.5rem;margin-bottom:1.25rem;display:flex;align-items:center;justify-content:space-between;">
        <div><span style="color:{_BLUE};font-weight:700;font-size:1rem;">🔗 Band Collaboration Timeline</span><span style="color:{_MUTED};font-size:0.75rem;margin-left:0.75rem;">case {case_id}</span></div>
        <span style="background:rgba(63,185,80,0.12);border:1px solid rgba(63,185,80,0.3);color:{_GREEN};font-size:0.65rem;font-weight:700;padding:3px 10px;border-radius:20px;">{len(ran)}/6 AGENTS RAN</span>
    </div>
    """, unsafe_allow_html=True)
    
    col_timeline, col_graph = st.columns([3, 2], gap="large")
    
    with col_timeline:
        st.markdown(f'<div style="color:{_MUTED};font-size:0.68rem;font-weight:700;margin-bottom:0.75rem;border-bottom:1px solid {_BORDER};padding-bottom:0.4rem;">Event Log</div>', unsafe_allow_html=True)
        for i, ag in enumerate(_AGENTS):
            did_run = ag["id"] in ran
            dot_col = ag["color"] if did_run else _MUTED
            border = ag["color"] if did_run else _BORDER
            status_text = "✓ Complete" if did_run else "— Not run"
            status_col = ag["color"] if did_run else _MUTED
            st.markdown(f"""
            <div style="display:flex;align-items:flex-start;gap:0.75rem;margin-bottom:0.5rem;">
                <div style="width:22px;height:22px;border-radius:50%;background:{dot_col};display:flex;align-items:center;justify-content:center;font-size:0.7rem;border:2px solid {_CARD};">{ag['icon']}</div>
                <div style="flex:1;border:1px solid {border};border-radius:8px;padding:0.5rem 0.9rem;">
                    <div style="display:flex;justify-content:space-between;"><span style="color:#f0f6fc;font-weight:600;">{ag['id']} — {ag['name']}</span><span style="color:{status_col};">{status_text}</span></div>
                    <div style="color:{_MUTED};font-size:0.72rem;">{ag['role']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col_graph:
        st.markdown(f'<div style="color:{_MUTED};font-size:0.68rem;font-weight:700;margin-bottom:0.75rem;border-bottom:1px solid {_BORDER};padding-bottom:0.4rem;">Collaboration Graph</div>', unsafe_allow_html=True)
        st.markdown("""
        <svg width="260" height="320" xmlns="http://www.w3.org/2000/svg" style="background:transparent;">
        <defs><marker id="arr" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#6e7681"/></marker></defs>
        <circle cx="130" cy="30" r="16" fill="#58a6ff" fill-opacity="0.2" stroke="#58a6ff" stroke-width="2"/><text x="130" y="31" text-anchor="middle" font-size="13">🔍</text><text x="130" y="55" text-anchor="middle" fill="#58a6ff" font-size="8.5">A1</text>
        <line x1="130" y1="46" x2="130" y2="94" stroke="#3fb950" stroke-width="1.5" marker-end="url(#arr)"/>
        <circle cx="130" cy="110" r="16" fill="#3fb950" fill-opacity="0.2" stroke="#3fb950" stroke-width="2"/><text x="130" y="111" text-anchor="middle" font-size="13">🧠</text><text x="130" y="135" text-anchor="middle" fill="#3fb950" font-size="8.5">A2</text>
        <line x1="114" y1="126" x2="71" y2="174" stroke="#bc8cff" stroke-width="1.5" marker-end="url(#arr)"/><text x="92" y="155" fill="#bc8cff" font-size="9">dx</text>
        <line x1="146" y1="126" x2="184" y2="174" stroke="#d29922" stroke-width="1.5" marker-end="url(#arr)"/><text x="165" y="155" fill="#d29922" font-size="9">dx</text>
        <circle cx="55" cy="190" r="16" fill="#bc8cff" fill-opacity="0.2" stroke="#bc8cff" stroke-width="2"/><text x="55" y="191" text-anchor="middle" font-size="13">📋</text><text x="55" y="215" text-anchor="middle" fill="#bc8cff" font-size="8.5">A3</text>
        <circle cx="200" cy="190" r="16" fill="#d29922" fill-opacity="0.2" stroke="#d29922" stroke-width="2"/><text x="200" y="191" text-anchor="middle" font-size="13">💊</text><text x="200" y="215" text-anchor="middle" fill="#d29922" font-size="8.5">A4</text>
        <line x1="55" y1="206" x2="55" y2="259" stroke="#58a6ff" stroke-width="1.5" marker-end="url(#arr)"/><text x="65" y="237" fill="#58a6ff" font-size="9">note</text>
        <line x1="200" y1="206" x2="200" y2="259" stroke="#3fb950" stroke-width="1.5" marker-end="url(#arr)"/><text x="210" y="237" fill="#3fb950" font-size="9">plan</text>
        <line x1="184" y1="206" x2="71" y2="259" stroke="#f85149" stroke-width="1.5" marker-end="url(#arr)"/><text x="128" y="237" fill="#f85149" font-size="9">review</text>
        <circle cx="55" cy="275" r="16" fill="#58a6ff" fill-opacity="0.2" stroke="#58a6ff" stroke-width="2"/><text x="55" y="276" text-anchor="middle" font-size="13">🗣️</text><text x="55" y="300" text-anchor="middle" fill="#58a6ff" font-size="8.5">A5</text>
        <circle cx="200" cy="275" r="16" fill="#f85149" fill-opacity="0.2" stroke="#f85149" stroke-width="2"/><text x="200" y="276" text-anchor="middle" font-size="13">✅</text><text x="200" y="300" text-anchor="middle" fill="#f85149" font-size="8.5">A6</text>
        <path d="M200,259 Q240,192 146,126" fill="none" stroke="#f85149" stroke-width="1.5" stroke-dasharray="4,3" marker-end="url(#arr)"/><text x="220" y="180" fill="#f85149" font-size="9">↩</text>
        </svg>
        """, unsafe_allow_html=True)
    
    reviewer_output = results.get("reviewer", "")
    if reviewer_output:
        st.markdown(f"""
        <div style="margin-top:1.25rem;background:rgba(248,81,73,0.06);border:1px solid rgba(248,81,73,0.25);border-left:4px solid {_RED};border-radius:8px;padding:0.9rem 1.1rem;">
            <div style="color:{_RED};font-weight:700;font-size:0.78rem;margin-bottom:0.5rem;">✅ A6 — CLINICAL REVIEWER FEEDBACK LOOP</div>
            <div style="color:#f0f6fc;font-size:0.8rem;white-space:pre-wrap;">{str(reviewer_output)[:500]}</div>
            <div style="color:{_MUTED};font-size:0.7rem;margin-top:0.5rem;">↩ Feedback sent back to A2 — Diagnostic Reasoner</div>
        </div>
        """, unsafe_allow_html=True)
    
    fact_trace = results.get("fact_trace_data", [])
    if fact_trace:
        st.markdown(f"""
        <div style="margin-top:1rem;background:{_SURFACE};border:1px solid {_BORDER};border-radius:8px;padding:0.75rem 1rem;">
            <span style="color:{_MUTED};font-size:0.72rem;font-weight:700;">FactTrace Audit — {len(fact_trace)} facts logged</span>
        </div>
        """, unsafe_allow_html=True)