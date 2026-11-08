import streamlit as st
from supabase import create_client
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload
import io
import json
import pandas as pd
import time
import hashlib
import urllib.parse
import difflib
import re
from datetime import datetime, timezone, timedelta
from collections import Counter
from google import genai
from google.genai import types

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="IEER Smart Lab & Design Grader",
    page_icon="🧪",
    layout="wide"
)

# ==========================================
# INITIALIZE CLIENTS FROM SECRETS
# ==========================================
@st.cache_resource
def init_supabase():
    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"]["key"]
    return create_client(url, key)

@st.cache_resource
def get_drive_service():
    creds = Credentials(
        None,
        refresh_token=st.secrets["google_drive"]["refresh_token"],
        client_id=st.secrets["google_drive"]["client_id"],
        client_secret=st.secrets["google_drive"]["client_secret"],
        token_uri="https://oauth2.googleapis.com/token"
    )
    return build('drive', 'v3', credentials=creds)

@st.cache_resource
def get_gemini_client():
    return genai.Client(api_key=st.secrets["gemini"]["api_key"])

supabase = init_supabase()
drive_service = get_drive_service()
gemini_client = get_gemini_client()

DRIVE_ROOT_KEY = "root_" + "folder_id"
COL_DRIVE_FOLDER = "drive_" + "folder_id"
COL_DRIVE_FILE = "drive_" + "file_id"
PARAM_FILE_KEY = "file" + "Id"

ROOT_DRIVE_CONTAINER = st.secrets["google_drive"][DRIVE_ROOT_KEY]
APP_PUBLIC_URL = "https://smartedtools-lab-report.streamlit.app"

# ==========================================
# DEFAULT 4-POINT LIKERT SCALE RUBRIC TEMPLATES
# ==========================================
DEFAULT_LAB_LIKERT_RUBRIC = [
    {
        "criterion": "1. Title",
        "max_marks": 4.0,
        "level_4": "Clear, concise, and accurately reflects the specific experiment performed.",
        "level_3": "Accurately reflects the experiment but could be more concise or specific.",
        "level_2": "Vague or too broad, lacking specific detail about the experiment.",
        "level_1": "Missing, completely inaccurate, or irrelevant to the lab."
    },
    {
        "criterion": "2. Chemical",
        "max_marks": 4.0,
        "level_4": "Comprehensive list of all chemicals used, including precise concentrations and quantities.",
        "level_3": "Lists most chemicals correctly; minor missing details regarding concentrations.",
        "level_2": "Incomplete list; major omissions regarding concentrations or specific reagents.",
        "level_1": "Missing entirely or completely inaccurate list of chemicals."
    },
    {
        "criterion": "3. Glassware",
        "max_marks": 4.0,
        "level_4": "Exhaustive list of all necessary glassware/equipment, including capacities/sizes.",
        "level_3": "Lists most glassware correctly; misses minor items or specific capacities.",
        "level_2": "Missing several key pieces of glassware or fails to specify sizes.",
        "level_1": "Missing entirely or lists completely irrelevant equipment."
    },
    {
        "criterion": "4. Objective",
        "max_marks": 4.0,
        "level_4": "Clearly and concisely states the specific aim(s) of the experiment in a logical manner.",
        "level_3": "States the aim, but the wording is slightly ambiguous or lacks precision.",
        "level_2": "Objective is overly broad, partially incorrect, or difficult to understand.",
        "level_1": "Missing, or does not align at all with the actual experiment performed."
    },
    {
        "criterion": "5. Theory",
        "max_marks": 4.0,
        "level_4": "Thoroughly explains the underlying scientific principles, relevant chemical equations, and background context.",
        "level_3": "Explains the main principles and equations but lacks depth in background context.",
        "level_2": "Superficial explanation of principles; missing key equations or contains minor factual errors.",
        "level_1": "Missing, fundamentally incorrect, or completely plagiarized without understanding."
    },
    {
        "criterion": "6. Procedure",
        "max_marks": 4.0,
        "level_4": "Step-by-step, logically ordered, and detailed enough for exact replication. Written in the correct tense/voice.",
        "level_3": "Logical order but misses minor steps or details that would aid in replication.",
        "level_2": "Confusing sequence of steps, major omissions, or written like a casual narrative.",
        "level_1": "Missing, completely disorganized, or impossible to follow."
    },
    {
        "criterion": "7. Calculations",
        "max_marks": 4.0,
        "level_4": "All formulas shown, math is completely accurate, proper units are used, and logical flow is easy to follow.",
        "level_3": "Formulas shown with minor math/rounding errors, or one missing unit.",
        "level_2": "Missing sample calculations, frequent math errors, or missing most units.",
        "level_1": "Missing entirely, completely inaccurate, or final answers provided without any shown work."
    },
    {
        "criterion": "8. Precautions",
        "max_marks": 4.0,
        "level_4": "Specific, highly relevant safety and operational precautions are clearly detailed for this exact lab.",
        "level_3": "Mentions relevant precautions, but some are slightly generic or lack specific detail.",
        "level_2": "Lists only overly generic precautions (e.g., 'wear gloves') missing lab-specific hazards.",
        "level_1": "Missing entirely or precautions are completely irrelevant to the chemicals/equipment used."
    },
    {
        "criterion": "9. Environmental Significance",
        "max_marks": 4.0,
        "level_4": "Demonstrates deep understanding of how the experiment/parameters relate to real-world environmental issues and regulations.",
        "level_3": "Explains the environmental relevance but lacks connection to specific real-world applications or standards.",
        "level_2": "Superficial mention of the environment; fails to connect the specific lab results to broader impacts.",
        "level_1": "Missing entirely or completely misunderstands the environmental context of the parameters."
    },
    {
        "criterion": "10. Conclusion",
        "max_marks": 4.0,
        "level_4": "Concisely summarizes key findings, directly answers the objective, and briefly notes any sources of error.",
        "level_3": "Summarizes findings and references the objective but does not discuss potential errors.",
        "level_2": "Vague summary that barely relates to the stated objective or merely repeats the theory.",
        "level_1": "Missing, abrupt, or concludes entirely unsupported by the data."
    }
]

DEFAULT_DESIGN_LIKERT_RUBRIC = [
    {
        "criterion": "1. Problem Statement & Objectives",
        "max_marks": 4.0,
        "level_4": "Clearly defines the engineering problem, scope, constraints, and specific design objectives.",
        "level_3": "Defines the problem and objectives clearly but misses minor boundary constraints.",
        "level_2": "Problem statement is vague or lacks quantifiable engineering objectives.",
        "level_1": "Missing or fails to identify the core engineering problem."
    },
    {
        "criterion": "2. Design Methodology & Assumptions",
        "max_marks": 4.0,
        "level_4": "All design criteria, governing standards, and engineering assumptions are explicitly justified.",
        "level_3": "Appropriate methodology and assumptions used, with minor gaps in justification.",
        "level_2": "Unrealistic assumptions or incomplete design methodology.",
        "level_1": "Missing methodology or fundamentally flawed engineering basis."
    },
    {
        "criterion": "3. Calculations & Process Design",
        "max_marks": 4.0,
        "level_4": "Complete mass/energy balances and sizing calculations; all formulas, units, and steps are accurate.",
        "level_3": "Correct design procedure with minor arithmetic or unit inconsistencies.",
        "level_2": "Incomplete sizing calculations or major formula errors.",
        "level_1": "Missing calculations or completely incorrect process design."
    },
    {
        "criterion": "4. Economic & Environmental Feasibility",
        "max_marks": 4.0,
        "level_4": "Thorough cost estimation and environmental impact/regulatory compliance analysis.",
        "level_3": "Covers basic economic and environmental factors but lacks detailed breakdown.",
        "level_2": "Superficial mention of cost or environmental impact without supporting data.",
        "level_1": "No economic or environmental feasibility analysis provided."
    },
    {
        "criterion": "5. Conclusions & Schematics",
        "max_marks": 4.0,
        "level_4": "Clear process flow diagrams/schematics and concise summary of final design specifications.",
        "level_3": "Includes schematics and conclusion, but diagrams lack minor labels or dimensions.",
        "level_2": "Poorly drawn schematics or vague conclusion disconnected from design results.",
        "level_1": "Missing schematics and conclusion."
    }
]

# ==========================================
# HELPER FUNCTIONS: AUTH, RUBRIC CSV, INTEGRITY, TIMERS, DRIVE & GEMINI
# ==========================================
def hash_password(password: str) -> str:
    return hashlib.sha256(password.strip().encode("utf-8")).hexdigest()

def ensure_default_admin():
    """Automatically seeds the default 'admin' account if not present."""
    try:
        res = supabase.table("app_users").select("user_id").eq("username", "admin").execute()
        if not res.data:
            supabase.table("app_users").insert({
                "username": "admin",
                "full_name": "System Administrator",
                "password_hash": hash_password("@IEERlabreport2026"),
                "role": "admin",
                "is_active": True
            }).execute()
    except Exception:
        pass

def get_remaining_seconds(poll_obj):
    """Calculates remaining countdown seconds from UTC timer_ends_at and auto-closes expired polls."""
    ends_str = poll_obj.get("timer_ends_at")
    if not ends_str:
        return None
    try:
        ends_dt = datetime.fromisoformat(str(ends_str).replace("Z", "+00:00"))
        if ends_dt.tzinfo is None:
            ends_dt = ends_dt.replace(tzinfo=timezone.utc)
        rem = int((ends_dt - datetime.now(timezone.utc)).total_seconds())
        if rem <= 0:
            if poll_obj.get("is_active"):
                try:
                    supabase.table("live_polls").update({"is_active": False}).eq("poll_id", poll_obj["poll_id"]).execute()
                    poll_obj["is_active"] = False
                except Exception:
                    pass
            return 0
        return rem
    except Exception:
        return None

def activate_slide_with_timer(course_id: str, poll_id: str, duration_sec: int):
    """Closes any other active slide in the course and opens the target slide with a fresh countdown timer."""
    ends_iso = (datetime.now(timezone.utc) + timedelta(seconds=int(duration_sec))).isoformat()
    supabase.table("live_polls").update({"is_active": False}).eq("course_id", course_id).execute()
    supabase.table("live_polls").update({
        "is_active": True,
        "show_results": False,
        "reveal_answer": False,
        "timer_seconds": int(duration_sec),
        "timer_ends_at": ends_iso
    }).eq("poll_id", poll_id).execute()

def normalize_text_for_similarity(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"[^a-z0-9\s]", " ", str(text).lower())
    return re.sub(r"\s+", " ", cleaned).strip()

def compute_pairwise_similarity(text_a: str, text_b: str) -> float:
    """Computes hybrid similarity (3-word shingle Jaccard + SequenceMatcher) between two student fingerprints."""
    norm_a = normalize_text_for_similarity(text_a)
    norm_b = normalize_text_for_similarity(text_b)
    if len(norm_a) < 35 or len(norm_b) < 35:
        return 0.0
    words_a = norm_a.split()
    words_b = norm_b.split()
    if len(words_a) < 6 or len(words_b) < 6:
        return 0.0
    shingles_a = {" ".join(words_a[i:i+3]) for i in range(len(words_a) - 2)}
    shingles_b = {" ".join(words_b[i:i+3]) for i in range(len(words_b) - 2)}
    union_len = len(shingles_a | shingles_b)
    jaccard = (len(shingles_a & shingles_b) / union_len) if union_len > 0 else 0.0
    seq_ratio = difflib.SequenceMatcher(None, norm_a[:4000], norm_b[:4000]).ratio()
    return round(max(jaccard, seq_ratio) * 100.0, 1)

def format_ai_risk_badge(ai_pct: float) -> str:
    if ai_pct >= 70:
        return f"🔴 High ({int(ai_pct)}%)"
    elif ai_pct >= 40:
        return f"🟡 Medium ({int(ai_pct)}%)"
    return f"🟢 Low ({int(ai_pct)}%)"

def format_similarity_badge(sim_pct: float, matched_reg: str) -> str:
    if sim_pct >= 75 and matched_reg:
        return f"🚨 {int(sim_pct)}% (Match: {matched_reg})"
    elif sim_pct >= 50 and matched_reg:
        return f"🟡 {int(sim_pct)}% (Match: {matched_reg})"
    return f"🟢 {int(sim_pct)}% (Unique)"

def save_evaluation_record(sub_id: str, criteria_scores, ai_score: float, final_score: float, overall_fb: str, integrity_dict: dict):
    """Safely saves evaluation and integrity data to Supabase, with automatic fallback if integrity_json column is absent."""
    payload = {
        "sub_id": sub_id,
        "criteria_scores_json": criteria_scores,
        "ai_total_score": ai_score,
        "final_total_score": final_score,
        "overall_feedback": overall_fb,
        "integrity_json": integrity_dict
    }
    try:
        supabase.table("evaluations").upsert(payload, on_conflict="sub_id").execute()
    except Exception:
        # Fallback: attach integrity_dict to first criterion metadata if SQL column wasn't added yet
        if criteria_scores and isinstance(criteria_scores, list):
            criteria_scores[0]["_integrity_meta"] = integrity_dict
        payload.pop("integrity_json", None)
        supabase.table("evaluations").upsert(payload, on_conflict="sub_id").execute()

def extract_integrity_from_ev(ev_row: dict) -> dict:
    """Retrieves integrity metadata from either integrity_json column or fallback criterion metadata."""
    if not ev_row:
        return {}
    integ = ev_row.get("integrity_json")
    if isinstance(integ, dict) and integ:
        return integ
    crit_list = ev_row.get("criteria_scores_json") or []
    if crit_list and isinstance(crit_list[0], dict) and "_integrity_meta" in crit_list[0]:
        return crit_list[0].get("_integrity_meta") or {}
    return {}

def run_cross_student_collusion_pass(submissions_list, evals_map_dict, apply_auto_penalty: bool):
    """Compares all graded manuals in an assessment for exact file hash or narrative copy-pasting."""
    records = []
    for sub in submissions_list:
        s_id = sub["sub_id"]
        ev = evals_map_dict.get(s_id)
        if not ev:
            continue
        integ = extract_integrity_from_ev(ev)
        records.append({
            "sub_id": s_id,
            "reg_no": sub["reg_no"],
            "student_name": sub["student_name"],
            "ev": ev,
            "integ": integ,
            "fingerprint": str(integ.get("student_unique_fingerprint", "")),
            "pdf_sha256": str(integ.get("pdf_sha256", ""))
        })
        
    for i, rec_a in enumerate(records):
        max_sim = 0.0
        best_match_reg = ""
        exact_file_match = False
        for j, rec_b in enumerate(records):
            if i == j:
                continue
            if rec_a["pdf_sha256"] and rec_a["pdf_sha256"] == rec_b["pdf_sha256"]:
                sim = 100.0
                exact_file_match = True
            else:
                sim = compute_pairwise_similarity(rec_a["fingerprint"], rec_b["fingerprint"])
            if sim > max_sim:
                max_sim = sim
                best_match_reg = rec_b["reg_no"]
                
        integ = rec_a["integ"]
        integ["similarity_score"] = round(max_sim, 1)
        integ["similarity_match_reg"] = best_match_reg
        integ["exact_file_duplicate"] = exact_file_match
        
        ev = rec_a["ev"]
        raw_ai_score = float(ev.get("ai_total_score", 0.0))
        base_fb = str(ev.get("overall_feedback", "")).split(" | 🚨 [AUTO-PENALTY")[0].split(" | ⚠️ [INTEGRITY FLAG")[0]
        
        ai_prob = float(integ.get("ai_Generation_probability", 0.0))
        ai_reason = str(integ.get("ai_forensic_summary", "Checked."))
        
        if max_sim >= 75.0 and best_match_reg:
            match_type = "Exact duplicate PDF file" if exact_file_match else f"{int(max_sim)}% text/calculation match"
            if apply_auto_penalty:
                penalized_score = round(raw_ai_score * 0.5, 2)
                updated_fb = f"{base_fb} | 🚨 [AUTO-PENALTY: 50% deduction applied for high copy-paste collusion ({match_type} with {best_match_reg})]"
                integ["penalty_applied"] = True
            else:
                penalized_score = float(ev.get("final_total_score", raw_ai_score))
                updated_fb = f"{base_fb} | ⚠️ [INTEGRITY FLAG: {match_type} with {best_match_reg}]"
                integ["penalty_applied"] = False
        else:
            penalized_score = raw_ai_score if apply_auto_penalty else float(ev.get("final_total_score", raw_ai_score))
            updated_fb = base_fb
            integ["penalty_applied"] = bool(apply_auto_penalty and ai_prob >= 70.0)
            
        integ["audit_summary"] = (
            f"AI Check: {ai_reason} | "
            f"Classmate Similarity: {int(max_sim)}%"
            + (f" (closest: {best_match_reg})" if best_match_reg else "")
        )
        
        save_evaluation_record(
            sub_id=rec_a["sub_id"],
            criteria_scores=ev.get("criteria_scores_json") or [],
            ai_score=raw_ai_score,
            final_score=penalized_score,
            overall_fb=updated_fb,
            integrity_dict=integ
        )

def parse_uploaded_rubric_file(uploaded_file):
    """Reads an uploaded CSV or Excel rubric file and normalizes column headers."""
    if uploaded_file.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
        
    col_mapping = {}
    for col in df.columns:
        c_low = str(col).strip().lower()
        if "crit" in c_low or "head" in c_low or "title" in c_low or "exp" in c_low:
            col_mapping[col] = "criterion"
        elif "mark" in c_low or "weight" in c_low or "score" in c_low:
            col_mapping[col] = "max_marks"
        elif "4" in c_low or "excel" in c_low:
            col_mapping[col] = "level_4"
        elif "3" in c_low or "prof" in c_low:
            col_mapping[col] = "level_3"
        elif "2" in c_low or "dev" in c_low:
            col_mapping[col] = "level_2"
        elif "1" in c_low or "need" in c_low or "improv" in c_low:
            col_mapping[col] = "level_1"
            
    df = df.rename(columns=col_mapping)
    
    if "criterion" not in df.columns:
        raise ValueError("Uploaded file must contain a 'criterion' (or 'Evaluation Criteria') column.")
        
    if "max_marks" not in df.columns:
        df["max_marks"] = 4.0
    if "level_4" not in df.columns:
        df["level_4"] = "Excellent performance; meets all requirements."
    if "level_3" not in df.columns:
        df["level_3"] = "Proficient performance; minor omissions or errors."
    if "level_2" not in df.columns:
        df["level_2"] = "Developing performance; major omissions or errors."
    if "level_1" not in df.columns:
        df["level_1"] = "Needs improvement or missing entirely."
        
    return df[["criterion", "max_marks", "level_4", "level_3", "level_2", "level_1"]]

def build_rubric_from_df(df_rubric: pd.DataFrame):
    """Validates an editable Likert rubric dataframe and computes weight percentages & total marks."""
    valid_rows = []
    for _, row in df_rubric.iterrows():
        crit = str(row.get("criterion", "")).strip()
        marks = row.get("max_marks", 4.0)
        if crit and crit.lower() != "nan" and pd.notna(marks) and float(marks) > 0:
            valid_rows.append({
                "criterion": crit,
                "max_marks": float(marks),
                "level_4": str(row.get("level_4", "Excellent performance; meets all requirements.")).strip(),
                "level_3": str(row.get("level_3", "Proficient performance; minor omissions or errors.")).strip(),
                "level_2": str(row.get("level_2", "Developing performance; major omissions or errors.")).strip(),
                "level_1": str(row.get("level_1", "Needs improvement or missing entirely.")).strip(),
            })
            
    if not valid_rows:
        return None, 0.0
        
    total_marks = sum(r["max_marks"] for r in valid_rows)
    rubric_json = [
        {
            "criterion": r["criterion"],
            "max_marks": round(r["max_marks"], 2),
            "weight_percentage": round((r["max_marks"] / total_marks) * 100.0, 2),
            "level_4": r["level_4"],
            "level_3": r["level_3"],
            "level_2": r["level_2"],
            "level_1": r["level_1"]
        }
        for r in valid_rows
    ]
    return rubric_json, round(total_marks, 2)

def format_rubric_for_display(rubric_json):
    """Formats rubric JSON into a clean Likert scale DataFrame for students and teachers."""
    rows = []
    for r in rubric_json:
        rows.append({
            "Evaluation Criteria": r.get("criterion", ""),
            "Max Marks": r.get("max_marks", 4.0),
            "4 – Excellent": r.get("level_4", "Complete, accurate, and thorough."),
            "3 – Proficient": r.get("level_3", "Mostly complete with minor omissions."),
            "2 – Developing": r.get("level_2", "Incomplete or contains notable errors."),
            "1 – Needs Improvement": r.get("level_1", "Missing, inaccurate, or irrelevant.")
        })
    return pd.DataFrame(rows)

def render_visual_wordcloud(words_list):
    """Renders an interactive HTML/CSS Word Cloud scaled by frequency."""
    if not words_list:
        st.info("Waiting for student word submissions...")
        return
    cleaned = [w.strip().upper() for w in words_list if w and w.strip()]
    counts = Counter(cleaned)
    max_c = max(counts.values()) if counts else 1
    
    palette = ["#1d4ed8", "#047857", "#b45309", "#7c3aed", "#be123c", "#0f766e", "#4338ca"]
    badges_html = []
    for idx, (word, count) in enumerate(counts.most_common(50)):
        size_px = int(18 + (count / max_c) * 28)
        color = palette[idx % len(palette)]
        badges_html.append(
            f"<span style='font-size:{size_px}px; font-weight:700; color:{color}; "
            f"background:#f8fafc; border:2px solid {color}33; padding:6px 16px; "
            f"border-radius:999px; margin:6px; display:inline-block; "
            f"box-shadow:0 2px 4px rgba(0,0,0,0.06);'>"
            f"{word} <sup style='font-size:13px; color:#475569;'>({count})</sup></span>"
        )
    cloud_box = (
        "<div style='display:flex; flex-wrap:wrap; justify-content:center; "
        "align-items:center; padding:28px; background:#ffffff; "
        "border:2px dashed #cbd5e1; border-radius:14px; min-height:180px;'>"
        + "".join(badges_html) +
        "</div>"
    )
    st.markdown(cloud_box, unsafe_allow_html=True)

def render_projector_option_cards(options_list, correct_answer=None, reveal_answer=False):
    """Displays large, classroom-readable Multiple Choice option cards on the projector screen."""
    if not options_list:
        return
    letters = ["A", "B", "C", "D", "E", "F"]
    cols = st.columns(2)
    for idx, opt in enumerate(options_list):
        letter = letters[idx] if idx < len(letters) else str(idx + 1)
        is_corr = reveal_answer and correct_answer and (str(opt).strip() == str(correct_answer).strip())
        border_col = "#16a34a" if is_corr else "#cbd5e1"
        bg_col = "#f0fdf4" if is_corr else "#f8fafc"
        badge_bg = "#16a34a" if is_corr else "#1e293b"
        check_mark = " ✅" if is_corr else ""
        card_html = (
            f"<div style='background:{bg_col}; border:2px solid {border_col}; border-radius:12px; "
            f"padding:14px 18px; margin-bottom:10px; display:flex; align-items:center; gap:14px;'>"
            f"<span style='background:{badge_bg}; color:white; font-weight:800; font-size:18px; "
            f"width:36px; height:36px; border-radius:8px; display:inline-flex; align-items:center; "
            f"justify-content:center; flex-shrink:0;'>{letter}</span>"
            f"<span style='font-size:18px; font-weight:600; color:#0f172a;'>{opt}{check_mark}</span>"
            f"</div>"
        )
        with cols[idx % 2]:
            st.markdown(card_html, unsafe_allow_html=True)

def get_or_create_folder(folder_name, parent_ref):
    if "folders/" in parent_ref:
        parent_ref = parent_ref.split("folders/")[-1].split("?")[0]
        
    query = f"mimeType='application/vnd.google-apps.folder' and name='{folder_name}' and '{parent_ref}' in parents and trashed=false"
    results = drive_service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
    folders = results.get('files', [])
    
    if folders:
        return folders[0]['id']
    else:
        file_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder',
            'parents': [parent_ref]
        }
        folder = drive_service.files().create(body=file_metadata, fields='id').execute()
        return folder.get('id')

def upload_or_update_file(file_bytes, file_name, mime_type, parent_ref, existing_ref=None):
    media = MediaIoBaseUpload(io.BytesIO(file_bytes), mimetype=mime_type, resumable=True)
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if existing_ref:
                update_kwargs = {PARAM_FILE_KEY: existing_ref, "media_body": media}
                drive_service.files().update(**update_kwargs).execute()
                return existing_ref
            else:
                file_metadata = {'name': file_name, 'parents': [parent_ref]}
                file = drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
                return file.get('id')
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(2)

def download_file_bytes_from_drive(target_ref):
    req_kwargs = {PARAM_FILE_KEY: target_ref}
    request = drive_service.files().get_media(**req_kwargs)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    fh.seek(0)
    return fh.read()

def evaluate_report_with_gemini(pdf_bytes, rubric_json, course_title, exp_title, exp_guidelines, total_marks, is_combined=False, apply_ai_penalty=False):
    num_criteria = len(rubric_json)
    task_desc = (
        "This is a COMPREHENSIVE or COMBINED lab manual/report (containing multiple experiments or a full Design Report) compiled into a single PDF. "
        "Carefully inspect ALL pages and experiments across the entire PDF document, evaluate the student's work against EVERY SINGLE criterion in the rubric, and provide a complete breakdown."
        if is_combined else 
        "Thoroughly inspect all pages of the attached student lab report PDF (reading observation tables, formulas, calculations, figures, and discussion) and evaluate against EVERY SINGLE criterion in the 4-point Likert scale rubric."
    )

    penalty_instruction = (
        "ACTIVE AUTO-PENALTY MODE: If you detect that a narrative section (e.g., Theory, Precautions, Environmental Significance, Conclusion) is generically AI-generated (ChatGPT/LLM boilerplate) and fails to reference the student's actual experimental numbers from their Observation/Calculation tables, you MUST cap that criterion at `likert_rating: 1` ('1 - Needs Improvement') and explicitly state '[AI/Generic Content Penalty]' in its feedback."
        if apply_ai_penalty else
        "FLAG-ONLY MODE: Grade the technical content fairly according to the rubric descriptors, and report any AI-generation or copy-paste indicators inside the `ai_generation_probability` and `ai_forensic_summary` fields so the instructor can review them."
    )

    prompt = f"""
    You are an expert university professor of environmental engineering grading student reports for the course: {course_title}.
    Assessment Title: {exp_title}
    Total Maximum Marks: {total_marks}
    
    Instructor Guidelines & Focus Notes:
    {exp_guidelines if exp_guidelines else "Evaluate based on technical accuracy, valid observation tables, correct calculations, and scientific discussion."}
    
    Here is the 4-Point Likert Scale Analytic Rubric containing {num_criteria} required criteria (4 = Excellent, 3 = Proficient, 2 = Developing, 1 = Needs Improvement):
    {json.dumps(rubric_json, indent=2)}
    
    CRITICAL SCORING & ACADEMIC INTEGRITY FORENSICS INSTRUCTIONS:
    1. {task_desc}
    2. Your `criteria_scores` array MUST contain EXACTLY {num_criteria} items—one for each criterion in the rubric above, using the exact "criterion" names provided. NEVER skip, merge, or omit any criterion.
    3. For each criterion, compare the student's work to `level_4`, `level_3`, `level_2`, and `level_1`, assign a `likert_rating` (4, 3, 2, or 1), and compute `marks_awarded` = `(likert_rating / 4.0) * max_marks`.
    4. PERFORM AI & COPY-PASTE FORENSICS:
       - Check whether the student's Discussion, Environmental Significance, and Conclusion cite the actual numerical results from their Observation Tables and Calculations, or if they read like a generic ChatGPT essay disconnected from their lab data.
       - Check for LLM linguistic footprints, fabricated intermediate calculation steps, or abrupt font/formatting paste blocks.
       - Estimate `ai_generation_probability` (0 to 100) and write a concise 1-sentence `ai_forensic_summary`.
       - {penalty_instruction}
    5. EXTRACT VERBATIM FINGERPRINT FOR CROSS-STUDENT COLLUSION CHECK:
       - In `student_unique_fingerprint`, copy verbatim 250 to 450 words of the student's actual written text from their Theory, Calculations, Precautions, Environmental Significance, and Conclusion sections (plus their calculated numerical results). Do NOT summarize—extract their exact words and numbers so Python can compare classmates for copy-pasting.
    
    Return strictly valid JSON with this exact structure:
    {{
      "criteria_scores": [
        {{
          "criterion": "Exact name of criterion from rubric",
          "likert_rating": 3,
          "rating_label": "3 - Proficient",
          "marks_awarded": 3.0,
          "max_marks": 4.0,
          "feedback": "Specific constructive feedback explaining why this earned 3 (Proficient)."
        }}
      ],
      "ai_total_score": 32.0,
      "overall_feedback": "Summary statement on the student's performance across the manual/report.",
      "ai_generation_probability": 15,
      "ai_forensic_summary": "Conclusion and environmental significance cite actual measured pH and alkalinity values.",
      "student_unique_fingerprint": "Verbatim text extracted from student's theory, calculations, environmental significance, and conclusion..."
    }}
    """
    
    pdf_part = types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf")
    models_to_try = [
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-3.8-flash",
        "gemini-3.5-flash",
        "gemini-1.5-flash"
    ]
    
    last_error = None
    for model_name in models_to_try:
        for attempt in range(3):
            try:
                response = gemini_client.models.generate_content(
                    model=model_name,
                    contents=[pdf_part, prompt],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                    ),
                )
                parsed = json.loads(response.text)
                
                returned_map = {
                    str(c.get("criterion", "")).strip().lower(): c
                    for c in (parsed.get("criteria_scores") or [])
                }
                complete_scores = []
                for idx_r, r_item in enumerate(rubric_json):
                    c_name = str(r_item["criterion"]).strip()
                    max_m = float(r_item["max_marks"])
                    matched = returned_map.get(c_name.lower())
                    if not matched and idx_r < len(parsed.get("criteria_scores", [])):
                        matched = parsed["criteria_scores"][idx_r]
                    if matched:
                        matched["criterion"] = c_name
                        matched["max_marks"] = max_m
                        complete_scores.append(matched)
                    else:
                        complete_scores.append({
                            "criterion": c_name,
                            "likert_rating": 1,
                            "rating_label": "1 - Needs Improvement",
                            "marks_awarded": round(0.25 * max_m, 2),
                            "max_marks": max_m,
                            "feedback": "Section missing or not clearly identifiable in the submitted PDF."
                        })
                parsed["criteria_scores"] = complete_scores
                parsed["ai_total_score"] = round(sum(float(x.get("marks_awarded", 0.0)) for x in complete_scores), 2)
                return parsed
            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                if "429" in err_str or "quota" in err_str or "exhausted" in err_str:
                    time.sleep(12 * (attempt + 1))
                else:
                    time.sleep(2)
                
    raise last_error

# Ensure default admin exists
ensure_default_admin()

# Shared Sample Rubric CSV bytes for template downloads
SAMPLE_RUBRIC_CSV_BYTES = pd.DataFrame(DEFAULT_LAB_LIKERT_RUBRIC).to_csv(index=False).encode("utf-8")

# ==========================================
# SIDEBAR NAVIGATION & QR URL PARAMETER ROUTING
# ==========================================
st.sidebar.title("🧪 IEER Smart Portal")
portal_list = [
    "Student Submission Portal",
    "🎯 Live Class Poll & Quiz",
    "Instructor / Admin Login"
]

qp_portal = st.query_params.get("portal", "").lower()
qp_course = st.query_params.get("course", "").upper()

default_portal_idx = 1 if qp_portal == "poll" else 0
portal_mode = st.sidebar.radio(
    "Select Portal",
    portal_list,
    index=default_portal_idx
)

# ==========================================
# PORTAL 1: SECURE STUDENT SUBMISSION PORTAL
# ==========================================
if portal_mode == "Student Submission Portal":
    st.title("📄 Secure Student Report Submission Portal")
    st.markdown("Enter your official university credentials to verify enrollment before uploading your report.")
    
    courses_res = supabase.table("courses").select("course_id, course_code, course_title, rubric_json, total_marks").execute()
    courses = courses_res.data
    
    if not courses:
        st.warning("No active courses configured yet. Please contact your instructor.")
    else:
        course_options = {f"{c['course_code']} - {c['course_title']}": c for c in courses}
        selected_course_label = st.selectbox("Select Course", list(course_options.keys()))
        selected_course_obj = course_options[selected_course_label]
        course_id = selected_course_obj['course_id']
        
        st.markdown("---")
        st.subheader("Step 1: Student Verification")
        with st.form("student_auth_form"):
            col_a, col_b = st.columns(2)
            with col_a:
                input_reg = st.text_input("Registration Number (e.g., 2023-MS-EN-201)").strip().upper()
            with col_b:
                input_email = st.text_input("University Email").strip().lower()
            
            verify_btn = st.form_submit_button("Verify Enrollment")
            
        if verify_btn:
            if not input_reg or not input_email:
                st.error("Please provide both your Registration Number and Email.")
            else:
                student_check = supabase.table("students").select("*").eq("course_id", course_id).eq("reg_no", input_reg).eq("email", input_email).execute().data
                
                if student_check:
                    st.session_state['verified_student'] = student_check[0]
                    st.success(f"Verification successful! Welcome, **{student_check[0]['student_name']}**.")
                else:
                    st.session_state.pop('verified_student', None)
                    st.error("❌ Credentials not found in the official class roster for this course.")
                    
        if 'verified_student' in st.session_state and st.session_state['verified_student'].get('course_id') == course_id:
            student_info = st.session_state['verified_student']
            
            exps_res = supabase.table("experiments").select("*").eq("course_id", course_id).eq("is_open", True).execute()
            experiments = exps_res.data
            
            if not experiments:
                st.info("No active assessments are currently accepting submissions for this course.")
            else:
                st.markdown("---")
                st.subheader(f"Step 2: Upload Report for {student_info['student_name']} ({student_info['reg_no']})")
                
                exp_options = {f"{'📦 [Combined/Design] ' if e.get('is_combined') else '🔬 '}{e['exp_title']}": e for e in experiments}
                selected_exp_label = st.selectbox("Select Assessment", list(exp_options.keys()))
                exp = exp_options[selected_exp_label]
                
                student_active_rubric = exp['custom_rubric_json'] if exp.get('custom_rubric_json') else selected_course_obj['rubric_json']
                if student_active_rubric:
                    with st.expander("📋 View Grading Rubric & Likert Scale Criteria (4 = Excellent to 1 = Needs Improvement)"):
                        st.dataframe(format_rubric_for_display(student_active_rubric), use_container_width=True, hide_index=True)
                
                with st.form("student_submission_form"):
                    uploaded_file = st.file_uploader("Upload Report (PDF only, max 10MB)", type=["pdf"])
                    submitted = st.form_submit_button("Submit Report")
                    
                    if submitted:
                        if not uploaded_file:
                            st.error("Please attach your PDF report.")
                        else:
                            with st.spinner("Processing upload and synchronizing with Google Drive..."):
                                try:
                                    course_data = supabase.table("courses").select(f"course_code, {COL_DRIVE_FOLDER}").eq("course_id", course_id).single().execute().data
                                    
                                    if not course_data.get(COL_DRIVE_FOLDER):
                                        c_folder_ref = get_or_create_folder(course_data['course_code'], ROOT_DRIVE_CONTAINER)
                                        supabase.table("courses").update({COL_DRIVE_FOLDER: c_folder_ref}).eq("course_id", course_id).execute()
                                    else:
                                        c_folder_ref = course_data[COL_DRIVE_FOLDER]
                                        
                                    if not exp.get(COL_DRIVE_FOLDER):
                                        e_folder_ref = get_or_create_folder(f"Assessment_{exp['exp_number']}", c_folder_ref)
                                        supabase.table("experiments").update({COL_DRIVE_FOLDER: e_folder_ref}).eq("exp_id", exp['exp_id']).execute()
                                    else:
                                        e_folder_ref = exp[COL_DRIVE_FOLDER]
                                    
                                    file_name = f"{course_data['course_code']}_Report_{student_info['reg_no']}.pdf"
                                    file_bytes = uploaded_file.getvalue()
                                    
                                    existing_sub = supabase.table("submissions").select(f"sub_id, {COL_DRIVE_FILE}, submission_count").eq("exp_id", exp['exp_id']).eq("reg_no", student_info['reg_no']).execute().data
                                    
                                    if existing_sub:
                                        sub_id = existing_sub[0]['sub_id']
                                        existing_drive_ref = existing_sub[0][COL_DRIVE_FILE]
                                        sub_count = existing_sub[0]['submission_count'] + 1
                                        
                                        saved_ref = upload_or_update_file(file_bytes, file_name, "application/pdf", e_folder_ref, existing_drive_ref)
                                        
                                        supabase.table("submissions").update({
                                            "student_name": student_info['student_name'],
                                            "submission_count": sub_count,
                                            "status": "pending"
                                        }).eq("sub_id", sub_id).execute()
                                        
                                        st.success(f"Success! Your report has been resubmitted and overwritten (Version {sub_count}).")
                                    else:
                                        saved_ref = upload_or_update_file(file_bytes, file_name, "application/pdf", e_folder_ref)
                                        
                                        supabase.table("submissions").insert({
                                            "exp_id": exp['exp_id'],
                                            "reg_no": student_info['reg_no'],
                                            "student_name": student_info['student_name'],
                                            COL_DRIVE_FILE: saved_ref,
                                            "file_name": file_name,
                                            "submission_count": 1,
                                            "status": "pending"
                                        }).execute()
                                        
                                        st.success("Report submitted successfully!")
                                except Exception as e:
                                    st.error(f"An error occurred during submission: {e}")

# ==========================================
# PORTAL 2: LIVE CLASS POLL & QUIZ (STUDENT MENTI PORTAL)
# ==========================================
elif portal_mode == "🎯 Live Class Poll & Quiz":
    st.title("🎯 Live Class Poll, Word Cloud & Quiz")
    st.markdown(
        "Verify your course enrollment once below. Your screen will **automatically sync with the classroom projector** as your instructor advances from slide to slide."
    )
    
    courses_res = supabase.table("courses").select("course_id, course_code, course_title").execute()
    courses = courses_res.data
    
    if not courses:
        st.warning("No active courses available.")
    else:
        course_labels = [f"{c['course_code']} - {c['course_title']}" for c in courses]
        course_options = {f"{c['course_code']} - {c['course_title']}": c for c in courses}
        
        preselect_idx = 0
        if qp_course:
            for idx_c, c_obj in enumerate(courses):
                if c_obj['course_code'].upper() == qp_course:
                    preselect_idx = idx_c
                    break
                    
        selected_course_label = st.selectbox("Select Your Course", course_labels, index=preselect_idx, key="poll_course_sel")
        selected_course_obj = course_options[selected_course_label]
        course_id = selected_course_obj['course_id']
        
        st.markdown("---")
        if 'verified_student' not in st.session_state or st.session_state['verified_student'].get('course_id') != course_id:
            st.subheader("Step 1: Verify Roster Identity")
            with st.form("poll_student_auth_form"):
                col_a, col_b = st.columns(2)
                with col_a:
                    input_reg = st.text_input("Registration Number (e.g., 2023-MS-EN-201)").strip().upper()
                with col_b:
                    input_email = st.text_input("University Email").strip().lower()
                verify_poll_btn = st.form_submit_button("🚀 Join Live Classroom Session", type="primary")
                
            if verify_poll_btn:
                if not input_reg or not input_email:
                    st.error("Please enter both your Registration Number and University Email.")
                else:
                    student_check = supabase.table("students").select("*").eq("course_id", course_id).eq("reg_no", input_reg).eq("email", input_email).execute().data
                    if student_check:
                        st.session_state['verified_student'] = student_check[0]
                        st.rerun()
                    else:
                        st.error("❌ Not found in the official roster for this course. Please contact your instructor.")
        else:
            student_info = st.session_state['verified_student']
            col_s_info, col_s_switch = st.columns([4, 1])
            with col_s_info:
                st.success(f"🟢 Connected: **{student_info['student_name']}** (`{student_info['reg_no']}`) — Course: **{selected_course_obj['course_code']}**")
            with col_s_switch:
                if st.button("Switch Student"):
                    st.session_state.pop('verified_student', None)
                    st.rerun()

            @st.fragment(run_every="3s")
            def render_student_live_slide(c_id, c_code, stu):
                active_polls = (
                    supabase.table("live_polls")
                    .select("*")
                    .eq("course_id", c_id)
                    .eq("is_active", True)
                    .order("slide_order")
                    .execute()
                    .data
                )
                
                if not active_polls:
                    st.info("⏳ **Waiting for instructor...** No slide is currently open for voting. Keep this screen open—it will update automatically as soon as the next question launches!")
                    return
                
                poll = active_polls[0]
                p_id = poll['poll_id']
                p_type = poll['poll_type']
                rem_sec = get_remaining_seconds(poll)
                
                if rem_sec is not None and rem_sec <= 0:
                    st.warning("⏰ **Time is up for this question!** Waiting for the instructor to advance to the next slide...")
                    return
                
                session_lbl = poll.get("session_title") or "Live Session"
                slide_no = poll.get("slide_order") or 1
                type_badge = {
                    "mcq": "📊 Live Multiple Choice Poll",
                    "quiz": f"🏆 Graded Quiz ({poll.get('marks', 1.0)} Marks)",
                    "wordcloud": "☁️ Live Word Cloud"
                }.get(p_type, "📊 Poll")
                
                with st.container(border=True):
                    top_c1, top_c2 = st.columns([3, 1])
                    with top_c1:
                        st.caption(f"📌 **{session_lbl} — Slide #{slide_no}** | {type_badge}")
                    with top_c2:
                        if rem_sec is not None:
                            st.markdown(f"### ⏱️ `{rem_sec}s`")
                            
                    st.markdown(f"### {poll['question_text']}")
                    
                    existing_vote = (
                        supabase.table("poll_responses")
                        .select("*")
                        .eq("poll_id", p_id)
                        .eq("reg_no", stu['reg_no'])
                        .execute()
                        .data
                    )
                    
                    if existing_vote:
                        my_resp = existing_vote[0]
                        st.success(
                            f"🔒 **Answer Locked:** You submitted **`{my_resp['response_text']}`**. "
                            f"Wait for the instructor to advance to the next slide."
                        )
                        if p_type == "quiz" and poll.get("reveal_answer"):
                            if my_resp.get("is_correct"):
                                st.info(f"✅ **Correct!** Answer: `{poll['correct_answer']}` | Marks Earned: **{my_resp['marks_awarded']} / {poll['marks']}**")
                            else:
                                st.warning(f"❌ **Incorrect.** Correct Answer: `{poll['correct_answer']}` | Marks Earned: **0 / {poll['marks']}**")
                    else:
                        with st.form(f"student_vote_form_{p_id}"):
                            if p_type in ["mcq", "quiz"]:
                                opts = poll.get("options_json") or []
                                choice = st.radio("Select your answer (Strictly ONE attempt):", opts, index=None, key=f"stu_radio_{p_id}")
                            else:
                                choice = st.text_input("Enter your word or short phrase (max 25 chars):", max_chars=25, key=f"stu_wc_{p_id}").strip()
                                
                            submit_vote = st.form_submit_button("🔒 Submit & Lock Answer", type="primary")
                            if submit_vote:
                                latest_poll = supabase.table("live_polls").select("*").eq("poll_id", p_id).single().execute().data
                                latest_rem = get_remaining_seconds(latest_poll)
                                if not latest_poll.get("is_active") or (latest_rem is not None and latest_rem <= 0):
                                    st.error("⏰ Voting for this slide has just closed!")
                                    st.rerun()
                                elif not choice:
                                    st.error("Please select or type your response before submitting.")
                                else:
                                    recheck = supabase.table("poll_responses").select("resp_id").eq("poll_id", p_id).eq("reg_no", stu['reg_no']).execute().data
                                    if not recheck:
                                        is_corr = False
                                        awarded = 0.0
                                        if p_type == "quiz":
                                            is_corr = (str(choice).strip() == str(latest_poll.get("correct_answer", "")).strip())
                                            awarded = float(latest_poll.get("marks", 1.0)) if is_corr else 0.0
                                        supabase.table("poll_responses").insert({
                                            "poll_id": p_id,
                                            "reg_no": stu['reg_no'],
                                            "student_name": stu['student_name'],
                                            "response_text": str(choice).strip(),
                                            "is_correct": is_corr,
                                            "marks_awarded": awarded
                                        }).execute()
                                    st.rerun()

            render_student_live_slide(course_id, selected_course_obj['course_code'], student_info)

# ==========================================
# PORTAL 3: INSTRUCTOR & ADMIN PORTAL
# ==========================================
elif portal_mode == "Instructor / Admin Login":
    st.title("🔒 Faculty & Administration Portal")
    
    if "auth_user" not in st.session_state:
        st.markdown("Please sign in with your faculty username and password.")
        with st.form("faculty_login_form"):
            u_name = st.text_input("Username").strip().lower()
            u_pass = st.text_input("Password", type="password")
            login_btn = st.form_submit_button("Sign In")
            
            if login_btn:
                if not u_name or not u_pass:
                    st.error("Please enter both username and password.")
                else:
                    user_res = supabase.table("app_users").select("*").eq("username", u_name).eq("password_hash", hash_password(u_pass)).eq("is_active", True).execute()
                    if user_res.data:
                        st.session_state["auth_user"] = user_res.data[0]
                        st.rerun()
                    else:
                        st.error("Invalid username/password or account is inactive.")
    else:
        current_user = st.session_state["auth_user"]
        is_admin = current_user["role"] == "admin"
        
        col_info, col_logout = st.columns([5, 1])
        with col_info:
            st.success(f"👤 Logged in as: **{current_user['full_name']}** (`{current_user['username']}`) | Role: **{current_user['role'].upper()}**")
        with col_logout:
            if st.button("🚪 Logout"):
                st.session_state.pop("auth_user", None)
                st.rerun()

        with st.expander("🔑 Change My Password"):
            with st.form("self_password_change_form"):
                cp_col1, cp_col2, cp_col3 = st.columns(3)
                with cp_col1:
                    curr_pwd = st.text_input("Current Password", type="password")
                with cp_col2:
                    new_pwd = st.text_input("New Password", type="password")
                with cp_col3:
                    conf_pwd = st.text_input("Confirm New Password", type="password")
                
                change_pwd_btn = st.form_submit_button("Update Password")
                if change_pwd_btn:
                    if not curr_pwd or not new_pwd or not conf_pwd:
                        st.error("Please fill in all three password fields.")
                    elif hash_password(curr_pwd) != current_user["password_hash"]:
                        st.error("❌ Current password is incorrect.")
                    elif new_pwd != conf_pwd:
                        st.error("❌ New passwords do not match.")
                    elif len(new_pwd.strip()) < 6:
                        st.error("❌ New password must be at least 6 characters long.")
                    else:
                        new_hash = hash_password(new_pwd)
                        supabase.table("app_users").update({
                            "password_hash": new_hash
                        }).eq("username", current_user["username"]).execute()
                        st.session_state["auth_user"]["password_hash"] = new_hash
                        st.success("✅ Your password has been updated successfully!")
                
        def get_accessible_courses():
            if is_admin:
                return supabase.table("courses").select("*").execute().data
            else:
                return supabase.table("courses").select("*").eq("teacher_username", current_user["username"]).execute().data

        # ==========================================
        # MULTI-ROW NAVIGATION BUTTON LAYOUT (3 per row)
        # ==========================================
        if is_admin:
            nav_options = [
                "🛡️ Manage Teachers", 
                "📚 Course & Rubric", 
                "👥 Class Roster", 
                "🔬 Assessments", 
                "📊 Grading & Export",
                "🎯 Live Polls & Quizzes"
            ]
        else:
            nav_options = [
                "📚 Course & Rubric", 
                "👥 Class Roster", 
                "🔬 Assessments", 
                "📊 Grading & Export",
                "🎯 Live Polls & Quizzes"
            ]
            
        if "faculty_nav_choice" not in st.session_state or st.session_state["faculty_nav_choice"] not in nav_options:
            st.session_state["faculty_nav_choice"] = nav_options[0]

        st.markdown("---")
        
        chunked_options = [nav_options[i:i + 3] for i in range(0, len(nav_options), 3)]
        for row in chunked_options:
            cols = st.columns(3)
            for idx, opt in enumerate(row):
                with cols[idx]:
                    is_selected = (st.session_state["faculty_nav_choice"] == opt)
                    btn_type = "primary" if is_selected else "secondary"
                    if st.button(opt, use_container_width=True, type=btn_type):
                        st.session_state["faculty_nav_choice"] = opt
                        st.rerun()
                        
        st.markdown("---")
        chosen_tab = st.session_state["faculty_nav_choice"]

        # ==========================================
        # SECTION 1: MANAGE TEACHERS (ADMIN ONLY)
        # ==========================================
        if chosen_tab == "🛡️ Manage Teachers":
            st.subheader("🛡️ Create or Update Teacher / Admin Accounts")
            st.markdown("Only Administrators can create or reset login credentials for faculty members.")
            
            with st.form("create_user_form"):
                col_u1, col_u2 = st.columns(2)
                with col_u1:
                    new_fullname = st.text_input("Teacher Full Name (e.g., Dr. Ahmad Khan)").strip()
                    new_username = st.text_input("Username (e.g., ahmad.khan)").strip().lower()
                with col_u2:
                    new_password = st.text_input("Set Initial / Reset Password", type="password")
                    new_role = st.selectbox("Account Role", ["teacher", "admin"])
                    
                is_active_user = st.checkbox("Account Active", value=True)
                save_user_btn = st.form_submit_button("➕ Create / Update User Account")
                
                if save_user_btn:
                    if not new_fullname or not new_username or not new_password:
                        st.error("Please fill in Full Name, Username, and Password.")
                    else:
                        try:
                            supabase.table("app_users").upsert({
                                "username": new_username,
                                "full_name": new_fullname,
                                "password_hash": hash_password(new_password),
                                "role": new_role,
                                "is_active": is_active_user
                            }, on_conflict="username").execute()
                            st.success(f"User account `{new_username}` ({new_role}) saved successfully!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error saving user: {e}")
                            
            st.markdown("---")
            st.subheader("Registered Faculty & Admin Accounts")
            all_users = supabase.table("app_users").select("username, full_name, role, is_active, created_at").order("created_at").execute().data
            if all_users:
                st.dataframe(pd.DataFrame(all_users), use_container_width=True)

        # ==========================================
        # SECTION 2: COURSE & LIKERT RUBRIC
        # ==========================================
        elif chosen_tab == "📚 Course & Rubric":
            st.subheader("Configure Course & 4-Point Likert Scale Rubric")
            st.markdown(
                "You can either use/edit the pre-loaded **10-Criterion Likert Scale Lab Rubric** below, or **upload your own Rubric CSV/Excel file**."
            )
            
            col_t1_dl, col_t1_up = st.columns([1, 2])
            with col_t1_dl:
                st.download_button(
                    label="📥 Download Sample Likert Rubric CSV Template",
                    data=SAMPLE_RUBRIC_CSV_BYTES,
                    file_name="likert_rubric_template.csv",
                    mime="text/csv",
                    key="dl_course_rubric_template"
                )
            with col_t1_up:
                uploaded_course_rubric_file = st.file_uploader(
                    "Optional: Upload Course Rubric CSV / Excel (.csv, .xlsx)",
                    type=["csv", "xlsx"],
                    key="course_rubric_csv_uploader"
                )
                
            if uploaded_course_rubric_file is not None:
                try:
                    default_course_rubric_df = parse_uploaded_rubric_file(uploaded_course_rubric_file)
                    st.success(f"✅ Loaded {len(default_course_rubric_df)} criteria from `{uploaded_course_rubric_file.name}` into the editor below!")
                except Exception as e:
                    st.error(f"Error reading uploaded rubric file: {e}")
                    default_course_rubric_df = pd.DataFrame(DEFAULT_LAB_LIKERT_RUBRIC)
            else:
                default_course_rubric_df = pd.DataFrame(DEFAULT_LAB_LIKERT_RUBRIC)
            
            with st.form("course_form"):
                col_c1, col_c2 = st.columns(2)
                with col_c1:
                    c_code = st.text_input("Course Code (e.g., ENV-504)").strip().upper()
                with col_c2:
                    c_title = st.text_input("Course Title (e.g., Advanced Environmental Engineering Lab)").strip()
                
                if is_admin:
                    teachers_list = supabase.table("app_users").select("username, full_name").execute().data
                    teacher_options = {f"{t['username']} ({t['full_name']})": t['username'] for t in teachers_list} if teachers_list else {"admin": "admin"}
                    assigned_teacher_label = st.selectbox("Assign Course Owner", list(teacher_options.keys()))
                    assigned_teacher = teacher_options[assigned_teacher_label]
                else:
                    assigned_teacher = current_user["username"]
                
                st.markdown("#### Interactive 4-Point Likert Rubric Matrix")
                edited_course_rubric = st.data_editor(
                    default_course_rubric_df,
                    column_config={
                        "criterion": st.column_config.TextColumn("Evaluation Criteria", required=True, width="medium"),
                        "max_marks": st.column_config.NumberColumn("Max Marks", min_value=1.0, max_value=100.0, step=0.5, required=True, width="small"),
                        "level_4": st.column_config.TextColumn("4 – Excellent", required=True, width="large"),
                        "level_3": st.column_config.TextColumn("3 – Proficient", required=True, width="large"),
                        "level_2": st.column_config.TextColumn("2 – Developing", required=True, width="large"),
                        "level_1": st.column_config.TextColumn("1 – Needs Improvement", required=True, width="large"),
                    },
                    num_rows="dynamic",
                    use_container_width=True,
                    key="course_rubric_editor"
                )
                
                submitted_course = st.form_submit_button("💾 Save Course & Likert Rubric")
                if submitted_course:
                    if not c_code or not c_title:
                        st.error("Please enter both Course Code and Course Title.")
                    else:
                        rubric_json, calculated_total = build_rubric_from_df(edited_course_rubric)
                        if not rubric_json:
                            st.error("Please provide at least one valid rubric criterion with marks greater than 0.")
                        else:
                            try:
                                supabase.table("courses").upsert({
                                    "course_code": c_code,
                                    "course_title": c_title,
                                    "total_marks": calculated_total,
                                    "rubric_json": rubric_json,
                                    "teacher_username": assigned_teacher
                                }, on_conflict="course_code").execute()
                                st.success(f"✅ Course `{c_code}` saved with Likert Scale Rubric! Total Marks: **{calculated_total}** across {len(rubric_json)} criteria.")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error saving course: {e}")

            existing_courses = get_accessible_courses()
            if existing_courses:
                st.markdown("---")
                st.subheader("Configured Courses & Default Likert Rubrics")
                for c in existing_courses:
                    with st.expander(f"📘 {c['course_code']} — {c['course_title']} (Default Total: {c['total_marks']} Marks | Owner: {c.get('teacher_username', 'admin')})"):
                        st.dataframe(format_rubric_for_display(c['rubric_json']), use_container_width=True, hide_index=True)

        # ==========================================
        # SECTION 3: CLASS ROSTER UPLOAD & STUDENT EDITOR
        # ==========================================
        elif chosen_tab == "👥 Class Roster":
            st.subheader("Upload & Manage Class Roster (CSV / Excel)")
            st.markdown("Upload a spreadsheet containing your student class list. The file must include these exact column headers: **`reg_no`**, **`student_name`**, and **`email`**.")
            
            sample_df = pd.DataFrame({
                "reg_no": ["2023-MS-EN-201", "2023-MS-EN-202"],
                "student_name": ["Muhammad Ali", "Ayesha Khan"],
                "email": ["ali@student.uet.edu.pk", "ayesha@student.uet.edu.pk"]
            })
            sample_csv = sample_df.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="📥 Download Sample Roster CSV Template",
                data=sample_csv,
                file_name="student_roster_template.csv",
                mime="text/csv"
            )
            st.markdown("---")
            
            my_courses = get_accessible_courses()
            if my_courses:
                c_dict = {c['course_code']: c['course_id'] for c in my_courses}
                selected_c_code = st.selectbox("Select Course for Roster Upload", list(c_dict.keys()), key="roster_course_select")
                selected_course_id = c_dict[selected_c_code]
                
                uploaded_roster = st.file_uploader("Upload Filled Roster File (.csv or .xlsx)", type=["csv", "xlsx"])
                
                if uploaded_roster:
                    try:
                        if uploaded_roster.name.endswith('.csv'):
                            df_roster = pd.read_csv(uploaded_roster)
                        else:
                            df_roster = pd.read_excel(uploaded_roster)
                            
                        df_roster.columns = [str(col).strip().lower() for col in df_roster.columns]
                        
                        required_cols = ['reg_no', 'student_name', 'email']
                        if not all(col in df_roster.columns for col in required_cols):
                            st.error(f"Missing required columns! Your file must contain: {required_cols}")
                        else:
                            st.dataframe(df_roster.head(), use_container_width=True)
                            if st.button("📥 Import All Students into Supabase Roster"):
                                count = 0
                                for _, row in df_roster.iterrows():
                                    supabase.table("students").upsert({
                                        "course_id": selected_course_id,
                                        "reg_no": str(row['reg_no']).strip().upper(),
                                        "student_name": str(row['student_name']).strip(),
                                        "email": str(row['email']).strip().lower()
                                    }, on_conflict="course_id, reg_no").execute()
                                    count += 1
                                st.success(f"Successfully imported {count} students into the roster!")
                                st.rerun()
                    except Exception as e:
                        st.error(f"Error parsing roster file: {e}")
                
                st.markdown("---")
                st.markdown("### Currently Enrolled Roster")
                roster_res = supabase.table("students").select("student_id, reg_no, student_name, email").eq("course_id", selected_course_id).order("reg_no").execute()
                if roster_res.data:
                    st.dataframe(pd.DataFrame(roster_res.data)[["reg_no", "student_name", "email"]], use_container_width=True, hide_index=True)
                    
                    with st.expander("🛠️ Fix / Edit or Delete a Student Record (Avoids Duplicate Key Errors)"):
                        stu_map = {f"{s['reg_no']} — {s['student_name']} ({s['email']})": s for s in roster_res.data}
                        sel_stu_label = st.selectbox("Select Student to Edit or Remove", list(stu_map.keys()))
                        sel_stu = stu_map[sel_stu_label]
                        
                        col_ed1, col_ed2, col_ed3 = st.columns(3)
                        with col_ed1:
                            new_s_reg = st.text_input("Updated Reg No", value=sel_stu['reg_no']).strip().upper()
                        with col_ed2:
                            new_s_name = st.text_input("Updated Full Name", value=sel_stu['student_name']).strip()
                        with col_ed3:
                            new_s_email = st.text_input("Updated Email", value=sel_stu['email']).strip().lower()
                            
                        col_btn_up, col_btn_del = st.columns(2)
                        with col_btn_up:
                            if st.button("💾 Save Student Changes", key="btn_upd_stu"):
                                try:
                                    clash = supabase.table("students").select("student_id").eq("course_id", selected_course_id).eq("reg_no", new_s_reg).neq("student_id", sel_stu["student_id"]).execute().data
                                    if clash:
                                        supabase.table("students").delete().eq("student_id", clash[0]["student_id"]).execute()
                                    supabase.table("students").update({
                                        "reg_no": new_s_reg,
                                        "student_name": new_s_name,
                                        "email": new_s_email
                                    }).eq("student_id", sel_stu["student_id"]).execute()
                                    st.success("Student record updated cleanly!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error updating student: {e}")
                        with col_btn_del:
                            if st.button("🗑️ Delete This Student from Roster", key="btn_del_stu"):
                                supabase.table("students").delete().eq("student_id", sel_stu["student_id"]).execute()
                                st.success("Student removed from roster.")
                                st.rerun()
                else:
                    st.info("No students uploaded for this course yet.")
            else:
                st.info("No courses assigned to your account yet. Please create a course first.")

        # ==========================================
        # SECTION 4: ASSESSMENTS & CUSTOM RUBRICS
        # ==========================================
        elif chosen_tab == "🔬 Assessments":
            st.subheader("Manage Experiments, Combined Packets & Design Reports")
            my_courses = get_accessible_courses()
            if my_courses:
                c_dict = {c['course_code']: c for c in my_courses}
                selected_c_code = st.selectbox("Select Course", list(c_dict.keys()), key="exp_course_select")
                course_data = c_dict[selected_c_code]
                
                col_e1, col_e2 = st.columns([1, 3])
                with col_e1:
                    exp_no = st.number_input("Assessment Number / ID", min_value=1, step=1)
                with col_e2:
                    exp_title = st.text_input("Assessment Title * (e.g., Exp 1: pH & Turbidity or Labs 1-11 Combined)").strip()
                
                is_combined = st.checkbox("📦 Is this a Combined Multi-Lab Manual or Design Report?", value=False)
                is_open = st.checkbox("🔓 Accepting Submissions (Open)", value=True)
                exp_guidelines = st.text_area("Optional Guidelines for AI Evaluator", placeholder="e.g., Verify mass balance equations and APHA standard compliance.")
                
                st.markdown("---")
                use_custom_rubric = st.checkbox("✨ Use Custom Likert Rubric for this Assessment (Overrides Course Default Rubric)", value=False)
                
                edited_exp_rubric = None
                if not use_custom_rubric:
                    st.info("ℹ️ This assessment will automatically use the **Course Default Likert Rubric** configured in Course & Rubric.")
                else:
                    st.markdown("#### Custom Likert Rubric Setup (Upload CSV/Excel or Edit Below)")
                    col_r_dl, col_r_up = st.columns([1, 2])
                    with col_r_dl:
                        st.download_button(
                            label="📥 Download Sample Likert Rubric CSV Template",
                            data=SAMPLE_RUBRIC_CSV_BYTES,
                            file_name="custom_likert_rubric_template.csv",
                            mime="text/csv",
                            key="dl_exp_rubric_template"
                        )
                    with col_r_up:
                        uploaded_exp_rubric_file = st.file_uploader(
                            "Upload Your Custom Rubric File (.csv or .xlsx)",
                            type=["csv", "xlsx"],
                            key="exp_rubric_csv_uploader"
                        )
                        
                    if uploaded_exp_rubric_file is not None:
                        try:
                            default_custom_rubric_df = parse_uploaded_rubric_file(uploaded_exp_rubric_file)
                            st.success(f"✅ Loaded {len(default_custom_rubric_df)} custom criteria from `{uploaded_exp_rubric_file.name}`!")
                        except Exception as e:
                            st.error(f"Error reading uploaded rubric file: {e}")
                            default_custom_rubric_df = pd.DataFrame(DEFAULT_LAB_LIKERT_RUBRIC)
                    else:
                        template_choice = st.radio(
                            "Or choose a starting template to edit manually:",
                            [
                                "10-Criterion Standard Lab Rubric (40 Marks)",
                                "5-Criterion Engineering Design Rubric (20 Marks)"
                            ],
                            horizontal=True,
                            key="exp_template_radio"
                        )
                        if "10-Criterion" in template_choice:
                            default_custom_rubric_df = pd.DataFrame(DEFAULT_LAB_LIKERT_RUBRIC)
                        else:
                            default_custom_rubric_df = pd.DataFrame(DEFAULT_DESIGN_LIKERT_RUBRIC)
                            
                    edited_exp_rubric = st.data_editor(
                        default_custom_rubric_df,
                        column_config={
                            "criterion": st.column_config.TextColumn("Evaluation Criteria", required=True, width="medium"),
                            "max_marks": st.column_config.NumberColumn("Max Marks", min_value=1.0, max_value=100.0, step=0.5, required=True, width="small"),
                            "level_4": st.column_config.TextColumn("4 – Excellent", required=True, width="large"),
                            "level_3": st.column_config.TextColumn("3 – Proficient", required=True, width="large"),
                            "level_2": st.column_config.TextColumn("2 – Developing", required=True, width="large"),
                            "level_1": st.column_config.TextColumn("1 – Needs Improvement", required=True, width="large"),
                        },
                        num_rows="dynamic",
                        use_container_width=True,
                        key="exp_custom_rubric_editor"
                    )
                
                if st.button("💾 Create / Update Assessment", type="primary", key="btn_save_assessment"):
                    if not exp_title:
                        st.error("❌ Assessment Title is required! Please enter a title above before saving.")
                    else:
                        custom_json = None
                        if use_custom_rubric and edited_exp_rubric is not None:
                            custom_json, custom_total = build_rubric_from_df(edited_exp_rubric)
                            if not custom_json:
                                st.error("Please provide at least one valid custom rubric criterion with marks > 0.")
                                st.stop()
                        
                        try:
                            supabase.table("experiments").upsert({
                                "course_id": course_data['course_id'],
                                "exp_number": exp_no,
                                "exp_title": exp_title,
                                "exp_guidelines": exp_guidelines,
                                "is_open": is_open,
                                "is_combined": is_combined,
                                "custom_rubric_json": custom_json
                            }, on_conflict="course_id, exp_number").execute()
                            st.success(f"✅ Assessment #{exp_no} (`{exp_title}`) saved successfully!")
                            time.sleep(1)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error saving assessment: {e}")
            else:
                st.info("Please set up a course first.")

        # ==========================================
        # SECTION 5: GRADING, AI/COPY-PASTE DETECTION & EXPORT
        # ==========================================
        elif chosen_tab == "📊 Grading & Export":
            st.subheader("Batch AI Grading Engine, AI/Copy-Paste Forensics & Gradebook")
            my_courses = get_accessible_courses()
            
            if my_courses:
                c_dict = {c['course_code']: c for c in my_courses}
                selected_c_code = st.selectbox("Select Course to Grade", list(c_dict.keys()), key="grade_course_select")
                course_info = c_dict[selected_c_code]
                
                exps_res = supabase.table("experiments").select("*").eq("course_id", course_info['course_id']).execute()
                
                if exps_res.data:
                    exp_dict = {f"{'📦 ' if e.get('is_combined') else '🔬 '}Exp {e['exp_number']}: {e['exp_title']}": e for e in exps_res.data}
                    selected_exp_label = st.selectbox("Select Assessment to Grade", list(exp_dict.keys()), key="grade_exp_select")
                    exp_info = exp_dict[selected_exp_label]
                    
                    active_rubric = exp_info['custom_rubric_json'] if exp_info.get('custom_rubric_json') else course_info['rubric_json']
                    active_total_marks = round(sum([float(item['max_marks']) for item in active_rubric]), 2)
                    
                    with st.expander(f"📋 View Active 4-Point Likert Rubric ({len(active_rubric)} Criteria | Total: {active_total_marks} Marks)"):
                        st.dataframe(format_rubric_for_display(active_rubric), use_container_width=True, hide_index=True)
                    
                    subs_res = supabase.table("submissions").select("*").eq("exp_id", exp_info['exp_id']).order("reg_no").execute()
                    submissions = subs_res.data or []
                    
                    sub_ids = [s['sub_id'] for s in submissions]
                    evals_map = {}
                    if sub_ids:
                        evals_query = supabase.table("evaluations").select("*").in_("sub_id", sub_ids).execute()
                        for ev in (evals_query.data or []):
                            evals_map[ev['sub_id']] = ev
                            
                    pending_subs = [s for s in submissions if s['sub_id'] not in evals_map or s.get('status') != 'graded']
                    graded_count = len(submissions) - len(pending_subs)
                    
                    m_col1, m_col2, m_col3 = st.columns(3)
                    m_col1.metric("Total Manuals Submitted", len(submissions))
                    m_col2.metric("✅ Graded in Analysis", graded_count)
                    m_col3.metric("⏳ Pending / Needs AI Grading", len(pending_subs))
                    
                    if submissions:
                        chk_col1, chk_col2 = st.columns(2)
                        with chk_col1:
                            only_ungraded = st.checkbox(
                                "⚡ Grade ONLY Pending / Un-evaluated Manuals (Skips already-graded reports)",
                                value=True
                            )
                        with chk_col2:
                            apply_integrity_penalty = st.checkbox(
                                "🛡️ Apply Automatic Penalty for High AI (>=70%) or Copy-Paste Collusion (>=75%)",
                                value=False,
                                help="When UNCHECKED, the system only flags AI Risk % and Copy-Paste similarity in the table. When CHECKED, generic AI sections are capped at Level 1 and reports with >=75% classmate similarity receive a 50% score deduction."
                            )
                            
                        target_queue = pending_subs if only_ungraded else submissions
                        
                        btn_g1, btn_g2 = st.columns([2, 1])
                        with btn_g1:
                            run_batch_clicked = st.button(
                                f"🚀 Run AI Grading & Integrity Forensics ({len(target_queue)} Manuals in Queue)",
                                type="primary",
                                use_container_width=True,
                                key="btn_run_batch_grading"
                            )
                        with btn_g2:
                            rerun_collusion_clicked = st.button(
                                "🔍 Re-Scan Class Copy-Paste Similarity & Apply Toggle Rule",
                                use_container_width=True,
                                key="btn_rescan_similarity"
                            )
                            
                        if rerun_collusion_clicked:
                            with st.spinner("Comparing all graded manuals across the class for copy-pasting and applying toggle rule..."):
                                run_cross_student_collusion_pass(submissions, evals_map, apply_integrity_penalty)
                                st.success("✅ Cross-student similarity scan and penalty rules updated!")
                                time.sleep(1)
                                st.rerun()
                                
                        if run_batch_clicked:
                            if not target_queue:
                                st.info("All submitted manuals are already graded! Click '🔍 Re-Scan Class Copy-Paste Similarity' to update flags, or uncheck 'Grade ONLY Pending' to re-grade all from scratch.")
                            else:
                                progress_bar = st.progress(0)
                                status_box = st.empty()
                                
                                for idx, sub in enumerate(target_queue):
                                    status_box.info(f"⏳ Grading & Running Forensics ({idx+1}/{len(target_queue)}): **{sub['student_name']}** (`{sub['reg_no']}`)...")
                                    try:
                                        pdf_bytes = download_file_bytes_from_drive(sub[COL_DRIVE_FILE])
                                        pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()
                                        
                                        eval_result = evaluate_report_with_gemini(
                                            pdf_bytes=pdf_bytes,
                                            rubric_json=active_rubric,
                                            course_title=course_info['course_title'],
                                            exp_title=exp_info['exp_title'],
                                            exp_guidelines=exp_info['exp_guidelines'],
                                            total_marks=active_total_marks,
                                            is_combined=exp_info.get('is_combined', False),
                                            apply_ai_penalty=apply_integrity_penalty
                                        )
                                        
                                        integrity_data = {
                                            "ai_generation_probability": float(eval_result.get("ai_generation_probability", 10)),
                                            "ai_forensic_summary": str(eval_result.get("ai_forensic_summary", "Evaluated.")),
                                            "student_unique_fingerprint": str(eval_result.get("student_unique_fingerprint", "")),
                                            "pdf_sha256": pdf_hash,
                                            "similarity_score": 0.0,
                                            "similarity_match_reg": "",
                                            "penalty_applied": bool(apply_integrity_penalty and float(eval_result.get("ai_generation_probability", 0)) >= 70)
                                        }
                                        
                                        save_evaluation_record(
                                            sub_id=sub['sub_id'],
                                            criteria_scores=eval_result['criteria_scores'],
                                            ai_score=eval_result['ai_total_score'],
                                            final_score=eval_result['ai_total_score'],
                                            overall_fb=eval_result['overall_feedback'],
                                            integrity_dict=integrity_data
                                        )
                                        
                                        supabase.table("submissions").update({"status": "graded"}).eq("sub_id", sub['sub_id']).execute()
                                        st.toast(f"✅ Graded {sub['reg_no']}: {eval_result['ai_total_score']}/{active_total_marks}")
                                        
                                        if idx < len(target_queue) - 1:
                                            time.sleep(4)
                                            
                                    except Exception as e:
                                        st.error(f"❌ Error grading {sub['reg_no']} ({sub['student_name']}): {e}")
                                    
                                    progress_bar.progress((idx + 1) / len(target_queue))
                                    
                                # Refresh evaluations map and run cross-student copy-paste comparison across all graded reports
                                status_box.info("🔍 Running cross-student copy-paste & duplicate file comparison across all manuals...")
                                refreshed_evals = supabase.table("evaluations").select("*").in_("sub_id", sub_ids).execute().data or []
                                refreshed_map = {e["sub_id"]: e for e in refreshed_evals}
                                run_cross_student_collusion_pass(submissions, refreshed_map, apply_integrity_penalty)
                                
                                status_box.success("✅ Batch grading and cross-student integrity audit completed!")
                                time.sleep(1)
                                st.rerun()
                                
                        st.markdown("---")
                        st.subheader(f"Graded Analysis, Integrity Flags & Likert Review (All {len(submissions)} Submissions)")
                        
                        rubric_col_map = {}
                        rubric_lookup = {}
                        ordered_criterion_cols = []
                        for r_item in active_rubric:
                            c_name = str(r_item['criterion']).strip()
                            col_label = f"{c_name} (/{r_item['max_marks']})"
                            rubric_col_map[c_name.lower()] = col_label
                            rubric_lookup[c_name.lower()] = r_item
                            ordered_criterion_cols.append(col_label)
                            
                        eval_df_data = []
                        detailed_feedback_rows = []
                        extra_criterion_cols = []
                        
                        for sub in submissions:
                            s_id = sub['sub_id']
                            reg_no_val = sub['reg_no']
                            student_name_val = sub['student_name']
                            ev = evals_map.get(s_id)
                            integ = extract_integrity_from_ev(ev)
                            
                            ai_pct = float(integ.get("ai_generation_probability", 0.0))
                            sim_pct = float(integ.get("similarity_score", 0.0))
                            sim_match = str(integ.get("similarity_match_reg", ""))
                            audit_notes = str(integ.get("audit_summary") or integ.get("ai_forensic_summary") or "—")
                            
                            row_dict = {
                                "sub_id": s_id,
                                "Reg No": reg_no_val,
                                "Student Name": student_name_val,
                                "Status": "✅ Graded" if ev else "⏳ Pending",
                                "🤖 AI Risk": format_ai_risk_badge(ai_pct) if ev else "—",
                                "📋 Copy-Paste Similarity": format_similarity_badge(sim_pct, sim_match) if ev else "—",
                            }
                            for col_label in ordered_criterion_cols:
                                row_dict[col_label] = 0.0
                                
                            if ev:
                                crit_list = ev.get('criteria_scores_json') or []
                                for idx_c, c_obj in enumerate(crit_list):
                                    raw_c_name = str(c_obj.get('criterion', f'Criterion {idx_c+1}')).strip()
                                    awarded = float(c_obj.get('marks_awarded', 0.0))
                                    max_m = float(c_obj.get('max_marks', 4.0))
                                    likert_val = c_obj.get('likert_rating', int(round((awarded / max_m) * 4)) if max_m > 0 else 1)
                                    r_label = str(c_obj.get('rating_label', f"{likert_val} / 4"))
                                    c_fb = str(c_obj.get('feedback', ''))
                                    
                                    r_def = rubric_lookup.get(raw_c_name.lower(), {})
                                    matched_desc = r_def.get(f"level_{likert_val}", "")
                                    
                                    matched_col = rubric_col_map.get(raw_c_name.lower())
                                    if not matched_col and idx_c < len(ordered_criterion_cols) and len(crit_list) == len(ordered_criterion_cols):
                                        matched_col = ordered_criterion_cols[idx_c]
                                        
                                    if matched_col:
                                        row_dict[matched_col] = awarded
                                    else:
                                        dyn_col = f"{raw_c_name} (/{max_m})" if max_m > 0 else raw_c_name
                                        row_dict[dyn_col] = awarded
                                        if dyn_col not in extra_criterion_cols:
                                            extra_criterion_cols.append(dyn_col)
                                            
                                    detailed_feedback_rows.append({
                                        "Reg No": reg_no_val,
                                        "Student Name": student_name_val,
                                        "Evaluation Criterion": raw_c_name,
                                        "Likert Rating (1-4)": likert_val,
                                        "Performance Level": r_label,
                                        "Marks Awarded": awarded,
                                        "Max Marks": max_m,
                                        "🤖 AI Risk": format_ai_risk_badge(ai_pct),
                                        "📋 Classmate Similarity": format_similarity_badge(sim_pct, sim_match),
                                        "Rubric Descriptor Matched": matched_desc,
                                        "Specific AI Feedback": c_fb
                                    })
                                    
                                row_dict["AI Score"] = float(ev['ai_total_score'])
                                row_dict["Final Score"] = float(ev['final_total_score'])
                                row_dict["🔍 Integrity Audit"] = audit_notes
                                row_dict["Feedback"] = ev['overall_feedback']
                            else:
                                row_dict["AI Score"] = 0.0
                                row_dict["Final Score"] = 0.0
                                row_dict["🔍 Integrity Audit"] = "Pending grading"
                                row_dict["Feedback"] = "⏳ Not graded yet — click 'Run AI Grading' above."
                                
                            eval_df_data.append(row_dict)
                            
                        if eval_df_data:
                            all_criterion_cols = ordered_criterion_cols + extra_criterion_cols
                            ordered_df_cols = (
                                ["sub_id", "Reg No", "Student Name", "Status", "🤖 AI Risk", "📋 Copy-Paste Similarity"]
                                + all_criterion_cols
                                + ["AI Score", "Final Score", "🔍 Integrity Audit", "Feedback"]
                            )
                            df_evals = pd.DataFrame(eval_df_data)
                            for col in ordered_df_cols:
                                if col not in df_evals.columns:
                                    df_evals[col] = 0.0
                            df_evals = df_evals[ordered_df_cols]
                            
                            st.markdown("### Edit Final Scores, Review Integrity Flags & Criterion Breakdown")
                            col_cfg = {
                                "sub_id": None,
                                "Reg No": st.column_config.TextColumn(disabled=True),
                                "Student Name": st.column_config.TextColumn(disabled=True),
                                "Status": st.column_config.TextColumn(disabled=True),
                                "🤖 AI Risk": st.column_config.TextColumn(disabled=True),
                                "📋 Copy-Paste Similarity": st.column_config.TextColumn(disabled=True),
                                "AI Score": st.column_config.NumberColumn(disabled=True),
                                "Final Score": st.column_config.NumberColumn(
                                    "Final Score (Editable)",
                                    min_value=0.0,
                                    max_value=float(active_total_marks),
                                    step=0.5
                                ),
                                "🔍 Integrity Audit": st.column_config.TextColumn(disabled=True),
                            }
                            for c_col in all_criterion_cols:
                                col_cfg[c_col] = st.column_config.NumberColumn(c_col, disabled=True)
                            
                            edited_df = st.data_editor(
                                df_evals,
                                column_config=col_cfg,
                                hide_index=True,
                                use_container_width=True
                            )
                            
                            if st.button("💾 Save Score Overrides"):
                                for index, row in edited_df.iterrows():
                                    if row['sub_id'] in evals_map:
                                        supabase.table("evaluations").update({
                                            "final_total_score": row['Final Score']
                                        }).eq("sub_id", row['sub_id']).execute()
                                st.success("Scores updated successfully!")
                                st.rerun()
                                
                            st.markdown("---")
                            st.markdown("### View Student Likert Breakdown (4 = Excellent to 1 = Needs Improvement)")
                            if detailed_feedback_rows:
                                st.dataframe(pd.DataFrame(detailed_feedback_rows), use_container_width=True, hide_index=True)
                            
                            st.markdown("---")
                            st.markdown("### Export Master Gradebook, Integrity Audit & Likert Feedback (.xlsx)")
                            
                            excel_buffer = io.BytesIO()
                            export_cols = (
                                ["Reg No", "Student Name", "Status", "🤖 AI Risk", "📋 Copy-Paste Similarity"]
                                + all_criterion_cols
                                + ["AI Score", "Final Score", "🔍 Integrity Audit", "Feedback"]
                            )
                            df_export = edited_df[export_cols]
                            
                            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                                sheet_1_name = f"{selected_c_code}_Assmt{exp_info['exp_number']}"[:31]
                                df_export.to_excel(writer, index=False, sheet_name=sheet_1_name)
                                
                                if detailed_feedback_rows:
                                    df_detailed = pd.DataFrame(detailed_feedback_rows)
                                    df_detailed.to_excel(writer, index=False, sheet_name="Detailed_Likert_Rubric")
                                    
                            excel_buffer.seek(0)
                            
                            st.download_button(
                                label="📥 Download Master Excel Grade Sheet, AI/Copy-Paste Audit & Likert Breakdown (.xlsx)",
                                data=excel_buffer,
                                file_name=f"{selected_c_code}_Assessment_{exp_info['exp_number']}_Likert_And_Integrity_Grades.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                    else:
                        st.info("No student submissions found for this assessment yet.")
                else:
                    st.info("No assessments created for this course yet.")
            else:
                st.info("Please set up a course first.")

        # ==========================================
        # SECTION 6: MENTIMETER LIVE SLIDE DECK, QR CODE, TIMER & FULL-ROSTER EXCEL
        # ==========================================
        elif chosen_tab == "🎯 Live Polls & Quizzes":
            st.subheader("🎯 Mentimeter Classroom Slide Deck, Live Timer & Roster Gradebook")
            my_courses = get_accessible_courses()
            
            if not my_courses:
                st.info("Please create a course first.")
            else:
                c_dict = {c['course_code']: c for c in my_courses}
                top_m1, top_m2 = st.columns([2, 2])
                with top_m1:
                    selected_poll_c_code = st.selectbox("Select Course", list(c_dict.keys()), key="menti_course_sel")
                poll_course_info = c_dict[selected_poll_c_code]
                poll_course_id = poll_course_info['course_id']
                
                all_course_polls = (
                    supabase.table("live_polls")
                    .select("*")
                    .eq("course_id", poll_course_id)
                    .order("slide_order")
                    .order("created_at")
                    .execute()
                    .data or []
                )
                
                existing_sessions = sorted(list({(p.get("session_title") or "Class Session 1").strip() for p in all_course_polls}))
                if not existing_sessions:
                    existing_sessions = ["Class Session 1"]
                    
                with top_m2:
                    selected_session = st.selectbox("Select Session / Lecture Deck", existing_sessions, key="menti_session_sel")
                
                # 1. SCAN-TO-JOIN QR CODE BANNER FOR PROJECTOR
                join_url = f"{APP_PUBLIC_URL}/?portal=poll&course={urllib.parse.quote(selected_poll_c_code)}"
                qr_img_url = f"https://api.qrserver.com/v1/create-qr-code/?size=240x240&data={urllib.parse.quote(join_url)}"
                
                with st.expander("📱 Show Scan-to-Join QR Code on Projector Screen", expanded=False):
                    qr_c1, qr_c2 = st.columns([1, 3])
                    with qr_c1:
                        st.image(qr_img_url, caption=f"Scan to Join {selected_poll_c_code}", width=220)
                    with qr_c2:
                        st.markdown(f"### Join Live Poll & Quiz: **{selected_poll_c_code}**")
                        st.markdown(f"1. Scan the QR code with your phone camera, **OR** open:\n   ### `{join_url}`")
                        st.markdown("2. Enter your **Registration Number** and **University Email** once.")
                        st.markdown("3. Keep your phone screen open—questions will advance automatically as the instructor clicks **Next Slide ➡️**.")

                # 2. ADD NEW SLIDE TO SESSION DECK
                session_slides = [
                    p for p in all_course_polls
                    if (p.get("session_title") or "Class Session 1").strip() == selected_session
                ]
                next_slide_num = len(session_slides) + 1
                
                with st.expander("➕ Add New Slide / Question to Session Deck", expanded=(len(all_course_polls) == 0)):
                    sc1, sc2, sc3 = st.columns([2, 1, 2])
                    with sc1:
                        session_input = st.text_input("Session / Lecture Deck Name", value=selected_session).strip()
                    with sc2:
                        slide_no_input = st.number_input("Slide Order #", min_value=1, value=next_slide_num, step=1)
                    with sc3:
                        timer_choice_lbl = st.radio(
                            "Countdown Timer",
                            ["30 sec", "60 sec", "90 sec"],
                            index=0,
                            horizontal=True,
                            key="create_timer_radio"
                        )
                        timer_sec_val = int(timer_choice_lbl.split()[0])
                        
                    p_type_label = st.radio(
                        "Question Type",
                        [
                            "🏆 Graded Quiz (MCQ with Correct Answer & Marks)",
                            "📊 Multiple Choice Poll (No Right/Wrong Answer)",
                            "☁️ Word Cloud (Short 1–3 Word Responses)"
                        ],
                        horizontal=True
                    )
                    
                    q_text = st.text_input(
                        "Question Prompt *",
                        placeholder="e.g., Which primary pollutant is the main precursor to tropospheric ozone in photochemical smog?"
                    ).strip()
                    
                    options_list = []
                    correct_opt = None
                    q_marks = 0.0
                    
                    if "Word Cloud" not in p_type_label:
                        p_type_code = "quiz" if "Graded Quiz" in p_type_label else "mcq"
                        st.markdown("**Enter Multiple Choice Options (Up to 5 options):**")
                        oc1, oc2 = st.columns(2)
                        with oc1:
                            opt1 = st.text_input("Option A", value="NOx and VOCs").strip()
                            opt2 = st.text_input("Option B", value="SO2 and Particulate Matter").strip()
                        with oc2:
                            opt3 = st.text_input("Option C", value="Carbon Monoxide only").strip()
                            opt4 = st.text_input("Option D", value="Methane and Ammonia").strip()
                        opt5 = st.text_input("Option E (Optional)", value="").strip()
                        
                        options_list = [o for o in [opt1, opt2, opt3, opt4, opt5] if o]
                        
                        if p_type_code == "quiz" and options_list:
                            qc1, qc2 = st.columns(2)
                            with qc1:
                                correct_opt = st.selectbox("Select Correct Answer", options_list)
                            with qc2:
                                q_marks = st.number_input("Marks for Correct Answer", min_value=0.5, max_value=20.0, value=1.0, step=0.5)
                    else:
                        p_type_code = "wordcloud"
                        
                    launch_immediately = st.checkbox("🟢 Launch Slide & Start Countdown Immediately", value=False)
                    
                    if st.button("💾 Save Slide to Deck", type="primary"):
                        if not q_text:
                            st.error("Please enter a Question Prompt.")
                        elif p_type_code in ["mcq", "quiz"] and len(options_list) < 2:
                            st.error("Please provide at least 2 options for a Multiple Choice question.")
                        else:
                            if launch_immediately:
                                supabase.table("live_polls").update({"is_active": False}).eq("course_id", poll_course_id).execute()
                                ends_iso = (datetime.now(timezone.utc) + timedelta(seconds=timer_sec_val)).isoformat()
                            else:
                                ends_iso = None
                                
                            supabase.table("live_polls").insert({
                                "course_id": poll_course_id,
                                "session_title": session_input or "Class Session 1",
                                "slide_order": int(slide_no_input),
                                "question_text": q_text,
                                "poll_type": p_type_code,
                                "options_json": options_list,
                                "correct_answer": correct_opt,
                                "marks": q_marks,
                                "is_active": launch_immediately,
                                "show_results": False,
                                "reveal_answer": False,
                                "timer_seconds": timer_sec_val,
                                "timer_ends_at": ends_iso
                            }).execute()
                            st.success(f"✅ Slide #{slide_no_input} added to `{session_input}`!")
                            st.rerun()

                # 3. MENTIMETER SLIDE DECK PRESENTER & CONTROLS
                st.markdown("---")
                if not session_slides:
                    st.info("No slides in this session yet. Expand the box above to add your first slide.")
                else:
                    slide_idx_key = f"slide_idx_{poll_course_id}_{selected_session}"
                    active_in_session = [idx for idx, s in enumerate(session_slides) if s.get("is_active")]
                    if slide_idx_key not in st.session_state:
                        st.session_state[slide_idx_key] = active_in_session[0] if active_in_session else 0
                    if st.session_state[slide_idx_key] >= len(session_slides):
                        st.session_state[slide_idx_key] = len(session_slides) - 1
                        
                    curr_idx = st.session_state[slide_idx_key]
                    
                    nav_c1, nav_c2, nav_c3 = st.columns([1, 3, 1])
                    with nav_c1:
                        if st.button("⬅️ Previous Slide", disabled=(curr_idx <= 0), use_container_width=True):
                            new_idx = curr_idx - 1
                            st.session_state[slide_idx_key] = new_idx
                            target_slide = session_slides[new_idx]
                            activate_slide_with_timer(
                                poll_course_id,
                                target_slide["poll_id"],
                                int(target_slide.get("timer_seconds") or 30)
                            )
                            st.rerun()
                    with nav_c2:
                        slide_labels = [
                            f"Slide {idx+1}/{len(session_slides)}: {'🟢 [LIVE] ' if s.get('is_active') else ''}[{s['poll_type'].upper()}] {s['question_text']}"
                            for idx, s in enumerate(session_slides)
                        ]
                        chosen_label = st.selectbox(
                            "Jump to Slide",
                            slide_labels,
                            index=curr_idx,
                            label_visibility="collapsed"
                        )
                        selected_idx = slide_labels.index(chosen_label)
                        if selected_idx != curr_idx:
                            st.session_state[slide_idx_key] = selected_idx
                            st.rerun()
                    with nav_c3:
                        if st.button("Next Slide ➡️", disabled=(curr_idx >= len(session_slides) - 1), use_container_width=True, type="primary"):
                            new_idx = curr_idx + 1
                            st.session_state[slide_idx_key] = new_idx
                            target_slide = session_slides[new_idx]
                            activate_slide_with_timer(
                                poll_course_id,
                                target_slide["poll_id"],
                                int(target_slide.get("timer_seconds") or 30)
                            )
                            st.rerun()

                    curr_poll = session_slides[st.session_state[slide_idx_key]]
                    curr_poll_id = curr_poll["poll_id"]
                    saved_timer_sec = int(curr_poll.get("timer_seconds") or 30)
                    if saved_timer_sec not in [30, 60, 90]:
                        saved_timer_sec = 30

                    bar1, bar2, bar3, bar4, bar5 = st.columns([1.4, 1.4, 1.4, 1.4, 1])
                    with bar1:
                        live_timer_sel = st.selectbox(
                            "Timer Duration",
                            [30, 60, 90],
                            index=[30, 60, 90].index(saved_timer_sec),
                            format_func=lambda x: f"⏱️ Timer: {x} sec",
                            key=f"timer_sel_{curr_poll_id}",
                            label_visibility="collapsed"
                        )
                    with bar2:
                        if curr_poll.get("is_active"):
                            if st.button("🔴 Stop Voting Now", use_container_width=True):
                                supabase.table("live_polls").update({"is_active": False}).eq("poll_id", curr_poll_id).execute()
                                st.rerun()
                        else:
                            if st.button(f"🟢 Start Voting ({live_timer_sel}s)", use_container_width=True, type="primary"):
                                activate_slide_with_timer(poll_course_id, curr_poll_id, live_timer_sel)
                                st.rerun()
                    with bar3:
                        if curr_poll.get("show_results"):
                            if st.button("🙈 Hide Results", use_container_width=True):
                                supabase.table("live_polls").update({"show_results": False}).eq("poll_id", curr_poll_id).execute()
                                st.rerun()
                        else:
                            if st.button("📊 Show Results", use_container_width=True):
                                supabase.table("live_polls").update({"show_results": True}).eq("poll_id", curr_poll_id).execute()
                                st.rerun()
                    with bar4:
                        if curr_poll["poll_type"] == "quiz":
                            if curr_poll.get("reveal_answer"):
                                if st.button("🙈 Hide Answer", use_container_width=True):
                                    supabase.table("live_polls").update({"reveal_answer": False}).eq("poll_id", curr_poll_id).execute()
                                    st.rerun()
                            else:
                                if st.button("👁️ Reveal Correct Answer", use_container_width=True):
                                    supabase.table("live_polls").update({"reveal_answer": True, "show_results": True}).eq("poll_id", curr_poll_id).execute()
                                    st.rerun()
                        else:
                            if st.button("🔄 Refresh Screen", use_container_width=True):
                                st.rerun()
                    with bar5:
                        if st.button("🗑️ Delete Slide", use_container_width=True):
                            supabase.table("live_polls").delete().eq("poll_id", curr_poll_id).execute()
                            st.rerun()

                    # 4. AUTO-REFRESHING PROJECTOR CANVAS
                    @st.fragment(run_every="2s" if curr_poll.get("is_active") else None)
                    def render_live_projector_canvas(poll_id_arg, total_slides_in_deck, slide_Position):
                        fresh_poll = supabase.table("live_polls").select("*").eq("poll_id", poll_id_arg).single().execute().data
                        resps = supabase.table("poll_responses").select("*").eq("poll_id", poll_id_arg).execute().data or []
                        roster_list = supabase.table("students").select("reg_no, student_name, email").eq("course_id", poll_course_id).order("reg_no").execute().data or []
                        total_enrolled = len(roster_list)
                        
                        rem_s = get_remaining_seconds(fresh_poll)
                        t_total = int(fresh_poll.get("timer_seconds") or 30)
                        
                        with st.container(border=True):
                            st.caption(f"🎯 **{fresh_poll.get('session_title', 'Session')} — Slide {slide_Position} of {total_slides_in_deck}**")
                            st.markdown(f"## 📢 {fresh_poll['question_text']}")
                            
                            pm1, pm2, pm3 = st.columns(3)
                            pm1.metric("🔒 Locked Student Responses", f"{len(resps)} / {total_enrolled if total_enrolled else '?'}")
                            if fresh_poll.get("is_active") and rem_s is not None and rem_s > 0:
                                pm2.metric("⏱️ Countdown Timer", f"{rem_s} sec remaining")
                                st.progress(min(max(rem_s / float(t_total), 0.0), 1.0))
                            else:
                                pm2.metric("⏱️ Voting Status", "🔴 Voting Closed / Time Up" if rem_s == 0 or not fresh_poll.get("is_active") else "🟢 Open")
                            pm3.metric("📊 Results Visibility", "📊 Shown on Projector" if fresh_poll.get("show_results") else "🙈 Hidden Until Clicked")

                            if fresh_poll["poll_type"] in ["mcq", "quiz"]:
                                render_projector_option_cards(
                                    fresh_poll.get("options_json") or [],
                                    correct_answer=fresh_poll.get("correct_answer"),
                                    reveal_answer=bool(fresh_poll.get("reveal_answer"))
                                )

                            st.markdown("---")
                            if not fresh_poll.get("show_results"):
                                st.info(
                                    "🙈 **Live Results Are Hidden:** Students are currently submitting answers independently. "
                                    "Click **📊 Show Results** above when you are ready to display the class distribution on the projector."
                                )
                            else:
                                if fresh_poll["poll_type"] in ["mcq", "quiz"]:
                                    opts = fresh_poll.get("options_json") or []
                                    counts = {o: 0 for o in opts}
                                    for r in resps:
                                        ans = r.get("response_text")
                                        if ans in counts:
                                            counts[ans] += 1
                                        elif ans:
                                            counts[ans] = counts.get(ans, 0) + 1
                                            
                                    chart_df = pd.DataFrame({
                                        "Option": list(counts.keys()),
                                        "Student Votes": list(counts.values())
                                    })
                                    st.bar_chart(chart_df.set_index("Option"), height=300)
                                    
                                    if fresh_poll["poll_type"] == "quiz" and fresh_poll.get("reveal_answer"):
                                        correct_cnt = sum(1 for r in resps if r.get("is_correct"))
                                        st.success(
                                            f"✅ **Correct Answer:** `{fresh_poll.get('correct_answer')}` "
                                            f"| **{correct_cnt} of {len(resps)}** students answered correctly ({fresh_poll.get('marks', 1.0)} Marks)."
                                        )
                                else:
                                    words = [r.get("response_text", "") for r in resps]
                                    render_visual_wordcloud(words)

                    render_live_projector_canvas(curr_poll_id, len(session_slides), st.session_state[slide_idx_key] + 1)

                    # 5. FULL-ROSTER ATTENDANCE & QUESTION-BY-QUESTION EXCEL EXPORT
                    st.markdown("---")
                    st.subheader(f"📥 Full-Roster Attendance & Question-by-Question Report (`{selected_session}`)")
                    
                    roster_students = (
                        supabase.table("students")
                        .select("reg_no, student_name, email")
                        .eq("course_id", poll_course_id)
                        .order("reg_no")
                        .execute()
                        .data or []
                    )
                    session_poll_ids = [s["poll_id"] for s in session_slides]
                    all_session_resps = []
                    if session_poll_ids:
                        all_session_resps = (
                            supabase.table("poll_responses")
                            .select("*")
                            .in_("poll_id", session_poll_ids)
                            .execute()
                            .data or []
                        )
                        
                    resp_lookup = {}
                    for r in all_session_resps:
                        resp_lookup[(str(r["reg_no"]).strip().upper(), r["poll_id"])] = r
                        
                    roster_reg_set = {str(s["reg_no"]).strip().upper() for s in roster_students}
                    combined_students = list(roster_students)
                    for r in all_session_resps:
                        r_reg = str(r["reg_no"]).strip().upper()
                        if r_reg not in roster_reg_set:
                            combined_students.append({
                                "reg_no": r_reg,
                                "student_name": r.get("student_name", ""),
                                "email": ""
                            })
                            roster_reg_set.add(r_reg)
                            
                    summary_rows = []
                    response_matrix_rows = []
                    total_possible_quiz_marks = sum(
                        float(s.get("marks", 0.0)) for s in session_slides if s["poll_type"] == "quiz"
                    )
                    
                    for stu in combined_students:
                        reg = str(stu["reg_no"]).strip().upper()
                        name = stu["student_name"]
                        
                        slides_answered = 0
                        student_quiz_total = 0.0
                        
                        s_row = {
                            "Reg No": reg,
                            "Student Name": name,
                        }
                        m_row = {
                            "Reg No": reg,
                            "Student Name": name,
                        }
                        
                        for idx_s, s_obj in enumerate(session_slides):
                            s_id = s_obj["poll_id"]
                            q_num = f"Q{idx_s + 1}"
                            short_q = s_obj["question_text"][:28] + ("..." if len(s_obj["question_text"]) > 28 else "")
                            r_match = resp_lookup.get((reg, s_id))
                            
                            if r_match:
                                slides_answered += 1
                                ans_text = r_match.get("response_text", "")
                                m_awarded = float(r_match.get("marks_awarded", 0.0))
                            else:
                                ans_text = "—"
                                m_awarded = 0.0
                                
                            if s_obj["poll_type"] == "quiz":
                                col_hdr = f"{q_num} Marks (/{s_obj.get('marks', 1.0)})"
                                s_row[col_hdr] = m_awarded
                                student_quiz_total += m_awarded
                            else:
                                col_hdr = f"{q_num} ({s_obj['poll_type'].upper()})"
                                s_row[col_hdr] = "Voted" if r_match else "Absent"
                                
                            m_row[f"{q_num}: {short_q}"] = ans_text
                            
                        attendance_status = "Present" if slides_answered > 0 else "Absent"
                        s_row["Attendance"] = attendance_status
                        s_row["Slides Answered"] = f"{slides_answered} / {len(session_slides)}"
                        s_row[f"Total Quiz Score (/{total_possible_quiz_marks})"] = round(student_quiz_total, 2)
                        
                        m_row["Attendance"] = attendance_status
                        m_row["Slides Answered"] = f"{slides_answered} / {len(session_slides)}"
                        
                        summary_rows.append(s_row)
                        response_matrix_rows.append(m_row)
                        
                    if summary_rows:
                        df_summary = pd.DataFrame(summary_rows)
                        fixed_first = ["Reg No", "Student Name", "Attendance", "Slides Answered"]
                        other_cols = [c for c in df_summary.columns if c not in fixed_first]
                        df_summary = df_summary[fixed_first + other_cols]
                        
                        df_matrix = pd.DataFrame(response_matrix_rows)
                        fixed_m_first = ["Reg No", "Student Name", "Attendance", "Slides Answered"]
                        other_m_cols = [c for c in df_matrix.columns if c not in fixed_m_first]
                        df_matrix = df_matrix[fixed_m_first + other_m_cols]
                        
                        with st.expander("📋 Preview Full-Roster Attendance & Question-by-Question Table"):
                            st.dataframe(df_summary, use_container_width=True, hide_index=True)
                            
                        session_excel_buf = io.BytesIO()
                        with pd.ExcelWriter(session_excel_buf, engine="openpyxl") as writer:
                            df_summary.to_excel(writer, index=False, sheet_name="Attendance_And_Marks")
                            df_matrix.to_excel(writer, index=False, sheet_name="Detailed_Student_Answers")
                        session_excel_buf.seek(0)
                        
                        clean_sess_fname = selected_session.replace(" ", "_")
                        st.download_button(
                            label=f"📥 Download Full-Roster Attendance & Question-by-Question Excel ({selected_session}.xlsx)",
                            data=session_excel_buf,
                            file_name=f"{selected_poll_c_code}_{clean_sess_fname}_Attendance_And_Quiz_Marks.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            type="primary"
                        )