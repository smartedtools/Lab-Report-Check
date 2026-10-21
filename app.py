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
# HELPER FUNCTIONS: AUTH, DRIVE & GEMINI
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

def build_rubric_from_df(df_rubric: pd.DataFrame):
    """Validates an editable Likert rubric dataframe and computes weight percentages & total marks."""
    valid_rows = []
    for _, row in df_rubric.iterrows():
        crit = str(row.get("criterion", "")).strip()
        marks = row.get("max_marks", 4.0)
        if crit and pd.notna(marks) and float(marks) > 0:
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

def evaluate_report_with_gemini(pdf_bytes, rubric_json, course_title, exp_title, exp_guidelines, total_marks, is_combined=False):
    task_desc = (
        "This is a COMPREHENSIVE or COMBINED report (containing multiple labs/experiments or a full Design Report) compiled into a single PDF. "
        "Locate each individual experiment or design section within the document, evaluate them against the 4-point Likert scale rubric criteria, and provide a structured breakdown for each criterion."
        if is_combined else 
        "Thoroughly inspect the attached student lab report PDF (reading observation tables, formulas, calculations, figures, and discussion) and evaluate against the 4-point Likert scale rubric."
    )

    prompt = f"""
    You are an expert university professor of environmental engineering grading student reports for the course: {course_title}.
    Assessment Title: {exp_title}
    Total Maximum Marks: {total_marks}
    
    Instructor Guidelines & Focus Notes:
    {exp_guidelines if exp_guidelines else "Evaluate based on technical accuracy, valid observation tables, correct calculations, and scientific discussion."}
    
    Here is the 4-Point Likert Scale Analytic Rubric (4 = Excellent, 3 = Proficient, 2 = Developing, 1 = Needs Improvement):
    {json.dumps(rubric_json, indent=2)}
    
    Scoring Instructions:
    1. {task_desc}
    2. Use the exact "criterion" names from the Rubric above.
    3. For each criterion, compare the student's work to the descriptors in `level_4`, `level_3`, `level_2`, and `level_1`.
    4. Assign a `likert_rating` (integer 4, 3, 2, or 1) and the corresponding `rating_label`:
       - "4 - Excellent"
       - "3 - Proficient"
       - "2 - Developing"
       - "1 - Needs Improvement"
    5. Calculate `marks_awarded` as: `(likert_rating / 4.0) * max_marks` (so if `max_marks` is 4.0, `marks_awarded` equals `likert_rating`).
    6. In `feedback`, clearly state WHY the student received that Likert level (referencing specific strengths or missing details in their report) so the student understands their mark.
    7. Ensure `ai_total_score` equals the exact sum of `marks_awarded` across all criteria.
    
    You MUST return your final evaluation strictly in valid JSON format with this exact structure:
    {{
      "criteria_scores": [
        {{
          "criterion": "Exact name of criterion from rubric",
          "likert_rating": 3,
          "rating_label": "3 - Proficient",
          "marks_awarded": 3.0,
          "max_marks": 4.0,
          "feedback": "Specific constructive feedback explaining why this earned 3 (Proficient) and what was missing for 4 (Excellent)."
        }}
      ],
      "ai_total_score": 32.0,
      "overall_feedback": "Summary statement on the student's performance, strengths, and areas of improvement."
    }}
    """
    
    pdf_part = types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf")
    models_to_try = ["gemini-3.8-flash", "gemini-3.5-flash", "gemini-3.8-pro"]
    
    last_error = None
    for model_name in models_to_try:
        for attempt in range(2):
            try:
                response = gemini_client.models.generate_content(
                    model=model_name,
                    contents=[pdf_part, prompt],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                    ),
                )
                return json.loads(response.text)
            except Exception as e:
                last_error = e
                time.sleep(2)
                
    raise last_error

# Ensure default admin exists
ensure_default_admin()

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("🧪 IEER Lab & Design Grader")
portal_mode = st.sidebar.radio("Select Portal", ["Student Submission Portal", "Instructor / Admin Login"])

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
                    
        if 'verified_student' in st.session_state:
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
                
                # Show the Likert Scale Grading Rubric to the student so they know how marks are awarded
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
# PORTAL 2: INSTRUCTOR & ADMIN PORTAL
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

        if is_admin:
            tab_admin, tab1, tab2, tab3, tab4 = st.tabs([
                "🛡️ Manage Teachers (Admin)", 
                "📚 Course & Likert Rubric", 
                "👥 Class Roster Upload", 
                "🔬 Assessments & Custom Rubrics", 
                "📊 Grading & Export"
            ])
        else:
            tab1, tab2, tab3, tab4 = st.tabs([
                "📚 Course & Likert Rubric", 
                "👥 Class Roster Upload", 
                "🔬 Assessments & Custom Rubrics", 
                "📊 Grading & Export"
            ])
            
        # ==========================================
        # ADMIN-ONLY TAB: MANAGE TEACHER ACCOUNTS
        # ==========================================
        if is_admin:
            with tab_admin:
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
        # TAB 1: COURSE SETUP WITH 4-POINT LIKERT RUBRIC MATRIX
        # ==========================================
        with tab1:
            st.subheader("Configure Course & 4-Point Likert Scale Rubric")
            st.markdown(
                "Pre-loaded below is your **10-Criterion Likert Scale Lab Rubric** (`4 – Excellent`, `3 – Proficient`, `2 – Developing`, `1 – Needs Improvement`). "
                "You can edit any cell, add/remove rows, or adjust `Max Marks` per criterion."
            )
            
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
        # TAB 2: CLASS ROSTER UPLOAD
        # ==========================================
        with tab2:
            st.subheader("Upload Class Roster (CSV / Excel)")
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
                roster_res = supabase.table("students").select("reg_no, student_name, email").eq("course_id", selected_course_id).execute()
                if roster_res.data:
                    st.dataframe(pd.DataFrame(roster_res.data), use_container_width=True)
                else:
                    st.info("No students uploaded for this course yet.")
            else:
                st.info("No courses assigned to your account yet. Please create a course in Tab 1.")

        # ==========================================
        # TAB 3: ASSESSMENTS & CUSTOM LIKERT RUBRIC BUILDER
        # ==========================================
        with tab3:
            st.subheader("Manage Experiments, Combined Packets & Design Reports")
            my_courses = get_accessible_courses()
            if my_courses:
                c_dict = {c['course_code']: c for c in my_courses}
                selected_c_code = st.selectbox("Select Course", list(c_dict.keys()), key="exp_course_select")
                course_data = c_dict[selected_c_code]
                
                template_choice = st.radio(
                    "Select Custom Rubric Starting Template (if overriding Course Default):",
                    ["10-Criterion Standard Lab Rubric (40 Marks)", "5-Criterion Engineering Design Rubric (20 Marks)"],
                    horizontal=True
                )
                
                if "10-Criterion" in template_choice:
                    default_custom_rubric_df = pd.DataFrame(DEFAULT_LAB_LIKERT_RUBRIC)
                else:
                    default_custom_rubric_df = pd.DataFrame(DEFAULT_DESIGN_LIKERT_RUBRIC)
                
                with st.form("exp_form"):
                    col_e1, col_e2 = st.columns([1, 3])
                    with col_e1:
                        exp_no = st.number_input("Assessment Number / ID", min_value=1, step=1)
                    with col_e2:
                        exp_title = st.text_input("Assessment Title (e.g., Exp 1: pH & Turbidity or Labs 1-6 Combined)").strip()
                    
                    is_combined = st.checkbox("📦 Is this a Combined Multi-Lab Packet or Design Report?", value=False)
                    is_open = st.checkbox("🔓 Accepting Submissions (Open)", value=True)
                    exp_guidelines = st.text_area("Optional Guidelines for AI Evaluator", placeholder="e.g., Verify mass balance equations and APHA standard compliance.")
                    
                    st.markdown("---")
                    use_custom_rubric = st.checkbox("✨ Use Custom Likert Rubric for this Assessment (Overrides Course Default Rubric)", value=False)
                    st.caption("If checked, the 4-point Likert table below will be used for this specific assessment. If unchecked, the Course Default Rubric will be used.")
                    
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
                    
                    submitted_exp = st.form_submit_button("💾 Create / Update Assessment")
                    if submitted_exp:
                        if not exp_title:
                            st.error("Please enter an Assessment Title.")
                        else:
                            custom_json = None
                            if use_custom_rubric:
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
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error saving assessment: {e}")
            else:
                st.info("Please set up a course first in Tab 1.")

        # ==========================================
        # TAB 4: GRADING & EXPORT (WITH LIKERT BREAKDOWN)
        # ==========================================
        with tab4:
            st.subheader("Batch AI Grading Engine & Likert Scale Gradebook")
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
                    
                    with st.expander(f"📋 View Active 4-Point Likert Rubric for this Assessment (Total: {active_total_marks} Marks)"):
                        st.dataframe(format_rubric_for_display(active_rubric), use_container_width=True, hide_index=True)
                    
                    subs_res = supabase.table("submissions").select("*").eq("exp_id", exp_info['exp_id']).execute()
                    submissions = subs_res.data
                    
                    st.write(f"Total Student Submissions Found: **{len(submissions)}**")
                    
                    if submissions:
                        if st.button("🚀 Run AI Batch Grading for All Submissions", key="btn_run_batch_grading"):
                            progress_bar = st.progress(0)
                            status_box = st.empty()
                            
                            for idx, sub in enumerate(submissions):
                                status_box.info(f"⏳ Processing ({idx+1}/{len(submissions)}): {sub['student_name']} ({sub['reg_no']})...")
                                try:
                                    pdf_bytes = download_file_bytes_from_drive(sub[COL_DRIVE_FILE])
                                    
                                    eval_result = evaluate_report_with_gemini(
                                        pdf_bytes=pdf_bytes,
                                        rubric_json=active_rubric,
                                        course_title=course_info['course_title'],
                                        exp_title=exp_info['exp_title'],
                                        exp_guidelines=exp_info['exp_guidelines'],
                                        total_marks=active_total_marks,
                                        is_combined=exp_info.get('is_combined', False)
                                    )
                                    
                                    supabase.table("evaluations").upsert({
                                        "sub_id": sub['sub_id'],
                                        "criteria_scores_json": eval_result['criteria_scores'],
                                        "ai_total_score": eval_result['ai_total_score'],
                                        "final_total_score": eval_result['ai_total_score'],
                                        "overall_feedback": eval_result['overall_feedback']
                                    }, on_conflict="sub_id").execute()
                                    
                                    supabase.table("submissions").update({"status": "graded"}).eq("sub_id", sub['sub_id']).execute()
                                    st.toast(f"✅ Graded {sub['reg_no']}: {eval_result['ai_total_score']}/{active_total_marks}")
                                    
                                except Exception as e:
                                    st.error(f"❌ Error grading {sub['reg_no']}: {e}")
                                
                                progress_bar.progress((idx + 1) / len(submissions))
                                
                            status_box.success("✅ Batch grading finished successfully!")
                            st.rerun()
                            
                        st.markdown("---")
                        st.subheader("Graded Results & Criterion-Wise Likert Review")
                        
                        sub_ids = [s['sub_id'] for s in submissions]
                        evals_query = supabase.table("evaluations").select("*, submissions(reg_no, student_name, file_name, status, sub_id)").in_("sub_id", sub_ids).execute()
                        
                        if evals_query.data:
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
                            
                            for ev in evals_query.data:
                                if ev.get('submissions'):
                                    reg_no_val = ev['submissions']['reg_no']
                                    student_name_val = ev['submissions']['student_name']
                                    
                                    row_dict = {
                                        "sub_id": ev['sub_id'],
                                        "Reg No": reg_no_val,
                                        "Student Name": student_name_val,
                                    }
                                    for col_label in ordered_criterion_cols:
                                        row_dict[col_label] = 0.0
                                        
                                    crit_list = ev.get('criteria_scores_json') or []
                                    for idx_c, c_obj in enumerate(crit_list):
                                        raw_c_name = str(c_obj.get('criterion', f'Criterion {idx_c+1}')).strip()
                                        awarded = float(c_obj.get('marks_awarded', 0.0))
                                        max_m = float(c_obj.get('max_marks', 4.0))
                                        likert_val = c_obj.get('likert_rating', int(round((awarded / max_m) * 4)) if max_m > 0 else 1)
                                        r_label = str(c_obj.get('rating_label', f"{likert_val} / 4"))
                                        c_fb = str(c_obj.get('feedback', ''))
                                        
                                        # Find matched rubric descriptor text for student clarity
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
                                            "Rubric Descriptor Matched": matched_desc,
                                            "Specific AI Feedback": c_fb
                                        })
                                        
                                    row_dict["AI Score"] = float(ev['ai_total_score'])
                                    row_dict["Final Score"] = float(ev['final_total_score'])
                                    row_dict["Feedback"] = ev['overall_feedback']
                                    eval_df_data.append(row_dict)
                                    
                            if eval_df_data:
                                all_criterion_cols = ordered_criterion_cols + extra_criterion_cols
                                ordered_df_cols = ["sub_id", "Reg No", "Student Name"] + all_criterion_cols + ["AI Score", "Final Score", "Feedback"]
                                df_evals = pd.DataFrame(eval_df_data)
                                for col in ordered_df_cols:
                                    if col not in df_evals.columns:
                                        df_evals[col] = 0.0
                                df_evals = df_evals[ordered_df_cols]
                                
                                st.markdown("### Edit Final Scores & Review Criterion Breakdown")
                                col_cfg = {
                                    "sub_id": None,
                                    "Reg No": st.column_config.TextColumn(disabled=True),
                                    "Student Name": st.column_config.TextColumn(disabled=True),
                                    "AI Score": st.column_config.NumberColumn(disabled=True),
                                    "Final Score": st.column_config.NumberColumn(
                                        "Final Score (Editable)",
                                        min_value=0.0,
                                        max_value=float(active_total_marks),
                                        step=0.5
                                    ),
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
                                st.markdown("### Export Master Gradebook & Likert Feedback (.xlsx)")
                                
                                excel_buffer = io.BytesIO()
                                export_cols = ["Reg No", "Student Name"] + all_criterion_cols + ["AI Score", "Final Score", "Feedback"]
                                df_export = edited_df[export_cols]
                                
                                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                                    sheet_1_name = f"{selected_c_code}_Assmt{exp_info['exp_number']}"[:31]
                                    df_export.to_excel(writer, index=False, sheet_name=sheet_1_name)
                                    
                                    if detailed_feedback_rows:
                                        df_detailed = pd.DataFrame(detailed_feedback_rows)
                                        df_detailed.to_excel(writer, index=False, sheet_name="Detailed_Likert_Rubric")
                                        
                                excel_buffer.seek(0)
                                
                                st.download_button(
                                    label="📥 Download Master Excel Grade Sheet & Likert Breakdown (.xlsx)",
                                    data=excel_buffer,
                                    file_name=f"{selected_c_code}_Assessment_{exp_info['exp_number']}_Likert_Grades.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
                    else:
                        st.info("No student submissions found for this assessment yet.")
                else:
                    st.info("No assessments created for this course yet.")
            else:
                st.info("Please set up a course first in Tab 1.")