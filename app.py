import os
import random
import re
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.express as px
from openai import OpenAI
from pypdf import PdfReader
from docx import Document

# ------------------------------------------------------------
# Nexus Unified API configuration
# Set NEXUS_API_KEY in Render's Environment settings.
# Never put the key directly in this file.
# ------------------------------------------------------------
API_KEY = os.getenv("NEXUS_API_KEY", "").strip()
client = OpenAI(
    api_key=API_KEY or "missing-api-key",
    base_url="https://nexusapi.navigatelabs.ai"
)
CHAT_MODEL = "nova-micro"
EMBED_MODEL = "text-embedding-3-small"

REQUIRED_COLUMNS = [
    "Complaint_ID", "Department", "Location",
    "Complaint", "Status", "Complaint_Date"
]
UNRESOLVED_STATUSES = ["Pending", "In Progress", "Reopened"]

# ------------------------------------------------------------
# Re-create the same deterministic 1,002-record demo dataset
# used in the Colab notebook.
# ------------------------------------------------------------
def create_default_dataset():
    random.seed(42)
    np.random.seed(42)

    complaint_types = {
        "Water": [
            "Irregular water supply in residential areas",
            "Drinking water contamination reported",
            "Water pipeline leakage remains unresolved",
            "Low water pressure in several streets",
            "Water tanker delivery is frequently delayed"
        ],
        "Electricity": [
            "Frequent power cuts in residential areas",
            "Damaged electric poles require repair",
            "Streetlights are not functioning",
            "Electricity voltage fluctuations reported",
            "Delayed resolution of power complaints"
        ],
        "Transport": [
            "Buses frequently arrive late",
            "Insufficient buses on busy routes",
            "Damaged roads causing traffic problems",
            "Bus stops lack proper facilities",
            "Poor maintenance of public transport"
        ],
        "Healthcare": [
            "Shortage of medicines in government hospital",
            "Long waiting time at government hospitals",
            "Insufficient medical staff at health centres",
            "Poor sanitation in hospital premises",
            "Delayed medical services for patients"
        ],
        "Sanitation": [
            "Garbage is not collected regularly",
            "Overflowing public dustbins",
            "Blocked drainage causing water stagnation",
            "Unclean public spaces",
            "Irregular waste disposal services"
        ],
        "Roads": [
            "Potholes on major roads",
            "Damaged roads require immediate repair",
            "Poor road maintenance in residential areas",
            "Road construction work remains incomplete",
            "Missing road safety signs"
        ],
        "Education": [
            "Insufficient classroom facilities",
            "Shortage of teachers in government schools",
            "Poor maintenance of school buildings",
            "Lack of drinking water in schools",
            "Inadequate sanitation facilities in schools"
        ],
        "Public Safety": [
            "Non-functional streetlights create safety concerns",
            "Damaged public safety barriers",
            "Unsafe pedestrian crossings",
            "Lack of traffic signals at busy junctions",
            "Delayed response to public safety complaints"
        ],
        "Revenue": [
            "Delay in processing government certificates",
            "Property tax payment records contain errors",
            "Long waiting time at public service offices",
            "Delay in issuing income certificates",
            "Pending applications at revenue offices"
        ],
        "Housing": [
            "Delay in processing housing scheme applications",
            "Poor maintenance of public housing",
            "Water leakage in government housing units",
            "Incomplete housing repair work",
            "Delay in housing benefit approval"
        ]
    }

    locations = [
        "Coimbatore", "Chennai", "Madurai", "Salem", "Erode",
        "Tiruppur", "Trichy", "Tirunelveli", "Vellore", "Thanjavur",
        "Dindigul", "Namakkal", "Karur", "Kanchipuram", "Theni",
        "Thoothukudi", "Krishnagiri", "Cuddalore", "Sivaganga", "Nilgiris"
    ]
    statuses = ["Pending", "Resolved", "Reopened", "In Progress"]
    weights = [0.35, 0.40, 0.10, 0.15]

    records = []
    for i in range(1000):
        dept = random.choice(list(complaint_types.keys()))
        records.append({
            "Complaint_ID": f"GRV{i+1:04d}",
            "Department": dept,
            "Location": random.choice(locations),
            "Complaint": random.choice(complaint_types[dept]),
            "Status": random.choices(statuses, weights=weights, k=1)[0],
            "Complaint_Date": pd.Timestamp("2025-01-01")
                + pd.Timedelta(days=random.randint(0, 320))
        })

    generated = pd.DataFrame(records)
    original_cases = pd.DataFrame([
        {
            "Complaint_ID": "G001",
            "Department": "Water",
            "Location": "Coimbatore",
            "Complaint": "Irregular water supply for several days",
            "Status": "Pending",
            "Complaint_Date": pd.Timestamp("2025-11-17")
        },
        {
            "Complaint_ID": "G006",
            "Department": "Transport",
            "Location": "Coimbatore",
            "Complaint": "Buses frequently arrive late",
            "Status": "Resolved",
            "Complaint_Date": pd.Timestamp("2025-11-16")
        }
    ])
    return pd.concat([original_cases, generated], ignore_index=True)

default_df = create_default_dataset()
df = default_df.copy()

# ------------------------------------------------------------
# Lightweight hosted embeddings: Nexus embedding endpoint +
# NumPy cosine similarity. This avoids loading a 90 MB local
# transformer and FAISS on Render's small free instance.
# ------------------------------------------------------------
def require_api_key():
    if not API_KEY:
        raise RuntimeError(
            "NEXUS_API_KEY is not configured. Add it under "
            "Render > Environment > Environment Variables."
        )

def embed_texts(texts, batch_size=48):
    require_api_key()
    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = [str(x)[:6000] for x in texts[start:start + batch_size]]
        result = client.embeddings.create(
            model=EMBED_MODEL,
            input=batch
        )
        ordered = sorted(result.data, key=lambda item: item.index)
        vectors.extend([item.embedding for item in ordered])
    arr = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, 1e-12)

complaint_vectors = None
document_chunks = []
document_vectors = None
embedding_error = None

def rebuild_complaint_index():
    global complaint_vectors, embedding_error
    try:
        texts = df["Complaint"].fillna("").astype(str).tolist()
        complaint_vectors = embed_texts(texts)
        embedding_error = None
        return f"Semantic index ready for {len(df)} complaint records."
    except Exception as exc:
        complaint_vectors = None
        embedding_error = str(exc)
        return f"Dataset loaded, but embedding setup failed: {exc}"

def retrieve_complaints(question, top_k=3):
    if complaint_vectors is None:
        raise RuntimeError(
            embedding_error or "The complaint semantic index is not ready."
        )
    query_vector = embed_texts([question])[0]
    scores = complaint_vectors @ query_vector
    top_ids = np.argsort(scores)[::-1][:min(top_k, len(df))]
    return df.iloc[top_ids].copy()

# Build initial semantic index if the secret is configured.
index_status = rebuild_complaint_index() if API_KEY else (
    "Add NEXUS_API_KEY in Render to enable semantic retrieval and AI."
)

# ------------------------------------------------------------
# Dataset validation and document extraction
# ------------------------------------------------------------
def prepare_uploaded_dataset(uploaded_df):
    data = uploaded_df.copy()
    data.columns = [
        str(col).strip().lower().replace(" ", "_")
        for col in data.columns
    ]
    aliases = {
        "complaint_id": "Complaint_ID",
        "department": "Department",
        "location": "Location",
        "complaint": "Complaint",
        "status": "Status",
        "complaint_date": "Complaint_Date"
    }
    data = data.rename(columns=aliases)

    required = ["Complaint_ID", "Department", "Location", "Complaint", "Status"]
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    if "Complaint_Date" not in data.columns:
        data["Complaint_Date"] = ""

    data = data[REQUIRED_COLUMNS].copy()
    for col in ["Complaint_ID", "Department", "Location", "Complaint", "Status"]:
        data[col] = data[col].fillna("").astype(str).str.strip()

    data = data[
        (data["Complaint"] != "")
        & (data["Department"] != "")
        & (data["Location"] != "")
    ].reset_index(drop=True)

    if data.empty:
        raise ValueError("No valid complaint rows were found.")
    return data

def extract_document_text(file_path):
    suffix = Path(file_path).suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(file_path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if suffix == ".docx":
        doc = Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    if suffix == ".txt":
        return Path(file_path).read_text(encoding="utf-8", errors="ignore")
    raise ValueError("Supported documents: PDF, DOCX and TXT.")

def split_into_chunks(text, chunk_size=900, overlap=120):
    clean = re.sub(r"\s+", " ", text).strip()
    if not clean:
        return []
    chunks = []
    start = 0
    while start < len(clean):
        end = min(start + chunk_size, len(clean))
        chunks.append(clean[start:end])
        if end == len(clean):
            break
        start = max(end - overlap, start + 1)
    return chunks

def upload_dataset(file_path):
    global df
    if not file_path:
        return "Choose a CSV or Excel file first.", *dashboard_values()
    suffix = Path(file_path).suffix.lower()
    if suffix == ".csv":
        raw = pd.read_csv(file_path)
    elif suffix in [".xlsx", ".xls"]:
        raw = pd.read_excel(file_path)
    else:
        return "For complaint datasets, upload CSV or Excel.", *dashboard_values()

    try:
        new_df = prepare_uploaded_dataset(raw)
        df = new_df
        status = rebuild_complaint_index()
        return (
            f"Uploaded dataset activated. Records: {len(df)}. {status}",
            *dashboard_values()
        )
    except Exception as exc:
        return f"Upload failed: {exc}", *dashboard_values()

def upload_document(file_path):
    global document_chunks, document_vectors
    if not file_path:
        return "Choose a PDF, DOCX or TXT document first."
    try:
        text = extract_document_text(file_path)
        chunks = split_into_chunks(text)
        if not chunks:
            return "No readable text was found in this document."
        vectors = embed_texts(chunks)
        document_chunks = chunks
        document_vectors = vectors
        return (
            f"Document indexed successfully: {len(chunks)} text sections. "
            "The document will be used as additional RAG evidence."
        )
    except Exception as exc:
        return f"Document processing failed: {exc}"

def restore_default_dataset():
    global df
    df = default_df.copy()
    status = rebuild_complaint_index()
    return f"Original default dataset restored ({len(df)} records). {status}", *dashboard_values()

# ------------------------------------------------------------
# Analytics and AI functions
# ------------------------------------------------------------
def prioritisation_agent():
    rows = []
    for dept, group in df.groupby("Department"):
        unresolved = int(group["Status"].isin(UNRESOLVED_STATUSES).sum())
        rows.append({
            "Department": dept,
            "Unresolved": unresolved,
            "Priority": "High" if unresolved >= 2 else "Low"
        })
    return pd.DataFrame(rows)

def pattern_statistics():
    return pd.crosstab(df["Department"], df["Status"])

def get_document_evidence(question, top_k=3):
    if document_vectors is None or not document_chunks:
        return []
    qvec = embed_texts([question])[0]
    scores = document_vectors @ qvec
    ids = np.argsort(scores)[::-1][:min(top_k, len(document_chunks))]
    return [document_chunks[i] for i in ids]

def call_nova(system_prompt, user_prompt):
    require_api_key()
    result = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2
    )
    return result.choices[0].message.content

def grievance_agent(question):
    if not question or not question.strip():
        return "Please enter a question."
    try:
        location = next(
            (loc for loc in df["Location"].dropna().unique()
             if str(loc).lower() in question.lower()),
            None
        )
        if location and any(term in question.lower() for term in [
            "unresolved", "pending", "reopened", "not resolved", "remain open"
        ]):
            evidence = df[
                (df["Location"] == location)
                & (df["Status"].isin(UNRESOLVED_STATUSES))
            ]
        else:
            evidence = retrieve_complaints(question)

        doc_evidence = get_document_evidence(question)
        prompt = (
            f"Question: {question}\n\n"
            f"Complaint records:\n{evidence.to_string(index=False)}\n\n"
            f"Additional uploaded document evidence:\n"
            + ("\n---\n".join(doc_evidence) if doc_evidence else "No document uploaded.")
        )
        return call_nova(
            "You are a public grievance analysis assistant. "
            "Answer only from the supplied complaint records and document evidence. "
            "Mention complaint IDs when available. Clearly say when evidence is insufficient. "
            "Do not invent facts. The default dataset is simulated academic data.",
            prompt
        )
    except Exception as exc:
        return f"Investigation could not be completed: {exc}"

def generate_pattern_insights():
    try:
        stats = pattern_statistics()
        return call_nova(
            "Analyse only the supplied simulated public grievance statistics. "
            "Identify departments with higher unresolved counts and possible patterns. "
            "Suggest practical actions. Do not invent facts or imply the records are official.",
            stats.to_string()
        )
    except Exception as exc:
        return f"AI pattern analysis is unavailable: {exc}"

def dashboard_values():
    total = len(df)
    pending = int((df["Status"] == "Pending").sum())
    resolved = int((df["Status"] == "Resolved").sum())
    reopened = int((df["Status"] == "Reopened").sum())
    in_progress = int((df["Status"] == "In Progress").sum())

    status_chart = px.pie(
        df, names="Status", title="Complaint Status Distribution", hole=0.48
    )
    counts = df["Department"].value_counts().rename_axis("Department").reset_index(name="Complaints")
    department_chart = px.bar(
        counts, x="Department", y="Complaints",
        title="Department-wise Complaints"
    )
    return (
        df.head(20),
        df,
        pattern_statistics(),
        prioritisation_agent(),
        total, pending, resolved, reopened, in_progress,
        status_chart, department_chart
    )

# ------------------------------------------------------------
# Gradio dashboard
# ------------------------------------------------------------
initial = dashboard_values()
initial_insights = generate_pattern_insights() if API_KEY else (
    "Configure NEXUS_API_KEY in Render to generate AI pattern insights."
)

with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="blue"),
    title="AI-Driven Public Grievance Intelligence"
) as demo:
    gr.Markdown(
        "# 🏛️ AI-Driven Public Grievance Intelligence\n"
        "### Smart monitoring • Pattern discovery • AI investigation"
    )

    with gr.Tab("📊 Overview"):
        with gr.Row():
            total_box = gr.Number(value=initial[4], label="Total Complaints")
            pending_box = gr.Number(value=initial[5], label="Pending")
            resolved_box = gr.Number(value=initial[6], label="Resolved")
            reopened_box = gr.Number(value=initial[7], label="Reopened")
            progress_box = gr.Number(value=initial[8], label="In Progress")
        with gr.Row():
            status_plot = gr.Plot(value=initial[9])
            department_plot = gr.Plot(value=initial[10])

    with gr.Tab("🔎 Complaint Explorer"):
        with gr.Row():
            dept_filter = gr.Dropdown(
                ["All"] + sorted(df["Department"].unique().tolist()),
                value="All", label="Department"
            )
            loc_filter = gr.Dropdown(
                ["All"] + sorted(df["Location"].unique().tolist()),
                value="All", label="Location"
            )
            status_filter = gr.Dropdown(
                ["All"] + sorted(df["Status"].unique().tolist()),
                value="All", label="Status"
            )
        complaint_table = gr.Dataframe(value=df, interactive=False)

        def filter_records(dept, loc, status):
            result = df.copy()
            if dept != "All":
                result = result[result["Department"] == dept]
            if loc != "All":
                result = result[result["Location"] == loc]
            if status != "All":
                result = result[result["Status"] == status]
            return result

        for control in [dept_filter, loc_filter, status_filter]:
            control.change(
                filter_records,
                inputs=[dept_filter, loc_filter, status_filter],
                outputs=complaint_table
            )

    with gr.Tab("🧠 Pattern Discovery"):
        gr.Markdown("## AI-Discovered Complaint Patterns")
        insights_box = gr.Markdown(value=initial_insights)
        refresh_insights = gr.Button("Refresh AI Pattern Insights")
        refresh_insights.click(generate_pattern_insights, outputs=insights_box)
        gr.Markdown("### Department-wise Complaint Statistics")
        pattern_box = gr.Dataframe(value=initial[2], interactive=False)

    with gr.Tab("⚠️ Priority Analysis"):
        gr.Markdown("## Department Priority Assessment")
        priority_box = gr.Dataframe(value=initial[3], interactive=False)
        gr.Markdown(
            "_Priority threshold is an experimental prototype rule, not an official government classification._"
        )

    with gr.Tab("🤖 AI Investigation"):
        gr.Markdown("## Ask the Grievance Intelligence Agent")
        question = gr.Textbox(
            label="Enter your question",
            placeholder="Which grievances remain unresolved in Coimbatore?"
        )
        investigate = gr.Button("🔍 Investigate", variant="primary")
        answer = gr.Markdown()
        investigate.click(grievance_agent, inputs=question, outputs=answer)

    with gr.Tab("📂 Dataset Management"):
        gr.Markdown(
            "## Manage Complaint Data\n"
            "The original 1,002-record simulated dataset loads by default. "
            "A valid uploaded CSV/Excel dataset replaces it for the current app session."
        )
        dataset_file = gr.File(
            label="Upload complaint dataset",
            file_types=[".csv", ".xlsx", ".xls"],
            type="filepath"
        )
        upload_button = gr.Button("📤 Upload and Analyse", variant="primary")
        restore_button = gr.Button("↩️ Restore Default Dataset")
        dataset_status = gr.Markdown(value=index_status)
        preview = gr.Dataframe(value=df.head(20), interactive=False)

        dataset_outputs = [
            dataset_status, preview, complaint_table, pattern_box, priority_box,
            total_box, pending_box, resolved_box, reopened_box, progress_box,
            status_plot, department_plot
        ]
        upload_button.click(
            upload_dataset, inputs=dataset_file, outputs=dataset_outputs
        )
        restore_button.click(
            restore_default_dataset, outputs=dataset_outputs
        )

        gr.Markdown("---\n### Add a complaint document as extra RAG evidence")
        gr.Markdown("Supported formats: PDF, DOCX and TXT. Document text is indexed separately and does not replace the structured complaint dataset.")
        document_file = gr.File(
            label="Upload supporting document",
            file_types=[".pdf", ".docx", ".txt"],
            type="filepath"
        )
        document_button = gr.Button("📄 Process Document")
        document_status = gr.Markdown()
        document_button.click(
            upload_document, inputs=document_file, outputs=document_status
        )

    gr.Markdown(
        "---\n"
        "**Default dataset:** 1,002 simulated public grievance records. "
        "This academic prototype is not connected to an official government database. "
        "Uploaded data is held in application memory for the current running session."
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "10000"))
    )
