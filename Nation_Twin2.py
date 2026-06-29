import streamlit as st
from groq import Groq
from pypdf import PdfReader
import sqlite3
import os
import uuid
import json
import base64
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from duckduckgo_search import DDGS
from sentence_transformers import SentenceTransformer
import chromadb

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Nation Twin",
    page_icon="🇳🇬",
    layout="wide"
)

# ---------------- API Key (safe — loaded from Streamlit Secrets) ----------------
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except (KeyError, FileNotFoundError):
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

if not GROQ_API_KEY:
    st.error(
        "⚠️ Groq API key not found. "
        "Please add GROQ_API_KEY to your Streamlit Secrets or environment variables."
    )
    st.stop()

groq_client = Groq(api_key=GROQ_API_KEY)


# ---------------- Founder / Identity System Prompt ----------------
FOUNDER_SYSTEM_PROMPT = """
You are Nation Twin, Nigeria's AI Digital Twin.
If a user asks any of the following (or close variations):
- Who founded Nation Twin?
- Who created Nation Twin?
- Who developed Nation Twin?
- Who owns Nation Twin?
- Tell me about the founder.

Answer:
Nation Twin was founded by Chrisantus Madueke Ebubechukwu, born on 18 May 2006.
He is a Nigerian from Ideato South Local Government Area of Imo State.
Chrisantus is a medical student studying Medicine and Surgery (MBBS) at the
Federal University of Health Sciences, Otukpo (FUHSO). He is also an AI
engineer, software developer, and technology innovator dedicated to building
intelligent solutions for Africa.
He created Nation Twin to serve as Nigeria's AI-powered digital twin—an
intelligent assistant designed to provide accurate, accessible, and insightful
information about Nigeria while promoting innovation, education, and digital
transformation.

Always answer confidently and professionally whenever questions about the
founder are asked.
"""

# ---------------- Constants ----------------
DB_FILE = "nation_twin.db"
CHUNK_SIZE = 800       # characters per chunk
CHUNK_OVERLAP = 150    # overlap between chunks
TOP_K_CHUNKS = 4       # how many chunks to retrieve per question
MODEL_NAME = "llama-3.3-70b-versatile"   # Free, fast, powerful on Groq
SHARED_DOC_TAG = "__shared__"  # metadata tag for documents visible to every user

# ---------------- Database Setup ----------------
def get_connection():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def _ensure_column(cur, table, column, col_type):
    """Lightweight migration helper: add a column if it doesn't already exist."""
    cur.execute(f"PRAGMA table_info({table})")
    existing_cols = [row[1] for row in cur.fetchall()]
    if column not in existing_cols:
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")

def init_db():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            title TEXT,
            created_at TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            content TEXT,
            agent TEXT,
            sources TEXT,
            image_b64 TEXT,
            created_at TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            fact TEXT,
            created_at TEXT
        )
    """)
    _ensure_column(cur, "sessions", "user_id", "TEXT")
    _ensure_column(cur, "facts", "user_id", "TEXT")
    _ensure_column(cur, "messages", "image_b64", "TEXT")
    conn.commit()
    conn.close()

def create_session(user_id, title="New Conversation"):
    session_id = str(uuid.uuid4())
    conn = get_connection()
    conn.execute(
        "INSERT INTO sessions (id, user_id, title, created_at) VALUES (?, ?, ?, ?)",
        (session_id, user_id, title, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()
    return session_id

def get_all_sessions(user_id):
    conn = get_connection()
    rows = conn.execute(
        "SELECT id, title, created_at FROM sessions WHERE user_id = ? ORDER BY created_at DESC",
        (user_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def rename_session_if_default(session_id, first_user_message):
    conn = get_connection()
    row = conn.execute(
        "SELECT title FROM sessions WHERE id = ?", (session_id,)
    ).fetchone()
    if row and row["title"] == "New Conversation":
        new_title = first_user_message.strip()[:50]
        if len(first_user_message.strip()) > 50:
            new_title += "..."
        conn.execute(
            "UPDATE sessions SET title = ? WHERE id = ?",
            (new_title, session_id)
        )
        conn.commit()
    conn.close()

def delete_session(user_id, session_id):
    conn = get_connection()
    conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
    conn.execute(
        "DELETE FROM sessions WHERE id = ? AND user_id = ?",
        (session_id, user_id)
    )
    conn.commit()
    conn.close()

def save_message(session_id, role, content, agent=None, sources=None, image_b64=None):
    conn = get_connection()
    conn.execute(
        "INSERT INTO messages (session_id, role, content, agent, sources, image_b64, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            session_id, role, content, agent,
            json.dumps(sources) if sources else None,
            image_b64,
            datetime.now().isoformat()
        )
    )
    conn.commit()
    conn.close()

def load_messages(session_id):
    conn = get_connection()
    rows = conn.execute(
        "SELECT role, content, agent, sources, image_b64 FROM messages "
        "WHERE session_id = ? ORDER BY id ASC",
        (session_id,)
    ).fetchall()
    conn.close()
    messages = []
    for r in rows:
        msg = {"role": r["role"], "content": r["content"]}
        if r["agent"]:
            msg["agent"] = r["agent"]
        if r["sources"]:
            msg["sources"] = json.loads(r["sources"])
        if r["image_b64"]:
            msg["image_b64"] = r["image_b64"]
        messages.append(msg)
    return messages

def count_user_questions(session_id):
    conn = get_connection()
    row = conn.execute(
        "SELECT COUNT(*) as c FROM messages WHERE session_id = ? AND role = 'user'",
        (session_id,)
    ).fetchone()
    conn.close()
    return row["c"] if row else 0

# ---------------- Persistent Facts ----------------
def save_fact(user_id, fact_text):
    conn = get_connection()
    conn.execute(
        "INSERT INTO facts (user_id, fact, created_at) VALUES (?, ?, ?)",
        (user_id, fact_text.strip(), datetime.now().isoformat())
    )
    conn.commit()
    conn.close()

def get_all_facts(user_id):
    conn = get_connection()
    rows = conn.execute(
        "SELECT id, fact FROM facts WHERE user_id = ? ORDER BY created_at DESC",
        (user_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def delete_fact(user_id, fact_id):
    conn = get_connection()
    conn.execute(
        "DELETE FROM facts WHERE id = ? AND user_id = ?",
        (fact_id, user_id)
    )
    conn.commit()
    conn.close()

init_db()

# ---------------- Lightweight User Identity ----------------
def get_or_create_user_id():
    try:
        existing = st.query_params.get("uid")
    except Exception:
        existing = None
    if existing:
        return existing
    new_id = uuid.uuid4().hex[:12]
    try:
        st.query_params["uid"] = new_id
    except Exception:
        pass
    return new_id

if "user_id" not in st.session_state:
    st.session_state.user_id = get_or_create_user_id()

USER_ID = st.session_state.user_id

# ---------------- Session State ----------------
if "current_session_id" not in st.session_state:
    sessions = get_all_sessions(USER_ID)
    if sessions:
        st.session_state.current_session_id = sessions[0]["id"]
    else:
        st.session_state.current_session_id = create_session(USER_ID)

if "messages" not in st.session_state:
    st.session_state.messages = load_messages(st.session_state.current_session_id)

if "questions_count" not in st.session_state:
    st.session_state.questions_count = count_user_questions(st.session_state.current_session_id)

if "documents" not in st.session_state:
    st.session_state.documents = []

# ---------------- Embedding + Vector Store Setup ----------------
@st.cache_resource
def get_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = get_embedding_model()

class MiniLMEmbeddingFunction:
    def __call__(self, input):
        return embedding_model.encode(input).tolist()

    def name(self):
        return "minilm-all-MiniLM-L6-v2"

@st.cache_resource
def get_chroma_collection():
    chroma_client = chromadb.PersistentClient(path="./nation_twin_vectors")
    try:
        return chroma_client.get_or_create_collection(
            name="documents",
            embedding_function=MiniLMEmbeddingFunction()
        )
    except Exception:
        try:
            chroma_client.delete_collection(name="documents")
        except Exception:
            pass
        return chroma_client.create_collection(
            name="documents",
            embedding_function=MiniLMEmbeddingFunction()
        )

collection = get_chroma_collection()

if "documents_loaded_from_disk" not in st.session_state:
    try:
        existing = collection.get(
            where={"user_id": {"$in": [USER_ID, SHARED_DOC_TAG]}}
        )
        seen_names = {}
        for meta in existing.get("metadatas", []):
            if meta and "doc_name" in meta:
                seen_names[meta["doc_name"]] = seen_names.get(meta["doc_name"], 0) + 1
        for name, count in seen_names.items():
            if not any(d["name"] == name for d in st.session_state.documents):
                st.session_state.documents.append({
                    "name": name,
                    "chunks": [{} for _ in range(count)],
                })
    except Exception:
        pass
    st.session_state.documents_loaded_from_disk = True

# ---------------- Seed Documents ----------------
SEED_FOLDER = "seed_documents"

def index_seed_documents():
    if not os.path.isdir(SEED_FOLDER):
        return

    already_seeded_names = {d["name"] for d in st.session_state.documents}

    for filename in os.listdir(SEED_FOLDER):
        if not filename.lower().endswith(".pdf"):
            continue
        seed_name = f"seed:{filename}"
        if seed_name in already_seeded_names:
            continue

        filepath = os.path.join(SEED_FOLDER, filename)
        try:
            with open(filepath, "rb") as f:
                doc_chunks, had_text = extract_pdf_chunks(f)
        except Exception:
            continue

        if not had_text:
            continue

        for i, chunk in enumerate(doc_chunks):
            chunk_id = f"{seed_name}::{i}::{uuid.uuid4().hex[:8]}"
            try:
                collection.add(
                    ids=[chunk_id],
                    documents=[chunk["text"]],
                    metadatas=[{
                        "page": chunk["page"],
                        "doc_name": seed_name,
                        "user_id": SHARED_DOC_TAG,
                    }]
                )
            except Exception:
                pass

        st.session_state.documents.append({"name": seed_name, "chunks": doc_chunks})

# ---------------- Text Processing Helpers ----------------
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunks.append(text[start:end])
        if end == text_len:
            break
        start = end - overlap
    return chunks

def extract_pdf_chunks(uploaded_file):
    reader = PdfReader(uploaded_file)
    doc_chunks = []
    had_text = False

    for page_num, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        page_text = page_text.strip()
        if page_text:
            had_text = True
            for piece in chunk_text(page_text):
                doc_chunks.append({"text": piece, "page": page_num})

    return doc_chunks, had_text

if "seed_docs_loaded" not in st.session_state:
    index_seed_documents()
    st.session_state.seed_docs_loaded = True

def retrieve_relevant_chunks(query, user_id, top_k=TOP_K_CHUNKS):
    if collection.count() == 0:
        return []

    try:
        results = collection.query(
            query_texts=[query],
            n_results=min(top_k, collection.count()),
            where={"user_id": {"$in": [user_id, SHARED_DOC_TAG]}}
        )
    except Exception:
        results = collection.query(
            query_texts=[query],
            n_results=min(top_k, collection.count())
        )

    chunks = []
    ids = results["ids"][0]
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results.get("distances", [[0] * len(ids)])[0]

    for i in range(len(ids)):
        distance = distances[i] if i < len(distances) else 0
        score = max(0.0, 1.0 - distance)
        chunks.append({
            "text": docs[i],
            "page": metas[i].get("page"),
            "doc_name": metas[i].get("doc_name"),
            "score": score,
        })

    return chunks

def already_uploaded(filename):
    return any(doc["name"] == filename for doc in st.session_state.documents)

def web_search(query):
    results = []
    try:
        with DDGS() as ddgs:
            search_results = ddgs.text(query, max_results=3)
            for r in search_results:
                results.append(f"{r['title']}\n{r['body']}")
    except Exception as e:
        print(f"[web_search] DuckDuckGo search failed: {e}")
    return "\n\n".join(results)

_CURRENT_INFO_PATTERNS = [
    "today", "tonight", "this week", "this month", "this year",
    "current", "currently", "latest", "recent", "recently",
    "right now", "as of", "up to date", "up-to-date",
    "news", "breaking", "price", "prices", "exchange rate",
    "stock", "weather", "election", "who is the", "who won",
]

def needs_current_info(query):
    q = query.lower()
    if any(pattern in q for pattern in _CURRENT_INFO_PATTERNS):
        return True
    for token in q.replace(",", " ").split():
        if token.isdigit() and len(token) == 4 and 2023 <= int(token) <= 2030:
            return True
    return False

# ---------------- Groq Chat Helper ----------------
def groq_chat(messages, stream=False):
    """
    Call Groq API. Strips any 'images' keys from messages since
    llama-3.3-70b-versatile is text-only. Returns a streaming iterator
    or a full completion depending on the stream flag.
    """
    clean_messages = []
    for m in messages:
        clean_msg = {"role": m["role"], "content": m["content"]}
        clean_messages.append(clean_msg)

    return groq_client.chat.completions.create(
        model=MODEL_NAME,
        messages=clean_messages,
        stream=stream,
        max_tokens=2048,
    )

# ---------------- Agents ----------------
# NOTE: FOUNDER_SYSTEM_PROMPT is appended to every agent's prompt so that any
# agent (not just Nation Twin) can correctly answer founder/ownership questions.
AGENTS = {
    "🇳🇬 Nation Twin": {
        "icon": "🇳🇬",
        "prompt": """
You are Nation Twin.
You are the AI adviser and digital twin of Nigeria.
Your mission is to help improve:
- Agriculture
- Healthcare
- Education
- Energy
- Security
- Transportation
- Economy
Provide thoughtful, intelligent, and practical recommendations.
""" + FOUNDER_SYSTEM_PROMPT,
    },
    "🌾 Agriculture Twin": {
        "icon": "🌾",
        "prompt": """
You are Agriculture Twin, a specialist adviser focused on Nigerian agriculture.
You understand crop production, farming practices, food security, agribusiness,
land use, irrigation, fertilizer access, supply chains, and rural livelihoods
in Nigeria. Provide practical, locally relevant recommendations to improve
agricultural productivity, sustainability, and farmer welfare.
""" + FOUNDER_SYSTEM_PROMPT,
    },
    "🏥 Health Twin": {
        "icon": "🏥",
        "prompt": """
You are Health Twin, a specialist adviser focused on Nigerian healthcare.
You understand public health systems, disease burden, hospital infrastructure,
maternal and child health, health financing, and access to care across Nigeria's
regions. Provide practical, evidence-informed recommendations to improve health
outcomes and the healthcare system.
""" + FOUNDER_SYSTEM_PROMPT,
    },
    "📚 Education Twin": {
        "icon": "📚",
        "prompt": """
You are Education Twin, a specialist adviser focused on Nigerian education.
You understand primary, secondary, and tertiary education, literacy, teacher
training, school infrastructure, curriculum, and access/equity issues across
Nigeria. Provide practical recommendations to improve learning outcomes and
access to quality education.
""" + FOUNDER_SYSTEM_PROMPT,
    },
    "⚡ Energy Twin": {
        "icon": "⚡",
        "prompt": """
You are Energy Twin, a specialist adviser focused on Nigerian energy.
You understand the power grid, electricity access, renewable energy, oil and gas,
energy policy, and infrastructure challenges in Nigeria. Provide practical
recommendations to improve energy access, reliability, and sustainability.
""" + FOUNDER_SYSTEM_PROMPT,
    },
    "💰 Economy Twin": {
        "icon": "💰",
        "prompt": """
You are Economy Twin, a specialist adviser focused on the Nigerian economy.
You understand macroeconomics, inflation, employment, trade, fiscal and
monetary policy, the informal sector, and economic development in Nigeria.
Provide practical, well-reasoned economic recommendations.
""" + FOUNDER_SYSTEM_PROMPT,
    },
    "🛡 Security Twin": {
        "icon": "🛡",
        "prompt": """
You are Security Twin.
You are an expert adviser on Nigerian security.
You understand:
- Crime prevention
- Policing
- Cybersecurity
- Border protection
- Intelligence systems
- National security
Provide practical and intelligent recommendations.
""" + FOUNDER_SYSTEM_PROMPT,
    },
    "🚗 Transportation Twin": {
        "icon": "🚗",
        "prompt": """
You are Transportation Twin.
You specialize in:
- Roads
- Rail systems
- Aviation
- Ports
- Logistics
- Urban mobility
Provide practical solutions for transportation challenges in Nigeria.
""" + FOUNDER_SYSTEM_PROMPT,
    },
}

if "active_agent" not in st.session_state:
    st.session_state.active_agent = "🇳🇬 Nation Twin"

# ---------------- Sidebar ----------------
with st.sidebar:
    st.title("🇳🇬 Nation Twin")
    st.markdown("---")
    st.subheader("Version")
    st.success("v0.9")
    st.subheader("Model")
    st.info(MODEL_NAME)
    st.subheader("Status")
    st.success("🟢 Online")
    st.metric("Questions Asked", st.session_state.questions_count)
    st.markdown("---")

    st.subheader("🤖 Choose Your Agent")
    selected_agent = st.selectbox(
        "Active agent",
        options=list(AGENTS.keys()),
        index=list(AGENTS.keys()).index(st.session_state.active_agent),
        label_visibility="collapsed",
    )
    st.session_state.active_agent = selected_agent
    st.caption(f"Currently talking to **{selected_agent}**")
    st.markdown("---")

    st.subheader("🌐 Web Search")
    if "web_search_enabled" not in st.session_state:
        st.session_state.web_search_enabled = True
    st.session_state.web_search_enabled = st.toggle(
        "Search the web for current info",
        value=st.session_state.web_search_enabled,
    )
    st.caption(
        "When enabled, only triggers when your question looks like it needs "
        "current/recent info (mentions 'today', 'latest', a recent year, "
        "prices, news, etc.) — not on every message."
    )
    st.markdown("---")

    st.subheader("🗂️ Remembered Facts")
    st.caption("These persist across your conversations (not visible to other users).")
    new_fact = st.text_input("Add something Nation Twin should remember", key="new_fact_input")
    if st.button("➕ Remember This", use_container_width=True):
        if new_fact.strip():
            save_fact(USER_ID, new_fact.strip())
            st.rerun()

    for f in get_all_facts(USER_ID):
        col1, col2 = st.columns([4, 1])
        with col1:
            st.caption(f"📌 {f['fact']}")
        with col2:
            if st.button("✕", key=f"del_fact_{f['id']}"):
                delete_fact(USER_ID, f["id"])
                st.rerun()
    st.markdown("---")

    st.subheader("🧠 Conversations")
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.current_session_id = create_session(USER_ID)
        st.session_state.messages = []
        st.session_state.questions_count = 0
        st.rerun()

    sessions = get_all_sessions(USER_ID)
    for s in sessions:
        is_active = s["id"] == st.session_state.current_session_id
        col1, col2 = st.columns([4, 1])
        with col1:
            label = ("🟢 " if is_active else "") + s["title"]
            if st.button(label, key=f"session_{s['id']}", use_container_width=True):
                st.session_state.current_session_id = s["id"]
                st.session_state.messages = load_messages(s["id"])
                st.session_state.questions_count = count_user_questions(s["id"])
                st.rerun()
        with col2:
            if st.button("✕", key=f"del_session_{s['id']}"):
                delete_session(USER_ID, s["id"])
                if is_active:
                    remaining = get_all_sessions(USER_ID)
                    if remaining:
                        st.session_state.current_session_id = remaining[0]["id"]
                        st.session_state.messages = load_messages(remaining[0]["id"])
                        st.session_state.questions_count = count_user_questions(remaining[0]["id"])
                    else:
                        new_id = create_session(USER_ID)
                        st.session_state.current_session_id = new_id
                        st.session_state.messages = []
                        st.session_state.questions_count = 0
                st.rerun()
    st.markdown("---")

    uploaded_files = st.file_uploader(
        "📄 Upload PDF(s)",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            if already_uploaded(uploaded_file.name):
                continue

            with st.spinner(f"Reading {uploaded_file.name}..."):
                doc_chunks, had_text = extract_pdf_chunks(uploaded_file)

            if not had_text:
                st.warning(
                    f"⚠️ '{uploaded_file.name}' appears to be a scanned PDF "
                    f"with no extractable text. OCR support is not yet available."
                )
                continue

            with st.spinner(f"Indexing {uploaded_file.name}..."):
                for i, chunk in enumerate(doc_chunks):
                    chunk_id = f"{uploaded_file.name}::{i}::{uuid.uuid4().hex[:8]}"
                    collection.add(
                        ids=[chunk_id],
                        documents=[chunk["text"]],
                        metadatas=[{
                            "page": chunk["page"],
                            "doc_name": uploaded_file.name,
                            "user_id": USER_ID,
                        }]
                    )

            st.session_state.documents.append({
                "name": uploaded_file.name,
                "chunks": doc_chunks,
            })
            st.success(
                f"✅ '{uploaded_file.name}' loaded "
                f"({len(doc_chunks)} chunks indexed)"
            )

    if st.session_state.documents:
        st.markdown("---")
        st.subheader("📚 Loaded Documents")
        for doc in st.session_state.documents:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.caption(f"📄 {doc['name']} ({len(doc['chunks'])} chunks)")
            with col2:
                if st.button("✕", key=f"remove_{doc['name']}"):
                    try:
                        collection.delete(where={"doc_name": doc["name"]})
                    except Exception:
                        pass
                    st.session_state.documents = [
                        d for d in st.session_state.documents
                        if d["name"] != doc["name"]
                    ]
                    st.rerun()

    st.markdown("---")
    st.subheader("📊 CSV Analytics")
    csv_file = st.file_uploader("Upload CSV", type=["csv"])

    if csv_file:
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            st.error(f"Couldn't read that CSV: {e}")
            df = None

        if df is not None:
            st.dataframe(df)
            numeric_cols = df.select_dtypes(include="number").columns
            if len(numeric_cols) > 0:
                fig, ax = plt.subplots()
                df[numeric_cols].plot(ax=ax)
                st.pyplot(fig)
            else:
                st.info("No numeric columns found to chart.")

    st.markdown("---")
    if st.button("📥 Export Chat"):
        chat_text = ""
        for m in st.session_state.messages:
            chat_text += f"{m['role']}:\n{m['content']}\n\n"
        st.download_button(
            "Download",
            data=chat_text,
            file_name="nation_twin_chat.txt"
        )

    st.markdown("---")
    if st.button("🗑 Clear Current Conversation"):
        conn = get_connection()
        owner = conn.execute(
            "SELECT user_id FROM sessions WHERE id = ?",
            (st.session_state.current_session_id,)
        ).fetchone()
        if owner and owner["user_id"] == USER_ID:
            conn.execute(
                "DELETE FROM messages WHERE session_id = ?",
                (st.session_state.current_session_id,)
            )
            conn.commit()
        conn.close()
        st.session_state.messages = []
        st.session_state.questions_count = 0
        st.rerun()

# ---------------- Main Page ----------------
st.title(st.session_state.active_agent)
st.caption("AI Adviser and Digital Twin of Nigeria")

# Show previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and message.get("agent"):
            st.caption(f"Answered by {message['agent']}")
        if message.get("image_b64"):
            st.image(base64.b64decode(message["image_b64"]), width=250)
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("📖 Sources used for this answer"):
                for src in message["sources"]:
                    score = src.get("score", 0.0)
                    st.caption(
                        f"📄 {src['doc_name']} — page {src['page']} "
                        f"(relevance: {score:.2f})"
                    )

# NOTE: Image upload is kept for display/storage purposes.
# The Groq model (llama-3.3-70b-versatile) is text-only, so attached images
# will be shown in the chat history but the model will describe them as
# "image attached" in its context. To re-enable full image reasoning,
# switch to a vision-capable Groq model like "meta-llama/llama-4-scout-17b-16e-instruct"
# when it becomes available on your Groq plan.
if "image_uploader_key" not in st.session_state:
    st.session_state.image_uploader_key = 0

attached_image = st.file_uploader(
    "🖼️ Attach an image to your next question (optional — displayed in chat, text description sent to model)",
    type=["png", "jpg", "jpeg"],
    key=f"image_upload_{st.session_state.image_uploader_key}",
)
if attached_image:
    st.image(attached_image, width=200, caption="Will be shown in chat (model sees text description)")

# User input
prompt = st.chat_input("Ask Nation Twin anything...")

if prompt:
    st.session_state.questions_count += 1

    image_b64 = None
    image_bytes = None
    if attached_image:
        image_bytes = attached_image.getvalue()
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    # Store user message
    st.session_state.messages.append({"role": "user", "content": prompt, "image_b64": image_b64})
    save_message(st.session_state.current_session_id, "user", prompt, image_b64=image_b64)
    rename_session_if_default(st.session_state.current_session_id, prompt)
    with st.chat_message("user"):
        if image_bytes:
            st.image(image_bytes, width=200)
        st.markdown(prompt)

    # ---------------- Retrieval ----------------
    relevant_chunks = retrieve_relevant_chunks(prompt, USER_ID)
    should_search = st.session_state.web_search_enabled and needs_current_info(prompt)
    internet_info = web_search(prompt) if should_search else ""

    # ---------------- Cross-Agent Collaboration ----------------
    cross_agent_notes = ""
    agent_key = st.session_state.active_agent

    if agent_key == "🇳🇬 Nation Twin":
        sector_keywords = {
            "🌾 Agriculture Twin": ["agricultur", "farm", "crop", "food security"],
            "🏥 Health Twin": ["health", "hospital", "disease", "medical"],
            "📚 Education Twin": ["education", "school", "literacy", "teacher"],
            "⚡ Energy Twin": ["energy", "electricity", "power grid", "fuel"],
            "💰 Economy Twin": ["econom", "inflation", "trade", "employment", "fiscal"],
            "🛡 Security Twin": ["security", "crime", "police", "cyber", "terror"],
            "🚗 Transportation Twin": ["transport", "road", "rail", "port", "aviation"],
        }
        prompt_lower = prompt.lower()
        matched_agents = [
            name for name, keywords in sector_keywords.items()
            if any(kw in prompt_lower for kw in keywords)
        ]

        if matched_agents:
            with st.spinner("Consulting specialist twins..."):
                for matched_name in matched_agents[:2]:
                    try:
                        specialist_response = groq_chat(
                            messages=[
                                {"role": "system", "content": AGENTS[matched_name]["prompt"]},
                                {"role": "user", "content": f"Briefly (2-3 sentences): {prompt}"},
                            ],
                            stream=False,
                        )
                        note = specialist_response.choices[0].message.content
                        cross_agent_notes += f"\n\n[{matched_name} perspective]: {note}"
                    except Exception as e:
                        print(f"[cross-agent] {matched_name} call failed: {e}")

    # ---------------- Remembered Facts ----------------
    remembered_facts = get_all_facts(USER_ID)
    facts_text = "\n".join(f"- {f['fact']}" for f in remembered_facts) if remembered_facts else ""

    # ---------------- System Personality ----------------
    system_prompt = AGENTS[agent_key]["prompt"] + """

GROUNDING RULES:
- If document context is provided below, ground your answer in it and mention
  which document/page the information came from where relevant.
- If no document context is provided, say so plainly and answer from your
  general knowledge of Nigeria instead of pretending to analyze a document
  that isn't there.

ACCURACY RULES (very important):
- Do NOT invent specific statistics, percentages, dates, or figures that are
  not present in the provided context or web information.
- If asked for a specific number you are not confident about and it is not in
  the provided context, say plainly that you do not have a verified figure,
  rather than guessing one.
- Clearly distinguish between facts grounded in provided context/sources and
  general reasoning or recommendations based on your own knowledge.

DISCLAIMER:
You are a decision-support tool, not a substitute for verified official data,
government sources, or expert review. For any high-stakes decision, recommend
the person confirm with primary sources.
"""

    messages = [{"role": "system", "content": system_prompt}]

    if facts_text:
        messages.append({
            "role": "system",
            "content": f"Remembered facts about this user/context (from previous sessions):\n{facts_text}"
        })

    if internet_info:
        messages.append({
            "role": "system",
            "content": f"Recent web information:\n\n{internet_info}"
        })

    if cross_agent_notes:
        messages.append({
            "role": "system",
            "content": f"Quick input from relevant specialist Twins:{cross_agent_notes}"
        })

    if relevant_chunks:
        context_text = "\n\n".join(
            f"[Source: {c['doc_name']}, page {c['page']}]\n{c['text']}"
            for c in relevant_chunks
        )
        messages.append({
            "role": "system",
            "content": f"Relevant excerpts from uploaded documents:\n\n{context_text}"
        })
    elif st.session_state.documents:
        messages.append({
            "role": "system",
            "content": (
                "Documents are loaded, but no closely matching excerpts were "
                "found for this question. Answer from general knowledge and "
                "note that the documents may not cover this topic."
            )
        })

    # If an image was attached, inform the model via text since we're using a text-only model
    image_note = ""
    if image_bytes:
        image_note = "\n\n[Note: The user has attached an image to this message. As a text-only model, you cannot see it directly. Acknowledge the image and ask them to describe what's in it if it's relevant to their question.]"

    # Add prior conversation history (text only, no image bytes)
    for m in st.session_state.messages[:-1]:  # exclude the just-added user message
        messages.append({"role": m["role"], "content": m["content"]})

    # Add the current user message (with optional image note)
    messages.append({"role": "user", "content": prompt + image_note})

    # ---------------- Generate Response (streaming) ----------------
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_reply = ""

        try:
            stream = groq_chat(messages=messages, stream=True)
            for chunk in stream:
                token = chunk.choices[0].delta.content or ""
                if token:
                    full_reply += token
                    placeholder.markdown(full_reply + "▌")
            placeholder.markdown(full_reply)
        except Exception as e:
            full_reply = (
                "⚠️ I couldn't reach the Groq API. Please check your API key in Streamlit Secrets.\n\n"
                f"Details: {e}"
            )
            placeholder.markdown(full_reply)

        reply = full_reply

        if relevant_chunks:
            with st.expander("📖 Sources used for this answer"):
                for c in relevant_chunks:
                    st.caption(
                        f"📄 {c['doc_name']} — page {c['page']} "
                        f"(relevance: {c['score']:.2f})"
                    )

    # Save assistant reply
    sources_list = [
        {"doc_name": c["doc_name"], "page": c["page"], "score": c["score"]}
        for c in relevant_chunks
    ] if relevant_chunks else []

    st.session_state.messages.append({
        "role": "assistant",
        "content": reply,
        "agent": agent_key,
        "sources": sources_list,
    })
    save_message(
        st.session_state.current_session_id,
        "assistant",
        reply,
        agent=agent_key,
        sources=sources_list,
    )

    if attached_image:
        st.session_state.image_uploader_key += 1
        st.rerun()


st.markdown("""
<hr style="margin-top:40px;">
<div style="text-align:center;padding:15px;">
    <h4 style="margin-bottom:5px;">NG Nation Twin</h4>
    <p style="color:#888;">
        Founded and Developed by <b>Chrisantus Madueke</b><br>
        Powered by Groq AI • Streamlit • ChromaDB
    </p>
</div>
""", unsafe_allow_html=True)



