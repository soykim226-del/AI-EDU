"""멀티유저 멀티세션 RAG 챗봇: user 테이블 로그인, Supabase 세션·벡터, multiref.py 스타일 UI."""

from __future__ import annotations

import hashlib
import logging
import os
import re
import secrets
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from supabase import Client, create_client

# ---------------------------------------------------------------------------
# Paths & environment
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / ".env"
LOGO_PATH = REPO_ROOT / "logo.png"
LOG_DIR = REPO_ROOT / "logs"

load_dotenv(dotenv_path=ENV_PATH)

LLM_MODEL_NAME = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
VECTOR_BATCH = 10

PBKDF2_ITERS = 390_000
USER_TABLE = "user"

CHAT_TITLE = "재정경제부 RAG 챗봇"


# ---------------------------------------------------------------------------
# Logging (WARNING/ERROR only)
# ---------------------------------------------------------------------------
def _setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_name = f"chatbot_{datetime.now().strftime('%Y%m%d')}.log"
    log_path = LOG_DIR / log_name

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.WARNING)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.WARNING)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(ch)

    for name in (
        "httpx",
        "httpcore",
        "urllib3",
        "openai",
        "langchain",
        "langchain_openai",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)

    return logging.getLogger("multiusers_rag")


logger = _setup_logging()


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
ANSWER_STYLE_SYSTEM = """당신은 친절하고 공손한 AI 어시스턴트입니다.

답변 규칙:
- 반드시 마크다운 헤딩(# ## ###)으로 구조화하세요. 주요 주제는 #, 세부는 ##, 구체 설명은 ###.
- 서술형으로 완전한 문장을 사용하고 존댓말로 작성하세요.
- 구분선(---, ===, ___)은 사용하지 마세요.
- 취소선(~~텍스트~~)은 사용하지 마세요.
- 참조 표시, 각주, 출처 문구, URL 인용 문장은 넣지 마세요.
"""


def remove_separators(text: str) -> str:
    out = re.sub(r"~~([^~]*)~~", r"\1", text)
    out = re.sub(r"(?m)^\s*-{3,}\s*$", "", out)
    out = re.sub(r"(?m)^\s*={3,}\s*$", "", out)
    out = re.sub(r"(?m)^\s*_{3,}\s*$", "", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def _overlay_streamlit_secrets_into_env() -> None:
    """Streamlit Cloud: secrets.toml 이 있으면 해당 값을 os.environ 에 덮어씁니다(우선).

    로컬에서 secrets 파일이 없으면 ``StreamlitSecretNotFoundError`` 가 나므로 무시하고
    이미 ``load_dotenv`` 로 채운 값만 사용합니다.
    """
    try:
        sec = st.secrets
        for key in ("SUPABASE_URL", "SUPABASE_ANON_KEY", "OPENAI_API_KEY"):
            if key not in sec:
                continue
            raw = sec[key]
            if isinstance(raw, str) and raw.strip():
                os.environ[key] = raw.strip()
    except StreamlitSecretNotFoundError:
        pass
    except (RuntimeError, OSError):
        # Streamlit 컨텍스트 밖에서 import 될 때 등
        pass


def _env_ok() -> tuple[bool, str]:
    missing: list[str] = []
    if not os.getenv("OPENAI_API_KEY", "").strip():
        missing.append("OPENAI_API_KEY")
    if not os.getenv("SUPABASE_URL", "").strip():
        missing.append("SUPABASE_URL")
    if not os.getenv("SUPABASE_ANON_KEY", "").strip():
        missing.append("SUPABASE_ANON_KEY")
    if missing:
        return False, (
            "다음 키가 필요합니다. Streamlit Cloud는 `st.secrets`에, 로컬은 `.env` 또는 환경 변수에 설정하세요: "
            + ", ".join(missing)
        )
    return True, ""


def get_supabase() -> Client | None:
    url = os.getenv("SUPABASE_URL", "").strip()
    key = os.getenv("SUPABASE_ANON_KEY", "").strip()
    if not url or not key:
        return None
    return create_client(url, key)


def hash_password(plain: str) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", plain.encode("utf-8"), salt, PBKDF2_ITERS)
    return f"pbkdf2_sha256${PBKDF2_ITERS}${salt.hex()}${dk.hex()}"


def verify_password(plain: str, stored: str) -> bool:
    try:
        algo, iters_s, salt_hex, hash_hex = stored.split("$", 3)
        if algo != "pbkdf2_sha256":
            return False
        iters = int(iters_s)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(hash_hex)
        dk = hashlib.pbkdf2_hmac("sha256", plain.encode("utf-8"), salt, iters)
        return secrets.compare_digest(dk, expected)
    except Exception:  # noqa: BLE001
        return False


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def session_belongs_to_user(supabase: Client, session_id: str, user_id: str) -> bool:
    try:
        r = (
            supabase.table("chat_sessions")
            .select("id")
            .eq("id", session_id)
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
        return bool(r.data)
    except Exception as exc:  # noqa: BLE001
        logger.warning("session_belongs_to_user: %s", exc)
        return False


def register_user(supabase: Client, login_id: str, password: str) -> tuple[bool, str]:
    lid = login_id.strip()
    if len(lid) < 2:
        return False, "로그인 ID는 2자 이상이어야 합니다."
    if len(password) < 4:
        return False, "비밀번호는 4자 이상이어야 합니다."
    ph = hash_password(password)
    try:
        supabase.table(USER_TABLE).insert({"login_id": lid, "password_hash": ph}).execute()
        return True, "회원가입이 완료되었습니다. 로그인해 주세요."
    except Exception as exc:  # noqa: BLE001
        logger.warning("register_user: %s", exc)
        err = str(exc).lower()
        if "unique" in err or "duplicate" in err:
            return False, "이미 사용 중인 로그인 ID입니다."
        return False, "회원가입에 실패했습니다. 잠시 후 다시 시도해 주세요."


def try_login(supabase: Client, login_id: str, password: str) -> tuple[bool, str, str | None, str | None]:
    lid = login_id.strip()
    if not lid or not password:
        return False, "로그인 ID와 비밀번호를 입력하세요.", None, None
    try:
        r = supabase.table(USER_TABLE).select("id,login_id,password_hash").eq("login_id", lid).limit(1).execute()
        rows = r.data or []
        if not rows:
            return False, "로그인 ID 또는 비밀번호가 올바르지 않습니다.", None, None
        row = rows[0]
        if not verify_password(password, row["password_hash"]):
            return False, "로그인 ID 또는 비밀번호가 올바르지 않습니다.", None, None
        return True, "", str(row["id"]), str(row["login_id"])
    except Exception as exc:  # noqa: BLE001
        logger.warning("try_login: %s", exc)
        return False, "로그인 처리 중 오류가 발생했습니다.", None, None


def _init_session_state() -> None:
    defaults: dict[str, Any] = {
        "chat_history": [],
        "conversation_memory": [],
        "current_session_id": None,
        "processed_names": [],
        "session_list_cache": [],
        "_session_sb_index": 0,
        "logged_in_user_id": None,
        "logged_in_login_id": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def fetch_sessions(supabase: Client, user_id: str) -> list[dict[str, Any]]:
    try:
        r = (
            supabase.table("chat_sessions")
            .select("id,title,updated_at")
            .eq("user_id", user_id)
            .order("updated_at", desc=True)
            .execute()
        )
        return list(r.data or [])
    except Exception as exc:  # noqa: BLE001
        logger.warning("fetch_sessions: %s", exc)
        return []


def fetch_messages(supabase: Client, session_id: str, user_id: str) -> list[dict[str, str]]:
    try:
        r = (
            supabase.table("chat_messages")
            .select("role,content,msg_index")
            .eq("session_id", session_id)
            .eq("user_id", user_id)
            .order("msg_index")
            .execute()
        )
        rows = r.data or []
        return [{"role": x["role"], "content": x["content"]} for x in rows]
    except Exception as exc:  # noqa: BLE001
        logger.warning("fetch_messages: %s", exc)
        return []


def fetch_distinct_vector_filenames(supabase: Client, session_id: str, user_id: str) -> list[str]:
    if not session_belongs_to_user(supabase, session_id, user_id):
        return []
    try:
        r = (
            supabase.table("vector_documents")
            .select("file_name")
            .eq("session_id", session_id)
            .execute()
        )
        names = sorted({row["file_name"] for row in (r.data or []) if row.get("file_name")})
        return list(names)
    except Exception as exc:  # noqa: BLE001
        logger.warning("fetch_distinct_vector_filenames: %s", exc)
        return []


def delete_messages_for_session(supabase: Client, session_id: str, user_id: str) -> None:
    supabase.table("chat_messages").delete().eq("session_id", session_id).eq("user_id", user_id).execute()


def persist_messages(
    supabase: Client, session_id: str, user_id: str, history: list[dict[str, str]]
) -> None:
    delete_messages_for_session(supabase, session_id, user_id)
    rows: list[dict[str, Any]] = []
    for i, m in enumerate(history):
        rows.append(
            {
                "session_id": session_id,
                "user_id": user_id,
                "role": m["role"],
                "content": m["content"],
                "msg_index": i,
            }
        )
    if not rows:
        return
    for i in range(0, len(rows), 50):
        supabase.table("chat_messages").insert(rows[i : i + 50]).execute()


def touch_session(supabase: Client, session_id: str, user_id: str) -> None:
    supabase.table("chat_sessions").update({"updated_at": _iso_now()}).eq("id", session_id).eq(
        "user_id", user_id
    ).execute()


def create_session_row(supabase: Client, title: str, user_id: str) -> str | None:
    new_id = str(uuid.uuid4())
    try:
        supabase.table("chat_sessions").insert(
            {"id": new_id, "title": title, "updated_at": _iso_now(), "user_id": user_id}
        ).execute()
        return new_id
    except Exception as exc:  # noqa: BLE001
        logger.warning("create_session_row: %s", exc)
    return None


def update_session_title(supabase: Client, session_id: str, user_id: str, title: str) -> None:
    supabase.table("chat_sessions").update({"title": title, "updated_at": _iso_now()}).eq(
        "id", session_id
    ).eq("user_id", user_id).execute()


def delete_session_cascade(supabase: Client, session_id: str, user_id: str) -> None:
    supabase.table("chat_sessions").delete().eq("id", session_id).eq("user_id", user_id).execute()


def copy_vectors_to_session(
    supabase: Client, source_session_id: str, target_session_id: str, user_id: str
) -> None:
    if not session_belongs_to_user(supabase, source_session_id, user_id):
        return
    if not session_belongs_to_user(supabase, target_session_id, user_id):
        return
    try:
        r = (
            supabase.table("vector_documents")
            .select("file_name,content,embedding,metadata")
            .eq("session_id", source_session_id)
            .execute()
        )
        rows_in = r.data or []
        batch: list[dict[str, Any]] = []
        for row in rows_in:
            emb = row.get("embedding")
            if emb is None:
                continue
            batch.append(
                {
                    "session_id": target_session_id,
                    "file_name": row["file_name"],
                    "content": row["content"],
                    "embedding": emb,
                    "metadata": row.get("metadata") or {},
                }
            )
            if len(batch) >= VECTOR_BATCH:
                supabase.table("vector_documents").insert(batch).execute()
                batch.clear()
        if batch:
            supabase.table("vector_documents").insert(batch).execute()
    except Exception as exc:  # noqa: BLE001
        logger.warning("copy_vectors_to_session: %s", exc)


def insert_vector_chunks(
    supabase: Client,
    session_id: str,
    splits: list[Document],
    embeddings: OpenAIEmbeddings,
) -> None:
    for i in range(0, len(splits), VECTOR_BATCH):
        batch_docs = splits[i : i + VECTOR_BATCH]
        texts = [d.page_content for d in batch_docs]
        vecs = embeddings.embed_documents(texts)
        rows: list[dict[str, Any]] = []
        for doc, vec in zip(batch_docs, vecs, strict=True):
            fname = (doc.metadata or {}).get("file_name") or "unknown.pdf"
            rows.append(
                {
                    "session_id": session_id,
                    "file_name": str(fname),
                    "content": doc.page_content,
                    "embedding": vec,
                    "metadata": {
                        k: v
                        for k, v in (doc.metadata or {}).items()
                        if isinstance(v, (str, int, float))
                    },
                }
            )
        supabase.table("vector_documents").insert(rows).execute()


def retrieve_with_rpc(
    supabase: Client,
    session_id: str,
    query: str,
    embeddings: OpenAIEmbeddings,
    k: int = 10,
) -> list[Document]:
    qvec = embeddings.embed_query(query)
    params: dict[str, Any] = {
        "query_embedding": qvec,
        "match_count": k,
        "filter_session_id": session_id,
    }
    try:
        r = supabase.rpc("match_vector_documents", params).execute()
    except Exception as exc:  # noqa: BLE001
        logger.warning("RPC match_vector_documents failed: %s", exc)
        try:
            r = supabase.rpc(
                "match_vector_documents",
                {
                    "query_embedding": "[" + ",".join(str(x) for x in qvec) + "]",
                    "match_count": k,
                    "filter_session_id": session_id,
                },
            ).execute()
        except Exception as exc2:  # noqa: BLE001
            logger.warning("RPC string embedding retry failed: %s", exc2)
            return []

    docs: list[Document] = []
    for row in r.data or []:
        docs.append(
            Document(
                page_content=row.get("content") or "",
                metadata={"file_name": row.get("file_name") or ""},
            )
        )
    return docs


def _format_memory_block(messages: list[dict[str, str]], max_items: int = 50) -> str:
    tail = messages[-max_items:] if len(messages) > max_items else messages
    lines: list[str] = []
    for m in tail:
        role = m.get("role", "")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        prefix = "사용자" if role == "user" else "어시스턴트"
        lines.append(f"{prefix}: {content}")
    return "\n".join(lines)


def _build_rag_messages(
    question: str, context: str, memory_text: str
) -> list[SystemMessage | HumanMessage]:
    sys = f"""{ANSWER_STYLE_SYSTEM}

아래 [대화 맥락]과 [참고 문서]를 활용해 답하세요. 참고 문서에 없는 내용은 추측하지 말고 한계를 밝히세요.
[대화 맥락]
{memory_text or "(없음)"}

[참고 문서]
{context}
"""
    return [SystemMessage(content=sys), HumanMessage(content=question)]


def _generate_followup_section(llm: ChatOpenAI, user_q: str, answer: str) -> str:
    trimmed = answer[:8000]
    prompt = (
        "다음 사용자 질문과 답변을 바탕으로, 이어서 물어볼 만한 후속 질문을 한국어로 정확히 3개만 작성하세요.\n"
        "형식:\n1. ...\n2. ...\n3. ...\n"
        "설명 문장이나 다른 텍스트는 출력하지 마세요.\n\n"
        f"[사용자 질문]\n{user_q}\n\n[답변]\n{trimmed}"
    )
    try:
        out = llm.invoke([HumanMessage(content=prompt)])
        raw = getattr(out, "content", str(out)) or ""
    except Exception as exc:  # noqa: BLE001
        logger.warning("Follow-up generation failed: %s", exc)
        return ""
    raw = remove_separators(str(raw))
    if not raw.strip():
        return ""
    return f"\n\n### 💡 다음에 물어볼 수 있는 질문들\n\n{raw.strip()}\n"


def generate_session_title_llm(openai_key: str, history: list[dict[str, str]]) -> str:
    first_user = ""
    first_asst = ""
    for m in history:
        if m["role"] == "user" and not first_user:
            first_user = m["content"][:2000]
        elif m["role"] == "assistant" and first_asst == "" and first_user:
            first_asst = m["content"][:2000]
            break
    if not first_user:
        return f"빈 세션 {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    llm = ChatOpenAI(model=LLM_MODEL_NAME, temperature=0.3, api_key=openai_key)
    prompt = (
        "첫 사용자 질문과 첫 어시스턴트 답변을 한 줄로 요약한 세션 제목을 한국어로 작성하세요. "
        "30자 이내, 따옴표나 접두어 없이 제목만 출력하세요.\n\n"
        f"[질문]\n{first_user}\n\n[답변]\n{first_asst or '(아직 답변 없음)'}"
    )
    try:
        out = llm.invoke([HumanMessage(content=prompt)])
        title = str(getattr(out, "content", "") or "").strip().split("\n")[0].strip()
        return title[:120] if title else "새 세션"
    except Exception as exc:  # noqa: BLE001
        logger.warning("generate_session_title_llm: %s", exc)
        return first_user[:40] + ("…" if len(first_user) > 40 else "")


def ensure_db_session(supabase: Client, openai_key: str, user_id: str) -> str:
    sid = st.session_state.current_session_id
    if sid and session_belongs_to_user(supabase, sid, user_id):
        return sid
    if sid and not session_belongs_to_user(supabase, sid, user_id):
        st.session_state.current_session_id = None
    title = f"임시 세션 {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    new_id = create_session_row(supabase, title, user_id)
    if new_id:
        st.session_state.current_session_id = new_id
        return new_id
    return ""


def auto_save_session(supabase: Client, openai_key: str, user_id: str) -> None:
    sid = st.session_state.current_session_id
    if not sid:
        sid = ensure_db_session(supabase, openai_key, user_id)
    if not sid:
        return
    persist_messages(supabase, sid, user_id, st.session_state.chat_history)
    hist = st.session_state.chat_history
    if len(hist) >= 2 and openai_key:
        has_pair = any(h["role"] == "user" for h in hist) and any(h["role"] == "assistant" for h in hist)
        if has_pair:
            new_title = generate_session_title_llm(openai_key, hist)
            if new_title:
                update_session_title(supabase, sid, user_id, new_title)
    touch_session(supabase, sid, user_id)
    st.session_state.session_list_cache = fetch_sessions(supabase, user_id)


def load_session_into_ui(supabase: Client, session_id: str, user_id: str) -> bool:
    if not session_belongs_to_user(supabase, session_id, user_id):
        return False
    msgs = fetch_messages(supabase, session_id, user_id)
    st.session_state.chat_history = msgs
    st.session_state.conversation_memory = msgs[-50:] if msgs else []
    st.session_state.current_session_id = session_id
    st.session_state.processed_names = fetch_distinct_vector_filenames(supabase, session_id, user_id)
    return True


def process_pdf_uploads_to_supabase(
    supabase: Client,
    session_id: str,
    uploaded_files: list[Any],
    openai_key: str,
    user_id: str,
) -> list[str]:
    if not session_belongs_to_user(supabase, session_id, user_id):
        return []
    if not uploaded_files:
        return []
    all_docs: list[Document] = []
    names: list[str] = []
    for uf in uploaded_files:
        suffix = Path(uf.name).suffix.lower() or ".pdf"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uf.getvalue())
            tmp_path = tmp.name
        try:
            loader = PyPDFLoader(tmp_path)
            for d in loader.load():
                d.metadata = dict(d.metadata or {})
                d.metadata["file_name"] = uf.name
                all_docs.append(d)
            names.append(uf.name)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    if not all_docs:
        return []
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    splits = splitter.split_documents(all_docs)
    emb = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=openai_key)
    insert_vector_chunks(supabase, session_id, splits, emb)
    return names


def _logout() -> None:
    st.session_state.logged_in_user_id = None
    st.session_state.logged_in_login_id = None
    st.session_state.chat_history = []
    st.session_state.conversation_memory = []
    st.session_state.current_session_id = None
    st.session_state.processed_names = []
    st.session_state.session_list_cache = []


def main() -> None:
    st.set_page_config(
        page_title=CHAT_TITLE,
        page_icon="📚",
        layout="wide",
    )
    _overlay_streamlit_secrets_into_env()
    _init_session_state()

    st.markdown(
        """
<style>
h1 { color: #ff69b4 !important; font-size: 1.4rem !important; }
h2 { color: #ffd700 !important; font-size: 1.2rem !important; }
h3 { color: #1f77b4 !important; font-size: 1.1rem !important; }
div.stButton > button:first-child {
  background-color: #ff69b4;
  color: #ffffff;
}
</style>
""",
        unsafe_allow_html=True,
    )

    ok, env_msg = _env_ok()
    supabase = get_supabase() if ok else None
    if ok and supabase is None:
        env_msg = "Supabase 클라이언트를 만들 수 없습니다."
        ok = False

    c1, c2, c3 = st.columns([1, 4, 1])
    with c1:
        if LOGO_PATH.is_file():
            st.image(str(LOGO_PATH), width=180)
        else:
            st.markdown("### 📚")
    with c2:
        st.markdown(
            f"""
<div style="text-align:center; margin:0;">
  <span style="font-size:4rem !important; font-weight:700;">
    <span style="color:#1f77b4 !important;">재정경제부</span>
    <span style="color:#ffd700 !important;">RAG 챗봇</span>
  </span>
</div>
""",
            unsafe_allow_html=True,
        )
    with c3:
        st.empty()

    if not ok:
        st.warning(env_msg)
        st.stop()

    assert supabase is not None
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()

    uid = st.session_state.logged_in_user_id

    with st.sidebar:
        st.markdown("### 계정")
        if not uid:
            st.caption("로그인 후 본인 세션·메시지·벡터만 접근합니다.")
            login_id_in = st.text_input("로그인 ID", key="mu_login_id")
            pw_in = st.text_input("비밀번호", type="password", key="mu_pw")
            if st.button("로그인", key="mu_btn_login"):
                ok_l, msg_l, new_uid, new_lid = try_login(supabase, login_id_in, pw_in)
                if ok_l and new_uid and new_lid:
                    st.session_state.logged_in_user_id = new_uid
                    st.session_state.logged_in_login_id = new_lid
                    st.session_state.session_list_cache = fetch_sessions(supabase, new_uid)
                    st.rerun()
                else:
                    st.error(msg_l)
            with st.expander("회원가입"):
                reg_id = st.text_input("새 로그인 ID", key="mu_reg_id")
                reg_pw = st.text_input("새 비밀번호", type="password", key="mu_reg_pw")
                reg_pw2 = st.text_input("비밀번호 확인", type="password", key="mu_reg_pw2")
                if st.button("가입하기", key="mu_btn_reg"):
                    if reg_pw != reg_pw2:
                        st.error("비밀번호 확인이 일치하지 않습니다.")
                    else:
                        ok_r, msg_r = register_user(supabase, reg_id, reg_pw)
                        if ok_r:
                            st.success(msg_r)
                        else:
                            st.error(msg_r)
            st.stop()

        st.write(f"**{st.session_state.logged_in_login_id}** 님")
        if st.button("로그아웃", key="mu_logout"):
            _logout()
            st.rerun()

    uid = st.session_state.logged_in_user_id
    assert uid is not None

    st.session_state.session_list_cache = fetch_sessions(supabase, uid)
    sessions = st.session_state.session_list_cache

    def _sync_select_index() -> None:
        cur = st.session_state.current_session_id
        ids = [s["id"] for s in sessions]
        if cur and cur in ids:
            st.session_state._session_sb_index = ids.index(cur)
        elif sessions:
            st.session_state._session_sb_index = min(
                st.session_state._session_sb_index, len(sessions) - 1
            )

    _sync_select_index()

    with st.sidebar:
        st.markdown("### 세션 관리")
        if sessions:
            labels = [f"{s.get('title', '제목 없음')}" for s in sessions]

            def _on_session_change() -> None:
                idx = int(st.session_state.session_select_sb_mu)
                if 0 <= idx < len(st.session_state.session_list_cache):
                    sid = st.session_state.session_list_cache[idx]["id"]
                    load_session_into_ui(supabase, sid, uid)

            sel_idx = st.selectbox(
                "세션 선택 (선택 시 자동 로드)",
                options=list(range(len(sessions))),
                format_func=lambda i: labels[i],
                index=st.session_state._session_sb_index,
                key="session_select_sb_mu",
                on_change=_on_session_change,
            )
            st.session_state._session_sb_index = sel_idx
        else:
            st.text("저장된 세션이 없습니다.")

        if st.button("세션저장", key="mu_save_sess"):
            if not openai_key:
                st.error("OPENAI_API_KEY가 필요합니다.")
            elif not st.session_state.chat_history:
                st.warning("저장할 대화가 없습니다.")
            else:
                title = generate_session_title_llm(openai_key, st.session_state.chat_history)
                new_id = create_session_row(supabase, title, uid)
                if not new_id:
                    st.error("세션 생성에 실패했습니다.")
                else:
                    persist_messages(supabase, new_id, uid, st.session_state.chat_history)
                    old_sid = st.session_state.current_session_id
                    if old_sid and old_sid != new_id:
                        copy_vectors_to_session(supabase, old_sid, new_id, uid)
                    st.session_state.current_session_id = new_id
                    st.session_state.processed_names = fetch_distinct_vector_filenames(supabase, new_id, uid)
                    st.session_state.session_list_cache = fetch_sessions(supabase, uid)
                    st.success("새 세션이 저장되었습니다.")
                    st.rerun()

        if st.button("세션로드", key="mu_load_sess") and sessions:
            idx = st.session_state.get("session_select_sb_mu", 0)
            idx = int(idx) if idx is not None else 0
            if 0 <= idx < len(sessions):
                if load_session_into_ui(supabase, sessions[idx]["id"], uid):
                    st.success("세션을 불러왔습니다.")
                    st.rerun()
                else:
                    st.error("세션을 불러올 수 없습니다.")

        if st.button("세션삭제", key="mu_del_sess"):
            if not sessions:
                st.warning("삭제할 세션이 없습니다.")
            else:
                idx = st.session_state.get("session_select_sb_mu", 0)
                idx = int(idx) if idx is not None else 0
                if 0 <= idx < len(sessions):
                    sid = sessions[idx]["id"]
                    delete_session_cascade(supabase, sid, uid)
                    if st.session_state.current_session_id == sid:
                        st.session_state.current_session_id = None
                        st.session_state.chat_history = []
                        st.session_state.conversation_memory = []
                        st.session_state.processed_names = []
                    st.session_state.session_list_cache = fetch_sessions(supabase, uid)
                    st.success("세션이 삭제되었습니다.")
                    st.rerun()

        if st.button("화면초기화", key="mu_clear_ui"):
            st.session_state.chat_history = []
            st.session_state.conversation_memory = []
            st.session_state.current_session_id = None
            st.session_state.processed_names = []
            st.rerun()

        if st.button("vectordb", key="mu_vecdb"):
            sid = st.session_state.current_session_id
            if not sid:
                st.info("활성 세션이 없습니다. 대화를 시작하거나 세션을 로드하세요.")
            elif not session_belongs_to_user(supabase, sid, uid):
                st.error("이 세션에 접근할 권한이 없습니다.")
            else:
                files = fetch_distinct_vector_filenames(supabase, sid, uid)
                if not files:
                    st.text("(이 세션에 저장된 벡터 파일명이 없습니다.)")
                else:
                    for fn in files:
                        st.text(f"- {fn}")

        st.markdown("### RAG (PDF)")
        model_choice = st.radio(
            "LLM 모델 선택",
            ("gpt-4o-mini", "gemini-3-pro-preview", "claude-sonnet-4-5"),
            index=0,
            key="mu_model_radio",
        )
        if model_choice != LLM_MODEL_NAME:
            st.caption("본 과제 명세에 따라 답변 생성에는 gpt-4o-mini만 사용합니다.")

        uploads = st.file_uploader(
            "PDF 파일 업로드",
            type=["pdf"],
            accept_multiple_files=True,
            key="mu_pdf_up",
        )
        if st.button("파일 처리하기", key="mu_proc_pdf"):
            if not uploads:
                st.warning("업로드된 PDF가 없습니다.")
            elif not openai_key:
                st.error("OPENAI_API_KEY가 필요합니다.")
            else:
                try:
                    sid = ensure_db_session(supabase, openai_key, uid)
                    if not sid:
                        st.error("세션을 만들 수 없습니다.")
                    else:
                        names = process_pdf_uploads_to_supabase(
                            supabase, sid, list(uploads), openai_key, uid
                        )
                        st.session_state.processed_names = sorted(
                            set(st.session_state.processed_names) | set(names)
                        )
                        auto_save_session(supabase, openai_key, uid)
                        st.success("PDF 처리 및 벡터 저장이 완료되었습니다.")
                except Exception as exc:  # noqa: BLE001
                    logger.warning("PDF 처리 실패: %s", exc)
                    st.error(f"PDF 처리 중 오류: {exc}")

        if st.session_state.processed_names:
            st.markdown("**처리된 파일**")
            for name in st.session_state.processed_names:
                st.text(f"- {name}")

        mem_count = len(st.session_state.conversation_memory)
        settings_text = (
            f"모델(답변): {LLM_MODEL_NAME}\n"
            f"임베딩: {EMBEDDING_MODEL}\n"
            f"현재 세션 ID: {st.session_state.current_session_id or '(없음)'}\n"
            f"처리된 PDF 파일 수: {len(st.session_state.processed_names)}\n"
            f"대화 기록(메시지) 수: {mem_count}"
        )
        st.text(settings_text)

    for msg in st.session_state.chat_history:
        role = msg["role"]
        content = remove_separators(msg["content"])
        with st.chat_message(role):
            st.markdown(content)

    user_input = st.chat_input("질문을 입력하세요")
    if not user_input:
        return

    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.conversation_memory.append({"role": "user", "content": user_input})
    if len(st.session_state.conversation_memory) > 50:
        st.session_state.conversation_memory = st.session_state.conversation_memory[-50:]

    with st.chat_message("user"):
        st.markdown(remove_separators(user_input))

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_answer = ""

        try:
            sid = ensure_db_session(supabase, openai_key, uid)
            if not sid:
                full_answer = "# 안내\n\n세션을 생성할 수 없어 답변을 저장할 수 없습니다."
                placeholder.markdown(remove_separators(full_answer))
            elif not session_belongs_to_user(supabase, sid, uid):
                full_answer = "# 안내\n\n현재 세션이 사용자에게 속하지 않습니다."
                placeholder.markdown(remove_separators(full_answer))
            elif not openai_key:
                full_answer = "# 안내\n\nOPENAI_API_KEY가 필요합니다."
                placeholder.markdown(remove_separators(full_answer))
            else:
                emb = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=openai_key)
                docs = retrieve_with_rpc(supabase, sid, user_input, emb, k=10)
                llm = ChatOpenAI(
                    model=LLM_MODEL_NAME,
                    temperature=0.7,
                    api_key=openai_key,
                    streaming=True,
                )
                if not docs:
                    mem_txt = _format_memory_block(st.session_state.conversation_memory[:-1])
                    sys = f"{ANSWER_STYLE_SYSTEM}\n\n[대화 맥락]\n{mem_txt or '(없음)'}"
                    msgs = [SystemMessage(content=sys), HumanMessage(content=user_input)]
                    acc = ""
                    for chunk in llm.stream(msgs):
                        piece = getattr(chunk, "content", "") or ""
                        if piece:
                            acc += piece
                            placeholder.markdown(remove_separators(acc) + "▌")
                    full_answer = remove_separators(acc)
                    placeholder.markdown(full_answer)
                else:
                    context = "\n\n".join(d.page_content for d in docs)
                    mem_txt = _format_memory_block(st.session_state.conversation_memory[:-1])
                    messages = _build_rag_messages(user_input, context, mem_txt)
                    acc = ""
                    for chunk in llm.stream(messages):
                        piece = getattr(chunk, "content", "") or ""
                        if piece:
                            acc += piece
                            placeholder.markdown(remove_separators(acc) + "▌")
                    full_answer = remove_separators(acc)
                    placeholder.markdown(full_answer)

                follow_llm = ChatOpenAI(model=LLM_MODEL_NAME, temperature=0.3, api_key=openai_key)
                follow = _generate_followup_section(follow_llm, user_input, full_answer)
                if follow:
                    full_answer += follow
                    placeholder.markdown(remove_separators(full_answer))

        except Exception as exc:  # noqa: BLE001
            logger.warning("답변 생성 실패: %s", exc)
            full_answer = f"# 오류\n\n요청을 처리하는 중 문제가 발생했습니다.\n\n`{exc}`"
            placeholder.markdown(remove_separators(full_answer))

        st.session_state.chat_history.append({"role": "assistant", "content": full_answer})
        st.session_state.conversation_memory.append({"role": "assistant", "content": full_answer})
        if len(st.session_state.conversation_memory) > 50:
            st.session_state.conversation_memory = st.session_state.conversation_memory[-50:]

        auto_save_session(supabase, openai_key, uid)


if __name__ == "__main__":
    main()
