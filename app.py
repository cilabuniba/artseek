"""
ArtSeek – interactive demo
==========================
Launch with:  streamlit run app.py

Needs a GPU and a running Qdrant server with the WikiFragments collection
(see README.md). Settings are read from the environment or from a `.env` file
in the repository root (see `.env.example`).
"""

import os

from dotenv import load_dotenv

# Must run before anything imports huggingface_hub, so that HF_HOME and the
# offline flags in .env are honoured.
load_dotenv()
# vLLM keeps a large CUDA memory pool on the GPU shared with the retriever;
# expandable segments reduce fragmentation between the two.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import io
import re

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from PIL import Image

# ── Page configuration ───────────────────────────────────────────────────────

st.set_page_config(
    page_title="ArtSeek",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown(
    """
<style>
/* tighter card badges */
.stProgress > div > div > div {
    height: 8px !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ── Model loading (cached across reruns) ─────────────────────────────────────


@st.cache_resource(show_spinner="Loading ArtSeek model… this may take a few minutes ⏳")
def load_model():
    from artseek.method.generate.config import get_model

    return get_model()


# ── Session-state defaults ───────────────────────────────────────────────────

_DEFAULTS: dict = {
    "messages": [],  # list[BaseMessage]
    "context_images": [],  # accumulated resized context imgs for the model
    "input_image": None,  # PIL.Image – the uploaded artwork
    "card": None,  # dict – classification results
    "tool_artifacts": {},  # tool_call_id → artifact dict
    "_last_upload": None,  # fingerprint to detect new uploads
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Helpers ──────────────────────────────────────────────────────────────────

TASK_EMOJI = {
    "artist": "🎨",
    "genre": "📚",
    "media": "🖌️",
    "style": "✨",
    "tag": "🏷️",
}

TASK_COLOR = {
    "artist": "#FF6B6B",
    "genre": "#4ECDC4",
    "media": "#45B7D1",
    "style": "#96CEB4",
    "tag": "#FFEAA7",
}


def _prob_value(prob) -> float:
    """Safely convert a tensor / float to a plain Python float."""
    return prob.item() if hasattr(prob, "item") else float(prob)


def render_card(card: dict):
    """Render the classification card as columns with progress bars."""
    cols = st.columns(len(card))
    for col, (task, preds) in zip(cols, card.items()):
        with col:
            emoji = TASK_EMOJI.get(task, "📌")
            st.markdown(f"**{emoji} {task.capitalize()}**")
            for pred, prob in preds:
                pct = int(_prob_value(prob) * 100)
                st.progress(pct / 100, text=f"{pred} — {pct}%")


def parse_ai_content(content: str) -> tuple[str | None, str]:
    """Split an AI message into *(thinking, visible_response)*.

    The model often omits the opening ``<think>`` tag and only emits
    ``</think>``, so we look for the closing tag and treat everything
    before it as the thinking portion.
    """
    idx = content.find("</think>")
    if idx != -1:
        thinking = content[:idx].replace("<think>", "").strip()
        visible = content[idx + len("</think>"):].strip()
    else:
        thinking = None
        visible = content.strip()
    return thinking, visible


def render_documents(artifact: dict | None):
    """Display the retrieved documents stored in a tool-call artifact."""
    if not artifact or "documents" not in artifact:
        st.info("Documents retrieved (details unavailable)")
        return

    for doc in artifact["documents"]:
        with st.container(border=True):
            score_pct = int(doc["score"] * 100)
            st.markdown(f"**📖 {doc['title']}** · relevance {score_pct}%")

            # Document images (thumbnails)
            imgs = doc.get("images", [])
            if imgs:
                img_cols = st.columns(min(len(imgs), 4))
                for i, entry in enumerate(imgs):
                    with img_cols[i % len(img_cols)]:
                        img = entry["image"]
                        if isinstance(img, dict):
                            img = Image.open(io.BytesIO(img["bytes"]))
                        if isinstance(img, Image.Image):
                            img.thumbnail((200, 200), Image.LANCZOS)
                        st.image(
                            img,
                            caption=entry.get("caption", ""),
                            use_container_width=False,
                        )

            # Document text
            with st.expander("📝 Full text"):
                st.markdown(doc.get("text", ""))


def render_timing_breakdown(step_durations: list[dict]):
    """Render a per-step timing breakdown for a full model turn."""
    total = sum(s["duration"] for s in step_durations)
    n = len(step_durations)
    with st.expander(
        f"⏱️ Answered in {total:.1f}s ({n} step{'s' if n != 1 else ''})",
        expanded=False,
    ):
        for i, step in enumerate(step_durations, start=1):
            pct = int(step["duration"] / total * 100) if total else 0
            if step["type"] == "generation":
                st.progress(
                    pct / 100,
                    text=f"{i}. 🧠 Generation/reasoning — {step['duration']:.1f}s",
                )
            else:
                q = step.get("query", "")
                st.progress(
                    pct / 100,
                    text=f"{i}. 🔍 Retrieving *{q}* — {step['duration']:.1f}s",
                )


def render_message(msg):
    """Render one message from the conversation history."""

    # ── User message ─────────────────────────────────────────────────────
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            if isinstance(msg.content, list):
                # Multimodal first message – only show the query part
                for part in msg.content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        txt = part["text"]
                        if "\n# Query\n" in txt:
                            st.markdown(txt.split("\n# Query\n", 1)[-1])
            else:
                st.markdown(msg.content)

    # ── AI message ───────────────────────────────────────────────────────
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant", avatar="🎨"):
            # Tool calls (if any)
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    q = tc["args"].get("query", "")
                    img_flag = " 🖼️" if tc["args"].get("requires_image") else ""
                    with st.expander(
                        f"🔍 Retrieving: *{q}*{img_flag}", expanded=False
                    ):
                        st.code(
                            f'query = {q!r}\nrequires_image = {tc["args"].get("requires_image", False)}',
                            language="python",
                        )

            # Content (thinking + response)
            if msg.content:
                thinking, response = parse_ai_content(msg.content)
                if thinking:
                    with st.expander("💭 Reasoning", expanded=False):
                        st.markdown(thinking)
                if response:
                    st.markdown(response)

            # Per-step generation time + full turn breakdown (set on the
            # final AIMessage of a turn, once tool-calling is done)
            duration = msg.response_metadata.get("duration")
            step_durations = msg.response_metadata.get("step_durations")
            if step_durations:
                render_timing_breakdown(step_durations)
            elif duration is not None:
                st.caption(f"⏱️ Generated in {duration:.1f}s")

    # ── Tool result message ──────────────────────────────────────────────
    elif isinstance(msg, ToolMessage):
        with st.chat_message("assistant", avatar="📚"):
            artifact = st.session_state.tool_artifacts.get(msg.tool_call_id)
            n = (
                len(artifact["documents"])
                if artifact and "documents" in artifact
                else "?"
            )
            duration = artifact.get("duration") if artifact else None
            label = f"📄 Retrieved {n} documents"
            if duration is not None:
                label += f" · ⏱️ {duration:.1f}s"
            with st.expander(label, expanded=False):
                render_documents(artifact)


def build_first_user_message(prompt: str, card: dict | None) -> HumanMessage:
    """Build the first user message (image placeholder, artwork card, query)."""
    from artseek.method.generate.prompts import build_user_content

    return HumanMessage(content=build_user_content(prompt, card))


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🎨 ArtSeek")
    st.caption("Interactive Artwork Analysis")

    uploaded = st.file_uploader(
        "Upload artwork",
        type=["jpg", "jpeg", "png", "webp"],
        help="Upload an artwork image to start a conversation.",
    )

    if uploaded is not None:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, use_container_width=True)

        # Detect a new upload (name + size fingerprint)
        upload_key = f"{uploaded.name}_{uploaded.size}"
        if st.session_state._last_upload != upload_key:
            st.session_state._last_upload = upload_key
            st.session_state.input_image = img
            # Reset conversation
            st.session_state.messages = []
            st.session_state.context_images = []
            st.session_state.tool_artifacts = {}
            st.session_state.card = None
            # Classify
            model = load_model()
            with st.spinner("🔍 Analysing artwork…"):
                st.session_state.card = model.classify(img)
            st.rerun()

    st.divider()
    classify = st.toggle("Use Classification", value=True, key="use_classify")
    retrieve = st.toggle("Use Retrieval (RAG)", value=True, key="use_retrieve")

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages.clear()
        st.session_state.context_images.clear()
        st.session_state.tool_artifacts.clear()
        st.rerun()

# ── Main area ────────────────────────────────────────────────────────────────

if st.session_state.input_image is None:
    st.title("🎨 ArtSeek")
    st.markdown(
        """
    Welcome to **ArtSeek**, an interactive art-analysis assistant powered by
    multimodal retrieval-augmented generation.

    **Getting started**
    1. 📤 Upload an artwork image in the sidebar
    2. 💬 Ask questions about the artwork
    3. 🔍 The model retrieves relevant art-historical documents and reasons
       over them before answering

    **Capabilities**
    | Feature | Description |
    |---------|-------------|
    | 🎨 Classification | Predicts artist, genre, style, media and tags |
    | 📚 RAG Retrieval | Fetches relevant art-historical documents |
    | 💭 Chain-of-thought | Transparent reasoning you can expand |
    | 🔄 Multi-turn chat | Ask follow-up questions naturally |
    """
    )
    st.stop()

# ── Classification card (main area, always visible) ──────────────────────────

if st.session_state.card and classify:
    with st.container(border=True):
        st.markdown("### 🎨 Artwork Classification")
        render_card(st.session_state.card)

# ── Chat history ─────────────────────────────────────────────────────────────

for msg in st.session_state.messages:
    render_message(msg)

# ── Chat input ───────────────────────────────────────────────────────────────

if prompt := st.chat_input("Ask about the artwork…"):
    model = load_model()

    # Build user message
    if not st.session_state.messages:
        card = st.session_state.card if classify else None
        user_msg = build_first_user_message(prompt, card)
    else:
        user_msg = HumanMessage(content=prompt)

    st.session_state.messages.append(user_msg)

    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.spinner("Thinking…"):
        new_msgs, new_ctx = model.chat_turn(
            messages=list(st.session_state.messages),
            input_image=st.session_state.input_image,
            classify=classify,
            retrieve=retrieve,
            context_images=list(st.session_state.context_images),
        )

    # Store new messages + artifacts
    for msg in new_msgs:
        st.session_state.messages.append(msg)
        if isinstance(msg, ToolMessage) and hasattr(msg, "artifact") and msg.artifact:
            st.session_state.tool_artifacts[msg.tool_call_id] = msg.artifact

    st.session_state.context_images.extend(new_ctx)

    # Rerun so the full history renders cleanly
    st.rerun()
