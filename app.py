"""
ArtSeek – Interactive Art Analysis Demo
========================================
Launch with:  streamlit run app.py
Requires:     pip install streamlit
"""

import os

os.environ["HF_HOME"] = "data_/hf"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

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


@st.cache_resource(show_spinner="Loading ArtSeek model… this may take a minute ⏳")
def load_model():
    from artseek.method.generate.pipe import MODEL

    return MODEL


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
    """Split an AI message into *(thinking, visible_response)*."""
    m = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
    thinking = m.group(1).strip() if m else None
    visible = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
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

            # Document images
            imgs = doc.get("images", [])
            if imgs:
                img_cols = st.columns(min(len(imgs), 4))
                for i, entry in enumerate(imgs):
                    with img_cols[i % len(img_cols)]:
                        img = entry["image"]
                        if isinstance(img, dict):
                            img = Image.open(io.BytesIO(img["bytes"]))
                        st.image(
                            img,
                            caption=entry.get("caption", ""),
                            use_container_width=True,
                        )

            # Document text
            with st.expander("📝 Full text"):
                st.markdown(doc.get("text", ""))


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

    # ── Tool result message ──────────────────────────────────────────────
    elif isinstance(msg, ToolMessage):
        with st.chat_message("assistant", avatar="📚"):
            artifact = st.session_state.tool_artifacts.get(msg.tool_call_id)
            n = (
                len(artifact["documents"])
                if artifact and "documents" in artifact
                else "?"
            )
            with st.expander(f"📄 Retrieved {n} documents", expanded=False):
                render_documents(artifact)


def build_first_user_message(prompt: str, card: dict | None) -> HumanMessage:
    """Build the first user message (includes the image placeholder + card)."""
    parts: list[dict] = [
        {"type": "text", "text": "# Current query image\n"},
        {"type": "image"},
    ]
    if card:
        card_text = "\n# Artwork card"
        for k, v in card.items():
            card_text += f"\n{k}: "
            for i, (pred, prob) in enumerate(v):
                card_text += f"{pred} ({int(_prob_value(prob) * 100)}%)"
                if i < len(v) - 1:
                    card_text += ", "
        parts.append({"type": "text", "text": card_text})
    parts.append({"type": "text", "text": f"\n# Query\n{prompt}"})
    return HumanMessage(content=parts)


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

    # Classification card
    if st.session_state.card and classify:
        st.divider()
        render_card(st.session_state.card)

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
