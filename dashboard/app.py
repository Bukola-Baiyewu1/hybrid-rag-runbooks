"""Athena dashboard: ask a question, see the cited answer, and compare retrieval modes.

    streamlit run dashboard/app.py

It talks to the Athena API (ATHENA_API_URL, default http://localhost:8000), so
it shows exactly what an API client or the Aegis agent would get.
"""

from __future__ import annotations

import os

import httpx
import streamlit as st

API = os.getenv("ATHENA_API_URL", "http://localhost:8000").rstrip("/")
MODES = {
    "Hybrid + rerank": "hybrid_rerank",
    "Hybrid (RRF)": "hybrid",
    "Dense only": "dense",
    "BM25 only": "sparse",
}

st.set_page_config(page_title="Athena", page_icon="🛡️", layout="wide")
st.title("Athena")
st.caption("Hybrid RAG over DevOps runbooks: dense + BM25, RRF fusion, reranking, verified citations.")


def post(path: str, payload: dict) -> dict:
    r = httpx.post(f"{API}{path}", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()


with st.sidebar:
    strategy = st.selectbox("Chunking strategy", ["headers", "semantic", "fixed"])
    compare = st.toggle("Compare with dense-only", value=True)
    try:
        info = httpx.get(f"{API}/ready", timeout=10).json()
        st.success(f"API ready: {info.get('chunks')} chunks, {info.get('embedder')}")
    except httpx.HTTPError as exc:
        st.error(f"API not reachable at {API}: {exc}")

question = st.text_input("Question", placeholder="How do I roll back the TLS certificate?")
mode_label = st.radio("Retrieval mode", list(MODES), horizontal=True)

if question:
    columns = st.columns(2) if compare and MODES[mode_label] != "dense" else [st.container()]
    configs = [(mode_label, MODES[mode_label])] + ([("Dense only", "dense")] if len(columns) == 2 else [])
    for col, (label, mode) in zip(columns, configs, strict=True):
        with col:
            st.subheader(label)
            try:
                result = post("/ask", {"question": question, "mode": mode, "strategy": strategy})
            except httpx.HTTPError as exc:
                st.error(str(exc))
                continue
            if result["abstained"]:
                st.warning(f"{result['answer']}  \nReason: {result['abstain_reason']}")
            else:
                st.markdown(result["answer"])
            st.metric("Confidence", f"{result['confidence']:.2f}")
            verification = result.get("verification") or {}
            for claim in verification.get("claims", []):
                icon = "✅" if claim["supported"] else "❌"
                st.write(f"{icon} {claim['text']}  `{claim['citations']}`")
            with st.expander("Retrieved passages"):
                for c in result["citations"]:
                    st.markdown(f"**[{c['n']}] {c['source']}**, {c['heading']}, lines {c['lines']}")
                    st.code(c["text"], language="markdown")
