import os

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Home Assistant RAG", page_icon="🏠")
st.title("🏠 Home Assistant RAG")
st.caption("Ask about Home Assistant, Zigbee2MQTT, or ESPHome")

with st.sidebar:
    mode = st.selectbox("Retrieval mode", ["hybrid", "vector", "text"], index=0)
    prompt_version = st.selectbox("Prompt version", ["v2", "v1"], index=0)
    use_rerank = st.checkbox("Rerank (cross-encoder)", value=True,
                              help="Evaluated to improve MRR@5 from 0.77 to 0.88 — see eval/evaluate_reranking.py")

if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        with st.expander("Sources"):
            for s in turn["sources"]:
                st.markdown(f"- [{s['title'] or s['url']}]({s['url']})")
        col1, col2 = st.columns(2)
        if col1.button("👍", key=f"up_{turn['conversation_id']}"):
            requests.post(f"{API_BASE_URL}/feedback",
                          json={"conversation_id": turn["conversation_id"], "rating": 1})
            st.toast("Thanks for the feedback!")
        if col2.button("👎", key=f"down_{turn['conversation_id']}"):
            requests.post(f"{API_BASE_URL}/feedback",
                          json={"conversation_id": turn["conversation_id"], "rating": -1})
            st.toast("Thanks, noted.")

question = st.chat_input("Ask a question...")
if question:
    with st.spinner("Thinking..."):
        resp = requests.post(
            f"{API_BASE_URL}/ask",
            json={"question": question, "mode": mode, "prompt_version": prompt_version,
                  "rerank": use_rerank},
            timeout=60,
        )
        resp.raise_for_status()
        st.session_state.history.append(resp.json())
    st.rerun()
