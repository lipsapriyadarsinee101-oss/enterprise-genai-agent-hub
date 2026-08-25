import os

import requests
import streamlit as st


st.set_page_config(page_title="Enterprise GenAI Agent Hub", page_icon="🧠", layout="wide")
st.title("Enterprise GenAI Agent Hub")
st.caption("Multi-agent RAG with grounded answers, citations, and quality checks")

with st.sidebar:
    st.header("Configuration")
    api_url = st.text_input("API URL", os.getenv("API_URL", "http://localhost:8000"))
    top_k = st.slider("Evidence chunks", 1, 6, 3)
    st.info("Demo mode works without an API key. Start the FastAPI service before asking a question.")

examples = [
    "What controls are required before deploying an AI model?",
    "How should a production RAG pipeline be monitored?",
    "What is the incident response process?",
]
question = st.selectbox("Try an example", [""] + examples)
question = st.text_area("Ask an enterprise knowledge question", value=question, height=100)

if st.button("Run agent workflow", type="primary", disabled=not question.strip()):
    try:
        with st.spinner("Agents are retrieving and validating evidence..."):
            response = requests.post(f"{api_url.rstrip('/')}/v1/query", json={"question": question, "top_k": top_k}, timeout=60)
            response.raise_for_status()
            result = response.json()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Route", result["route"].title())
        c2.metric("Confidence", f'{result["confidence"]:.0%}')
        c3.metric("Grounded", "Yes" if result["grounded"] else "No")
        c4.metric("Latency", f'{result["latency_ms"]} ms')
        st.subheader("Answer")
        st.markdown(result["answer"])
        with st.expander("Agent trace", expanded=True):
            for index, step in enumerate(result["trace"], 1):
                st.markdown(f'**{index}. {step["agent"].replace("_", " ").title()}** — {step["action"]}: {step["detail"]}')
        with st.expander("Retrieved evidence"):
            for citation in result["citations"]:
                st.markdown(f'**{citation["source"]}** · similarity `{citation["score"]:.3f}`')
                st.write(citation["excerpt"])
    except requests.RequestException as exc:
        st.error(f"Could not reach the API: {exc}")
