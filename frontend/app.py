import streamlit as st

from api_client import ApiError, ask_question, check_health

st.set_page_config(page_title="Data Viz Course Assistant", page_icon="📊")

st.title("📊 Data Visualization Course Assistant")
st.caption("Ask a question about the lecture notes — answers are grounded in the course material, with sources cited.")

if not check_health():
    st.warning(
        "Can't reach the backend (or the vector store isn't loaded yet). "
        "Start it with `uvicorn app.main:app --reload` from the backend/ folder, "
        "and make sure Ollama is running."
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            st.caption("Sources: " + ", ".join(message["sources"]))

question = st.chat_input("Ask about the lecture notes...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating an answer..."):
            try:
                result = ask_question(question)
                st.markdown(result["answer"])
                if result["sources"]:
                    st.caption("Sources: " + ", ".join(result["sources"]))
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                })
            except ApiError as exc:
                st.error(str(exc))
                st.session_state.messages.append({"role": "assistant", "content": f"⚠️ {exc}"})
