import streamlit as st
from embedding import EmbeddingRetriever
from llm_interface import LocalLLM

st.set_page_config(page_title="RAG Spese", layout="centered")

st.title("💬 Interroga le tue spese!")

query = st.text_input("Scrivi una domanda sui tuoi dati:")

if query:
    # Retrieval
    retriever = EmbeddingRetriever.from_csv("spese.csv")
    context = retriever.get_context(query)

    # Prompt
    prompt = f"""
    Contesto:
    ---
    {context}
    ---
    Domanda:
    {query}
    Risposta:
    """

    # LLM
    llm = LocalLLM()
    risposta = llm.ask(prompt)

    st.markdown("### ✨ Risposta")
    st.write(risposta)
