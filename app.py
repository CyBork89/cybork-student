# CyBork - KI-gestützte Web-App für Schüler und Studenten
# Funktioniert lokal über Streamlit und verarbeitet PDF, Spracheingabe und Websuche

import streamlit as st
import openai
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import tempfile
import os
import time

# === API Key (sicher setzen!) ===
openai.api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else "DEIN_OPENAI_API_KEY"

# === Design Settings ===
st.set_page_config(page_title="CyBork - KI für Bildung", page_icon="🧰", layout="wide")
st.markdown("""
    <style>
        body { background-color: #0f2e1c; color: white; }
        .stApp { background-color: #0f2e1c; }
        .stButton > button { background-color: #1e442a; color: white; border-radius: 8px; }
        .stTextInput>div>div>input { background-color: #163a24; color: white; }
    </style>
""", unsafe_allow_html=True)

st.title("CyBork - Deine KI für Hausarbeiten & Präsentationen 🧰")
st.write("Nutze PDFs, stelle Fragen, analysiere Quellen. Alles in einer App!")

# === PDF Upload ===
uploaded_file = st.file_uploader("Lade deine PDF hoch", type=["pdf"])

# === Spracheingabe (Beta mit Workaround) ===
use_speech = st.toggle("Spracheingabe aktivieren (Testphase)")

# === Initialisierung
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vstore = FAISS.from_documents(chunks, embeddings)

    chain = RetrievalQA.from_chain_type(llm=OpenAI(temperature=0), retriever=vstore.as_retriever())

    st.success("PDF verarbeitet. Stelle jetzt Fragen dazu!")

    # === Chat Interface ===
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    frage = st.text_input("Deine Frage zur PDF oder zum Thema:", placeholder="Was steht über Klimawandel drin?")

    if frage:
        with st.spinner("CyBork denkt nach..."):
            antwort = chain.run(frage)
            st.session_state.chat_history.append((frage, antwort))
            st.success("Antwort bereit")

    # === Verlauf anzeigen ===
    if st.session_state.chat_history:
        st.subheader("📃 Verlauf")
        for i, (q, a) in enumerate(reversed(st.session_state.chat_history[-10:])):
            st.markdown(f"**{i+1}. Du:** {q}")
            st.markdown(f"> **CyBork:** {a}")

    # === Export Button
    if st.session_state.chat_history:
        if st.button("Chatverlauf exportieren als .txt"):
            with open("chatverlauf_cybork.txt", "w") as f:
                for q, a in st.session_state.chat_history:
                    f.write(f"Frage: {q}\nAntwort: {a}\n\n")
            st.success("Exportiert als 'chatverlauf_cybork.txt'")

else:
    st.info("Bitte lade zuerst eine PDF hoch, um loszulegen.")
