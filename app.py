import os
import json
import numpy as np
import pandas as pd
from PIL import Image
import pickle
import requests
import streamlit as st
import torch  # Import torch to set the device
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import UnstructuredFileLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter

# Set CUDA to not use any GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "" 

# Set device to CPU
device = torch.device("cpu")

# API anahtarını ayarlama
def set_api_key():
    working_dir = os.path.dirname(os.path.abspath(__file__))
    config_data = json.load(open(f"{working_dir}/config.json"))
    groq_api_key = config_data["GROQ_API_KEY"]
    os.environ["GROQ_API_KEY"] = groq_api_key

# Ollama API üzerinden Gemma modeline istek gönderme
def get_gemma_explanation(prompt):
    try:
        ollama_endpoint = "http://localhost:11434/api/generate"
        payload = json.dumps({"model": "gemma:2b", "prompt": prompt, "stream": False})
        response = requests.post(ollama_endpoint, data=payload)
        response.raise_for_status()
        return response.json().get("response", "No response from Ollama.")
    except requests.exceptions.RequestException as e:
        return f"Error contacting Ollama API: {str(e)}"

# Eğitimli modeli ve scaler'ı yükleme
def load_model_and_scaler():
    with open('covid.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('covid_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

# COVID-19 tahmini yapma
def covid_prediction(model, image, scaler):
    image = image.convert("L")
    image = image.resize((28, 28))
    image_array = np.array(image).flatten()

    if scaler:
        image_array = scaler.transform([image_array])
    
    prediction = model.predict(image_array)
    return prediction[0]

# Belgeleri yükleme
def load_documents():
    loader = DirectoryLoader(path="data", glob="./*.pdf", loader_cls=UnstructuredFileLoader)
    documents = loader.load()
    return documents

# Metinleri parçalara ayırma
def split_texts(documents):
    text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
    return text_splitter.split_documents(documents)

# Vektör veritabanını oluşturma
def setup_vectorstore(documents):
    embeddings = HuggingFaceEmbeddings()  # CPU kullan
    persist_directory = "vector_db_dir"
    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory
    )

# ChatChain oluşturma
def chat_chain(vectorstore):
    llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0)
    retriever = vectorstore.as_retriever()
    memory = ConversationBufferMemory(
        llm=llm,
        output_key="answer",
        memory_key="chat_history",
        return_messages=True
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        memory=memory,
        verbose=True,
        return_source_documents=True
    )

# Streamlit uygulaması
def app():
    set_api_key()
    rf, sc = load_model_and_scaler()

    st.set_page_config(page_title="COVID-19 Detection & Multi Documents Chatbot", page_icon="📚", layout="centered")
    st.title("📚 COVID-19 Detection & Multi Documents Chatbot")

    # COVID-19 tespiti
    st.subheader("Upload an X-ray Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        label = covid_prediction(rf, image, sc)
        if label == 0:
            st.error("Unfortunately, you are infected by COVID-19")
            base_prompt = "COVID-19 infection detected in the x-ray image."
        else:
            st.success("Everything is OK")
            base_prompt = "No sign of COVID-19 infection in the x-ray image."
        
        # ELI5 seçeneği
        eli5_mode = st.checkbox("Explain like I'm 5 (ELI5)")
        if eli5_mode:
            prompt = f"{base_prompt} Please explain it like I'm 5."
        else:
            prompt = f"{base_prompt} Please provide a detailed explanation."

        explanation = get_gemma_explanation(prompt)
        st.write(explanation)

    # Belgeleri yükleyip vektör veritabanını oluşturma
    documents = load_documents()
    text_chunks = split_texts(documents)
    vectorstore = setup_vectorstore(text_chunks)

    if "conversational_chain" not in st.session_state:
        st.session_state.conversational_chain = chat_chain(vectorstore)

    # Kullanıcıdan gelen soruları işleme
    user_input = st.text_input("Ask AI about your documents...")

    if user_input:
        response = st.session_state.conversational_chain({"question": user_input})
        assistant_response = response["answer"]
        st.write(assistant_response)

if __name__ == "__main__":
    app()
