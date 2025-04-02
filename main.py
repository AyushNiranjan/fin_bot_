import streamlit as st
import os
import pickle
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
import yfinance as yf
import pandas as pd
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Initialize LLM
llm = ChatGroq(model_name="llama3-8b-8192", groq_api_key="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")

# Streamlit UI
st.title("FinBot📈📈")

submit = st.sidebar.button("Submit")

col1, col2 = st.columns(2)
with col1:
    btn_fundamental = st.button("FUNDAMENTAL ANALYSIS")
with col2:
    btn_technical = st.button("TECHNICAL ANALYSIS")



# Initialize Session State Variables
for key in ["technical_vectorstore", "fundamental_vectorstore", "technical_qa_chain", "fundamental_qa_chain", "query", "response"]:
    if key not in st.session_state:
        st.session_state[key] = None

# Paths for FAISS index
technical_faiss_path = "technical_faiss.pkl"
fundamental_faiss_path = "fundamental_faiss.pkl"


# Function to Load and Process PDF
def process_pdf(pdf_path, faiss_path, vectorstore_key, qa_chain_key):
    loader = PyPDFLoader(pdf_path)
    data = loader.load()
    st.text(f"Loading data from {pdf_path}...")

    text_splitter = RecursiveCharacterTextSplitter(separators=['\n\n', '\n', '.', ','], chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(data)
    docs = [Document(page_content=chunk.page_content) for chunk in chunks]

    st.text("Generating Embeddings...")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if os.path.exists(faiss_path):
        st.text("Loading Existing FAISS Index...")
        with open(faiss_path, "rb") as f:
            st.session_state[vectorstore_key] = pickle.load(f)
    else:
        st.text("Creating FAISS Index...")
        st.session_state[vectorstore_key] = FAISS.from_documents(docs, embedding_model)
        with open(faiss_path, "wb") as f:
            pickle.dump(st.session_state[vectorstore_key], f)

    retriever = st.session_state[vectorstore_key].as_retriever(search_kwargs={"k": 3})
    st.session_state[qa_chain_key] = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff", return_source_documents=True)
    st.text("Analysis Setup Complete ✅")

# **Technical Analysis**
if btn_technical:
    st.subheader("📊 Performing Technical Analysis...")
    process_pdf("C:\\Users\\ayush\\OneDrive\\Desktop\\Module 2_Technical Analysis zerodha-Ayush project.pdf", technical_faiss_path, "technical_vectorstore", "technical_qa_chain")

# **Fundamental Analysis**
if btn_fundamental:
    st.subheader("📊 Performing Fundamental Analysis...")
    process_pdf("C:\\Users\\ayush\\OneDrive\\Desktop\\Module 3_Fundamental Analysis ayush.pdf", fundamental_faiss_path, "fundamental_vectorstore", "fundamental_qa_chain")




# Query input
st.session_state.query = st.text_input("Enter your question:", value=st.session_state.query)

# Submit Button
if st.button("Submit it"):
    if st.session_state.query:
        if st.session_state.technical_qa_chain:
            st.text("🔍 Searching Technical Analysis FAISS Index...")
            st.session_state.response = st.session_state.technical_qa_chain.invoke(st.session_state.query)
        elif st.session_state.fundamental_qa_chain:
            st.text("🔍 Searching Fundamental Analysis FAISS Index...")
            st.session_state.response = st.session_state.fundamental_qa_chain.invoke(st.session_state.query)


# Display response if available
if st.session_state.response:
    st.write("### Answer:")
    st.write(st.session_state.response["result"])
    if "source_documents" in st.session_state.response:
        st.write("### Sources:")
        for idx, doc in enumerate(st.session_state.response["source_documents"]):
            st.write(f"**Source {idx+1}:** {doc.page_content[:500]}...")

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

# LSTM Model Definition
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        return self.fc(x[:, -1, :])

# Function to Fetch Stock Data
def fetch_stock_data_df(ticker_symbol):
    try:
        stock = yf.Ticker(ticker_symbol)
        history = stock.history(period="6mo")
        if history.empty:
            return None
        history.reset_index(inplace=True)
        history["Date"] = history["Date"].dt.strftime("%Y-%m-%d")
        return history[["Date", "Close"]]
    except Exception as e:
        st.error(f"Error fetching stock data: {str(e)}")
        return None

# Function to Train LSTM Model
def train_lstm(df, epochs=50):
    scaler = MinMaxScaler()
    data = scaler.fit_transform(df["Close"].values.reshape(-1, 1))

    X, y = [], []
    for i in range(30, len(data) - 1):
        X.append(data[i - 30:i])
        y.append(data[i + 1])

    X, y = np.array(X), np.array(y)
    X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    model = LSTMModel(1, 64, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()

    return model, scaler

# Function to Predict Future Prices
def predict_lstm(model, scaler, df, days=7):
    data = scaler.transform(df["Close"].values.reshape(-1, 1))
    inputs = torch.tensor(data[-30:].reshape(1, 30, 1), dtype=torch.float32)

    future_prices = []
    for _ in range(days):
        with torch.no_grad():
            pred = model(inputs).numpy()
        future_prices.append(pred[0][0])
        inputs = torch.cat([inputs[:, 1:, :], torch.tensor(pred).reshape(1, 1, 1)], axis=1)

    predicted_prices = scaler.inverse_transform(np.array(future_prices).reshape(-1, 1))

    # Generate future dates
    last_date = datetime.strptime(df["Date"].iloc[-1], "%Y-%m-%d")
    future_dates = [(last_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, days + 1)]

    return pd.DataFrame({"Date": future_dates, "Predicted Close Price": predicted_prices.flatten()})

# Streamlit UI
st.title("Stock Price Prediction 📈")

ticker_symbol = st.text_input("Enter Stock Ticker").upper()
btn_predict = st.button("Predict Future Prices")

# Session State Initialization
if "stock_data" not in st.session_state:
    st.session_state.stock_data = None
if "lstm_model" not in st.session_state:
    st.session_state.lstm_model = None
if "scaler" not in st.session_state:
    st.session_state.scaler = None

# Fetch Data and Train Model
if btn_predict:
    if ticker_symbol:
        st.session_state.stock_data = fetch_stock_data_df(ticker_symbol)

        if st.session_state.stock_data is not None:
            st.session_state.lstm_model, st.session_state.scaler = train_lstm(st.session_state.stock_data)
            st.success("✅ Model trained successfully!")

            # Predict and Display Results
            predicted_df = predict_lstm(st.session_state.lstm_model, st.session_state.scaler, st.session_state.stock_data)

            st.subheader("📈 Predicted Future Prices")
            st.write(predicted_df)

            # Plot the results
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(predicted_df["Date"], predicted_df["Predicted Close Price"], marker='o', linestyle='-', color='b', label="Predicted Prices")
            plt.xticks(rotation=45)
            plt.xlabel("Date")
            plt.ylabel("Stock Price (USD)")
            plt.title(f"Predicted Stock Prices for {ticker_symbol}")
            plt.legend()
            st.pyplot(fig)

        else:
            st.error("⚠️ Invalid stock ticker. Please try again.")
    else:
        st.warning("⚠️ Please enter a stock ticker.")
