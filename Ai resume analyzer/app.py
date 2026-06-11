import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import tempfile
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="AI Resume Analyzer", page_icon="📄")
st.title("📄 AI Resume Analyzer for Payoneer")
st.write("Upload resume + paste JD → Get match score using RAG")

with st.sidebar:
    st.header("Setup")
    openai_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY") or "")
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key

uploaded_file = st.file_uploader("1. Upload Resume PDF", type="pdf")
jd = st.text_area("2. Paste Payoneer Job Description", height=200)

if st.button("Analyze Match", type="primary") and uploaded_file and jd and openai_key:
    with st.spinner("Reading resume + building vector DB..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            loader = PyPDFLoader(tmp.name)
            docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = splitter.split_documents(docs)

        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(splits, embeddings)

        qa = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(temperature=0),
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )

        query = f"""Analyze fit for this JD.

1. Match Score out of 100
2. Top 3 Strengths
3. Top 3 Missing Skills
4. Advice

Job Description: {jd}"""

        result = qa.run(query)

    st.success("Analysis Complete")
    st.markdown(result)
