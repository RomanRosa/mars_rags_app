import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv, find_dotenv
import openai

# Assuming necessary imports from langchain are correctly installed and imported
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# Load environment variables
load_dotenv(find_dotenv(), override=True)
openai_api_key = os.environ.get("OPENAI_API_KEY")

# Ensure OpenAI API key is set
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in the environment variables.")

# Streamlit App Configuration - This should be the first Streamlit command
st.set_page_config(page_title="Q&A ChatBot Using RAGs & LLMs - Custom",
                   page_icon='🤖',
                   layout='centered',
                   initial_sidebar_state='collapsed')

@st.cache(allow_output_mutation=True)
def load_data_and_initialize_components(file_name):
    try:
        df = pd.read_csv(file_name, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(file_name, encoding='latin1')
    
    loader = CSVLoader(file_path=file_name, encoding='latin1')
    try:
        docs = loader.load()
    except UnicodeDecodeError:
        loader = CSVLoader(file_path=file_name, encoding='ISO-8859-1')
        docs = loader.load()

    chunk_size = 512
    chunk_overlap = 32

    c_text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )

    r_text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )

    pages = c_text_splitter.split_documents(docs)
    pages += r_text_splitter.split_documents(docs)

    embedding = OpenAIEmbeddings()
    persist_directory = 'persist_chroma'
    vectordb = Chroma.from_documents(
        documents=pages,
        embedding=embedding,
        persist_directory=persist_directory
    )
    vectordb.persist()

    llm_name = "gpt-3.5-turbo"
    llm = ChatOpenAI(model_name=llm_name, temperature=1)

    qa_chain_default = RetrievalQA.from_chain_type(
        llm,
        retriever=vectordb.as_retriever(search_kwargs={"k":3}),
        chain_type="stuff",
        return_source_documents=True
    )

    return qa_chain_default

def get_chatbot_response(question, qa_chain):
    response = qa_chain({"query": question})
    return response


# Load data and initialize components
with st.spinner('Initializing ChatBot Components...'):
    qa_chain_default = load_data_and_initialize_components(
        r"C:\github_repos\mars_question_answering_app_rags\data\MARS_Benchmark_Diciembre (1).csv"
    )
    st.success('ChatBot Ready!')

# User Interaction
#user_question = st.text_input("Ask your Question:", "")
#if user_question:
#    with st.spinner('Generating Answer...'):
#        chatbot_answer = get_chatbot_response(user_question, qa_chain_default)
#        st.write(chatbot_answer)

# User Interaction
st.markdown("# 🤖 BIUBot With RAG...")
st.markdown("## 🤔 Ask your Question")
user_question = st.text_input("", placeholder="Type your question here...")

if user_question:
    with st.spinner('🤖 Generating Answer...'):
        chatbot_answer = get_chatbot_response(user_question, qa_chain_default)
        st.markdown("## 📚 Answer")
        st.write(chatbot_answer)