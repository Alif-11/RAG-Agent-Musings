import streamlit as st
import os 
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain import hub
from langchain_core.documents import Document
from typing_extensions import List, TypedDict

os.environ["LANGSMITH_TRACING"] = "true"
st.title("RAG Agent")
website_to_scrape_link = st.text_input("Input The Website You Want The Agent To Scrape For Context", type="default")
user_query = st.text_input("Input Your Query Here", type="default")

langsmith_api_key = ""
google_gemini_api_key = ""

with st.sidebar:
  st.header("API Keys")
  langsmith_api_key = st.text_input("LangSmith API Key", type="password", help="A LangSmith API Key is required to use this app.")
  google_gemini_api_key = st.text_input("Google Gemini Flash 2.0 API Key", type="password", help="A Google Gemini Flash 2.0 API Key is required to use this app.")
  
if langsmith_api_key != "":
  print(f"langsmith api key: {langsmith_api_key}")
  os.environ["LANGSMITH_API_KEY"] = langsmith_api_key

if google_gemini_api_key != "":
  print(f"google gemini api key: {google_gemini_api_key}")
  os.environ["GOOGLE_API_KEY"] = google_gemini_api_key

if langsmith_api_key == "" or google_gemini_api_key == "":
  st.warning("Please enter both your LangSmit and Google Gemini Flash API Keys to use this app.")
else:
  pass