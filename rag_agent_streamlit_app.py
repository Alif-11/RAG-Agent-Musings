import streamlit as st

st.title("RAG Agent")
website_to_scrape_link = st.text_input("Input The Website You Want The Agent To Scrape For Context", type="default")
user_query = st.text_input("Input Your Query Here", type="default")

api_key = ""

with st.sidebar:
  st.header("API Keys")
  langsmith_api_key = st.text_input("LangSmith API Key", type="password", help="A LangSmith API Key is required to use this app.")
  google_gemini_api_key = st.text_input("Google Gemini Flash 2.0 API Key", type="password", help="A Google Gemini Flash 2.0 API Key is required to use this app.")
  
if langsmith_api_key != "":
  print(f"langsmith api key: {langsmith_api_key}")

if google_gemini_api_key != "":
  print(f"google gemini api key: {google_gemini_api_key}")