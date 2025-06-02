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
from PIL import Image


global global_llm, global_rag_prompt, global_vector_store

class RAG_State(TypedDict):
  question: str
  context: List[Document] # remember, context should be several chunks of a document
  answer: str

def retrieve(rag_state: RAG_State):
  """
  Retrieves relevant data, based on our user question, from the vector store

  Args:
    rag_state: A dictionary which, for this problem, contains the user question, in the key 'question'. Has to be of type RAG_State
  
  Returns:
    All relevant context document chunks. Should be a dictionary of the following form: {str ; List[Document]}
  """
  global global_vector_store
  similar_document_chunks = global_vector_store.similarity_search(rag_state["question"])
  return {"context" : similar_document_chunks}

def generate(rag_state: RAG_State):
  """
  Generate appropriate RAG response to the user query.

  Args:
    rag_state: A dictionary which, for this problem, contains the user question, in the key 'question', and the context of the question, in the key 'context'. Has to be of type RAG_State
  
  Returns:
    The model's response. Should be a dictionary of the following form: {str ; ?}
  """
  global global_llm, global_rag_prompt
  formalized_RAG_prompt = global_rag_prompt.invoke({"question": rag_state["question"], "context": rag_state["context"]})
  llm_RAG_response = global_llm.invoke(formalized_RAG_prompt)
  return {"answer" : llm_RAG_response.content}

os.environ["LANGSMITH_TRACING"] = "true"
st.title("RAG Agent")

langsmith_api_key = ""
google_gemini_api_key = ""

ai_models_initialized = False

def initialize_ai_models():
  global ai_models_initialized
  if not ai_models_initialized:
    print("Initializing chat model!")
    llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
    print("Chat model initialized!")

    print("Initializing embedder model!")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    print("Embedder model initialized!")

    print("Initialize vector store!")
    vector_store = InMemoryVectorStore(embeddings)
    print("Vector store initialized!")
    rag_prompt = hub.pull("rlm/rag-prompt")
    ai_models_initialized = True
  return llm, embeddings, vector_store, rag_prompt

def submit_button_text(tab_type):
  """
  This function will handle the submit button functionality under either tab.
  """
  global langsmith_api_key, google_gemini_api_key
  global global_llm, global_rag_prompt, global_vector_store
  user_query = st.text_input("Input Your Query Here", type="default", key=f"user query key: {tab_type}")
  if st.button("Submit", key=tab_type):
    if langsmith_api_key == "" or google_gemini_api_key == "":
      st.warning("Please enter both your LangSmit and Google Gemini Flash API Keys to use this app.")
    elif website_to_scrape_link == "":
      st.warning("Please enter a website to scrape to use this app.")
    elif user_query == "":
      st.warning("Please enter a query to use this app.")
    else:
      with st.spinner("Initializing AI Models..."):
        llm, embeddings, vector_store, rag_prompt = initialize_ai_models()
      global_llm = llm
      global_vector_store = vector_store
      global_rag_prompt = rag_prompt
      if tab_type == "website":
        with st.spinner("Initializing Website Loader..."):
          bs4_filterer = bs4.SoupStrainer()
          webpage_loader = WebBaseLoader(
          web_paths=(website_to_scrape_link,),
          bs_kwargs={"parse_only": bs4_filterer}
          )
        with st.spinner("Loading Website..."):
          documents = webpage_loader.load()
        with st.spinner("Splitting Text From Website..."):
          recursive_text_splitter = RecursiveCharacterTextSplitter(
          chunk_size=1000,
          chunk_overlap=200,
          add_start_index=True,
          )

        splitted_text = recursive_text_splitter.split_documents(documents)
        vector_store.add_documents(splitted_text)

        from langgraph.graph import START, StateGraph # START is a special Start Node

        rag_agent_maker = StateGraph(RAG_State).add_sequence([retrieve, generate])
        rag_agent_maker.add_edge(START, "retrieve") # links the special Start node to the retrieve node sequence we defined above.
        rag_agent = rag_agent_maker.compile() # now we have made our RAG agent.

        with st.spinner("Generating Response..."):
          rag_agent_response = rag_agent.invoke({"question": user_query})

        st.write(f"Context: {rag_agent_response['context'][0].page_content}")
        st.write(f"Answer: {rag_agent_response['answer']}")


      elif tab_type == "pdf":
        pass

with st.sidebar:
  st.header("API Keys")
  langsmith_api_key = st.text_input("LangSmith API Key", type="password", help="A LangSmith API Key is required to use this app.")
  google_gemini_api_key = st.text_input("Google Gemini Flash 2.0 API Key", type="password", help="A Google Gemini Flash 2.0 API Key is required to use this app.")
  
if langsmith_api_key != "":
  #print(f"langsmith api key: {langsmith_api_key}")
  os.environ["LANGSMITH_API_KEY"] = langsmith_api_key

if google_gemini_api_key != "":
  #print(f"google gemini api key: {google_gemini_api_key}")
  os.environ["GOOGLE_API_KEY"] = google_gemini_api_key

website_tab, pdf_tab = st.tabs(["Website To Scrape", "PDF To Scrape"])



with website_tab:
  website_to_scrape_link = st.text_input("Input The Website You Want The Agent To Scrape For Context", type="default")

with website_tab:
  submit_button_text("website")

with pdf_tab:
  uploaded_pdf = st.file_uploader("Upload the desired text-containing PDF", type=["pdf"])
  submit_button_text("pdf")



