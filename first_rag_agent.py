import getpass
import os

# Set LangSmith Environment Variables
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = getpass.getpass()


# Set up google llm model for RAG usage
os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")
from langchain.chat_models import init_chat_model
llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

from langchain_google_vertexai import VertexAIEmbeddings

embeddings = VertexAIEmbeddings(model="text-embedding-004")