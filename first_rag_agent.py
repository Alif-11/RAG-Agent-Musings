import getpass
import os

# Set LangSmith Environment Variables
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter API key for LangSmith")


# Set up google llm model for RAG usage
os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")


print("Initializing chat model!")
from langchain.chat_models import init_chat_model

llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
print("Chat model initialization over!")


print("Initializing embedder model!")
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
print("Embedder model initialized!")

print("Initialize vector store!")
from langchain_core.vectorstores import InMemoryVectorStore

vector_store = InMemoryVectorStore(embeddings)
print("Vector store initialized!")