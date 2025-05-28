[The tutorial you are following to learn RAG](https://python.langchain.com/docs/tutorials/rag/)

To activate the conda environment associated with this repository, enter this command:
conda activate rag-agent-musings

You are using Google Gemini for your LLM, HuggingFace for your embeddings model, and an In Memory Vector Store for your vector store.

[Using this link for this RAG agent](https://lilianweng.github.io/posts/2023-06-23-agent/)

To use streamlit, run this command:
python -m streamlit run <<desired_streamlit_app>>.py

OR

streamlit run <<desired_streamlit_app>>.py 
(if this doesn't work, or throws a "ModuleNotFound: toml" error, be sure to deactivate your current environment several times and then activate the rag-agent-musings conda environment.)

[A way to add support to handle reading in PDFs, so we can use our RAG agent on the PDF text.](https://medium.com/@dr.booma19/extracting-text-from-pdf-files-using-ocr-a-step-by-step-guide-with-python-code-becf221529ef)