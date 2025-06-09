[Extending the below chat tutorial by adding memory to your chat bot](https://python.langchain.com/docs/tutorials/qa_chat_history/)
---

On the "This model for state is so versatile..." section for the code.


To activate the conda environment associated with this repository, enter this command:
conda activate rag-agent-musings

You are using Google Gemini for your LLM, HuggingFace for your embeddings model, and an In Memory Vector Store for your vector store.

[Using this link for this RAG agent](https://lilianweng.github.io/posts/2023-06-23-agent/)

To use streamlit, run this command:
python -m streamlit run <<desired_streamlit_app>>.py

OR

streamlit run <<desired_streamlit_app>>.py 
(if this doesn't work, or throws a "ModuleNotFound: toml" error, be sure to deactivate your current environment several times and then activate the rag-agent-musings conda environment.)

[A way to add support to handle reading in PDFs, so we can use our RAG agent on the PDF text.](https://github.com/nainiayoub/pdf-text-data-extractor/blob/main/functions.py#L15)

Please look at the images_to_txt function.

Please note that you must take the pdf output of the file uploader, then run .read() on it to get the bytes object. You can then pass that bytes object, along with the key parameter "eng" to the images_to_txt function to properly read in the PDF text.