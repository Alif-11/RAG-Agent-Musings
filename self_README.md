LangChain Project Ideas (obtained from [here](https://www.reddit.com/r/LangChain/comments/13bw60e/looking_for_project_ideas_for_learning_langchain/)):

[DONE] - beginner: question answering bot with any existing data source integration


[DID THROUGH PDF] - intermediate: do some web scraping to get your own data(use language model to help you parse the html), combine multiple data sources, e.g. review aggregator(present data as charts, word cloud), a news aggregator that collect news on a particular topic(e.g. new open source LLMs announcement, use LLMs to identify relevant posts), waiter bot that can take order based on an input menu(only accepts order of items on the menu, calculate the total price, can check inventory, etc), travel planning app(ask users for preference, combine data from tour guide, hotel/flight booking sites, reviews, museum/park/restaurant sites), auto scrap a website based on site map and create a chatbot, make an AI fact check site like theflipside.io

[The site link I'm going to use for the expert project](https://huggingface.co/blog/dvgodoy/fine-tuning-llm-hugging-face)

expert: fine tune your own LLM and embedding, integrate with other models, use pix2struct to parse a document image/use whisper to transcribe a podcast, make several bots that do different tasks and can interact with each other, make a bot that can automatically do online shopping based on a shopping list/fill in an application form based on your resume, use tts and ai avatar to present your output.


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