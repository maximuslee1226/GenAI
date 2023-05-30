from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import os
import pinecone
from langchain.vectorstores import Pinecone
from tqdm.autonotebook import tqdm

def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF 💬")
    
    # upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    
    # extract the text
    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        info = pdf_reader.get_fields()

        print(info)
        pdf_reader.pages[0]
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        print(text)
        # split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap  = 0,
        )

        texts = text_splitter.create_documents([text])

        print(texts[0])

if __name__ == '__main__':
    main()