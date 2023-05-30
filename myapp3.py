from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from tqdm.autonotebook import tqdm
from sentence_transformers import SentenceTransformer
from scalable_semantic_search import ScalableSemanticSearch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline
from langchain.llms import HuggingFacePipeline

def main():

    load_dotenv()
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF 💬")
    st.sidebar.title('I love Marlene, Ask her anything...')
    
    # upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    
    # extract the text
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() 
        # split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=0,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        semantic_search = ScalableSemanticSearch(device="mps")

        # create embeddings
        embedding_index_path = '/Users/brandonl/projects/NLP/notebooks/embedding_index'
        embeddings = semantic_search.encode(chunks)
        #knowledge_base = semantic_search.build_index(embeddings)
        semantic_search.build_index(embeddings)
        knowledge_base = semantic_search.save_index(embedding_index_path)
        #print(knowledge_base)
        
        # show user input
        user_question = st.text_input("Ask a question about your PDF:")
        if user_question:
            top_indices, top_scroes = semantic_search.search(user_question, 5)
            #top_sentences = ScalableSemanticSearch.get_top_sentences(semantic_search.hashmap_index_sentence, top_indices)
            top_sentences = ScalableSemanticSearch.get_top_sentences(semantic_search.hashmap_index_sentence, top_indices)
            tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
            offload_weights = '/Users/brandonl/projects/NLP/notebooks/large_laguage_models/offload_weights'
            model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl", device_map="auto", offload_folder=offload_weights)
            pipe = pipeline(
                "text2text-generation", model=model, tokenizer=tokenizer,
                max_new_tokens=1000, early_stopping=True, no_repeat_ngram_size=2
            )
            st.balloons()
            st.subheader('Progress bar')
            st.progress(10)
            llm = HuggingFacePipeline(pipeline=pipe)
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=top_sentences, question=user_question)
           
            st.write(response)
    

if __name__ == '__main__':
    main()