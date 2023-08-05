# Simple AI pdf reader
# Using langchain and python
from langchain.vectorstores import FAISS
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    NLTKTextSplitter,
    TokenTextSplitter,
)
from langchain.chat_models import AzureChatOpenAI
from dotenv import load_dotenv
import os
load_dotenv()

print("teste")
llm = AzureChatOpenAI(
    deployment_name="gpt-35-turbo",
    model_name="gpt-35-turbo",
    openai_api_version=os.environ["OPENAI_API_VERSION"],
    openai_api_base=os.environ["OPENAI_API_BASE"],
    openai_api_key=os.environ["OPENAI_API_KEY"],
    openai_api_type=os.environ["OPENAI_API_TYPE"]
    #max_tokens=128,
    #top_p=0.95, # alternative to the temperature parameter, which uses nucleus sampling to determine the tokens to consider
    #frequency_penalty=0,
    #presence_penalty=0,
    #n=1
)
print("teste")
def load_pdf_file_and_save_vector_database(filename, vector_database_name):
    """
    Load the pdf file and save it locally in a vector database.
    """
    from langchain.document_loaders import PyPDFLoader
    from langchain.vectorstores import FAISS
    from langchain.embeddings.openai import OpenAIEmbeddings

    loader = PyPDFLoader(filename)
    pages = loader.load_and_split(text_splitter=RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=200))

    faiss_index = FAISS.from_documents(pages, embedding=OpenAIEmbeddings(deployment="ada-embedding", #pip install faiss-cpu
        model="text-embedding-ada-002",
        openai_api_version=os.environ["OPENAI_API_VERSION"],
        openai_api_base=os.environ["OPENAI_API_BASE"],
        openai_api_key=os.environ["OPENAI_API_KEY"],
        openai_api_type=os.environ["OPENAI_API_TYPE"],
        chunk_size=1
    ))
    
    faiss_index.save_local(vector_database_name)
    print("making question...")
    #docs = faiss_index.similarity_search("What are the supported operators of 'Affected Version' field?", k=1)
    #for doc in docs:
        #print(str(doc.metadata["page"]) + ":", doc.page_content[:300])


print("teste")

def question_in_vector_database(vector_database_name, query):
    from langchain.vectorstores import FAISS
    from langchain.chains.qa_with_sources import load_qa_with_sources_chain
    from langchain.llms import OpenAI
    from langchain.embeddings.openai import OpenAIEmbeddings

    db = FAISS.load_local(vector_database_name, embeddings=OpenAIEmbeddings(deployment="ada-embedding", #pip install faiss-cpu
        model="text-embedding-ada-002",
        openai_api_version=os.environ["OPENAI_API_VERSION"],
        openai_api_base=os.environ["OPENAI_API_BASE"],
        openai_api_key=os.environ["OPENAI_API_KEY"],
        openai_api_type=os.environ["OPENAI_API_TYPE"],
        chunk_size=1
    ))
    docs = db.similarity_search(query)
    print(docs)
    #docsearch = FAISS.from_embeddings()
    #docs = docsearch.similarity_search(query)

    chain = load_qa_with_sources_chain(llm, chain_type="stuff")
    print(chain({"input_documents": docs, "question": query}, return_only_outputs=True))

print("teste")
load_pdf_file_and_save_vector_database("./simondon.pdf", "simondon_index")
print("teste")
question_in_vector_database("simondon_index", "Resuma o primeiro paragrafo")