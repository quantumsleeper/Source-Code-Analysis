import os 
from src.helper import (load_embedding, 
                        load_repo, text_splitter, repo_ingestion
                        )
from dotenv import load_dotenv
from langchain.vectorstores import Chroma

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# repo ingestion will happen inside app.py 

documents = load_repo('repo/')

text_chunks = text_splitter(documents)

embeddings = load_embedding()

# storing vectors in ChromaDB
vectordb = Chroma.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    persist_directory="./db"
)
vectordb.persist()
