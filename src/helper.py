import os
from git import Repo    # for cloning GitHub codebases
from langchain.text_splitter import Language   # for context-aware splitting
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings


# cloning user-provided repository
def repo_ingestion(repo_url):
    os.makedirs("repo", exist_ok=True)
    repo_path = "repo/"
    Repo.clone_from(repo_url, to_path=repo_path)


# loading the repository as documents 
def load_repo(repo_path):
    loader = GenericLoader.from_filesystem(
        path=repo_path + 'src/mlProject',
        glob="**/*",
        suffixes=[".py"],
        parser=LanguageParser(
        language=Language.PYTHON,
        parser_threshold=500
        )
    )

    documents = loader.load()
    return documents


# splitting the documents into chunks
def text_splitter(documents):
    documents_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=2000,
    chunk_overlap=200
    )

    text_chunks = documents_splitter.split_documents(documents)
    return text_chunks


# loading the embedding model
def load_embedding():
    embeddings = OpenAIEmbeddings(disallowed_special=())
    return embeddings




