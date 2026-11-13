import os
from dotenv import load_dotenv

# from langchain_community.document_loaders import TextLoader --> deprecated

from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from pathlib import Path

load_dotenv()


if __name__ == "__main__":
    print("Ingesting...")

    loader = UnstructuredLoader(
        file_path=str(Path(__file__).parent / "mediumblog1.txt"),
        chunking_strategy="basic",
        max_characters=1000000,
    )
    docs = loader.load()

    print("Splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = text_splitter.split_documents(docs)
    print(f"created {len(chunks)} chunks")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    print("ingesting...")
    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=os.environ["INDEX_NAME"],
    )
    # PineconeVectorStore.from_documents(chunks, embeddings, index_name=os.environ['INDEX_NAME'])
    print("Done")
