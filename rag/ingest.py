from pathlib import Path
import shutil

from dotenv import load_dotenv
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

load_dotenv(override=True)

KNOWLEDGE_DIR = Path("knowledge")
CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "digital_twin"


def load_documents():
    documents = []

    # PDF
    pdf_file = KNOWLEDGE_DIR / "linkedin.pdf"

    if pdf_file.exists():
        documents.extend(
            PyPDFLoader(str(pdf_file)).load()
        )

    # Text files
    for filename in [
        "summary.txt",
        "projects.txt",
        "skills.txt",
        "experience.txt",
    ]:
        file_path = KNOWLEDGE_DIR / filename

        if file_path.exists():
            documents.extend(
                TextLoader(
                    str(file_path),
                    encoding="utf-8",
                ).load()
            )

    if not documents:
        raise FileNotFoundError(
            "No knowledge files found in knowledge/. "
            "Add linkedin.pdf and/or text files."
        )

    print(f"Loaded {len(documents)} source documents/pages.")
    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks.")
    return chunks


def create_vectorstore(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
    )

    # Rebuild the database whenever ingestion is run.
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION_NAME,
    )

    print("Chroma vector database created.")
    return vectorstore


if __name__ == "__main__":
    documents = load_documents()
    chunks = split_documents(documents)
    create_vectorstore(chunks)

    print("RAG ingestion completed successfully.")
