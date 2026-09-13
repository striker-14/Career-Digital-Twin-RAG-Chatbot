from pathlib import Path

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "digital_twin"


def get_retriever():
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            "chroma_db does not exist. Run: "
            "uv run python rag/ingest.py"
        )

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
    )

    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )

    return vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )
