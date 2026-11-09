"""תהליך ה-Ingestion: מסמכים -> Chunks -> וקטורים -> ChromaDB.

מריצים רק בהקמה, או כשהמסמכים, גודל ה-Chunk או מודל ה-Embedding משתנים.
"""
from pathlib import Path

import chromadb
from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore

import config


def _doc_title(path: str) -> dict:
    """הכותרת הראשונה של הקובץ משמשת כשם המקור בתשובה."""
    first_line = Path(path).read_text(encoding="utf-8").splitlines()[0]
    return {"title": first_line.lstrip("# ").strip()}


def get_collection(client=None, reset: bool = False):
    client = client or chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    if reset:
        try:
            client.delete_collection(config.COLLECTION_NAME)
        except Exception:  # הקולקציה עדיין לא קיימת
            pass
    # cosine ולא l2 (ברירת המחדל), כדי שהסף MIN_SIMILARITY יהיה בעל משמעות
    return client.get_or_create_collection(
        config.COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )


def build_index() -> int:
    """בונה את האינדקס מאפס (כך אין כפילויות בהרצה חוזרת) ומחזיר את מספר ה-Chunks."""
    documents = SimpleDirectoryReader(
        str(config.DATA_DIR), required_exts=[".md"], file_metadata=_doc_title
    ).load_data()

    splitter = TokenTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    nodes = splitter.get_nodes_from_documents(documents)

    store = ChromaVectorStore(chroma_collection=get_collection(reset=True))
    VectorStoreIndex(
        nodes,
        storage_context=StorageContext.from_defaults(vector_store=store),
        embed_model=Settings.embed_model,
    )
    return len(nodes)


if __name__ == "__main__":
    from models import setup_models

    setup_models()
    print(f"האינדקס נבנה: {build_index()} קטעים נשמרו ב-ChromaDB")
