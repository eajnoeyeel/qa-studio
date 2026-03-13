"""Document endpoints."""
from typing import Optional

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ...db.repository import DocumentRepository
from ...models.schemas import DocumentMeta
from ...rag.retriever import RAGRetriever
from ..deps import build_rag_indexer, get_db, set_rag_retriever

router = APIRouter()


@router.get("/documents")
async def list_documents(
    version: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """List documents in the RAG index."""
    repo = DocumentRepository(db)
    if version:
        docs = repo.get_by_version(version)
    else:
        docs = repo.get_all()
    return {"documents": [d.model_dump() for d in docs]}


@router.post("/documents/reindex")
async def reindex_documents(db: Session = Depends(get_db)):
    """Rebuild the RAG index and sync documents to DB."""
    indexer = build_rag_indexer()
    success = indexer.build_index()

    if success:
        doc_repo = DocumentRepository(db)
        indexed_doc_ids = []
        for doc in indexer.documents:
            indexed_doc_ids.append(doc["doc_id"])
            doc_repo.upsert(
                DocumentMeta(
                    doc_id=doc["doc_id"],
                    title=doc["title"],
                    source_url=doc.get("source_url"),
                    version=doc.get("version", "v1"),
                    tags=doc.get("tags", []),
                    category=doc["category"],
                ),
                content=doc["content"],
            )
        doc_repo.delete_not_in_doc_ids(indexed_doc_ids)

    if success:
        set_rag_retriever(RAGRetriever(indexer))

    return {"success": success, "document_count": len(indexer.documents)}
