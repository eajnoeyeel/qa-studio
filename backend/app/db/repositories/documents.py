"""Repositories for documents and trace logs."""
from typing import List, Optional

from sqlalchemy.orm import Session

from ...models.database import DocumentModel, TraceLogModel, json_serializer
from ...models.schemas import DocumentInDB, DocumentMeta
from .common import generate_id


class DocumentRepository:
    """Repository for document operations."""

    def __init__(self, db: Session):
        self.db = db

    def get(self, doc_id: str) -> Optional[DocumentInDB]:
        model = self.db.query(DocumentModel).filter(DocumentModel.doc_id == doc_id).first()
        if not model:
            return None
        return self._to_schema(model)

    def get_by_version(self, version: str) -> List[DocumentInDB]:
        models = self.db.query(DocumentModel).filter(DocumentModel.version == version).all()
        return [self._to_schema(m) for m in models]

    def get_all(self) -> List[DocumentInDB]:
        models = self.db.query(DocumentModel).all()
        return [self._to_schema(m) for m in models]

    def create(self, doc: DocumentMeta, content: str) -> DocumentInDB:
        model = DocumentModel(
            doc_id=doc.doc_id,
            title=doc.title,
            content=content,
            source_url=doc.source_url,
            version=doc.version,
            tags_json=json_serializer(doc.tags),
            category=doc.category,
        )
        self.db.add(model)
        self.db.commit()
        self.db.refresh(model)
        return self._to_schema(model)

    def upsert(self, doc: DocumentMeta, content: str) -> DocumentInDB:
        model = self.db.query(DocumentModel).filter(DocumentModel.doc_id == doc.doc_id).first()
        if not model:
            return self.create(doc, content)

        model.title = doc.title
        model.content = content
        model.source_url = doc.source_url
        model.version = doc.version
        model.tags_json = json_serializer(doc.tags)
        model.category = doc.category
        self.db.commit()
        self.db.refresh(model)
        return self._to_schema(model)

    def delete_not_in_doc_ids(self, doc_ids: List[str]) -> int:
        if not doc_ids:
            return 0
        query = self.db.query(DocumentModel).filter(~DocumentModel.doc_id.in_(doc_ids))
        deleted = query.delete(synchronize_session=False)
        self.db.commit()
        return deleted

    def _to_schema(self, model: DocumentModel) -> DocumentInDB:
        return DocumentInDB(
            doc_id=model.doc_id,
            title=model.title,
            content=model.content,
            source_url=model.source_url,
            version=model.version,
            tags=model.tags,
            category=model.category,
            created_at=model.created_at,
        )


class TraceLogRepository:
    """Repository for trace log operations (Langfuse fallback)."""

    def __init__(self, db: Session):
        self.db = db

    def create(
        self,
        trace_id: str,
        span_name: str,
        input_data: Optional[dict] = None,
        output_data: Optional[dict] = None,
        latency_ms: Optional[float] = None,
        error: Optional[str] = None,
        commit: bool = True,
    ):
        model = TraceLogModel(
            id=generate_id(),
            trace_id=trace_id,
            span_name=span_name,
            input_json=json_serializer(input_data) if input_data else None,
            output_json=json_serializer(output_data) if output_data else None,
            latency_ms=latency_ms,
            error=error,
        )
        self.db.add(model)
        if commit:
            self.db.commit()
        else:
            self.db.flush()
        return model

    def get_by_trace(self, trace_id: str):
        return self.db.query(TraceLogModel).filter(TraceLogModel.trace_id == trace_id).all()
