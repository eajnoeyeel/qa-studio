"""RAG retriever for document retrieval."""
import re
from typing import List, Optional, Dict, Any
import logging

from ..models.schemas import RAGDocument, RAGResult

logger = logging.getLogger(__name__)


class RAGRetriever:
    """Retrieves relevant documents for queries."""

    def __init__(self, indexer):
        self.indexer = indexer
        self.use_mock = indexer.use_mock

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_tags: Optional[List[str]] = None,
        filter_version: Optional[str] = None
    ) -> RAGResult:
        """Retrieve relevant documents for a query."""
        if self.use_mock:
            documents = self._mock_retrieve(query, top_k, filter_tags, filter_version)
        else:
            documents = self._real_retrieve(query, top_k, filter_tags, filter_version)

        return RAGResult(query=query, documents=documents)

    def _mock_retrieve(
        self,
        query: str,
        top_k: int,
        filter_tags: Optional[List[str]],
        filter_version: Optional[str]
    ) -> List[RAGDocument]:
        """Mock retrieval using simple text matching."""
        query_lower = query.lower()
        query_terms = set(re.findall(r'\w+', query_lower))

        scored_docs = []
        for doc in self.indexer.documents:
            # Apply filters
            if filter_version and doc.get("version") != filter_version:
                continue
            if filter_tags:
                doc_tags = doc.get("tags", [])
                if not any(t in doc_tags for t in filter_tags):
                    continue

            # Calculate simple relevance score
            content_lower = doc["content"].lower()
            title_lower = doc["title"].lower()

            # Score based on term matches
            content_terms = set(re.findall(r'\w+', content_lower))
            title_terms = set(re.findall(r'\w+', title_lower))

            # Title matches worth more
            title_matches = len(query_terms & title_terms)
            content_matches = len(query_terms & content_terms)

            score = title_matches * 3 + content_matches

            # Boost for exact phrase match
            if query_lower in content_lower:
                score += 5
            if query_lower in title_lower:
                score += 10

            if score > 0:
                scored_docs.append((score, doc))

        # Sort by score descending
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        # Convert to RAGDocument
        results = []
        for score, doc in scored_docs[:top_k]:
            results.append(RAGDocument(
                doc_id=doc["doc_id"],
                title=doc["title"],
                content=doc["content"][:500] + "..." if len(doc["content"]) > 500 else doc["content"],
                source_url=doc.get("source_url"),
                version=doc["version"],
                tags=doc.get("tags", []),
                relevance_score=min(score / 10, 1.0),  # Normalize score
            ))

        return results

    def _real_retrieve(
        self,
        query: str,
        top_k: int,
        filter_tags: Optional[List[str]],
        filter_version: Optional[str]
    ) -> List[RAGDocument]:
        """Real retrieval using vector similarity."""
        try:
            if not self.indexer.vector_store:
                logger.warning("Vector store not loaded, falling back to mock")
                return self._mock_retrieve(query, top_k, filter_tags, filter_version)

            # Build filter dict
            filter_dict = {}
            if filter_version:
                filter_dict["version"] = filter_version

            # Search with or without filter
            if filter_dict:
                results = self.indexer.vector_store.similarity_search_with_score(
                    query,
                    k=top_k * 2,  # Get more to filter
                    filter=filter_dict
                )
            else:
                results = self.indexer.vector_store.similarity_search_with_score(
                    query,
                    k=top_k * 2
                )

            # Process results
            documents = []
            for doc, score in results:
                # Apply tag filter
                if filter_tags:
                    doc_tags = doc.metadata.get("tags", [])
                    if not any(t in doc_tags for t in filter_tags):
                        continue

                documents.append(RAGDocument(
                    doc_id=doc.metadata["doc_id"],
                    title=doc.metadata["title"],
                    content=doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                    source_url=doc.metadata.get("source_url"),
                    version=doc.metadata["version"],
                    tags=doc.metadata.get("tags", []),
                    relevance_score=1 / (1 + score),  # Convert distance to similarity
                ))

                if len(documents) >= top_k:
                    break

            return documents

        except Exception as e:
            logger.error(f"Error in real retrieval: {e}")
            return self._mock_retrieve(query, top_k, filter_tags, filter_version)

    def check_claim(
        self,
        claim: str,
        top_k: int = 3
    ) -> Dict[str, Any]:
        """Check if a claim is supported by documentation."""
        result = self.retrieve(claim, top_k=top_k)

        if not result.documents:
            return {
                "supported": False,
                "confidence": 0.0,
                "reason": "No relevant documentation found",
                "citations": [],
            }

        # Check relevance of top documents
        top_doc = result.documents[0]

        # Simple heuristic: if relevance score is high, claim might be supported
        if top_doc.relevance_score > 0.7:
            return {
                "supported": True,
                "confidence": top_doc.relevance_score,
                "reason": f"Found supporting documentation in '{top_doc.title}'",
                "citations": [d.doc_id for d in result.documents],
            }
        elif top_doc.relevance_score > 0.3:
            return {
                "supported": None,  # Uncertain
                "confidence": top_doc.relevance_score,
                "reason": f"Partially related documentation found in '{top_doc.title}'",
                "citations": [d.doc_id for d in result.documents],
            }
        else:
            return {
                "supported": False,
                "confidence": top_doc.relevance_score,
                "reason": "Documentation does not support this claim",
                "citations": [],
            }

    def get_context_for_evaluation(
        self,
        question: str,
        response: str,
        taxonomy_label: Optional[str] = None,
        docs_version: Optional[str] = None,
        top_k: int = 5
    ) -> RAGResult:
        """Get relevant context for evaluating a response."""
        # Combine question and response for query
        query = f"{question}\n\nResponse: {response}"

        # Filter by taxonomy label if provided
        filter_tags = [taxonomy_label] if taxonomy_label else None

        return self.retrieve(
            query,
            top_k=top_k,
            filter_tags=filter_tags,
            filter_version=docs_version,
        )
