"""RAG indexer for building vector store from documents."""
import os
import json
import hashlib
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Loads markdown documents from docs/ directory."""

    def __init__(self, docs_path: str):
        self.docs_path = Path(docs_path)

    def load_all(self) -> List[Dict[str, Any]]:
        """Load all markdown documents with metadata."""
        documents = []

        # Prefer rubric docs for the current generic Q/A evaluation domain.
        # Fall back to legacy CS corpora only when rubric docs are absent.
        categories = ["rubrics"]
        if not (self.docs_path / "rubrics").exists():
            categories = ["policies", "help_center", "rubrics"]

        for category in categories:
            category_path = self.docs_path / category
            if not category_path.exists():
                continue

            for md_file in category_path.glob("*.md"):
                doc = self._load_document(md_file, category)
                if doc:
                    documents.append(doc)

        logger.info(f"Loaded {len(documents)} documents from {self.docs_path}")
        return documents

    def _load_document(self, file_path: Path, category: str) -> Optional[Dict[str, Any]]:
        """Load a single markdown document with frontmatter."""
        try:
            content = file_path.read_text(encoding="utf-8")

            # Parse frontmatter if present (YAML between ---)
            metadata = {}
            body = content

            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    frontmatter = parts[1].strip()
                    body = parts[2].strip()

                    # Simple YAML parsing
                    for line in frontmatter.split("\n"):
                        if ":" in line:
                            key, value = line.split(":", 1)
                            key = key.strip()
                            value = value.strip().strip('"').strip("'")
                            if value.startswith("[") and value.endswith("]"):
                                value = [v.strip().strip('"').strip("'") for v in value[1:-1].split(",")]
                            metadata[key] = value

            # Generate doc_id from filename
            doc_id = file_path.stem

            return {
                "doc_id": metadata.get("doc_id", doc_id),
                "title": metadata.get("title", doc_id.replace("_", " ").title()),
                "content": body,
                "source_url": metadata.get("source_url"),
                "version": metadata.get("version", "v1"),
                "tags": metadata.get("tags", []),
                "category": category,
            }
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {e}")
            return None


class RAGIndexer:
    """Builds and manages vector index for RAG."""

    def __init__(
        self,
        docs_path: str,
        vector_store_path: str,
        embedding_model: str = "text-embedding-3-small",
        use_mock: bool = True,
        openai_api_key: Optional[str] = None,
    ):
        self.docs_path = docs_path
        self.vector_store_path = Path(vector_store_path)
        self.embedding_model = embedding_model
        self.use_mock = use_mock
        self.openai_api_key = openai_api_key
        self.documents: List[Dict[str, Any]] = []
        self.embeddings = None
        self.vector_store = None

    def build_index(self) -> bool:
        """Build the vector index from documents."""
        try:
            # Load documents
            loader = DocumentLoader(self.docs_path)
            self.documents = loader.load_all()

            if not self.documents:
                logger.warning("No documents found to index")
                return False

            # Create embeddings
            if self.use_mock:
                self._build_mock_index()
            else:
                self._build_real_index()

            logger.info(f"Built index with {len(self.documents)} documents")
            return True

        except Exception as e:
            logger.error(f"Error building index: {e}")
            return False

    def _build_mock_index(self):
        """Build a mock index using simple text matching."""
        # For mock mode, we just store documents and use text similarity
        self.vector_store_path.mkdir(parents=True, exist_ok=True)

        # Save documents as JSON for quick loading
        index_file = self.vector_store_path / "mock_index.json"
        with open(index_file, "w") as f:
            json.dump(self.documents, f, indent=2)

        logger.info("Built mock index (text-based matching)")

    def _build_real_index(self):
        """Build real index using LangChain and embeddings."""
        try:
            from langchain_openai import OpenAIEmbeddings
            from langchain_community.vectorstores import FAISS
            from langchain_core.documents import Document

            # Initialize embeddings (pass key explicitly since pydantic-settings may not set os.environ)
            self.embeddings = OpenAIEmbeddings(
                model=self.embedding_model,
                api_key=self.openai_api_key,
            )

            # Convert to LangChain documents
            lc_docs = []
            for doc in self.documents:
                lc_doc = Document(
                    page_content=doc["content"],
                    metadata={
                        "doc_id": doc["doc_id"],
                        "title": doc["title"],
                        "source_url": doc.get("source_url"),
                        "version": doc["version"],
                        "tags": doc["tags"],
                        "category": doc["category"],
                    }
                )
                lc_docs.append(lc_doc)

            # Create FAISS index
            self.vector_store = FAISS.from_documents(lc_docs, self.embeddings)

            # Save index
            self.vector_store_path.mkdir(parents=True, exist_ok=True)
            self.vector_store.save_local(str(self.vector_store_path / "faiss_index"))

            logger.info("Built FAISS index with OpenAI embeddings")

        except ImportError as e:
            logger.warning(f"LangChain dependencies not available: {e}")
            logger.info("Falling back to mock index")
            self._build_mock_index()

    def load_index(self) -> bool:
        """Load existing index."""
        try:
            if self.use_mock:
                return self._load_mock_index()
            else:
                return self._load_real_index()
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            return False

    def _load_mock_index(self) -> bool:
        """Load mock index."""
        index_file = self.vector_store_path / "mock_index.json"
        if not index_file.exists():
            return False

        with open(index_file) as f:
            self.documents = json.load(f)

        logger.info(f"Loaded mock index with {len(self.documents)} documents")
        return True

    def _load_real_index(self) -> bool:
        """Load real FAISS index."""
        try:
            from langchain_openai import OpenAIEmbeddings
            from langchain_community.vectorstores import FAISS

            index_path = self.vector_store_path / "faiss_index"
            if not index_path.exists():
                return False

            self.embeddings = OpenAIEmbeddings(
                model=self.embedding_model,
                api_key=self.openai_api_key,
            )
            self.vector_store = FAISS.load_local(
                str(index_path),
                self.embeddings,
                allow_dangerous_deserialization=True
            )

            logger.info("Loaded FAISS index")
            return True

        except ImportError:
            logger.warning("LangChain not available, falling back to mock")
            return self._load_mock_index()

    def get_documents(self) -> List[Dict[str, Any]]:
        """Get all indexed documents."""
        return self.documents
