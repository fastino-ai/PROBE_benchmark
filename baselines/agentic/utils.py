"""
Shared datastore utilities for baseline agents.

Provides semantic search and SQL-like querying capabilities over document collections.
Enhanced with vector embeddings, auto-annotation, and intelligent tagging.
"""

import json
import sqlite3
from typing import Dict, List, Any, Optional
import tempfile
import os
import logging
import numpy as np
from openai import OpenAI

logger = logging.getLogger(__name__)


class DocumentDatastore:
    """In-memory datastore with semantic search and SQL query capabilities."""

    def __init__(
        self,
        documents: List[Dict[str, Any]],
        use_embeddings: bool = True,
        openai_api_key: Optional[str] = None,
        mock_mode: bool = False,
        auto_annotate: bool = False,
    ):
        """
        Initialize datastore with documents.

        Args:
            documents: List of document dictionaries
            use_embeddings: Enable vector embeddings for semantic search (default: True)
            openai_api_key: OpenAI API key for embeddings/annotations
            mock_mode: Use mock embeddings instead of OpenAI API
            auto_annotate: Auto-annotate documents with LLM
        """
        # Clean documents to only have id, type, payload (no metadata)
        cleaned_docs = []
        for doc in documents:
            cleaned_doc = {
                "id": doc.get("id", ""),
                "type": doc.get("type", "document"),
                "payload": doc.get("payload", {}),
            }
            # If there's metadata with info we need, merge it into payload
            if "metadata" in doc and doc["metadata"]:
                # Merge useful metadata fields into payload
                for key, value in doc["metadata"].items():
                    if key not in cleaned_doc["payload"]:
                        cleaned_doc["payload"][key] = value
            cleaned_docs.append(cleaned_doc)

        self.documents = {doc["id"]: doc for doc in cleaned_docs}
        self.doc_list = cleaned_docs
        self.use_embeddings = use_embeddings
        self.mock_mode = mock_mode
        self.auto_annotate = auto_annotate
        self.embeddings = {}  # Store document embeddings

        # Setup OpenAI client if needed
        self.client = None
        if use_embeddings and not mock_mode:
            # Get API key from parameter or environment variable
            api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.warning(
                    "No OpenAI API key provided. Set OPENAI_API_KEY environment variable. "
                    "Falling back to keyword search."
                )
                self.use_embeddings = False
            else:
                self.client = OpenAI(api_key=api_key)
                self.embedding_model = "text-embedding-3-small"

        # Create temporary SQLite database for SQL queries
        self.db_path = None
        self._setup_sql_db()

        # Generate embeddings if enabled
        if self.use_embeddings:
            self._generate_embeddings()

    def _setup_sql_db(self):
        """Setup SQLite database for SQL querying with simplified schema."""
        # Create temporary database
        fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create simplified documents table
        cursor.execute(
            """
            CREATE TABLE documents (
                id TEXT PRIMARY KEY,
                type TEXT,
                payload TEXT  -- JSON object
            )
        """
        )

        # Create indices for performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_type ON documents (type)")

        # Insert documents
        for doc in self.doc_list:
            payload = doc.get("payload", {})

            # Auto-annotate if enabled (store in payload)
            if self.auto_annotate and not payload.get("annotations"):
                # Handle email body vs content
                content = payload.get("content") or payload.get("body", "")
                title = payload.get("title") or payload.get("subject", "")

                annotations = self._annotate_document(content, title)
                # Store annotations directly in payload
                payload["annotations"] = annotations
                doc["payload"] = payload

            # Store embeddings in payload if available
            if self.use_embeddings and doc["id"] in self.embeddings:
                payload["_embedding"] = self.embeddings[doc["id"]]
                doc["payload"] = payload

            cursor.execute(
                """
                INSERT INTO documents VALUES (?, ?, ?)
            """,
                (doc.get("id", ""), doc.get("type", ""), json.dumps(payload)),
            )

        conn.commit()
        conn.close()

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using OpenAI API or mock."""
        if self.mock_mode:
            # Mock embedding for testing - simple hash-based vector
            np.random.seed(hash(text) % 2**32)
            return np.random.randn(1536).tolist()

        if self.client:
            try:
                response = self.client.embeddings.create(
                    model=self.embedding_model, input=text
                )
                return response.data[0].embedding
            except Exception as e:
                logger.error(f"Error getting embedding: {e}")

        # Fallback to zero vector
        return [0.0] * 1536

    def _generate_embeddings(self):
        """Generate embeddings for all documents."""
        for doc_id, doc in self.documents.items():
            payload = doc.get("payload", {})
            # Handle both regular documents and emails
            content = payload.get("content") or payload.get("body", "")
            title = payload.get("title") or payload.get("subject", "")
            # Combine title and content for embedding
            text = f"{title}\n{content}" if title else content
            if text:
                self.embeddings[doc_id] = self._get_embedding(text)

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def _annotate_document(self, content: str, title: str = None) -> Dict[str, Any]:
        """Auto-annotate document with LLM."""
        if self.mock_mode or not self.client:
            # Return mock annotations for testing or when no client
            return {
                "company_goal": None,
                "content_urgency": "Evergreen",
                "information_accessibility": "Internal",
                "decision_status": "No decision needed",
                "action_items": "No",
                "sentiment": "Neutral",
                "intents": ["Inform"],
                "topic": None,
                "keywords": [],
                "summary": content[:200] + "..." if len(content) > 200 else content,
            }

        try:
            prompt = f"""Analyze this document and classify it:

Title: {title or 'Untitled'}
Content: {content[:1000]}...

Provide classification in JSON format with these fields:
- company_goal: Main goal (e.g., "Growth", "Customer Success", "Product Development")
- content_urgency: "Time-sensitive" or "Evergreen"
- information_accessibility: "Private", "Internal", or "Public"
- decision_status: "Requires decision", "Decision made", or "No decision needed"
- action_items: "Yes" or "No"
- sentiment: "Very negative", "Negative", "Neutral", "Positive", or "Very positive"
- intents: Array of intents ["Inform", "Request", "Propose", etc.]
- topic: Main topic
- keywords: Array of 3-5 keywords
- summary: 2-3 sentence summary"""

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a document classifier. Respond only with valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                response_format={"type": "json_object"},
            )

            return json.loads(response.choices[0].message.content)

        except Exception as e:
            logger.error(f"Error annotating document: {e}")
            return self._annotate_document("", "")  # Return mock annotations

    def semantic_search(
        self, query: str, top_k: int = 5, threshold: float = 0.0
    ) -> List[str]:
        """
        Perform semantic search over documents.
        Returns list of document IDs ranked by relevance.
        Uses vector similarity if embeddings are available, otherwise falls back to keyword search.
        """
        if self.use_embeddings and self.embeddings:
            # Vector-based semantic search
            query_embedding = self._get_embedding(query)
            scores = {}

            for doc_id, doc_embedding in self.embeddings.items():
                similarity = self._cosine_similarity(query_embedding, doc_embedding)
                if similarity >= threshold:
                    scores[doc_id] = similarity

            # Sort by similarity score
            sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            return [doc_id for doc_id, _ in sorted_docs[:top_k]]
        else:
            # Fallback to keyword-based search
            query_terms = query.lower().split()
            scores = {}

            for doc_id, doc in self.documents.items():
                payload = doc.get("payload", {})
                # Handle both documents and emails
                content = (payload.get("content") or payload.get("body", "")).lower()
                title = (payload.get("title") or payload.get("subject", "")).lower()

                score = 0
                for term in query_terms:
                    # Weight title matches higher
                    score += title.count(term) * 3
                    score += content.count(term) * 1

                if score > 0:
                    scores[doc_id] = score

            # Return top_k document IDs sorted by score
            sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            return [doc_id for doc_id, _ in sorted_docs[:top_k]]

    def similarity_search(
        self, query: str, top_k: int = 5, threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Perform similarity search and return documents with scores.
        This is an enhanced version that returns full document info with similarity scores.
        """
        doc_ids = self.semantic_search(query, top_k, threshold)
        results = []

        if self.use_embeddings and self.embeddings:
            query_embedding = self._get_embedding(query)
            for doc_id in doc_ids:
                doc = self.get_document(doc_id)
                if doc and doc_id in self.embeddings:
                    similarity = self._cosine_similarity(
                        query_embedding, self.embeddings[doc_id]
                    )
                    results.append(
                        {"id": doc_id, "document": doc, "similarity": similarity}
                    )
        else:
            # For keyword search, include normalized scores
            for i, doc_id in enumerate(doc_ids):
                doc = self.get_document(doc_id)
                if doc:
                    # Approximate similarity score based on ranking
                    similarity = 1.0 - (i * 0.1)
                    results.append(
                        {
                            "id": doc_id,
                            "document": doc,
                            "similarity": max(0.0, similarity),
                        }
                    )

        return results

    def sql_query(self, query: str) -> List[Dict[str, Any]]:
        """
        Execute SQL query against the document database.
        Returns list of result dictionaries with parsed JSON fields.
        """
        if not self.db_path or not os.path.exists(self.db_path):
            return []

        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # Enable column access by name
            cursor = conn.cursor()

            cursor.execute(query)
            results = []
            for row in cursor.fetchall():
                result = dict(row)
                # Parse JSON fields if they exist
                if "payload" in result and result["payload"]:
                    result["payload"] = json.loads(result["payload"])
                results.append(result)

            conn.close()
            return results
        except Exception as e:
            print(f"SQL query error: {e}")
            return []

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get full document by ID."""
        return self.documents.get(doc_id)

    def get_document_content(self, doc_id: str) -> str:
        """Get just the content of a document (handles both content and body for emails)."""
        doc = self.documents.get(doc_id, {})
        payload = doc.get("payload", {})
        return payload.get("content") or payload.get("body", "")

    def list_document_ids(self) -> List[str]:
        """Get all document IDs."""
        return list(self.documents.keys())

    def add_document(
        self,
        doc_id: str = None,
        doc_type: str = "document",
        payload: Dict[str, Any] = None,
        auto_annotate: Optional[bool] = None,
    ) -> str:
        """
        Add a new document to the datastore.

        Args:
            doc_id: Unique document ID (auto-generated if None)
            doc_type: Type of document (e.g., 'document', 'email')
            payload: Document payload (title, content, etc.)
            auto_annotate: Whether to auto-annotate (uses instance default if None)

        Returns:
            Document ID
        """
        import uuid

        # Generate ID if not provided
        if not doc_id:
            doc_id = str(uuid.uuid4())

        # Create document structure
        doc = {"id": doc_id, "type": doc_type, "payload": payload or {}}

        # Auto-annotate if enabled
        if auto_annotate is None:
            auto_annotate = self.auto_annotate

        if auto_annotate and not doc["payload"].get("annotations"):
            # Handle both content/body and title/subject
            content = doc["payload"].get("content") or doc["payload"].get("body", "")
            title = doc["payload"].get("title") or doc["payload"].get("subject", "")
            annotations = self._annotate_document(content, title)
            doc["payload"]["annotations"] = annotations

        # Add to documents
        self.documents[doc_id] = doc
        self.doc_list.append(doc)

        # Generate embedding if enabled
        if self.use_embeddings:
            content = doc["payload"].get("content") or doc["payload"].get("body", "")
            title = doc["payload"].get("title") or doc["payload"].get("subject", "")
            text = f"{title}\n{content}" if title else content
            if text:
                self.embeddings[doc_id] = self._get_embedding(text)

        # Recreate SQL database with new document
        self._setup_sql_db()

        return doc_id

    def query_by_tags(self, **kwargs) -> List[Dict[str, Any]]:
        """
        Query documents by tags in payload.

        Args:
            type: Filter by document type
            Any payload field can be queried

        Returns:
            List of matching documents
        """
        results = []

        for doc_id, doc in self.documents.items():
            payload = doc.get("payload", {})
            annotations = payload.get("annotations", {})

            # Check all filter conditions
            match = True
            for key, value in kwargs.items():
                # Check type at document level
                if key == "type":
                    doc_value = doc.get("type")
                else:
                    # Check in payload first, then annotations
                    doc_value = payload.get(key) or annotations.get(key)

                if doc_value != value:
                    match = False
                    break

            if match:
                results.append(doc)

        return results

    def get_document_with_annotations(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document with all annotations in payload."""
        # Annotations are already stored in payload, so just return the document
        return self.get_document(doc_id)

    def __del__(self):
        """Cleanup temporary database file."""
        if self.db_path and os.path.exists(self.db_path):
            try:
                os.unlink(self.db_path)
            except:
                pass


def create_datastore(
    documents: List[Dict[str, Any]],
    use_embeddings: bool = True,
    openai_api_key: Optional[str] = None,
    mock_mode: bool = False,
    auto_annotate: bool = False,
) -> DocumentDatastore:
    """
    Create a new document datastore with optional enhanced features.

    Args:
        documents: List of document dictionaries
        use_embeddings: Enable vector embeddings for semantic search (default: True)
        openai_api_key: OpenAI API key for embeddings/annotations
        mock_mode: Use mock embeddings instead of OpenAI API
        auto_annotate: Auto-annotate documents with LLM

    Returns:
        DocumentDatastore instance
    """
    return DocumentDatastore(
        documents=documents,
        use_embeddings=use_embeddings,
        openai_api_key=openai_api_key,
        mock_mode=mock_mode,
        auto_annotate=auto_annotate,
    )
