"""
Note Storage Service using ChromaDB

Manages person notes with semantic search capabilities using ChromaDB.
Notes are automatically embedded and can be retrieved by person or semantic similarity.
"""

import time
import logging
from typing import List, Optional, Dict, Any
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)


import os

class NoteService:
    """
    Note storage and retrieval service using ChromaDB.

    Features:
    - Persistent storage on disk
    - Automatic embedding generation for semantic search
    - Fast retrieval by person_id
    - Support for semantic similarity search (future enhancement)
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the note service.

        Args:
            db_path: Path to ChromaDB persistent storage directory. If None, uses default relative to this file.
        """
        if db_path is None:
            # Resolve to backend/person_memory/chroma_db regardless of CWD
            base_dir = os.path.dirname(os.path.abspath(__file__))
            self.db_path = os.path.join(base_dir, "person_memory", "chroma_db")
        else:
            self.db_path = db_path

        # Initialize ChromaDB with persistent storage
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(
                anonymized_telemetry=False  # Disable telemetry for privacy
            )
        )

        # Get or create collection for person notes
        self.collection = self.client.get_or_create_collection(
            name="person_notes",
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )

        logger.info(f"NoteService initialized with ChromaDB at {db_path}")
        logger.info(f"Current note count: {self.collection.count()}")

    def preload_models(self):
        """
        Preload ChromaDB embedding models at startup to avoid runtime delays.

        This triggers the download of:
        - all-MiniLM-L6-v2 ONNX model (~80MB)
        - ONNX runtime dependencies
        """
        try:
            logger.info("🔄 Preloading ChromaDB embedding models (all-MiniLM-L6-v2)...")

            # Trigger model download by adding and removing a dummy document
            dummy_id = "_preload_dummy_"

            self.collection.add(
                documents=["Preloading embedding model"],
                ids=[dummy_id],
                metadatas=[{"type": "preload"}]
            )

            # Remove the dummy document
            self.collection.delete(ids=[dummy_id])

            logger.info("✅ ChromaDB embedding models preloaded successfully")

        except Exception as e:
            logger.warning(f"⚠️  ChromaDB preload failed (will load on first use): {e}")

    def add_note(self, person_id: str, person_name: str, note_text: str) -> str:
        """
        Store a note for a person.

        ChromaDB automatically generates embeddings for semantic search.

        Args:
            person_id: UUID of the person
            person_name: Person's name for metadata
            note_text: The note content

        Returns:
            note_id: Unique identifier for the note
        """
        try:
            timestamp = time.time()
            note_id = f"{person_id}_{int(timestamp)}"

            # Add note to ChromaDB (embeddings generated automatically)
            self.collection.add(
                documents=[note_text],
                metadatas=[{
                    "person_id": person_id,
                    "person_name": person_name,
                    "timestamp": timestamp,
                    "type": "user_note"
                }],
                ids=[note_id]
            )

            logger.info(f"Saved note for {person_name}: {note_text[:50]}...")
            return note_id

        except Exception as e:
            logger.error(f"Error saving note: {e}", exc_info=True)
            raise

    def get_notes_for_person(self, person_id: str, limit: int = 5) -> List[str]:
        """
        Retrieve recent notes for a person (ordered by timestamp).

        Args:
            person_id: UUID of the person
            limit: Maximum number of notes to retrieve

        Returns:
            List of note texts (most recent first)
        """
        try:
            # Query all notes for this person
            results = self.collection.get(
                where={"person_id": person_id}
            )

            if not results or not results['documents']:
                logger.debug(f"No notes found for person {person_id}")
                return []

            # Combine documents with their metadata for sorting
            notes_with_metadata = list(zip(
                results['documents'],
                results['metadatas']
            ))

            # Sort by timestamp (most recent first)
            notes_with_metadata.sort(
                key=lambda x: x[1]['timestamp'],
                reverse=True
            )

            # Extract just the note texts (limited)
            notes = [note[0] for note in notes_with_metadata[:limit]]

            logger.debug(f"Retrieved {len(notes)} notes for person {person_id}")
            return notes

        except Exception as e:
            logger.error(f"Error retrieving notes: {e}", exc_info=True)
            return []

    def search_notes(self, person_id: str, query: str, limit: int = 3) -> List[str]:
        """
        Semantic search for relevant notes based on a query.

        This enables finding notes by meaning, not just exact text match.
        Useful for future enhancements like "What did I say about John's diet?"

        Args:
            person_id: UUID of the person
            query: Search query text
            limit: Maximum number of results

        Returns:
            List of relevant note texts (most similar first)
        """
        try:
            # Query with semantic similarity search
            results = self.collection.query(
                query_texts=[query],
                where={"person_id": person_id},
                n_results=limit
            )

            if not results or not results['documents'] or not results['documents'][0]:
                logger.debug(f"No matching notes found for query: {query}")
                return []

            notes = results['documents'][0]  # First query result

            logger.debug(f"Found {len(notes)} matching notes for query: {query}")
            return notes

        except Exception as e:
            logger.error(f"Error searching notes: {e}", exc_info=True)
            return []

    def delete_note(self, note_id: str) -> bool:
        """
        Delete a specific note.

        Args:
            note_id: Unique identifier of the note

        Returns:
            True if deleted, False if not found
        """
        try:
            self.collection.delete(ids=[note_id])
            logger.info(f"Deleted note: {note_id}")
            return True

        except Exception as e:
            logger.error(f"Error deleting note: {e}", exc_info=True)
            return False

    def delete_all_notes_for_person(self, person_id: str) -> int:
        """
        Delete all notes for a person.

        Useful for implementing "forget [person]" command.

        Args:
            person_id: UUID of the person

        Returns:
            Number of notes deleted
        """
        try:
            # Get all notes for this person first
            results = self.collection.get(
                where={"person_id": person_id}
            )

            if not results or not results['ids']:
                logger.debug(f"No notes to delete for person {person_id}")
                return 0

            note_ids = results['ids']

            # Delete all notes
            self.collection.delete(ids=note_ids)

            logger.info(f"Deleted {len(note_ids)} notes for person {person_id}")
            return len(note_ids)

        except Exception as e:
            logger.error(f"Error deleting notes for person: {e}", exc_info=True)
            return 0

    def get_note_count(self, person_id: Optional[str] = None) -> int:
        """
        Get total count of notes (optionally filtered by person).

        Args:
            person_id: Optional person UUID to filter by

        Returns:
            Number of notes
        """
        try:
            if person_id:
                results = self.collection.get(
                    where={"person_id": person_id}
                )
                return len(results['ids']) if results and results['ids'] else 0
            else:
                return self.collection.count()

        except Exception as e:
            logger.error(f"Error getting note count: {e}", exc_info=True)
            return 0

    def export_notes_for_person(self, person_id: str) -> List[Dict[str, Any]]:
        """
        Export all notes for a person with metadata.

        Useful for data portability and GDPR compliance.

        Args:
            person_id: UUID of the person

        Returns:
            List of note dictionaries with text and metadata
        """
        try:
            results = self.collection.get(
                where={"person_id": person_id}
            )

            if not results or not results['documents']:
                return []

            notes = []
            for i in range(len(results['documents'])):
                note = {
                    'id': results['ids'][i],
                    'text': results['documents'][i],
                    'metadata': results['metadatas'][i]
                }
                notes.append(note)

            # Sort by timestamp
            notes.sort(key=lambda x: x['metadata']['timestamp'], reverse=True)

            logger.info(f"Exported {len(notes)} notes for person {person_id}")
            return notes

        except Exception as e:
            logger.error(f"Error exporting notes: {e}", exc_info=True)
            return []
