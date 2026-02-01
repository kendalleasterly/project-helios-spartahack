"""
Face Recognition Service using DeepFace

Handles face detection, recognition, and person management for the person memory feature.
Uses DeepFace with ArcFace model for high accuracy face recognition.
"""

import os
import sqlite3
import uuid
import time
import logging
import pickle
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
import numpy as np
import cv2
from deepface import DeepFace

logger = logging.getLogger(__name__)


@dataclass
class Person:
    """Represents a known person in the database."""
    person_id: str
    name: str
    face_encoding: np.ndarray
    first_seen: float
    last_seen: float
    total_interactions: int


@dataclass
class FaceDetection:
    """Represents a detected face in a frame."""
    person_id: Optional[str]
    person_name: Optional[str]
    confidence: float
    bbox: Dict[str, int]  # Bounding box (x, y, w, h)
    distance: Optional[float] = None  # Cosine distance for debugging


class FaceService:
    """
    Face recognition service using DeepFace with ArcFace model.

    Features:
    - Face detection and recognition using DeepFace + ArcFace
    - SQLite database for face encodings
    - In-memory cache for fast lookups
    - GPU-accelerated inference (~200-300ms)
    """

    def __init__(self, db_path: str = "./person_memory/faces.db"):
        """
        Initialize the face service.

        Args:
            db_path: Path to SQLite database for storing face encodings
        """
        self.db_path = db_path
        self.model_name = "ArcFace"  # Best accuracy model (512D embeddings)
        self.detector_backend = "opencv"  # Fast and reliable
        self.distance_metric = "cosine"

        # ArcFace default threshold for cosine distance
        # Lower distance = more similar faces
        # Match if distance < threshold
        self.recognition_threshold = 0.68

        # In-memory cache of known faces for fast lookup
        self.known_faces_cache: Dict[str, Person] = {}

        # Initialize database
        self._init_database()

        # Load known faces into memory cache
        self._load_known_faces()

        logger.info(f"✅ FaceService initialized with {len(self.known_faces_cache)} known people")

    def _init_database(self):
        """Initialize SQLite database with schema."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS people (
                person_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                face_encoding BLOB NOT NULL,
                first_seen REAL NOT NULL,
                last_seen REAL NOT NULL,
                total_interactions INTEGER DEFAULT 0
            )
        """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_name ON people(name)")

        conn.commit()
        conn.close()

        logger.info(f"Database initialized at {self.db_path}")

    def _load_known_faces(self):
        """Load all known faces from database into memory cache."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT person_id, name, face_encoding, first_seen, last_seen, total_interactions FROM people")
        rows = cursor.fetchall()

        for row in rows:
            person_id, name, encoding_blob, first_seen, last_seen, total_interactions = row
            face_encoding = pickle.loads(encoding_blob)

            person = Person(
                person_id=person_id,
                name=name,
                face_encoding=face_encoding,
                first_seen=first_seen,
                last_seen=last_seen,
                total_interactions=total_interactions
            )

            self.known_faces_cache[person_id] = person

        conn.close()
        logger.info(f"Loaded {len(self.known_faces_cache)} people from database")

    def detect_and_recognize(self, image: np.ndarray) -> List[FaceDetection]:
        """
        Detect and recognize faces in an image.

        Args:
            image: OpenCV image (BGR format)

        Returns:
            List of FaceDetection objects (person_id is None if unknown)
        """
        if not isinstance(image, np.ndarray):
            logger.error("Invalid image format - expected numpy array")
            return []

        start_time = time.perf_counter()

        try:
            # Step 1: Detect faces in the image
            face_objs = DeepFace.extract_faces(
                img_path=image,
                detector_backend=self.detector_backend,
                enforce_detection=False,  # Don't raise error if no face found
                align=True
            )

            if not face_objs:
                logger.debug("No faces detected in frame")
                return []

            detections = []

            # Step 2: Process each detected face
            for face_obj in face_objs:
                facial_area = face_obj['facial_area']
                detection_confidence = face_obj['confidence']

                # Skip low-confidence detections
                if detection_confidence < 0.9:
                    logger.debug(f"Skipping low-confidence face detection: {detection_confidence:.2f}")
                    continue

                # Extract the face region
                face_region = face_obj['face']

                # Step 3: Generate embedding for this face
                try:
                    embedding_objs = DeepFace.represent(
                        img_path=face_region,
                        model_name=self.model_name,
                        enforce_detection=False
                    )

                    if not embedding_objs:
                        logger.debug("Failed to generate embedding for detected face")
                        continue

                    # Extract the embedding vector (512D for ArcFace)
                    face_embedding = np.array(embedding_objs[0]["embedding"])

                    # Step 4: Match against known faces
                    person_id, person_name, match_distance = self._match_face(face_embedding)

                    # Create detection result
                    detection = FaceDetection(
                        person_id=person_id,
                        person_name=person_name,
                        confidence=1.0 - match_distance if person_id else detection_confidence,
                        bbox=facial_area,
                        distance=match_distance if person_id else None
                    )

                    detections.append(detection)

                    if person_id:
                        logger.info(f"✅ Recognized: {person_name} (distance: {match_distance:.3f})")
                    else:
                        logger.debug(f"Unknown face detected (best distance: {match_distance:.3f}, threshold: {self.recognition_threshold})")

                except Exception as e:
                    logger.error(f"Error generating embedding for face: {e}")
                    continue

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.debug(f"⚡ Face detection completed in {elapsed_ms:.0f}ms ({len(detections)} faces)")

            return detections

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"❌ Face detection error after {elapsed_ms:.0f}ms: {e}", exc_info=True)
            return []

    def _match_face(self, face_embedding: np.ndarray) -> Tuple[Optional[str], Optional[str], float]:
        """
        Match a face embedding against known faces in the database.

        Args:
            face_embedding: Face embedding vector (512D for ArcFace)

        Returns:
            Tuple of (person_id, person_name, distance)
            Returns (None, None, inf) if no match found
        """
        if not self.known_faces_cache:
            logger.debug("No known faces in cache")
            return None, None, float('inf')

        best_match_id = None
        best_match_name = None
        best_distance = float('inf')

        # Compare against all known faces
        for person_id, person in self.known_faces_cache.items():
            # Calculate cosine distance between embeddings
            distance = self._cosine_distance(face_embedding, person.face_encoding)

            if distance < best_distance:
                best_distance = distance
                best_match_id = person_id
                best_match_name = person.name

        # Check if best match is below threshold
        if best_distance < self.recognition_threshold:
            logger.debug(f"Match found: {best_match_name} (distance: {best_distance:.3f} < threshold: {self.recognition_threshold})")
            return best_match_id, best_match_name, best_distance
        else:
            logger.debug(f"No match (best distance: {best_distance:.3f} >= threshold: {self.recognition_threshold})")
            return None, None, best_distance

    def _cosine_distance(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine distance between two embeddings.

        Cosine distance = 1 - cosine_similarity
        Range: [0, 2] where 0 = identical, 2 = opposite

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine distance (0 = identical, higher = more different)
        """
        # Calculate dot product
        dot_product = np.dot(embedding1, embedding2)

        # Calculate norms
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        # Handle zero norms
        if norm1 == 0 or norm2 == 0:
            return 1.0

        # Cosine similarity = dot_product / (norm1 * norm2)
        cosine_similarity = dot_product / (norm1 * norm2)

        # Cosine distance = 1 - cosine_similarity
        cosine_distance = 1.0 - cosine_similarity

        return cosine_distance

    def add_known_person(self, name: str, face_image: np.ndarray) -> str:
        """
        Add a new person to the database.

        Args:
            name: Person's name
            face_image: Image containing the person's face (BGR format)

        Returns:
            person_id (UUID string)

        Raises:
            ValueError: If no face detected or embedding generation fails
        """
        try:
            logger.info(f"Adding new person: {name}")

            # Step 1: Extract face from image
            face_objs = DeepFace.extract_faces(
                img_path=face_image,
                detector_backend=self.detector_backend,
                enforce_detection=True,  # Require face detection
                align=True
            )

            if not face_objs:
                raise ValueError("No face detected in image")

            # Use the first (largest) detected face
            face_region = face_objs[0]['face']

            # Step 2: Generate embedding
            embedding_objs = DeepFace.represent(
                img_path=face_region,
                model_name=self.model_name,
                enforce_detection=False
            )

            if not embedding_objs:
                raise ValueError("Failed to generate face embedding")

            face_encoding = np.array(embedding_objs[0]["embedding"])

            # Step 3: Create new person record
            person_id = str(uuid.uuid4())
            current_time = time.time()

            # Step 4: Save to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            encoding_blob = pickle.dumps(face_encoding)

            cursor.execute("""
                INSERT INTO people (person_id, name, face_encoding, first_seen, last_seen, total_interactions)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (person_id, name, encoding_blob, current_time, current_time, 0))

            conn.commit()
            conn.close()

            # Step 5: Add to in-memory cache
            person = Person(
                person_id=person_id,
                name=name,
                face_encoding=face_encoding,
                first_seen=current_time,
                last_seen=current_time,
                total_interactions=0
            )
            self.known_faces_cache[person_id] = person

            logger.info(f"✅ Added person: {name} (ID: {person_id})")
            return person_id

        except Exception as e:
            logger.error(f"❌ Error adding person '{name}': {e}", exc_info=True)
            raise

    def get_person_by_name(self, name: str) -> Optional[Person]:
        """
        Lookup person by name (case-insensitive).

        Args:
            name: Person's name

        Returns:
            Person object or None if not found
        """
        name_lower = name.lower()

        for person in self.known_faces_cache.values():
            if person.name.lower() == name_lower:
                return person

        return None

    def get_person_by_id(self, person_id: str) -> Optional[Person]:
        """
        Lookup person by ID.

        Args:
            person_id: Person's UUID

        Returns:
            Person object or None if not found
        """
        return self.known_faces_cache.get(person_id)

    def update_last_seen(self, person_id: str):
        """
        Update last_seen timestamp and increment interaction count.

        Args:
            person_id: Person's UUID
        """
        if person_id not in self.known_faces_cache:
            logger.warning(f"Cannot update last_seen for unknown person: {person_id}")
            return

        current_time = time.time()

        # Update database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE people
            SET last_seen = ?, total_interactions = total_interactions + 1
            WHERE person_id = ?
        """, (current_time, person_id))

        conn.commit()
        conn.close()

        # Update in-memory cache
        person = self.known_faces_cache[person_id]
        person.last_seen = current_time
        person.total_interactions += 1

        logger.debug(f"Updated last_seen for {person.name}")
