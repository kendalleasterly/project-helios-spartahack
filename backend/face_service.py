"""
Fast and reliable face detection service using InsightFace.

InsightFace provides:
- 95%+ detection rate (vs DeepFace's ~70%)
- 3x faster processing (30-80ms vs 130-200ms)
- Better handling of angles, lighting, motion blur
- Production-ready performance
"""

import os
import sqlite3
import uuid
import time
import logging
import pickle
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
import numpy as np
from insightface.app import FaceAnalysis

logger = logging.getLogger(__name__)


# Configuration Constants
RECOGNITION_THRESHOLD = 0.40         # Cosine distance threshold (60% similarity required)
DB_PATH = "person_memory/faces.db"   # Relative to backend/


@dataclass
class Person:
    """Known person in database."""
    person_id: str
    name: str
    face_encoding: np.ndarray
    first_seen: float
    last_seen: float


@dataclass
class FaceDetection:
    """Detected face result."""
    person_id: Optional[str]
    person_name: Optional[str]
    confidence: float
    bbox: dict
    distance: Optional[float] = None


class FaceService:
    """
    FAST and RELIABLE face recognition using InsightFace.

    Key Features:
    - Buffalo_L model (balanced speed/accuracy)
    - SCRFD detector (95%+ detection rate)
    - ArcFace embeddings (512D, same as before)
    - Production-tested at scale
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize face service.

        Args:
            db_path: Path to SQLite database (default: person_memory/faces.db)
        """
        self.db_path = db_path or DB_PATH
        self.known_faces_cache: Dict[str, Person] = {}

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        # Initialize InsightFace analyzer (will be loaded in preload_models)
        self.app = None

        # Initialize database and load cache
        self._init_database()
        self._load_known_faces()

        logger.info(f"FaceService initialized with {len(self.known_faces_cache)} known faces")

    def preload_models(self) -> None:
        """
        Preload InsightFace models at startup to avoid delays on first detection.
        Downloads models (~200MB) on first run if not cached.
        """
        try:
            logger.info("Preloading InsightFace models (buffalo_l)...")

            # Initialize FaceAnalysis with buffalo_l model (good balance)
            self.app = FaceAnalysis(
                name='buffalo_l',
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )

            # Prepare the model (downloads if needed)
            self.app.prepare(ctx_id=0, det_size=(640, 640))

            logger.info("InsightFace models preloaded successfully")
        except Exception as e:
            logger.warning(f"Model preload warning (will lazy-load on first use): {e}")

    def detect_and_recognize(self, image: np.ndarray) -> List[FaceDetection]:
        """
        Main detection pipeline: Find and identify ALL faces in frame.

        Args:
            image: OpenCV BGR image (numpy array)

        Returns:
            List of FaceDetection objects (includes both known and unknown faces)
            Returns empty list on error or if no faces found
        """
        try:
            # Validate input
            if image is None or not isinstance(image, np.ndarray):
                logger.warning("Invalid image input")
                return []

            # Lazy load if not preloaded
            if self.app is None:
                self.preload_models()

            start_time = time.time()
            detections = []

            # Step 1: Detect and extract all faces with embeddings
            # InsightFace returns: bbox, kps (keypoints), det_score, embedding, gender, age
            faces = self.app.get(image)

            if not faces:
                logger.debug("No faces detected")
                return []

            # Step 2: Process each detected face
            for face in faces:
                # Extract face data
                bbox = face.bbox.astype(int)  # [x1, y1, x2, y2]
                confidence = float(face.det_score)
                embedding = face.normed_embedding  # Already normalized 512D vector

                # Convert bbox to our format
                x1, y1, x2, y2 = bbox
                face_bbox = {
                    'x': int(x1),
                    'y': int(y1),
                    'w': int(x2 - x1),
                    'h': int(y2 - y1)
                }

                # Step 3: Match against known faces
                person_id, person_name, distance = self._match_face(embedding)

                # Step 4: Create detection result
                detection = FaceDetection(
                    person_id=person_id,
                    person_name=person_name,
                    confidence=confidence,
                    bbox=face_bbox,
                    distance=distance
                )

                detections.append(detection)

                if person_name:
                    logger.info(f"Recognized {person_name} (distance: {distance:.3f}, conf: {confidence:.2f})")
                else:
                    logger.debug(f"Unknown face (best distance: {distance:.3f}, conf: {confidence:.2f})")

            elapsed = (time.time() - start_time) * 1000
            logger.info(f"Detected {len(detections)} faces in {elapsed:.0f}ms")

            return detections

        except Exception as e:
            logger.error(f"Error in detect_and_recognize: {e}", exc_info=True)
            return []

    def add_known_person(self, name: str, face_image: np.ndarray) -> str:
        """
        Register a new person's face to the database.

        Args:
            name: Person's name
            face_image: OpenCV BGR image containing the face

        Returns:
            person_id (UUID string)

        Raises:
            ValueError: If no face detected or embedding generation fails
        """
        try:
            logger.info(f"Adding known person: {name}")

            # Lazy load if not preloaded
            if self.app is None:
                self.preload_models()

            # Step 1: Detect face and extract embedding
            faces = self.app.get(face_image)

            if not faces or len(faces) == 0:
                raise ValueError("No face detected in image")

            # Use the first (largest/most prominent) face
            face = faces[0]
            embedding = face.normed_embedding  # 512D normalized vector

            # Step 2: Create person record
            person_id = str(uuid.uuid4())
            timestamp = time.time()

            person = Person(
                person_id=person_id,
                name=name,
                face_encoding=embedding,
                first_seen=timestamp,
                last_seen=timestamp
            )

            # Step 3: Save to database
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO people (person_id, name, face_encoding, first_seen, last_seen)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (person_id, name, pickle.dumps(embedding), timestamp, timestamp)
                )
                conn.commit()
            finally:
                conn.close()

            # Step 4: Update cache
            self.known_faces_cache[person_id] = person

            logger.info(f"Successfully added {name} with ID {person_id}")
            return person_id

        except Exception as e:
            logger.error(f"Error adding person {name}: {e}", exc_info=True)
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

    def update_last_seen(self, person_id: str) -> None:
        """
        Update the last_seen timestamp for a person.

        Args:
            person_id: UUID of the person
        """
        if person_id not in self.known_faces_cache:
            logger.warning(f"Attempted to update last_seen for unknown person: {person_id}")
            return

        timestamp = time.time()

        # Update cache
        person = self.known_faces_cache[person_id]
        person.last_seen = timestamp

        # Update database
        try:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE people SET last_seen = ? WHERE person_id = ?",
                    (timestamp, person_id)
                )
                conn.commit()
            finally:
                conn.close()
        except Exception as e:
            logger.error(f"Error updating last_seen for {person_id}: {e}")

    def _match_face(self, embedding: np.ndarray) -> Tuple[Optional[str], Optional[str], float]:
        """
        Compare embedding against all known faces in cache.

        Args:
            embedding: 512D normalized embedding from InsightFace

        Returns:
            (person_id, person_name, distance) tuple
            Returns (None, None, distance) if no match below threshold
        """
        if not self.known_faces_cache:
            return (None, None, float('inf'))

        best_distance = float('inf')
        best_id = None
        best_name = None

        # O(N) search through cache (acceptable for <100 people)
        for person_id, person in self.known_faces_cache.items():
            # Cosine distance (embeddings are already normalized)
            distance = self._cosine_distance(embedding, person.face_encoding)

            if distance < best_distance:
                best_distance = distance
                best_id = person_id
                best_name = person.name

        # Apply recognition threshold
        if best_distance < RECOGNITION_THRESHOLD:
            return (best_id, best_name, best_distance)
        else:
            return (None, None, best_distance)

    def _cosine_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate cosine distance between two normalized embeddings.

        Args:
            a, b: 512D normalized embeddings

        Returns:
            Distance in range [0, 2] where 0 = identical
        """
        # For normalized vectors: cosine_distance = 1 - dot_product
        similarity = np.dot(a, b)
        distance = 1 - similarity
        return float(distance)

    def _init_database(self) -> None:
        """Create database schema if not exists."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()

            # Create people table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS people (
                    person_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    face_encoding BLOB NOT NULL,
                    first_seen REAL NOT NULL,
                    last_seen REAL NOT NULL
                )
                """
            )

            # Create index for name lookups
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_name ON people(name)
                """
            )

            conn.commit()
            logger.debug("Database schema initialized")

        finally:
            conn.close()

    def _load_known_faces(self) -> None:
        """Load all known faces from database into memory cache."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT person_id, name, face_encoding, first_seen, last_seen
                FROM people
                """
            )

            rows = cursor.fetchall()

            for row in rows:
                person_id, name, encoding_blob, first_seen, last_seen = row

                # Unpickle face encoding
                face_encoding = pickle.loads(encoding_blob)

                # Add to cache
                person = Person(
                    person_id=person_id,
                    name=name,
                    face_encoding=face_encoding,
                    first_seen=first_seen,
                    last_seen=last_seen
                )

                self.known_faces_cache[person_id] = person

            logger.debug(f"Loaded {len(rows)} known faces from database")

        finally:
            conn.close()
