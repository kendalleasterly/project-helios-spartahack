import sqlite3
import json

# Check faces database
print("=" * 60)
print("FACES DATABASE")
print("=" * 60)

conn = sqlite3.connect('backend/person_memory/faces.db')
cursor = conn.cursor()

# Get tables
cursor.execute('SELECT name FROM sqlite_master WHERE type="table"')
tables = cursor.fetchall()
print(f"\nTables: {tables}")

# Get people table structure
cursor.execute('PRAGMA table_info(people)')
columns = cursor.fetchall()
print(f"\nPeople table columns:")
for col in columns:
    print(f"  - {col[1]} ({col[2]})")

# Get all persons
cursor.execute('SELECT * FROM people')
rows = cursor.fetchall()
print(f"\nTotal persons: {len(rows)}")
print("\nPersons data:")
for row in rows:
    print(f"  ID: {row[0]}")
    print(f"  Name: {row[1]}")
    print(f"  Embedding size: {len(row[2]) if row[2] else 0} bytes")
    print(f"  Created: {row[3]}")
    print()

conn.close()

# Check ChromaDB notes
print("=" * 60)
print("NOTES DATABASE (ChromaDB)")
print("=" * 60)

try:
    import chromadb

    chroma_client = chromadb.PersistentClient(path="backend/person_memory/chroma_db")

    # Get or create collection
    try:
        collection = chroma_client.get_collection(name="person_notes")

        # Get all notes
        results = collection.get()

        print(f"\nTotal notes: {len(results['ids'])}")
        print("\nNotes data:")
        for i, note_id in enumerate(results['ids']):
            print(f"  Note ID: {note_id}")
            print(f"  Text: {results['documents'][i]}")
            if results['metadatas'][i]:
                print(f"  Metadata: {results['metadatas'][i]}")
            print()
    except Exception as e:
        print(f"No notes collection found or error: {e}")

except ImportError:
    print("ChromaDB not installed, skipping notes check")
