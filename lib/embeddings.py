import os
import json
import numpy as np
from pathlib import Path
from typing import Optional, List
import faiss
from sentence_transformers import SentenceTransformer

EMBEDDINGS_DIR = Path("transcripts/embeddings")
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

INDEX_FILE = EMBEDDINGS_DIR / "index.faiss"
META_FILE = EMBEDDINGS_DIR / "metadata.json"


class EmbeddingsManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.index = None
        self.metadata = []
        self._load_index()

    def _load_index(self):
        if INDEX_FILE.exists() and META_FILE.exists():
            self.index = faiss.read_index(str(INDEX_FILE))
            with open(META_FILE, "r") as f:
                self.metadata = json.load(f)
        else:
            dimension = self._get_embedding_dimension()
            self.index = faiss.IndexFlatL2(dimension)
            self.metadata = []

    def _get_embedding_dimension(self) -> int:
        model = self._get_model()
        return model.get_sentence_embedding_dimension()

    def _get_model(self):
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)
        return self.model

    def add_transcription(self, job_id: str, text: str, filename: str) -> bool:
        model = self._get_model()

        sentences = self._split_into_sentences(text)
        embeddings = model.encode(sentences)

        self.index.add(embeddings)

        for i, sentence in enumerate(sentences):
            self.metadata.append(
                {
                    "job_id": job_id,
                    "filename": filename,
                    "text": sentence,
                    "chunk_id": i,
                }
            )

        self._save_index()
        return True

    def _split_into_sentences(self, text: str, chunk_size: int = 500) -> List[str]:
        sentences = text.replace("\n", " ").split(". ")
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if current_length + len(sentence) > chunk_size:
                if current_chunk:
                    chunks.append(". ".join(current_chunk) + ".")
                    current_chunk = []
                    current_length = 0

            current_chunk.append(sentence)
            current_length += len(sentence)

        if current_chunk:
            chunks.append(". ".join(current_chunk) + ".")

        return chunks if chunks else [text[:chunk_size]]

    def _save_index(self):
        faiss.write_index(self.index, str(INDEX_FILE))
        with open(META_FILE, "w") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def search(self, query: str, top_k: int = 5) -> List[dict]:
        if self.index.ntotal == 0:
            return []

        model = self._get_model()
        query_embedding = model.encode([query])

        distances, indices = self.index.search(
            query_embedding, min(top_k, self.index.ntotal)
        )

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.metadata):
                results.append(
                    {
                        "text": self.metadata[idx]["text"],
                        "filename": self.metadata[idx]["filename"],
                        "job_id": self.metadata[idx]["job_id"],
                        "chunk_id": self.metadata[idx]["chunk_id"],
                        "distance": float(dist),
                    }
                )

        return results

    def get_all_embedded(self) -> List[dict]:
        return self.metadata

    def get_job_embeddings(self, job_id: str) -> List[dict]:
        return [m for m in self.metadata if m.get("job_id") == job_id]

    def clear_index(self):
        dimension = self._get_embedding_dimension()
        self.index = faiss.IndexFlatL2(dimension)
        self.metadata = []
        self._save_index()
