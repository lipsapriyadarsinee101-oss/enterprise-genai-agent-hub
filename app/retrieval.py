from dataclasses import dataclass
from pathlib import Path
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class Chunk:
    source: str
    text: str


@dataclass(frozen=True)
class SearchResult:
    source: str
    text: str
    score: float


class KnowledgeBase:
    def __init__(self, directory: str):
        self.directory = Path(directory)
        self.chunks = self._load()
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform([c.text for c in self.chunks]) if self.chunks else None

    def _load(self) -> list[Chunk]:
        chunks: list[Chunk] = []
        if not self.directory.exists():
            return chunks
        for path in sorted(self.directory.glob("*")):
            if path.suffix.lower() not in {".md", ".txt"}:
                continue
            text = path.read_text(encoding="utf-8")
            sections = [s.strip() for s in re.split(r"\n(?=#{1,3} )|\n\s*\n", text) if len(s.strip()) > 40]
            chunks.extend(Chunk(source=path.name, text=s) for s in sections)
        return chunks

    @property
    def documents(self) -> list[str]:
        return sorted({chunk.source for chunk in self.chunks})

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        if self.matrix is None or not query.strip():
            return []
        scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        ranked = scores.argsort()[::-1][:top_k]
        return [
            SearchResult(self.chunks[i].source, self.chunks[i].text, round(float(scores[i]), 4))
            for i in ranked
            if scores[i] > 0
        ]
