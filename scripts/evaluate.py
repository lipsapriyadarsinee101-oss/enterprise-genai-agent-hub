from statistics import mean

from app.config import Settings
from app.service import AgentHub


CASES = [
    ("What controls are needed before AI deployment?", "ai-governance.md"),
    ("How do we monitor a RAG system?", "rag-operations.md"),
    ("What should happen after a critical AI incident?", "incident-response.md"),
]


def main() -> None:
    hub = AgentHub(Settings(demo_mode=True, knowledge_dir="data/knowledge"))
    results = [hub.query(question) for question, _ in CASES]
    retrieval_hits = [expected in {citation.source for citation in result.citations} for result, (_, expected) in zip(results, CASES)]
    print("Enterprise GenAI Agent Hub — Evaluation")
    print(f"Retrieval hit rate: {mean(retrieval_hits):.0%}")
    print(f"Grounded answer rate: {mean(r.grounded for r in results):.0%}")
    print(f"Mean confidence: {mean(r.confidence for r in results):.0%}")
    print(f"Mean latency: {mean(r.latency_ms for r in results):.1f} ms")


if __name__ == "__main__":
    main()
