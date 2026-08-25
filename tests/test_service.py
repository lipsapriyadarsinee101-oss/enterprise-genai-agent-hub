from app.config import Settings
from app.service import AgentHub


def make_hub() -> AgentHub:
    return AgentHub(Settings(demo_mode=True, knowledge_dir="data/knowledge"))


def test_query_is_grounded_and_cited():
    result = make_hub().query("What controls are required before deploying an AI model?")
    assert result.route == "policy"
    assert result.grounded is True
    assert result.citations
    assert "[ai-governance.md]" in result.answer


def test_unknown_question_abstains():
    result = make_hub().query("How many cafeterias are in the Madrid office?")
    assert result.grounded is False
    assert "could not find sufficient evidence" in result.answer


def test_trace_contains_four_agent_steps():
    result = make_hub().query("How should the RAG pipeline be monitored?")
    assert [step.agent for step in result.trace] == ["router", "technical_specialist", "synthesis", "quality"]
