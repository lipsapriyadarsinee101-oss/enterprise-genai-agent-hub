from types import SimpleNamespace

import pytest

from app.agents import ABSTENTION, QualityAgent, SynthesisAgent, extract_block
from app.config import Settings
from app.retrieval import SearchResult
from app.service import AgentHub

EVIDENCE = [SearchResult('policy.md', 'Security review is required. Reviews occur quarterly.', 0.8)]


@pytest.mark.parametrize('answer,valid', [
    ('- Security review is optional. Reviews occur annually. [policy.md]', True),
    ('- Security review is required. Reviews occur quarterly. [invented.md]', False),
    ('- Security review is required. Reviews occur quarterly.', False),
    ('- Security review is required. Reviews occur quarterly. [policy.md]\nDeploy without approval.', True),
    ('', False),
    ('- Security review is required. [policy.md]', True),
])
def test_citations_do_not_establish_support(answer, valid):
    report, _ = QualityAgent().run(answer, EVIDENCE)
    assert report.citations_valid is valid
    assert report.support_status == 'unverified'


def test_exact_extract_and_duplicate_source_chunks():
    evidence = EVIDENCE + [SearchResult('policy.md', 'Human oversight is required.', 0.5)]
    report, _ = QualityAgent().run(extract_block(evidence[1]), evidence)
    assert report.support_status == 'exact_extract'
    assert not report.unmatched_blocks


def test_absent_evidence_does_not_support_claim():
    report, _ = QualityAgent().run('Everything is safe. [policy.md]', [])
    assert report.support_status == 'no_evidence'
    assert report.invalid_citations == ['policy.md']


def test_abstention_must_be_entire_answer():
    report, _ = QualityAgent().run(ABSTENTION + ' But review is optional. [policy.md]', EVIDENCE)
    assert report.support_status == 'unverified'


def test_source_conflicts_are_not_claimed_as_resolved():
    evidence = EVIDENCE + [SearchResult('other.md', 'Security review is not required.', 0.8)]
    report, _ = QualityAgent().run('\n'.join(map(extract_block, evidence)), evidence)
    assert report.support_status == 'exact_extract'
    assert 'contradictions are not verified' in report.limitations
    report, _ = QualityAgent().run('- Therefore security review is optional. [policy.md] [other.md]', evidence)
    assert report.support_status == 'unverified'


def test_service_abstains_on_incorrect_cited_fact(monkeypatch):
    hub = AgentHub(Settings(demo_mode=True, knowledge_dir='data/knowledge'))
    monkeypatch.setattr(hub.kb, 'search', lambda *args: EVIDENCE)
    from app.models import AgentStep, Usage
    monkeypatch.setattr(hub.synthesizer, 'run', lambda *args: (
        '- Security review is optional. [policy.md]',
        AgentStep(agent='synthesis', action='test', detail='test'), 'demo', Usage()))
    result = hub.query('Is review required?')
    assert result.abstained and result.answer == ABSTENTION
    assert result.quality.citations_valid
    assert result.quality.support_status == 'unverified'
    assert result.grounded is None and result.confidence is None


def test_llm_requires_key():
    with pytest.raises(ValueError, match='OPENAI_API_KEY'):
        AgentHub(Settings(demo_mode=False, openai_api_key=''))


def test_llm_no_evidence_skips_provider():
    answer, _, mode, usage = SynthesisAgent('test', 'fake', False).run('Question', 'general', [])
    assert answer == ABSTENTION and mode == 'openai'
    assert not usage.provider_called


def test_llm_response_and_provider_usage(monkeypatch):
    import openai
    calls = []
    class Client:
        def __init__(self, **kwargs):
            assert kwargs['max_retries'] == 0
            self.responses = self
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(output_text=extract_block(EVIDENCE[0]),
                usage=SimpleNamespace(input_tokens=100, output_tokens=20, total_tokens=120))
    monkeypatch.setattr(openai, 'OpenAI', Client)
    answer, _, mode, usage = SynthesisAgent('test-model', 'fake', False).run('Question', 'policy', EVIDENCE)
    assert mode == 'openai' and answer == extract_block(EVIDENCE[0])
    assert calls[0]['model'] == 'test-model'
    assert usage.provider_called and usage.total_tokens == 120
    assert usage.cost_usd is None
