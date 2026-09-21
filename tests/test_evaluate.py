import json

import pytest

from app.config import Settings
from app.service import AgentHub
from scripts.evaluate import DEFAULT_CASES, evaluate, main, score_case


def test_annotations_check_content_not_just_source():
    hub = AgentHub(Settings(demo_mode=True, knowledge_dir='data/knowledge'))
    result = hub.query('What controls are needed before AI deployment?')
    case = {'expected_evidence': [{'source': 'ai-governance.md', 'contains': 'invented requirement'}],
            'expect_abstention': False, 'required_answer_spans': ['not present'], 'forbidden_answer_spans': []}
    failures = score_case(case, result)
    assert len(failures) == 2


def test_evaluation_reports_individual_execution_failures():
    class BrokenHub:
        def query(self, *args):
            raise RuntimeError('private provider detail')
    rows = evaluate(BrokenHub(), [{'question': 'test'}])
    assert not rows[0]['passed'] and rows[0]['failures'] == ['execution_error']
    assert 'private provider detail' not in json.dumps(rows)


def test_offline_report_records_known_abstention_failure(tmp_path):
    output = tmp_path / 'results.json'
    assert main(['--mode', 'demo', '--output', str(output)]) == 0
    report = json.loads(output.read_text(encoding='utf-8'))
    assert report['summary']['cases'] == 10
    row = next(r for r in report['results'] if r['case']['id'] == 'unknown-budget-overlap')
    assert 'expected_abstention' in row['failures']
    assert report['summary']['actual_cost_usd'] is None
    assert report['reproducibility']['cases_sha256']
    assert all(not r['response']['usage']['provider_called'] for r in report['results'])
    assert main(['--mode', 'demo', '--output', str(output), '--fail-on-case-failure']) == 1


def test_llm_mode_never_falls_back(monkeypatch, tmp_path):
    monkeypatch.setenv('OPENAI_API_KEY', '')
    with pytest.raises(SystemExit) as exc:
        main(['--mode', 'llm', '--output', str(tmp_path / 'results.json')])
    assert exc.value.code == 2
