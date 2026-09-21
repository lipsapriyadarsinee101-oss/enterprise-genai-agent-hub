from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import time

from app.config import Settings
from app.service import AgentHub

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / 'data/benchmark.json'


def score_case(case, result):
    failures = []
    evidence = result.citations
    for expected in case['expected_evidence']:
        if not any(e.source == expected['source'] and expected['contains'] in e.excerpt for e in evidence):
            failures.append('missing_evidence: ' + expected['contains'])
    if result.abstained != case['expect_abstention']:
        failures.append('unexpected_abstention' if result.abstained else 'expected_abstention')
    for requirement in case['required_answer_spans']:
        if requirement not in result.answer:
            failures.append('missing_answer_span: ' + requirement)
    for forbidden in case['forbidden_answer_spans']:
        if forbidden in result.answer:
            failures.append('forbidden_answer_span: ' + forbidden)
    if not result.abstained and result.quality.support_status != 'exact_extract':
        failures.append('unverified_output')
    return failures


def evaluate(hub, cases):
    rows = []
    for case in cases:
        started = time.perf_counter()
        try:
            result = hub.query(case['question'], case.get('top_k', 3))
            failures = score_case(case, result)
            row = dict(case=case, passed=not failures, failures=failures, response=result.model_dump())
        except Exception as exc:
            # Do not copy provider messages (which may contain sensitive request data).
            row = dict(case=case, passed=False, failures=['execution_error'], error_type=type(exc).__name__, response=None)
        row['elapsed_ms'] = round((time.perf_counter() - started) * 1000, 3)
        rows.append(row)
    return rows


def fingerprint(directory):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(directory).glob('*')) if p.suffix.lower() in {'.txt', '.md'}}


def main(argv=None):
    parser = argparse.ArgumentParser(description='Annotated extraction benchmark; not a factual-grounding metric.')
    parser.add_argument('--mode', choices=['demo', 'llm'], default='demo')
    parser.add_argument('--cases', type=Path, default=DEFAULT_CASES)
    parser.add_argument('--knowledge-dir', type=Path, default=ROOT / 'data/knowledge')
    parser.add_argument('--model', default=None)
    parser.add_argument('--output', type=Path, default=Path('evaluation-results.json'))
    parser.add_argument('--fail-on-case-failure', action='store_true')
    args = parser.parse_args(argv)
    settings = Settings(demo_mode=args.mode == 'demo', knowledge_dir=str(args.knowledge_dir))
    if args.model:
        settings.openai_model = args.model
    if args.mode == 'llm' and not settings.openai_api_key:
        parser.error('--mode llm requires OPENAI_API_KEY; no demo fallback')
    suite = json.loads(args.cases.read_text(encoding='utf-8'))
    started_at = datetime.now(timezone.utc).isoformat()
    rows = evaluate(AgentHub(settings), suite['cases'])
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    report = {
        'schema_version': 1, 'benchmark_version': suite['version'],
        'started_at_utc': started_at,
        'settings': {'mode': args.mode, 'model': settings.openai_model if args.mode == 'llm' else None,
                     'retriever': 'TF-IDF word unigrams/bigrams, English stop words, score > 0',
                     'top_k': 'per case, default 3', 'generation': 'verbatim excerpt selection',
                     'max_retries': 0, 'timeout_seconds': 60, 'sampling': 'provider default; LLM output is nondeterministic'},
        'reproducibility': {'git_revision': revision,
            'code_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for folder in ('app', 'scripts') for p in sorted((ROOT / folder).glob('*.py'))},
            'cases_sha256': hashlib.sha256(args.cases.read_bytes()).hexdigest(),
            'knowledge_sha256': fingerprint(args.knowledge_dir), 'python': platform.python_version(),
            'packages': {name: importlib.metadata.version(name) for name in ('scikit-learn', 'pydantic', 'openai')}},
        'limitations': 'Small authored benchmark, not a held-out accuracy estimate. Literal span checks do not establish semantic correctness. Demo returns retrieved excerpts without reasoning. Exact extracts can be irrelevant or mutually contradictory.',
        'summary': {'cases': len(rows), 'passed': sum(r['passed'] for r in rows),
                    'failed': sum(not r['passed'] for r in rows),
                    'mean_latency_ms': sum(r['elapsed_ms'] for r in rows) / len(rows) if rows else None,
                    'actual_cost_usd': None, 'cost_status': 'unavailable: billed costs are not returned by the provider'},
        'results': rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(report['summary']))
    return int(any('execution_error' in r['failures'] for r in rows) or
               (args.fail_on_case_failure and any(not r['passed'] for r in rows)))


if __name__ == '__main__':
    raise SystemExit(main())
