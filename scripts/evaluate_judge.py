import argparse
import json
from pathlib import Path
from statistics import mean
from statsmodels.stats.proportion import proportion_confint
from langchain_openai import ChatOpenAI
from app.config import Settings
from app.judge import judge_answer
from app.service import AgentHub


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='reports/judge.json')
    parser.add_argument('--model', required=True, help='Choose an available judge model independent of synthesis')
    args = parser.parse_args()
    settings = Settings()
    if settings.demo_mode or not settings.openai_api_key:
        raise SystemExit('Judge evaluation requires DEMO_MODE=false and OPENAI_API_KEY; no demo scores are emitted')
    hub = AgentHub(settings)
    model = ChatOpenAI(model=args.model, api_key=settings.openai_api_key, temperature=0)
    cases = json.loads(Path('data/evaluation.json').read_text())
    rows = []
    for case in cases:
        result = hub.query(case['question'])
        evidence = hub.kb.search(case['question'], 3)
        score = judge_answer(model, case['question'], result.answer,
                             [e.text for e in evidence], case['reference'])
        rows.append({'question': case['question'], 'answer': result.answer, **score.model_dump()})
    passed = sum(r['correctness'] >= .8 and r['grounding'] >= .8 for r in rows)
    low, high = proportion_confint(passed, len(rows), method='wilson')
    report = {'judge_model': args.model, 'generator_model': settings.openai_model,
              'backend': settings.retrieval_backend, 'cases': rows,
              'pass_rate': passed/len(rows), 'wilson_95_ci': [float(low), float(high)],
              'mean_grounding': mean(r['grounding'] for r in rows),
              'limitation': 'Tiny illustrative dataset; scores need human review and are not a production benchmark.'}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
