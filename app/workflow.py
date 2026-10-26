from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from app.models import AgentStep


class State(TypedDict, total=False):
    question: str
    top_k: int
    route: str
    evidence: list
    answer: str
    mode: str
    confidence: float
    grounded: bool
    trace: list


def build_graph(hub):
    def route(state):
        selected, step = hub.router.run(state['question'])
        return {'route': selected, 'trace': [step]}

    def retrieve(state):
        evidence = hub.kb.search(state['question'], state['top_k'])
        step = AgentStep(agent=f"{state['route']}_specialist", action='retrieve',
                         detail=f'Retrieved {len(evidence)} evidence chunks')
        return {'evidence': evidence, 'trace': state['trace'] + [step]}

    def synthesize(state):
        answer, step, mode = hub.synthesizer.run(state['question'], state['route'], state['evidence'])
        return {'answer': answer, 'mode': mode, 'trace': state['trace'] + [step]}

    def quality(state):
        confidence, grounded, step = hub.quality.run(state['answer'], state['evidence'])
        return {'confidence': confidence, 'grounded': grounded, 'trace': state['trace'] + [step]}

    graph = StateGraph(State)
    graph.add_node('router', route)
    for specialist in ('policy', 'technical', 'general'):
        graph.add_node(specialist, retrieve)
        graph.add_edge(specialist, 'synthesis')
    graph.add_node('synthesis', synthesize)
    graph.add_node('quality', quality)
    graph.add_edge(START, 'router')
    graph.add_conditional_edges('router', lambda s: s['route'],
                                {name: name for name in ('policy', 'technical', 'general')})
    graph.add_edge('synthesis', 'quality')
    graph.add_edge('quality', END)
    return graph.compile()
