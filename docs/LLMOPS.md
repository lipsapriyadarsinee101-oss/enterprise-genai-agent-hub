# LLMOps project guide

## What each tool does

| Tool | Implementation | How to verify |
|---|---|---|
| LangGraph | app/workflow.py: conditional specialist routes, synthesis, quality | Existing API/service tests run the graph |
| LangChain | app/agents.py: prompt -> ChatOpenAI -> parser | Run a real query with demo disabled |
| FAISS | app/vector_store.py: embeddings and local similarity index | Unit test uses real FAISS with deterministic test embeddings |
| OpenSearch | Explicit ingestion, knn_vector mapping and k-NN query | Provision server, ingest, query |
| Langfuse | Graph callback configured in app/service.py | Enable credentials, query and inspect a trace |
| LLM-as-a-judge | Separate structured correctness and grounding evaluation | Run scripts.evaluate_judge with credentials |
| statsmodels | Wilson 95% interval on evaluation pass rate | Inspect judge report |
| Terraform | Local Docker OpenSearch resource | terraform init, validate, plan, apply |
| Jenkins / GitLab CI | Tests and offline evaluation pipelines | Import repo into a configured runner |

These are LLMOps practices: versioned evaluation, quality checks, observability, reproducible infrastructure and CI. A configuration file alone does not mean a deployment has been tested.

## Install and run

Use Python 3.11 or 3.12. From the repository root:

```bash
python -m venv .venv
# Windows PowerShell:
.venv/Scripts/Activate.ps1
# macOS/Linux instead: source .venv/bin/activate
pip install -r requirements.txt
# PowerShell: Copy-Item .env.example .env
# macOS/Linux: cp .env.example .env
python -m pytest -q
python -m scripts.evaluate
uvicorn app.api:app --reload
```

The default TF-IDF retrieval and extractive demo are offline. This baseline is kept so the API and UI remain easy to demonstrate. LangGraph runs in every mode.

## Dense retrieval with FAISS

Set these values in `.env`, then restart the API:

```dotenv
OPENAI_API_KEY=your-own-key
DEMO_MODE=false
RETRIEVAL_BACKEND=faiss
EMBEDDING_MODEL=text-embedding-3-small
```

The app embeds the sample documents at startup and builds an in-memory FAISS index. Restarting re-embeds them and may incur API charges. FAISS similarity is reported as a distance-derived score; it is not a probability of correctness. Dense retrieval always returns nearest neighbors, so irrelevant questions can still retrieve content. Human review and stronger abstention calibration remain necessary.

## OpenSearch and Terraform

Install Docker and Terraform first. Run from the repository root:

```bash
terraform -chdir=infra init
terraform -chdir=infra fmt -check
terraform -chdir=infra validate
terraform -chdir=infra plan
terraform -chdir=infra apply
```

This creates only a local development OpenSearch container bound to 127.0.0.1:9200. Authentication is disabled for the local example. Wait until `http://localhost:9200` responds. Linux Docker hosts may need `vm.max_map_count=262144` configured by their administrator. Do not run Compose and Terraform OpenSearch containers on the same port.

Set `.env`:

```dotenv
RETRIEVAL_BACKEND=opensearch
OPENSEARCH_URL=http://localhost:9200
OPENSEARCH_INDEX=agent-hub-v1
```

Keep your API key and embedding model settings. Run `python -m scripts.ingest` once, then restart the API and submit a query. Ingestion refuses to replace an existing index. Choose a new index name when changing the corpus or embedding model. The API never creates/deletes indexes automatically. Run `terraform -chdir=infra destroy` to remove this local example when finished. Terraform uses local state; do not commit state files. No cloud resources are created by this configuration.

For a remote authenticated service, use a TLS URL and configure appropriate authentication in the client adapter before use. This example has no AWS IAM signing or multi-tenant source authorization.

## Langfuse tracing

Create a Langfuse project and set `LANGFUSE_ENABLED=true`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` and `LANGFUSE_BASE_URL` in `.env`. Restart the API, submit a query, and inspect router, specialist, synthesis and quality spans in Langfuse. With real generation enabled, the nested LangChain calls are captured too. Use sample documents: enabling tracing exports prompts and answers to the configured service. Leave tracing disabled for sensitive data until redaction and access policies are implemented.

Integration reference: [Langfuse LangChain callbacks](https://langfuse.com/integrations/frameworks/langchain).

## LLM-as-a-judge and statsmodels

Set `DEMO_MODE=false` and your API key, then:

```bash
python -m scripts.evaluate_judge --model YOUR_AVAILABLE_JUDGE_MODEL --output reports/judge.json
```

Choose a judge different from the generator where possible. The evaluator gives correctness and grounding scores in [0,1] and an explanation. A case passes if both scores are at least 0.8; statsmodels computes a Wilson 95% confidence interval for the pass rate. The three cases are illustrative and far too small to establish production quality. Review scores manually and expand the dataset with realistic questions, contradictions, abstentions and injection cases. Judge evaluation is separate from the API's fast citation heuristic. The API `grounded` flag means citation presence, not verified factual correctness. No live judge scores are fabricated when credentials are missing.

## CI

GitHub Actions remains the active CI. Jenkinsfile requires a Jenkins agent with the Docker Pipeline and JUnit plugins; `.gitlab-ci.yml` requires a GitLab project and runner. Both run the same tests and offline evaluation. Live judge runs are deliberately manual because they need secrets and incur usage. Store credentials in the CI secret manager, never in source files. Terraform plan/apply is manual.

## Interview exercises and CV wording

Run each integration yourself. Explain graph conditional edges, embedding model/index compatibility, nearest-neighbor limitations, callback traces, judge bias and statistical uncertainty. Save actual results and screenshots privately, investigate failures, and make your own changes.

After you have run and understood the features, a defensible CV bullet is:

> Extended an enterprise RAG agent project with LangGraph orchestration, LangChain generation, dense retrieval, and LLMOps evaluation and tracing; implemented an independent LLM judge and statistical reporting.

Only name OpenSearch, Langfuse, Terraform or external CI systems as hands-on experience after running those integrations. No ATS percentage is guaranteed by these additions.

## References

- [LangGraph graph API](https://reference.langchain.com/python/langgraph/graph)
- [OpenSearch vector mapping](https://docs.opensearch.org/latest/mappings/supported-field-types/knn-vector/)
- [statsmodels Wilson intervals](https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.proportion_confint.html)
- [Terraform Docker provider](https://registry.terraform.io/providers/kreuzwerker/docker/latest/docs/resources/container)
