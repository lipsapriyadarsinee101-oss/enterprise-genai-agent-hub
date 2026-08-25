# Production RAG Operations

## Pipeline design
A production RAG pipeline separates ingestion, chunking, indexing, retrieval, generation, and evaluation. Each stage should expose version metadata so an answer can be traced to the prompt, model, index, and source documents used.

## Monitoring
Teams monitor retrieval relevance, answer groundedness, citation coverage, latency, token use, user feedback, and refusal rate. Alerts should detect ingestion failures, stale indexes, sudden quality drops, and cost anomalies.

## Evaluation
Every release is tested against a versioned evaluation set containing representative questions, expected sources, and risk cases. A release is blocked when groundedness or safety performance falls below its agreed threshold.

## Security
Retrieved content is untrusted input. The system must isolate it from system instructions, filter unauthorized sources before retrieval, and test for prompt injection and data exfiltration.
