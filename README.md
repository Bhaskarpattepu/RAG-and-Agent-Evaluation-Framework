# RAG + Agent Evaluation Framework

A runnable learning project combining FastAPI, LangChain, LangGraph tool calling, ChromaDB, Pytest, DeepEval, and LangSmith.

## Prerequisites
- Python 3.11 or 3.12
- OpenAI API key (embedding + chat + DeepEval judge API costs may apply)
- Optional LangSmith account/key

## Setup
```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env          # Windows: copy .env.example .env
```
Edit `.env` and add `OPENAI_API_KEY`. Optional: set `LANGSMITH_TRACING=true` and add `LANGSMITH_API_KEY` to send traces to LangSmith.

## Run
```bash
python -m scripts.ingest_demo
uvicorn app.main:app --reload
```
Open http://127.0.0.1:8000/docs and use POST `/ask` with:
```json
{"question":"Are digital gift cards refundable?"}
```
Use POST `/ingest` with `{"filename":"refund_policy.txt"}` for files already present under `data/documents/`.

## Tests
```bash
python -m pytest -q tests/ -k "not test_agent_selects_ticket_tool and not test_agent_selects_retrieval_tool"  # offline tests
python -m pytest -q tests/test_agent_tools.py -k "test_agent_selects_ticket_tool or test_agent_selects_retrieval_tool"  # live OpenAI tests
python -m scripts.evaluate_agent           # live agent tool/source check
python -m scripts.run_evaluation           # DeepEval semantic evaluations, requires API key
```
The live OpenAI tests and evaluation scripts require `OPENAI_API_KEY` and may incur API charges. The CI workflow runs offline tests and a Docker build first, then requires `OPENAI_API_KEY` to run live agent tests on pushes and same-repository pull requests; it fails fast if the secret is missing. Fork pull requests run only the offline checks. LangSmith tracing is enabled only when the corresponding environment variable is set.

## Docker
```bash
docker build -t rag-eval .
docker run --env-file .env -p 8000:8000 rag-eval
```
The image includes demo documents but local Chroma persistence is ephemeral unless a volume is mounted. For persistence, mount `/code/data/chroma` as a volume and ingest documents after starting the container.

## Architecture
- `app/rag.py`: PDF/text ingestion, chunking, embeddings, vector search.
- `app/agent.py`: LangGraph tool-calling loop with a recursion limit and source collection.
- `app/main.py`: FastAPI endpoints and input validation.
- `tests/`: deterministic API/retrieval tests.
- `scripts/evaluate_agent.py`: live agent tool selection and source checks.
- `scripts/run_evaluation.py`: DeepEval faithfulness and answer relevancy.

## Limitations and next steps
This is a learning/demo framework, not a hardened production service. In particular:
- `/ingest` is operator-only by design and assumes trusted local files. Add auth, file size limits, content scanning, tenant isolation and upload handling before deployment.
- Ingestion of the same file multiple times creates duplicate chunks; implement stable IDs/upsert/delete-by-source for production.
- LangGraph state is request-scoped; add checkpointer and durable persistence for multi-turn conversations.
- `used_tool` checks that a tool executed, not that the model chose the correct tool or that the final answer is accurate.
- DeepEval may require a judge API key and its API/metric signatures may change across major versions; pin tested dependency versions for CI.
- Live evaluations are intentionally not run in the default GitHub Actions job because they require secrets and incur costs.
- Improve retrieval with hybrid search, reranking, retrieval metrics and grounded source-span citations.
