# LongMemEval → HyperBinder

Ingestion and evaluation pipeline for the **LongMemEval** benchmark using HyperBinder's dual-slot semantic search.

---

## Overview

LongMemEval tests a system's ability to retrieve accurate answers from long, multi-session conversation histories. HyperBinder ingests each dataset into a dedicated namespace and answers questions by searching across two semantic slots — the question itself and the relevant content chunk — with weighted scoring.

---

## Supported Datasets

| Index | Dataset |
|---|---|
| 0 | `implicit_preference_v2` |
| 1 | `knowledge_update` |
| 2 | `knowledge_update` (100-sample) |
| 3 | `multi_session_synthesis` |
| 4 | `single_hop` |
| 5 | `temp_reasoning_explicit` |
| 6 | `temp_reasoning_implicit` |
| 7 | `two_hop` |
| 8 | `longmemeval_s` |

All files are expected in a `data/` directory relative to the script.

---

## Schema Design

Each row is stored with the following field encodings:

| Field | Encoding |
|---|---|
| `question` | Semantic |
| `content_chunk` | Semantic |
| `answer` | Exact |
| `session_id` | Exact |
| `question_type` | Exact |
| `question_date` | Temporal |

The primary key is a composite `fact_id` formed from `session_id` + `chunk_index`.

---

## Usage

### 1. Ingest

```bash
python ingest.py
```

Prompts you to select a dataset index (0–8) and enter a namespace name, then:

- Runs `LongMemEvalIngestor` to preprocess the JSON into rows
- Builds a DataFrame and uploads it as a CSV to HyperBinder
- Reports mode, namespace, rows added, and vector source on completion

### 2. Evaluate

```bash
python eval.py
```

Prompts for the same dataset index and namespace, then:

- Loads all questions from the source JSON
- Runs a **weighted dual-slot search** per question:
  - `question` slot at weight `0.7`
  - `content_chunk` slot at weight `0.3`
- Scores predictions via exact string match against ground truth
- Prints per-question results and a final accuracy summary

---

## Configuration

Set the following environment variables (or use a `.env` file):

| Variable | Description |
|---|---|
| `SERVER_URL` | HyperBinder server URL |
| `API_KEY` | API key for authentication |

---

## Output

Each evaluated question prints:

```
ID   : <question_id> | type=<question_type>
Q    : <question text>
GT   : <ground truth answer>
PRED : <predicted answer>  (score=0.XXX)  ✅ / ❌
```

Followed by a final summary:

```
Accuracy: X/Y = Z%
```

---

## Get Access

Request an API key at [questions@semantic-reach.io](mailto:questions@semantic-reach.io)



