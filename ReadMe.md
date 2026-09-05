# LongMemEval → HyperBinder

Ingestion and evaluation pipeline for the **LongMemEval** benchmark using HyperBinder's dual-slot semantic search.

---

## Overview

LongMemEval tests a system's ability to retrieve accurate answers from long, multi-session conversation histories.

HyperBinder ingests each dataset into a dedicated namespace and answers questions by searching across two semantic slots — the question itself and the relevant content chunk — with weighted scoring.

---

## Supported Datasets

| Index | Dataset |
|------:|---------|
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
|-------|----------|
| `question` | Semantic |
| `content_chunk` | Semantic |
| `answer` | Exact |
| `session_id` | Exact |
| `question_type` | Exact |
| `question_date` | Temporal |

The primary key is a composite `fact_id` formed from:

`session_id + chunk_index`

---

## Usage

### 1. Ingest

    python benchmark_ingest.py

The script prompts you to select a dataset index (`0–8`) and enter a namespace name.

It then:

- Runs `LongMemEvalIngestor` to preprocess the JSON into rows
- Builds a DataFrame
- Uploads the resulting CSV to HyperBinder
- Reports the ingestion mode
- Reports the namespace
- Reports the number of rows added
- Reports the vector source

### 2. Evaluate

    python query.py

The script prompts for the same dataset index and namespace.

For each question, it:

- Loads all questions from the source JSON
- Runs a **weighted dual-slot search**
- Searches the `question` slot with a weight of `0.7`
- Searches the `content_chunk` slot with a weight of `0.3`
- Produces a predicted answer
- Compares the prediction against the ground-truth answer
- Reports per-question results
- Prints a final accuracy summary

The default scoring weights are:

| Slot | Weight |
|------|-------:|
| `question` | `0.7` |
| `content_chunk` | `0.3` |

---

## Evaluation

Predictions are evaluated using exact string matching against the ground-truth answer.

Each evaluated question prints:

    ID   : <question_id> | type=<question_type>

    Q    : <question text>

    GT   : <ground truth answer>

    PRED : <predicted answer> (score=0.XXX)  ✅ / ❌

After all questions have been evaluated, the final accuracy is displayed:

    Accuracy: X/Y = Z%

---

## Configuration

Set the following environment variables or provide them in a `.env` file:

| Variable | Description |
|----------|-------------|
| `SERVER_URL` | HyperBinder server URL |
| `API_KEY` | API key for authentication |

Example:

    SERVER_URL=http://your-server:8000
    API_KEY=your_api_key

---

## Pipeline

    LongMemEval JSON
           │
           ▼
    LongMemEvalIngestor
           │
           ▼
    Preprocessed Rows
           │
           ▼
    HyperBinder Namespace
           │
           ├── question ──────── 0.7
           │
           └── content_chunk ─── 0.3
                      │
                      ▼
               Weighted Search
                      │
                      ▼
               Predicted Answer
                      │
                      ▼
            Ground Truth Comparison
                      │
                      ▼
                  Accuracy

---

## Get Access

Visit [semantic-reach.io](https://semantic-reach.io) for access and API credentials.