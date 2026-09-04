"""
ingest_client.py — LongMemEval HyperBinder Ingest
===================================================

Ingests LongMemEval-style haystack sessions as Row-schema facts
(question / answer / content_chunk / session_id / question_type / question_date).

Workflow:
    # Interactive (prompts for dataset, namespace, db_name, wipe)
    python ingest_client.py

    # Non-interactive, e.g. from a script or CI
    python ingest_client.py --file 8 --namespace longmemeval_s --wipe
"""

from __future__ import annotations

import argparse
import io
import json
import os
import time
from pathlib import Path

import pandas as pd
import requests
from preproc import LongMemEvalIngestor
from dotenv import load_dotenv

# ── Config ────────────────────────────────────────────────────────────────────
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

load_dotenv()

SERVER_URL = os.getenv("SERVER_URL")
API_KEY    = os.getenv("API_KEY")

# NOTE on db_name / dim: the server caches open engine handles keyed by
# db_name independently of on-disk/metadata state. DELETE /db/{db_name}
# clears metadata and disk files but NOT that cached handle, so re-ingesting
# into a db_name that was ever opened at a different `dim` in this server
# process's lifetime can silently reuse the stale handle instead of honoring
# the `dim` you pass. Until that's fixed server-side, prefer a db_name this
# process hasn't used before whenever you change `dim`. (Confirmed pattern:
# the working CUAD ingest script uses "fraud_db_v2" for the same reason.)
DB_NAME = os.environ.get("HB_DB_NAME", "long_mem_eval_db_v2")
DIM = int(os.environ.get("HB_DIM", 2048))  # capacity = dim/2; must cover phase_dim (default 512)

BATCH_SIZE = 500
TIMEOUT = 300
MAX_RETRIES = 3

DATASETS = {
    0: "202410_custom_haystack1_implicit_preference_v2_8-12haysess_user0.5sharegpt0.25ultrachat0.25.json",
    1: "202410_custom_haystack1_knowledge_update_8-12haysess_user0.5sharegpt0.25ultrachat0.25.json",
    2: "202410_custom_haystack1_knowledge_update_8-12haysess_user0.5sharegpt0.25ultrachat0.25_100.json",
    3: "202410_custom_haystack1_multi_session_synthesis_8-12haysess_user0.5sharegpt0.25ultrachat0.25.json",
    4: "202410_custom_haystack1_single_hop_8-12haysess_user0.5sharegpt0.25ultrachat0.25.json",
    5: "202410_custom_haystack1_temp_reasoning_explicit_8-12haysess_user0.5sharegpt0.25ultrachat0.25.json",
    6: "202410_custom_haystack1_temp_reasoning_implicit_8-12haysess_user0.5sharegpt0.25ultrachat0.25.json",
    7: "202410_custom_haystack1_two_hop_8-12haysess_user0.5sharegpt0.25ultrachat0.25.json",
    8: "longmemeval_s.json",
}

# ✅ Row schema, verified against the engine's actual Schema::from_json
# (encoding_gateway.rs): a Row schema only requires
#   - "primary_key": {"name": ..., "encoding": "exact"}
#   - "fields": a JSON object of slot_name -> {"name", "encoding", ...}
#   - "field_order": a JSON array of slot names
# There is NO rule against the primary key's name also appearing inside
# "fields"/"field_order" (parse_one_slot / from_json never cross-check
# against primary_key) -- the CUAD ingest script does exactly that and
# it works. Kept consistent with that pattern here.
TEMPLATE_SCHEMA = json.dumps({
    "molecule": "Row",
    "primary_key": {"name": "fact_id", "encoding": "exact"},
    "fields": {
        "fact_id": {"name": "fact_id", "encoding": "exact"},
        "question": {"name": "question", "encoding": "semantic"},
        "answer": {"name": "answer", "encoding": "exact"},
        "content_chunk": {"name": "content_chunk", "encoding": "semantic"},
        "session_id": {"name": "session_id", "encoding": "exact"},
        "question_type": {"name": "question_type", "encoding": "exact"},
        "question_date": {"name": "question_date", "encoding": "temporal"},
    },
    "field_order": [
        "fact_id", "question", "answer", "content_chunk",
        "session_id", "question_type", "question_date"
    ]
})


# ── Database management ──────────────────────────────────────────────────────

def delete_database(db_name: str) -> None:
    """Delete an entire database."""
    print(f"  Deleting database '{db_name}'...")
    resp = requests.delete(
        f"{SERVER_URL}/db/{db_name}",
        headers={"X-API-Key": API_KEY},
        timeout=30,
    )
    if resp.status_code in (200, 404):
        print(f"  ✓ Deleted (status {resp.status_code})")
    else:
        print(f"  ⚠️  {resp.status_code}: {resp.text[:200]}")


def get_namespace_count(db_name: str, namespace: str) -> int:
    try:
        resp = requests.get(
            f"{SERVER_URL}/namespace/{db_name}/{namespace}/count",
            headers={"X-API-Key": API_KEY},
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json().get("count", 0)
    except Exception:
        pass
    return -1


# ── Feature building ──────────────────────────────────────────────────────────

def build_rows(json_path: Path) -> list[dict]:
    """Run the ingestor and shape rows to exactly the schema's fields."""
    ingestor = LongMemEvalIngestor(json_path=json_path)
    rust_rows = ingestor.run_ingestion()

    df = pd.DataFrame([row[1] for row in rust_rows])
    df["fact_id"] = df["session_id"].astype(str) + "_" + df["chunk_index"].astype(str)
    df = df[["fact_id", "question", "answer", "content_chunk",
             "session_id", "question_type", "question_date"]]

    return df.to_dict(orient="records")


# ── Batched ingest ────────────────────────────────────────────────────────────

def ingest_batch(
    rows: list[dict],
    batch_num: int,
    db_name: str,
    namespace: str,
) -> int:
    if not rows:
        return 0

    df = pd.DataFrame(rows)
    buf = io.BytesIO()
    df.to_csv(buf, index=False)

    for attempt in range(1, MAX_RETRIES + 1):
        buf.seek(0)
        try:
            resp = requests.post(
                f"{SERVER_URL}/build_ingest_data/",
                headers={"X-API-Key": API_KEY},
                files={"file": (f"batch_{batch_num:04d}.csv", buf, "text/csv")},
                data={
                    "dim": DIM,
                    "seed": 42,
                    "depth": 3,
                    "db_name": db_name,
                    "namespace": namespace,
                    "template_schema": TEMPLATE_SCHEMA,
                    "on_conflict": "error",
                },
                timeout=TIMEOUT,
            )

            if resp.status_code == 200:
                result = resp.json()
                rows_added = result.get("rows_added", len(rows))
                if batch_num == 1:
                    print(f"  ✓ vector_source = {result.get('vector_source', 'unknown')}")
                return rows_added

            # Non-200: surface the real server error, don't just retry blind.
            print(f"  ⚠️  Batch {batch_num} attempt {attempt} — {resp.status_code}:")
            try:
                print(f"     {resp.json()}")
            except ValueError:
                print(f"     {resp.text[:300]}")

        except requests.exceptions.ReadTimeout:
            print(f"  ⏱️  Batch {batch_num} attempt {attempt} timed out")
        except requests.exceptions.ConnectionError as e:
            print(f"  ✗  Connection error: {e}")

        if attempt < MAX_RETRIES:
            wait = 10 * attempt
            print(f"     Retrying in {wait}s...")
            time.sleep(wait)

    return 0


# ── Main ingest ───────────────────────────────────────────────────────────────

def ingest_dataset(
    file_index: int,
    namespace: str,
    data_dir: Path,
    wipe_db: bool = False,
    db_name: str = DB_NAME,
    batch_size: int = BATCH_SIZE,
) -> None:
    filename = DATASETS[file_index]
    json_path = data_dir / filename

    if not json_path.exists():
        raise FileNotFoundError(f"❌ File not found: {json_path}")

    print("\n" + "=" * 65)
    print("  ingest_client.py — LongMemEval Ingest")
    print(f"  Server    : {SERVER_URL}")
    print(f"  DB        : {db_name} / {namespace}")
    print(f"  File      : {filename}")
    print(f"  Dim       : {DIM}")
    print(f"  Wipe      : {wipe_db}")
    print("=" * 65)

    if wipe_db:
        delete_database(db_name)
        print(f"  ✅ Database '{db_name}' deleted. Will be recreated on ingest.")

    print(f"\n  Building rows from {filename}...")
    rows = build_rows(json_path)
    print(f"  ✓ {len(rows):,} rows to ingest")

    if not rows:
        print("  ✗ No rows built")
        return

    total_added = 0
    total_batches = (len(rows) + batch_size - 1) // batch_size
    t0 = time.time()

    print(f"\n  Ingesting {len(rows):,} rows in {total_batches} batch(es)...\n")

    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        batch_num = i // batch_size + 1

        added = ingest_batch(batch, batch_num, db_name, namespace)
        total_added += added

        pct = batch_num / total_batches * 100
        elapsed = time.time() - t0
        rate = total_added / elapsed if elapsed > 0 else 0
        remaining = (len(rows) - total_added) / rate if rate > 0 else 0

        print(
            f"  Batch {batch_num:3d}/{total_batches}  "
            f"[{pct:5.1f}%]  "
            f"+{added:4d} rows  "
            f"{rate:5.0f}/s  "
            f"ETA {remaining / 60:.1f}m"
        )

    elapsed = time.time() - t0
    final_count = get_namespace_count(db_name, namespace)

    print(f"\n{'=' * 65}")
    print(f"  ✓ INGEST COMPLETE")
    print(f"  Rows ingested   : {total_added:,}")
    print(f"  Total time      : {elapsed:.1f}s")
    if final_count >= 0:
        print(f"  Namespace count : {final_count:,}")
    print(f"{'=' * 65}\n")


# ── CLI / interactive entry point ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Ingest LongMemEval haystack data into HyperBinder"
    )
    parser.add_argument("--file", type=int, choices=list(DATASETS.keys()), default=None,
                        help="Dataset index (0-8). Omit for interactive prompt.")
    parser.add_argument("--namespace", default=None,
                        help="Target namespace. Omit for interactive prompt.")
    parser.add_argument("--db", default=DB_NAME, help=f"Database name (default: {DB_NAME})")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--wipe", "--new-db", dest="wipe", action="store_true",
                        help="Delete and recreate the database before ingesting")
    args = parser.parse_args()

    data_dir = (Path(__file__).parent / "data").resolve()

    if args.file is not None:
        file_idx = args.file
    else:
        print("\n" + "=" * 70)
        print("      HYPERBINDER API INGESTOR")
        print("=" * 70)
        for i, filename in DATASETS.items():
            print(f" [{i}] {filename}")
        print("=" * 70)
        idx = input("\nSelect file index (0-8): ").strip()
        if not idx.isdigit() or int(idx) not in DATASETS:
            print("❌ Invalid index.")
            return
        file_idx = int(idx)

    if args.namespace:
        namespace = args.namespace
    else:
        parts = DATASETS[file_idx].replace(".json", "").split("_")
        default_ns = parts[3] if len(parts) > 3 else parts[0]
        namespace = input(f"Enter namespace [default: {default_ns}]: ").strip() or default_ns

    db_name = args.db
    wipe = args.wipe
    if args.file is None:  # only prompt for wipe in interactive mode
        wipe = wipe or (input("Delete and recreate database? (y/N): ").strip().lower() == 'y')

    ingest_dataset(
        file_idx, namespace, data_dir,
        wipe_db=wipe, db_name=db_name, batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()