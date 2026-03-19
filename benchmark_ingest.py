import os
import json
import tempfile
import pandas as pd
import requests
from pathlib import Path
from preproc import LongMemEvalIngestor
from dotenv import load_dotenv

# ── Config ────────────────────────────────────────────────────────────────────
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


load_dotenv()

SERVER_URL = os.getenv("SERVER_URL")
API_KEY    = os.getenv("API_KEY")

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

TEMPLATE_SCHEMA = json.dumps({
    "molecule": "Row",
    "semantic_fields": ["question", "content_chunk"],
    "primary_key": "fact_id",
    "fields": {
        "question":      {"encoding": "semantic"},
        "answer":        {"encoding": "exact"},
        "content_chunk": {"encoding": "semantic"},
        "session_id":    {"encoding": "exact"},
        "question_type": {"encoding": "exact"},
        "question_date": {"encoding": "temporal"},
    },
    "field_order": ["question", "answer", "content_chunk", "session_id", "question_type", "question_date"]
})

# ── Core ──────────────────────────────────────────────────────────────────────
def ingest_dataset(file_index: int, namespace: str, data_dir: Path) -> dict:
    filename = DATASETS[file_index]
    json_path = data_dir / filename

    if not json_path.exists():
        raise FileNotFoundError(f"❌ File not found: {json_path}")

    print(f"\n⚙️  Processing: {filename}")

    # 1. Run ingestor → rows
    ingestor = LongMemEvalIngestor(json_path=json_path)
    rust_rows = ingestor.run_ingestion()

    # 2. Build DataFrame
    df = pd.DataFrame([row[1] for row in rust_rows])
    df["fact_id"] = df["session_id"].astype(str) + "_" + df["chunk_index"].astype(str)

    # Keep only schema columns
    df = df[["fact_id", "question", "answer", "content_chunk",
             "session_id", "question_type", "question_date"]]

    # 3. Serialize to temp CSV and upload
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8"
    ) as tmp:
        df.to_csv(tmp, index=False)
        tmp_path = tmp.name

    print(f"📤 Uploading {len(df)} rows → namespace: '{namespace}' ...")

    try:
        with open(tmp_path, "rb") as f:
            response = requests.post(
                f"{SERVER_URL}/build_ingest_data/",
                headers={"X-API-Key": API_KEY},
                files={"file": (Path(tmp_path).name, f, "text/csv")},
                data={
                    "dim":             384,
                    "seed":            42,
                    "depth":           3,
                    "db_name":         "long_mem_eval_db",
                    "namespace":       namespace,
                    "template_schema": TEMPLATE_SCHEMA,
                },
            )
        response.raise_for_status()
    finally:
        os.unlink(tmp_path)

    result = response.json()
    print(f"✅ Success!")
    print(f"   mode       : {result['mode']}")
    print(f"   namespace  : {result['namespace']}")
    print(f"   rows_added : {result['rows_added']}")
    print(f"   vector_src : {result['vector_source']}")

    return result


def main():
    data_dir = (Path(__file__).parent / "data").resolve()

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

    file_idx   = int(idx)
    parts      = DATASETS[file_idx].replace(".json", "").split("_")
    default_ns = parts[3] if len(parts) > 3 else parts[0]
    namespace  = input(f"Enter namespace [default: {default_ns}]: ").strip() or default_ns

    ingest_dataset(file_idx, namespace, data_dir)


if __name__ == "__main__":
    main()