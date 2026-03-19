import json
import requests
from pathlib import Path
from dotenv import load_dotenv
import os


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

def load_questions(json_path: Path) -> list:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [
        {
            "question_id":   entry["question_id"],
            "question":      entry["question"],
            "answer":        entry["answer"],
            "question_type": entry["question_type"],
            "question_date": entry["question_date"],
        }
        for entry in data
    ]

def query_longmem(question: str, namespace: str, top_k: int = 1) -> dict:
    response = requests.post(
        f"{SERVER_URL}/compose/search_slots/long_mem_eval_db/{namespace}",
        headers={"X-API-Key": API_KEY},
        json={
            "slot_queries": {
                "question":      {"query": question, "weight": 0.7, "encoding": "semantic"},
                "content_chunk": {"query": question, "weight": 0.3, "encoding": "semantic"},
            },
            "top_k": top_k,
        },
    )
    response.raise_for_status()
    return response.json()

def main():
    data_dir = (Path(__file__).parent / "data").resolve()

    print("\n" + "=" * 70)
    print("      HYPERBINDER QUERY EVALUATOR")
    print("=" * 70)
    for i, filename in DATASETS.items():
        print(f" [{i}] {filename}")
    print("=" * 70)

    idx = input("\nSelect file index (0-8): ").strip()
    if not idx.isdigit() or int(idx) not in DATASETS:
        print("❌ Invalid index.")
        return

    file_idx   = int(idx)
    filename   = DATASETS[file_idx]
    parts      = filename.replace(".json", "").split("_")
    default_ns = parts[3] if len(parts) > 3 else parts[0]
    namespace  = input(f"Enter namespace [default: {default_ns}]: ").strip() or default_ns

    json_path = data_dir / filename
    questions = load_questions(json_path)
    print(f"\n📋 Loaded {len(questions)} questions from {filename}\n")

    correct = 0
    for q in questions:
        result = query_longmem(q["question"], namespace)
        top    = result["results"][0] if result["results"] else None

        predicted = top["data"].get("answer") if top else "N/A"
        score     = top["_score"]             if top else 0.0
        match = predicted.strip().lower() == str(q["answer"]).strip().lower()
        if match:
            correct += 1

        print(f"ID   : {q['question_id']} | type={q['question_type']}")
        print(f"Q    : {q['question']}")
        print(f"GT   : {q['answer']}")
        print(f"PRED : {predicted}  (score={score:.3f})  {'✅' if match else '❌'}")
        print()

    print("=" * 70)
    print(f"Accuracy: {correct}/{len(questions)} = {correct/len(questions):.1%}")
    print("=" * 70)

if __name__ == "__main__":
    main()