import json
from pathlib import Path

path = Path("data/longmemeval_s.json")
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Total entries: {len(data)}")
print(f"First question: {data[0]['question']}")
print(f"First answer: {data[0]['answer']}")
print(f"Num sessions: {len(data[0]['haystack_sessions'])}")