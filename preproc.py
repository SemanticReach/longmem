import json
from typing import Tuple, Dict, Any, List, Optional
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import spacy
import re
from datetime import datetime

@dataclass
class MultisessionTriplet:
    question: str
    answer: str
    session_id: str
    question_date: str
    question_type: str
    supporting_chunks: List[str]
    chunk_dates: List[str]
    chunk_sessions: List[int]
    semantic_scores: Optional[List[float]] = None
    temporal_scores: Optional[List[float]] = None
    total_scores: Optional[List[float]] = None

class LongMemEvalIngestor:
    def __init__(
        self,
        json_path: Path,
        namespace: str = "longmemeval_multisession",
        fragment_limit: int = 20,
        semantic_model: str = "all-MiniLM-L6-v2",
        use_entity_matching: bool = True,
        temporal_decay_days: int = 30,
        cross_session_boost: float = 1.2,
        semantic_weight: float = 0.5,
        temporal_weight: float = 0.3,
        cross_session_weight: float = 0.2,
        debug: bool = False,
    ):
        self.json_path = json_path
        self.namespace = namespace
        self.fragment_limit = fragment_limit
        self.use_entity_matching = use_entity_matching
        self.temporal_decay_days = temporal_decay_days
        self.cross_session_boost = cross_session_boost
        self.semantic_weight = semantic_weight
        self.temporal_weight = temporal_weight
        self.cross_session_weight = cross_session_weight
        self.debug = debug

        # Semantic model
        print("🧠 Loading semantic model...")
        self.encoder = SentenceTransformer(semantic_model)

        # SpaCy for entity extraction
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.spacy_available = True
        except OSError:
            print("⚠️ spaCy model not found. Using regex fallback.")
            self.nlp = None
            self.spacy_available = False

    def parse_date(self, date_str: str) -> datetime:
        for fmt in ("%Y-%m-%d", "%m/%d/%Y"):
            try:
                return datetime.strptime(date_str, fmt)
            except:
                continue
        return datetime.now()

    def temporal_score(self, question_date: str, chunk_date: str) -> float:
        qd = self.parse_date(question_date)
        cd = self.parse_date(chunk_date)
        days_diff = abs((qd - cd).days)
        return float(np.exp(-days_diff / self.temporal_decay_days))

    def extract_entities(self, text: str) -> List[str]:
        entities = []
        if self.spacy_available and self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'ORG', 'GPE', 'DATE', 'PRODUCT', 'EVENT']:
                    entities.append(ent.text.lower())
            for token in doc:
                if token.pos_ == 'PROPN' and len(token.text) > 2:
                    entities.append(token.text.lower())
        else:
            capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
            entities.extend([e.lower() for e in capitalized])
            quoted = re.findall(r'"([^"]*)"', text)
            entities.extend([q.lower() for q in quoted if len(q) > 2])
        return list(set(entities))

    def batch_encode(self, texts: List[str], batch_size: int = 256) -> np.ndarray:
        embeddings = self.encoder.encode(texts, convert_to_numpy=True, show_progress_bar=True, batch_size=batch_size)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embeddings / norms

    def cross_session_bonus(self, session_idx: int, total_sessions: int) -> float:
        if total_sessions <= 1:
            return 1.0
        diversity = session_idx / max(1, total_sessions - 1)
        return 1.0 + (self.cross_session_boost - 1.0) * diversity

    def load_json(self) -> List[Dict[str, Any]]:
        with open(self.json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def flatten_sessions(self, haystack_sessions: List[List[Dict]], haystack_dates: List[str]) -> List[Dict[str, Any]]:
        flattened = []
        for sess_idx, (session, date) in enumerate(zip(haystack_sessions, haystack_dates)):
            for turn_idx, turn in enumerate(session):
                if isinstance(turn, dict) and 'role' in turn and 'content' in turn:
                    flattened.append({
                        'role': turn['role'],
                        'content': turn['content'],
                        'date': date,
                        'session_index': sess_idx,
                        'turn_index': turn_idx,
                        'formatted_turn': f"[{date}] {turn['role']}: {turn['content']}"
                    })
        return flattened

    def extract_dataframes(self, data: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        memory_rows, question_rows = [], []
        for entry in data:
            question_id = entry.get('question_id', 'unknown')
            question = entry.get('question', '')
            answer = entry.get('answer', '')
            question_date = entry.get('question_date', '')
            question_type = entry.get('question_type', 'unknown')
            haystack_sessions = entry.get('haystack_sessions', [])
            haystack_dates = entry.get('haystack_dates', [])
            flattened_turns = self.flatten_sessions(haystack_sessions, haystack_dates)
            for turn in flattened_turns:
                turn.update({
                    'question_id': question_id,
                    'associated_question': question,
                    'associated_answer': answer,
                    'question_date': question_date,
                    'question_type': question_type
                })
                memory_rows.append(turn)
            question_rows.append({
                'question_id': question_id,
                'question': question,
                'answer': answer,
                'question_date': question_date,
                'question_type': question_type,
                'num_sessions': len(haystack_sessions)
            })
        return pd.DataFrame(memory_rows), pd.DataFrame(question_rows)

    def build_triplets(self, memory_df: pd.DataFrame, questions_df: pd.DataFrame) -> List[MultisessionTriplet]:
        # Batch-encode all memory chunks and questions upfront
        print("Encoding memory chunks...")
        chunk_texts = memory_df['content'].tolist()
        chunk_embeddings = self.batch_encode(chunk_texts)

        print("Encoding questions...")
        question_texts = questions_df['question'].tolist()
        question_embeddings = self.batch_encode(question_texts)

        # Build index mapping: question_id -> memory_df row indices
        memory_indices = memory_df.groupby('question_id').indices

        triplets = []
        for q_idx, (_, row) in enumerate(tqdm(questions_df.iterrows(), total=len(questions_df), desc="Building triplets")):
            question_id = row['question_id']
            question_date = row['question_date']
            num_sessions = row['num_sessions']

            mem_idxs = memory_indices.get(question_id, np.array([], dtype=int))
            if len(mem_idxs) == 0:
                continue

            # Vectorized cosine similarity (embeddings already normalized)
            mem_embs = chunk_embeddings[mem_idxs]
            sem_scores = mem_embs @ question_embeddings[q_idx]

            # Compute temporal and cross-session scores
            mem_rows = memory_df.iloc[mem_idxs]
            temp_scores = np.array([self.temporal_score(question_date, d) for d in mem_rows['date']])
            cross_scores = np.array([self.cross_session_bonus(s, num_sessions) for s in mem_rows['session_index']])

            total_scores = (self.semantic_weight * sem_scores +
                            self.temporal_weight * temp_scores +
                            self.cross_session_weight * cross_scores)

            # Top N by total score
            top_k = min(self.fragment_limit, len(total_scores))
            top_idxs = np.argpartition(total_scores, -top_k)[-top_k:]
            top_idxs = top_idxs[np.argsort(total_scores[top_idxs])[::-1]]

            triplets.append(MultisessionTriplet(
                question=row['question'],
                answer=row['answer'],
                session_id=question_id,
                question_date=question_date,
                question_type=row['question_type'],
                supporting_chunks=[chunk_texts[mem_idxs[i]] for i in top_idxs],
                chunk_dates=[mem_rows.iloc[i]['date'] for i in top_idxs],
                chunk_sessions=[int(mem_rows.iloc[i]['session_index']) for i in top_idxs],
                semantic_scores=[float(sem_scores[i]) for i in top_idxs],
                temporal_scores=[float(temp_scores[i]) for i in top_idxs],
                total_scores=[float(total_scores[i]) for i in top_idxs],
            ))
        return triplets

    def prepare_rust_rows(self, triplets: List[MultisessionTriplet]) -> List[Tuple[int, Dict[str, str]]]:
        rows = []
        for i, trip in enumerate(triplets):
            for idx, chunk in enumerate(trip.supporting_chunks):
                row = {
                    "question": trip.question,
                    "answer": trip.answer,
                    "content_chunk": chunk,
                    "chunk_index": str(idx),
                    "session_id": trip.session_id,
                    "question_date": trip.question_date,
                    "question_type": trip.question_type,
                    "chunk_date": trip.chunk_dates[idx],
                    "chunk_session_index": trip.chunk_sessions[idx],
                    "semantic_score": str(trip.semantic_scores[idx]),
                    "temporal_score": str(trip.temporal_scores[idx]),
                    "total_score": str(trip.total_scores[idx]),
                }
                rows.append((i, row))
        return rows

    def run_ingestion(self) -> List[Tuple[int, Dict[str, str]]]:
        print("Loading JSON data...")
        data = self.load_json()
        print("Flattening sessions...")
        memory_df, questions_df = self.extract_dataframes(data)
        print(f"Memory turns: {len(memory_df)}, Questions: {len(questions_df)}")
        
        triplets = self.build_triplets(memory_df, questions_df)
        print(f"Built {len(triplets)} multisession triplets")
        
        rust_rows = self.prepare_rust_rows(triplets)

        # Ensure PyO3-compatible formatting: (int, dict[str,str])
        rust_rows_clean = []
        for i, row in rust_rows:
            clean_row = {k: str(v) for k, v in row.items()}  # force all values to strings
            rust_rows_clean.append((i, clean_row))

        print(f"Prepared {len(rust_rows_clean)} Rust-ready rows")
        return rust_rows_clean

