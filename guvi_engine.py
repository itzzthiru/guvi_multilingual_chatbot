# guvi_engine.py
from sentence_transformers import SentenceTransformer, util
import re
import os

class GuviEngine:
    def __init__(self, file_path="guvi.txt", model_name="all-MiniLM-L6-v2", top_k=3, device=None):
        self.model = SentenceTransformer(model_name)
        self.top_k = top_k
        if not os.path.exists(file_path):
            self.chunks = []
            self.embeddings = None
            return

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Split into paragraphs by blank lines and clean short noise
        self.chunks = re.split(r'\n\s*\n', text)
        self.chunks = [c.strip().replace('\n', ' ') for c in self.chunks if len(c.strip()) > 40]

        # Precompute embeddings (tensor)
        self.embeddings = self.model.encode(self.chunks, convert_to_tensor=True)

    def get_top_k_matches(self, query, top_k=None, threshold=0.35):
        if not self.chunks or self.embeddings is None:
            return []

        k = top_k if top_k is not None else self.top_k
        query_emb = self.model.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(query_emb, self.embeddings)[0]  # tensor
        # topk
        topk = min(k, len(self.chunks))
        values, indices = scores.topk(topk)
        results = []
        for val, idx in zip(values, indices):
            score = float(val)
            if score >= threshold:
                results.append((self.chunks[int(idx)], score))
        return results
