# faq_engine.py
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import os

class FAQEngine:
    def __init__(self, faq_path="guvi_faq.json", model_name="sentence-transformers/all-MiniLM-L6-v2", device=None):
        # device: None -> CPU; integer 0+ maps to GPU device index
        self.device = torch.device("cuda" if (torch.cuda.is_available() and device is not None and device >= 0) else "cpu")

        # sentence embedding model from huggingface (AutoModel + AutoTokenizer)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

        # Load FAQs
        if not os.path.exists(faq_path):
            self.faq_data = []
            self.questions = []
            self.answers = []
            self.question_embeddings = torch.empty((0, 384))
            return

        with open(faq_path, "r", encoding="utf-8") as f:
            self.faq_data = json.load(f)

        self.questions = [item["question"] for item in self.faq_data]
        self.answers = [item["answer"] for item in self.faq_data]

        # Precompute embeddings
        self.question_embeddings = self._encode(self.questions)  # CPU tensor

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(token_embeddings.dtype)
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def _encode(self, texts):
        # tokenizes and computes embeddings; returns CPU tensor to avoid GPU memory pressure
        if len(texts) == 0:
            return torch.empty((0, 384))
        encoded = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            output = self.model(**encoded)
        emb = self._mean_pooling(output, encoded["attention_mask"])
        return emb.cpu()

    def get_top_k_answers(self, query, top_k=3, threshold=0.4):
        """
        Returns list of (answer_text, score) sorted by score desc.
        If score < threshold they are filtered out.
        """
        if len(self.questions) == 0:
            return []

        q_emb = self._encode([query])[0]  # CPU
        sims = F.cosine_similarity(q_emb.unsqueeze(0), self.question_embeddings)  # vector of scores
        # convert to python floats and sort
        scores = sims.tolist()
        idx_scores = sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in idx_scores[:top_k]:
            if score >= threshold:
                results.append((self.answers[idx], float(score)))
        return results
