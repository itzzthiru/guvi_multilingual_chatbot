import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

class FAQEngine:
    def __init__(self):
        # CPU by default in free Spaces; use CUDA if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Sentence embeddings model (small, fast)
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

        # Load FAQs
        with open("guvi_faq.json", "r", encoding="utf-8") as f:
            self.faq_data = json.load(f)
        self.questions = [item["question"] for item in self.faq_data]
        self.answers = [item["answer"] for item in self.faq_data]

        # Precompute question embeddings once
        self.question_embeddings = self._encode(self.questions)

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def _encode(self, texts):
        encoded = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            output = self.model(**encoded)
        # store on CPU to avoid GPU memory pressure
        return self._mean_pooling(output, encoded["attention_mask"]).cpu()

    def get_answer(self, user_question: str) -> str:  # Removed top_k parameter
        query_emb = self._encode([user_question])[0]
        sims = F.cosine_similarity(query_emb.unsqueeze(0), self.question_embeddings)
        best_idx = torch.argmax(sims).item()
        return self.answers[best_idx]