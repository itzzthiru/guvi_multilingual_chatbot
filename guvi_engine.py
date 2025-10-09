from sentence_transformers import SentenceTransformer, util
import re

class GuviEngine:
    def __init__(self, file_path="guvi.txt", top_k=3):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.top_k = top_k

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Break into chunks (not just sentences)
        self.chunks = re.split(r'\n\s*\n', text)  # Split by paragraphs
        self.chunks = [chunk.strip().replace('\n', ' ') for chunk in self.chunks if len(chunk.strip()) > 40]

        self.embeddings = self.model.encode(self.chunks, convert_to_tensor=True)

    def get_best_match(self, query, threshold=0.4):
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(query_embedding, self.embeddings)[0]

        best_idx = scores.argmax().item()
        best_score = scores[best_idx].item()

        print("Best Score:", best_score)

        if best_score >= threshold:
            return self.chunks[best_idx]
        return None

# Test block
if __name__ == "__main__":
    engine = GuviEngine()
    query = "How can GUVI help me get a job?"
    print("Q:", query)
    print("A:", engine.get_best_match(query) or "No match found.")