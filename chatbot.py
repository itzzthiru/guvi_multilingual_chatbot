# chatbot.py
import torch
from translator import Translator
from faq_engine import FAQEngine
from guvi_engine import GuviEngine
from transformers import pipeline

class Chatbot:
    def __init__(self, device=None):
        # device selection
        self.device = device if device is not None else (0 if torch.cuda.is_available() else -1)

        # Initialize components
        self.translator = Translator(device=self.device if self.device >= 0 else None)
        self.faq_engine = FAQEngine(device=self.device if self.device >= 0 else None)
        self.guvi_engine = GuviEngine(device=self.device if self.device >= 0 else None)

        # Lightweight generative fallback (use small model to avoid OOM)
        # Use device index when available; pipeline expects device int (-1 CPU, 0 GPU)
        try:
            self.generator = pipeline(
                "text-generation",
                model="distilgpt2",
                tokenizer="distilgpt2",
                device=self.device if self.device >= 0 else -1,
                max_length=150,
                do_sample=True,
                top_p=0.95,
                temperature=0.75,
            )
        except Exception:
            # pipeline fallback: None (still fine)
            self.generator = None

    def get_response(self, user_input: str, top_k: int = 3, faq_threshold: float = 0.45, guvi_threshold: float = 0.40):
        """
        Returns structured dict:
          detected_lang_code, translated_input,
          faq_answers: list[(text, score, "FAQ")],
          guvi_paragraphs: list[(text, score)],
          generative_answers: list[(text, score)]
        """
        # 1. Translate to English (detect language too)
        translated_input, lang_code = self.translator.translate_to_english(user_input)

        # 2. FAQ retrieval (top_k)
        faq_results = self.faq_engine.get_top_k_answers(translated_input, top_k=top_k, threshold=faq_threshold)
        # faq_results -> list of (answer_text, score)

        # 3. GUVI retrieval (top_k)
        guvi_results = self.guvi_engine.get_top_k_matches(translated_input, top_k=top_k, threshold=guvi_threshold)
        # guvi_results -> list of (paragraph_text, score)

        # 4. Generative fallback (only if low confidence)
        generative_results = []
        # choose fallback when both top scores are low or no hits
        best_faq_score = faq_results[0][1] if faq_results else 0.0
        best_guvi_score = guvi_results[0][1] if guvi_results else 0.0
        if max(best_faq_score, best_guvi_score) < 0.5 and self.generator is not None:
            # get one generative reply in English and then translate back later
            gen_out = self.generator(translated_input, max_length=120, num_return_sequences=1)
            gen_text = gen_out[0]["generated_text"] if gen_out else ""
            generative_results.append((gen_text.strip(), 0.0))  # score unknown (set 0.0 placeholder)

        # 5. Translate results back to user's language
        def translate_back_list(items):
            out = []
            for text, score in items:
                try:
                    tb = self.translator.translate_from_english(text, lang_code)
                except Exception:
                    tb = text  # fallback to english if translation fails
                out.append((tb, score))
            return out

        faq_translated = [(self.translator.translate_from_english(a, lang_code), s, "FAQ") for a, s in faq_results]
        guvi_translated = [(self.translator.translate_from_english(a, lang_code), s) for a, s in guvi_results]
        generative_translated = [(self.translator.translate_from_english(a, lang_code), s) for a, s in generative_results]

        return {
            "detected_lang_code": lang_code,
            "translated_input": translated_input,
            "faq_answers": faq_translated,
            "guvi_paragraphs": guvi_translated,
            "generative_answers": generative_translated,
        }
