from translator import Translator
from faq_engine import FAQEngine
from guvi_engine import GuviEngine

class Chatbot:
    def __init__(self):
        self.translator = Translator()
        self.faq_engine = FAQEngine()
        self.guvi_engine = GuviEngine()

    def get_response(self, user_input: str, top_k: int = 3):
        """
        Returns a dictionary with FAQ answers and GUVI text results.
        """
        # Step 1: Translate input to English
        translated_input, lang_code = self.translator.translate_to_english(user_input)

        # Step 2: Search FAQ - fixed method call
        faq_answer = self.faq_engine.get_answer(translated_input)

        # Step 3: Search GUVI text - fixed method call
        guvi_match = self.guvi_engine.get_best_match(translated_input)

        # Step 4: Translate answers back to user language
        final_faq = []
        if faq_answer:
            translated_back = self.translator.translate_from_english(faq_answer, lang_code)
            final_faq.append((translated_back, 1.0, "FAQ"))

        final_guvi = []
        if guvi_match:
            translated_back = self.translator.translate_from_english(guvi_match, lang_code)
            final_guvi.append((translated_back, 1.0))

        return {
            "detected_lang_code": lang_code,
            "translated_input": translated_input,
            "faq_answers": final_faq,
            "guvi_paragraphs": final_guvi
        }