from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0

class Translator:
    def __init__(self):
        self.model_name = "facebook/nllb-200-distilled-600M"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

        # ISO639 → NLLB token map (subset incl. Indian langs)
        self.lang_code_map = {
            'af':'afr_Latn','ar':'arb_Arab','bn':'ben_Beng','en':'eng_Latn','es':'spa_Latn',
            'fr':'fra_Latn','gu':'guj_Gujr','hi':'hin_Deva','id':'ind_Latn','it':'ita_Latn',
            'ja':'jpn_Jpan','kn':'kan_Knda','ko':'kor_Hang','ml':'mal_Mlym','mr':'mar_Deva',
            'ne':'npi_Deva','nl':'nld_Latn','pa':'pan_Guru','pt':'por_Latn','ru':'rus_Cyrl',
            'ta':'tam_Taml','te':'tel_Telu','tr':'tur_Latn','ur':'urd_Arab',
            'zh':'zho_Hans','zh-cn':'zho_Hans'
        }

    def detect_lang_code(self, text: str) -> str:
        try:
            iso = detect(text).lower()
            return self.lang_code_map.get(iso, "eng_Latn")
        except Exception:
            return "eng_Latn"

    def lang_token_id(self, code: str) -> int:
        return self.tokenizer.convert_tokens_to_ids(code)

    def translate_to_english(self, text: str):
        src = self.detect_lang_code(text)
        self.tokenizer.src_lang = src
        inputs = self.tokenizer(text, return_tensors="pt")
        output = self.model.generate(**inputs, forced_bos_token_id=self.lang_token_id("eng_Latn"))
        return self.tokenizer.decode(output[0], skip_special_tokens=True), src

    def translate_from_english(self, text: str, target_lang_code: str) -> str:
        self.tokenizer.src_lang = "eng_Latn"
        inputs = self.tokenizer(text, return_tensors="pt")
        output = self.model.generate(**inputs, forced_bos_token_id=self.lang_token_id(target_lang_code))
        return self.tokenizer.decode(output[0], skip_special_tokens=True)