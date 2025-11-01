# translator.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langdetect import detect, DetectorFactory
import torch

DetectorFactory.seed = 0

class Translator:
    def __init__(self, model_name="facebook/nllb-200-distilled-600M", device=None):
        # Keep model moderate-sized but multilingual
        self.model_name = model_name
        self.device = torch.device("cuda" if (torch.cuda.is_available() and device is not None and device >= 0) else "cpu")
        # Load once
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)

        # A mapping from common iso639 to NLLB tokens — include fallback to eng_Latn
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

    def _lang_to_bos_id(self, lang_token: str) -> int:
        """
        Prefer tokenizer.lang_code_to_id if available (some seq2seq tokenizers include it),
        else try converting token via tokenizer.convert_tokens_to_ids.
        """
        try:
            mapping = getattr(self.tokenizer, "lang_code_to_id", None)
            if mapping and lang_token in mapping:
                return int(mapping[lang_token])
        except Exception:
            pass
        # fallback: try to find token id of language token string
        try:
            return int(self.tokenizer.convert_tokens_to_ids(lang_token))
        except Exception:
            # last resort: return bos_token_id
            return int(self.tokenizer.bos_token_id) if self.tokenizer.bos_token_id is not None else 0

    def translate_to_english(self, text: str):
        src_code = self.detect_lang_code(text)
        # For NLLB-style tokenizer we usually set forced_bos_token_id to target language token id
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True).to(self.device)
        forced_bos = self._lang_to_bos_id("eng_Latn")
        output = self.model.generate(**inputs, forced_bos_token_id=forced_bos, max_new_tokens=256)
        decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return decoded, src_code

    def translate_from_english(self, text: str, target_lang_code: str) -> str:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True).to(self.device)
        forced_bos = self._lang_to_bos_id(target_lang_code)
        output = self.model.generate(**inputs, forced_bos_token_id=forced_bos, max_new_tokens=256)
        decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return decoded
