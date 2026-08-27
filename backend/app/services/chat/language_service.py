"""
Multilingual Language Detection & Selection Service
Supports 20+ Indian Languages, Script-Level Detection, and Romanized Tanglish Heuristics
"""

import re
import unicodedata
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class LanguageDefinition:
    """Metadata representing a supported language."""
    code: str
    name: str
    native_name: str
    script: str
    unicode_ranges: List[Tuple[int, int]] = field(default_factory=list)
    vocabulary_markers: List[str] = field(default_factory=list)


# ----------------------------------------------------------------------
# 1. 20+ Supported Languages Registry
# ----------------------------------------------------------------------

SUPPORTED_LANGUAGES: Dict[str, LanguageDefinition] = {
    # English
    "en": LanguageDefinition(
        code="en",
        name="English",
        native_name="English",
        script="Latin",
        unicode_ranges=[(0x0020, 0x007F)],
        vocabulary_markers=["cow", "milk", "feed", "silage", "cattle", "ration", "yield", "fodder", "nutrition"]
    ),
    # Dravidian Languages
    "ta": LanguageDefinition(
        code="ta",
        name="Tamil",
        native_name="தமிழ்",
        script="Tamil",
        unicode_ranges=[(0x0B80, 0x0BFF)],
        vocabulary_markers=[
            "மாடு", "பால்", "தீவனம்", "சைலேஜ்", "பசு", "கறவை", "உணவு", "என்ன", "எடை", "கொடுக்க", "லிட்டர்",
            "maadu", "paal", "theevanam", "theevana", "silage", "pasu", "karavai", "enna", "kudukuthu", "kudukanum",
            "epdi", "nalla", "iruka", "saapadu", "solunga", "romba", "illai", "venum", "kudunga"
        ]
    ),
    "te": LanguageDefinition(
        code="te",
        name="Telugu",
        native_name="తెలుగు",
        script="Telugu",
        unicode_ranges=[(0x0C00, 0x0C7F)],
        vocabulary_markers=[
            "ఆవు", "పాలు", "మేత", "సైలేజ్", "దాణా", "గేదె", "ఎంత", "ఏమిటి", "ఇవ్వాలి",
            "aavu", "paalu", "meetha", "daana", "ela", "ivvali", "entha", "kavali", "cheppandi", "gedhe"
        ]
    ),
    "kn": LanguageDefinition(
        code="kn",
        name="Kannada",
        native_name="ಕನ್ನಡ",
        script="Kannada",
        unicode_ranges=[(0x0C80, 0x0CFF)],
        vocabulary_markers=[
            "ಹಸು", "ಹಾಲು", "ಮೇವು", "ಸೈಲೇಜ್", "ಆಹಾರ", "ಎಷ್ಟು", "ಕೊಡಬೇಕು", "ಹೇಗೆ",
            "hasu", "haalu", "mevu", "aahara", "yeshtu", "kodabeku", "hege", "kodi"
        ]
    ),
    "ml": LanguageDefinition(
        code="ml",
        name="Malayalam",
        native_name="മലയാളം",
        script="Malayalam",
        unicode_ranges=[(0x0D00, 0x0D7F)],
        vocabulary_markers=[
            "പശു", "പാൽ", "തീറ്റ", "സൈലേജ്", "എത്ര", "എന്താണ്", "കൊടുക്കേണ്ടത്", "കാലിത്തീറ്റ",
            "pashu", "paal", "theetta", "enthanu", "kodukkendathu", "ethra", "venam", "kalitheetta"
        ]
    ),
    # Indo-Aryan Languages (Devanagari & variants)
    "hi": LanguageDefinition(
        code="hi",
        name="Hindi",
        native_name="हिन्दी",
        script="Devanagari",
        unicode_ranges=[(0x0900, 0x097F)],
        vocabulary_markers=[
            "मेरी", "मेरा", "गाय", "कितना", "कितनी", "खाए", "खिलाएं", "क्या", "कैसे", "बढ़ाएं", "चाहिए", "है", "होता", "देना",
            "gaay", "gai", "doodh", "chara", "pashu", "kitna", "khaye", "khilaye", "kya", "kaise", "badhaye", "batao", "chahiye"
        ]
    ),
    "mr": LanguageDefinition(
        code="mr",
        name="Marathi",
        native_name="मराठी",
        script="Devanagari",
        unicode_ranges=[(0x0900, 0x097F)],
        vocabulary_markers=["आहे", "कसे", "गाईला", "द्यावे", "पाहिजे", "वाढवण्यासाठी", "गायीचे", "किती", "द्यावा", "द्यावी", "ळ", "करावे"]
    ),
    "sa": LanguageDefinition(
        code="sa",
        name="Sanskrit",
        native_name="संस्कृतम्",
        script="Devanagari",
        unicode_ranges=[(0x0900, 0x097F)],
        vocabulary_markers=["धेनु", "क्षीरम्", "अस्ति", "भोजनम्", "किम्", "दातव्यम्", "कथम्", "गोपाल", "पशुपालनम्", "भवति"]
    ),
    "ne": LanguageDefinition(
        code="ne",
        name="Nepali",
        native_name="नेपाली",
        script="Devanagari",
        unicode_ranges=[(0x0900, 0x097F)],
        vocabulary_markers=["गाईलाई", "घाँस", "कति", "दिने", "हुने", "छ", "गर्नु"]
    ),
    "kok": LanguageDefinition(
        code="kok",
        name="Konkani",
        native_name="कोंकणी",
        script="Devanagari",
        unicode_ranges=[(0x0900, 0x097F)],
        vocabulary_markers=["गायक", "कितें", "खावड", "दिवचें", "आसा", "गोरवां"]
    ),
    "mai": LanguageDefinition(
        code="mai",
        name="Maithili",
        native_name="मैथिली",
        script="Devanagari",
        unicode_ranges=[(0x0900, 0x097F)],
        vocabulary_markers=["अछि", "दियौक", "कतेक", "खिलाबी", "अहाँ"]
    ),

    # Eastern & Western Indic
    "bn": LanguageDefinition(
        code="bn",
        name="Bengali",
        native_name="বাংলা",
        script="Bengali",
        unicode_ranges=[(0x0980, 0x09FF)],
        vocabulary_markers=["গরু", "দুধ", "ঘাস", "খাবার", "সাইলেজ", "কত", "কি", "খাওয়ানো", "পশু"]
    ),
    "as": LanguageDefinition(
        code="as",
        name="Assamese",
        native_name="অসমীয়া",
        script="Bengali",
        unicode_ranges=[(0x0980, 0x09FF)],
        vocabulary_markers=["গাভী", "গাখীৰ", "ঘাঁহ", "খাদ্য", "কেনেদৰে", "কিমান", "চাইলেজ", "ৰ", "ৱ"]
    ),
    "gu": LanguageDefinition(
        code="gu",
        name="Gujarati",
        native_name="ગુજરાતી",
        script="Gujarati",
        unicode_ranges=[(0x0A80, 0x0AFF)],
        vocabulary_markers=["ગાય", "દૂધ", "ચારો", "ખોરાક", "સાઈલેજ", "કેટલું", "શું", "આપવું", "પશુ"]
    ),
    "pa": LanguageDefinition(
        code="pa",
        name="Punjabi",
        native_name="ਪੰਜਾਬੀ",
        script="Gurmukhi",
        unicode_ranges=[(0x0A00, 0x0A7F)],
        vocabulary_markers=["ਗਾਂ", "ਦੁੱਧ", "ਚਾਰਾ", "ਸਾਈਲੇਜ", "ਖੁਰਾਕ", "ਕਿੰਨਾ", "ਕੀ", "ਦੇਣਾ", "ਪਸ਼ੂ"]
    ),
    "or": LanguageDefinition(
        code="or",
        name="Odia",
        native_name="ଓଡ଼ିଆ",
        script="Odia",
        unicode_ranges=[(0x0B00, 0x0B7F)],
        vocabulary_markers=["ଗାଈ", "କ୍ଷୀର", "ଘାସ", "ଖାଦ୍ୟ", "ସାଇଲେଜ", "କେତେ", "କଣ", "ଦେବା", "ପଶୁ"]
    ),
    # Perso-Arabic Script
    "ur": LanguageDefinition(
        code="ur",
        name="Urdu",
        native_name="اردو",
        script="Arabic",
        unicode_ranges=[(0x0600, 0x06FF)],
        vocabulary_markers=["گائے", "دودھ", "چارہ", "خوراک", "کتنا", "کیا", "سائلیج", "دینا"]
    ),
    "ks": LanguageDefinition(
        code="ks",
        name="Kashmiri",
        native_name="کٲشُر / डोगरी",
        script="Arabic/Devanagari",
        unicode_ranges=[(0x0600, 0x06FF), (0x0900, 0x097F)],
        vocabulary_markers=["گاو", "دود", "گاسہ", "کیتھ", "गाई", "दूद", "घास"]
    ),
    "sd": LanguageDefinition(
        code="sd",
        name="Sindhi",
        native_name="سنڌي / सिंधी",
        script="Arabic/Devanagari",
        unicode_ranges=[(0x0600, 0x06FF), (0x0900, 0x097F)],
        vocabulary_markers=["ڳئون", "کير", "چارو", "گايون", "दूध", "चारो"]
    ),
    # Sino-Tibetan / Meitei
    "mni": LanguageDefinition(
        code="mni",
        name="Manipuri / Meitei",
        native_name="মৈতৈলোন্ / ꯃꯤꯇꯩꯂꯣꯟ",
        script="Meetei/Bengali",
        unicode_ranges=[(0xABC0, 0xABFF), (0xAAE0, 0xAAFF), (0x0980, 0x09FF)],
        vocabulary_markers=["ꯁꯟ", "ꯁꯡꯒꯣꯝ", "ꯆꯤꯟꯖꯥꯛ", "শন", "দুধ", "চাৰা"]
    )
}


class LanguageService:
    """Production Language Detection, Normalization, and Management Service."""

    def __init__(self):
        self.languages = SUPPORTED_LANGUAGES

    def is_supported(self, code: Optional[str]) -> bool:
        """Check if a language code is directly supported."""
        if not code:
            return False
        return code.strip().lower() in self.languages

    def register_language(self, definition: LanguageDefinition) -> None:
        """Extensible method to add more regional languages dynamically."""
        self.languages[definition.code.lower()] = definition

    def get_language_name(self, code: str) -> str:
        """Get human-readable language name."""
        lang = self.languages.get(code.lower())
        return lang.name if lang else "English"

    def detect_script(self, text: str) -> str:
        """
        Identify the dominant Unicode script of the input text.
        """
        counts: Dict[str, int] = {}
        for ch in text:
            if not ch.isalpha():
                continue
            code_point = ord(ch)
            script_found = "Latin"
            for lang_code, lang_def in self.languages.items():
                for start, end in lang_def.unicode_ranges:
                    if start <= code_point <= end:
                        script_found = lang_def.script
                        break
                if script_found != "Latin":
                    break
            counts[script_found] = counts.get(script_found, 0) + 1

        if not counts:
            return "Latin"
        return max(counts, key=counts.get)

    def detect_language(self, text: str) -> Tuple[str, float]:
        """
        Detect language of the text. Returns (language_code, confidence).
        Handles Native Scripts, Romanized Tanglish/Hinglish, and Latin English.
        """
        clean_text = text.strip()
        if not clean_text:
            return "en", 0.0

        lower_text = clean_text.lower()

        # 1. Check Native Script Unicode Blocks
        script_scores: Dict[str, int] = {}
        for ch in clean_text:
            if not ch.isalpha():
                continue
            cp = ord(ch)
            for code, lang_def in self.languages.items():
                if code == "en":
                    continue
                for start, end in lang_def.unicode_ranges:
                    if start <= cp <= end:
                        script_scores[code] = script_scores.get(code, 0) + 1

        if script_scores:
            top_lang = max(script_scores, key=script_scores.get)
            total_chars = sum(script_scores.values())
            confidence = round(min(1.0, script_scores[top_lang] / max(1, total_chars)), 2)

            # Disambiguate shared scripts:
            # A. Devanagari script: hi, mr, sa, ne, kok, mai
            devanagari_codes = {"hi", "mr", "sa", "ne", "kok", "mai"}
            if top_lang in devanagari_codes or top_lang == "hi":
                dev_scores: Dict[str, int] = {}
                for sub_code in ["hi", "mr", "sa", "ne", "kok", "mai"]:
                    markers = self.languages[sub_code].vocabulary_markers
                    matches = sum(1 for m in markers if m in clean_text)
                    if matches > 0:
                        dev_scores[sub_code] = matches
                if dev_scores:
                    best_dev = max(dev_scores, key=dev_scores.get)
                    return best_dev, 0.95
                return "hi", confidence


            # B. Bengali script: bn, as, mni
            if top_lang in {"bn", "as", "mni"}:
                # Assamese specific characters: ৰ (0x09F0), ৱ (0x09F1) or Assamese vocabulary
                if "\u09F0" in clean_text or "\u09F1" in clean_text or any(m in clean_text for m in self.languages["as"].vocabulary_markers):
                    return "as", 0.95
                return "bn", confidence

            # C. Arabic script: ur, ks, sd
            if top_lang in {"ur", "ks", "sd"}:
                for sub_code in ["ks", "sd"]:
                    if any(m in clean_text for m in self.languages[sub_code].vocabulary_markers):
                        return sub_code, 0.90
                return "ur", confidence

            return top_lang, confidence

        # 2. Check Romanized Indian languages (Tanglish / Hinglish / Romanized Indic)
        words = re.findall(r"\b[a-zA-Z]{2,}\b", lower_text)
        indic_roman_scores: Dict[str, int] = {}

        for code in ["ta", "hi", "te", "kn", "ml"]:
            markers = self.languages[code].vocabulary_markers
            match_count = 0
            for w in words:
                for m in markers:
                    if m.isalpha() and (w == m or (len(m) >= 4 and (w.startswith(m) or w.endswith(m)))):
                        match_count += 1
            if match_count > 0:
                indic_roman_scores[code] = match_count

        if indic_roman_scores:
            top_roman = max(indic_roman_scores, key=indic_roman_scores.get)
            return top_roman, 0.85

        # 3. Default fallback to English
        return "en", 0.70

    def resolve_effective_language(
        self,
        explicit_language: Optional[str],
        text: str
    ) -> Tuple[str, str]:
        """
        Determine target response language and detected input language.
        Rules:
        - If explicit_language is valid and supported -> target_language = explicit_language.
        - Otherwise target_language = detected_language.
        - detected_language is always computed from text.
        """
        detected_lang, _ = self.detect_language(text)

        if explicit_language:
            cleaned_code = explicit_language.strip().lower()
            if self.is_supported(cleaned_code):
                return cleaned_code, detected_lang

        # Fallback to detected language or default
        return detected_lang or "en", detected_lang or "en"


# Global singleton service
language_service = LanguageService()
