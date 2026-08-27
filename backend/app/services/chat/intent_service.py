"""
Intent Classification & Module Routing Service
Accurately categorizes farmer queries across 20+ Indian languages, Romanized Tanglish/Hinglish, and English
"""

import re
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class IntentMatch:
    intent: str
    module: str
    confidence: float
    matched_keywords: List[str]


# ----------------------------------------------------------------------
# Multilingual Intent Lexicons & Regex Patterns
# ----------------------------------------------------------------------

INTENT_LEXICON: Dict[str, Dict[str, List[str]]] = {
    "greeting": {
        "keywords": [
            "hi", "hello", "hey", "good morning", "good evening", "good afternoon",
            "vanakkam", "namaste", "namaskar", "namaskara", "namaskaram", "nomoshkar",
            "pranam", "adaab", "sat sri akal", "sasriyakaal", "khem cho", "ram ram",
            "வணக்கம்", "காலை வணக்கம்", "ஹலோ", "ஹாய்",
            "नमस्ते", "प्रणाम", "नमस्कार", "राम राम", "हेलो", "हाय",
            "నమస్కారం", "నమస్తే", "హలో",
            "ನಮಸ್ಕಾರ", "ಹಲೋ", "ಹಾಯ್",
            "നമസ്കാരം", "ഹലോ", "ഹായ്",
            "নমস্কার", "নমস্কারান্তে", "হ্যালো",
            "নমস্কাৰ", "হ্যালো",
            "નમસ્તે", "કેમ છો", "હેલો",
            "ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ", "ਹੈਲੋ",
            "ନମସ୍କାର", "ଜୟ ଜଗନ୍ନାଥ",
            "السلام علیکم", "آداب", "ہیلو",
            "नमस्कारः", "प्रणामाः",
            "नमस्कार", "हेलो",
            "ꯈꯨꯔꯨꯃꯖꯔꯤ"
        ]
    },
    "silage_quality": {
        "keywords": [
            "silage", "fermentation", "fqi", "silo", "pit silage", "bunker silage", "silage ph",
            "silage moisture", "mold in silage", "spoilage", "chopping length", "compaction",
            "சைலேஜ்", "புல் ஊறுகாய்", "சைலேஜ் தரம்", "ஈரப்பதம்", "நொதித்தல்", "அமிலத்தன்மை", "பிஎச்",
            "साइलेज", "साइलेज की गुणवत्ता", "किण्वन", "नमी", "साइलो", "पीएच", "खट्टापन",
            "సైలేజ్", "సైలేజ్ నాణ్యత", "పులియబెట్టడం", "తేమ", "పిహెచ్",
            "ಸೈಲೇಜ್", "ಹುಲ್ಲು ಉಪ್ಪಿನಕಾಯಿ", "ಸೈಲೇಜ್ ಗುಣಮಟ್ಟ", "ತೇವಾಂಶ",
            "സൈലേജ്", "സൈലേജ് ഗുണനിലവാരം", "ഈർപ്പം", "പുളിപ്പിക്കൽ",
            "সাইলেজ", "সাইলেজের মান", "ফার্মেন্টেশন", "আর্দ্রতা",
            "চাইলেজ", "চাইলেজৰ মান",
            "સાઈલેજ", "સાઈલેજ ગુણવત્તા", "આથો", "ભેજ",
            "ਸਾਈਲੇਜ", "ਸਾਈਲੇਜ ਦੀ ਗੁਣਵੱਤਾ", "ਨਮੀ",
            "ସାଇଲେଜ", "ସାଇଲେଜ ଗୁଣବତ୍ତା", "ଆର୍ଦ୍ରତା",
            "سائلیج", "خمیر", "نمی",
            "सायलेज", "सायलेज प्रत", "किण्वन"
        ]
    },
    "nutrition": {
        "keywords": [
            "ration", "feed formula", "feed requirement", "nutrition", "diet", "dmi", "dry matter intake",
            "crude protein", "feed calculation", "what to feed for 15 litre", "lactating ration", "feed quantity",
            "feed per day", "daily feed", "nutrient requirement", "body weight feed", "balanced ration", "what to feed",
            "தீவன அளவு", "ரேஷன்", "ஊட்டச்சத்து", "எடைக்கு தீவனம்", "பால் உற்பத்திக்கு தீவனம்", "தினசரி தீவனம்",
            "உணவு தேவை", "கறவை மாட்டு உணவு", "கணக்கீடு", "எவ்வளவு தீவனம்", "தீவன தேவை",
            "संतुलित आहार", "राशन", "पोषण", "खुराक", "दूध के लिए चारा", "दैनिक आहार", "वजन के अनुसार आहार",
            "दाना कितना दें", "आहार गणना", "पोषण जरूरत", "आहार",
            "సమతుల్య ఆహారం", "రేషన్", "పోషకాహారం", "మేత పరిమాణం", "దాణా ఎంత ఇవ్వాలి",
            "ಸಮತೋಲಿತ ಆಹಾರ", "ರೇಷನ್", "ಪೋಷಕಾಂಶ", "ಮೇವು ಪ್ರಮಾಣ", "ಹಾಲು ಹೆಚ್ಚಿಸಲು ಆಹಾರ",
            "സമീകൃതാഹാരം", "റേഷൻ", "പോഷകാഹാരം", "തീറ്റയുടെ അളവ്", "പാലിനുള്ള തീറ്റ",
            "সুষম খাদ্য", "রেশন", "পুষ্টি", "খাবারের পরিমাণ", "দুধের জন্য খাবার",
            "সুষম আহাৰ", "পুষ্টি", "খাদ্যৰ পৰিমাণ",
            "સંતુલિત આહાર", "રાશન", "પોષણ", "ખોરાકનું પ્રમાણ",
            "ਸੰਤੁਲਿਤ ਖੁਰਾਕ", "ਰਾਸ਼ਨ", "ਪੋਸ਼ਣ", "ਖੁਰਾਕ ਦੀ ਮਾਤਰਾ",
            "ସନ୍ତୁଳିତ ଖାଦ୍ୟ", "ପୁଷ୍ଟିକର ଖାଦ୍ୟ", "ଖାଦ୍ୟ ପରିମାଣ",
            "متوازن غذا", "راشن", "غذائیت", "خوراک کی مقدار",
            "संतुलित आहार", "चारा प्रमाण", "पोषण गरज"
        ]
    },
    "feed": {
        "keywords": [
            "feed", "feeding", "fodder", "green grass", "dry fodder", "straw", "maize", "corn", "sorghum", "napier", "co4", "co-4",
            "azolla", "berseem", "lucerne", "concentrate", "cotton seed cake", "mustard cake", "wheat bran",
            "mineral mixture", "bypass fat", "urea treated straw", "feed ingredient", "pasture", "grazing", "diet", "food",
            "தீவனம்", "உணவு", "பசுந்தீவனம்", "உலர்தீவனம்", "மக்காச்சோளம்", "நேப்பியர்", "அசோலா", "அடர்தீவனம்", "தாது உப்பு",
            "பருத்தி கொட்டை", "தவிடு", "வைக்கோல்", "புல்", "தீவன வகை",
            "चारा", "खुराक", "हरा चारा", "सूखा चारा", "मक्का", "नेपियर", "अजोला", "बरसीम", "खली", "चोकर", "खनिज मिश्रण",
            "भूसा", "घास", "पशु आहार घटक", "दाना",
            "మేత", "పచ్చి మేత", "ఎండు మేత", "మొక్కజొన్న", "అజోల్లా", "ఖనిజ మిశ్రమం", "తవుడు", "దాణా",
            "ಮೇವು", "ಹಸಿರು ಮೇವು", "ಒಣ ಮೇವು", "ಮೆಕ್ಕೆಜೋಳ", "ಅಜೋಲ್ಲಾ", "ಖನಿಜ ಮಿಶ್ರಣ", "ಆಹಾರ",
            "തീറ്റ", "പച്ചപ്പുല്ല്", "വൈക്കോൽ", "ചോളം", "അസോള", "ധാതുലവണ മിശ്രിതം",
            "খাবার", "ঘাস", "সবুজ ঘাস", "শুকনো ঘাস", "ভুট্টা", "অ্যাজোলা", "খৈল", "ভুসি",
            "খাদ্য", "সেউজীয়া ঘাঁহ", "মাকৈ", "আজোলা",
            "ચારો", "લીલો ચારો", "સૂકો ચારો", "મકાઈ", "અઝોલા", "ખાણ",
            "ਚਾਰਾ", "ਹਰਾ ਚਾਰਾ", "ਸੁੱਕਾ ਚਾਰਾ", "ਮੱਕੀ", "ਅਜ਼ੋਲਾ", "ਖਲ", "ਚੋਕਰ", "ਖੁਰਾਕ",
            "ଘାସ", "ଖାଦ୍ୟ", "ସବୁଜ ଘାସ", "ଶୁଖିଲା ଘାସ", "ମକା", "ଆଜୋଲା",
            "چارہ", "خوراک", "ہرا چارہ", "خشک چارہ", "مکئی", "ازولا", "کھل", "چوکر",
            "चारा", "हिरवा चारा", "वाळलेला चारा", "मका", "अझोला", "पेंड", "कोंडा"
        ]
    },

    "cattle_health_general": {
        "keywords": [
            "disease", "fever", "mastitis", "lumpy skin", "lsd", "foot and mouth", "fmd", "black quarter",
            "bq", "hemorrhagic septicemia", "hs", "brucellosis", "diarrhea", "wound", "tick", "parasite",
            "loss of appetite", "not eating", "bloat", "vaccine", "vaccination", "ill", "sick", "vet",
            "veterinary", "infection", "eye discharge", "salivation", "limping",
            "நோய்", "காய்ச்சல்", "மடிநோய்", "கோமாரி", "சரும கட்டி", "சாப்பிடவில்லை", "தீவனம் எடுக்கவில்லை",
            "வயிற்றுப்போக்கு", "தடுப்பூசி", "மருத்துவர்", "புண்", "உண்ணி",
            "बीमारी", "बुखार", "थनैला", "गलघोंटू", "खुरपका", "मुंहपका", "लंपी", "दस्त", "टीकाकरण",
            "खाना नहीं खा रही", "सुस्त", "पशु चिकित्सक", "घाव", "संक्रमण",
            "వ్యాధి", "జ్వరం", "పొదుగు వాపు", "గాలికుంటు", "టీకా", "మేత తినడం లేదు",
            "ರೋಗ", "ಜ್ವರ", "ಕೆಚ್ಚಲು ಬಾವು", "ಕಾಲುಬಾಯಿ ರೋಗ", "ಲಸಿಕೆ", "ಹುಲ್ಲು ತಿನ್ನುತ್ತಿಲ್ಲ",
            "രോഗം", "പനി", "അകിടുവീക്കം", "കുളമ്പുരോഗം", "വാക്സിൻ", "തീറ്റ തിന്നുന്നില്ല",
            "রোগ", "জ্বর", "ওলান ফোলা", "খুরারোগ", "টিকা", "খাচ্ছে না",
            "বেমাৰ", "জ্বৰ", "টীকাকৰণ", "খোৱা নাই",
            "રોગ", "તાવ", "મસ્ટાઈટીસ", "ખરવા મોવાસા", "રસી", "ખાતી નથી",
            "ਬਿਮਾਰੀ", "ਬੁਖਾਰ", "ਸੜਿਆਂਗ", "ਮੂੰਹਖੁਰ", "ਟੀਕਾਕਰਨ", "ਨਹੀਂ ਖਾ ਰਹੀ",
            "ରୋଗ", "ଜ୍ୱର", "ଥନେଲା", "ଫାଟୁଆ", "ଟୀକା", "ଖାଉ ନାହିଁ",
            "بیماری", "بخار", "سڑن", "منہ کھر", "ٹیکہ کاری", "نہیں کھا رہی",
            "आजारी", "ताप", "स्तनदाह", "लाळ्या खुरकूत", "लसीकरण", "खात नाही"
        ]
    },
    "milk_production": {
        "keywords": [
            "milk yield", "milk production", "increase milk", "more milk", "fat percentage", "snf",
            "milking interval", "lactation yield", "improve milk", "low milk", "drop in milk",
            "பால் உற்பத்தி", "பால் அதிகரிக்க", "கொழுப்பு சத்து", "பால் குறைவு", "பால் பெருக்கம்", "கறவை முறை",
            "दूध उत्पादन", "दूध बढ़ाएं", "दूध में फैट", "दूध कम होना", "दूध कैसे बढ़ाएं", "लैक्टेशन",
            "పాలు పెంచడం", "పాలు దిగుబడి", "పాలల్లో వెన్న", "పాలు తగ్గడం",
            "ಹಾಲು ಇಳುವರಿ", "ಹಾಲು ಹೆಚ್ಚಿಸುವುದು", "ಹಾಲಿನ ಕೊಬ್ಬು", "ಹಾಲು ಕಡಿಮೆಯಾಗಿದೆ",
            "പാൽ ഉത്പാദനം", "പാൽ കൂടാൻ", "പാലിലെ കൊഴുപ്പ്", "പാൽ കുറഞ്ഞു",
            "দুধ বৃদ্ধি", "দুধের উৎপাদন", "দুধে ফ্যাট", "দুধ কমে গেছে",
            "গাখীৰ বৃদ্ধি", "গাখীৰৰ উৎপাদন", "গাখীৰত চৰ্বি",
            "દૂધ વધારવું", "દૂધ ઉત્પાદન", "દૂધમાં ફેટ", "દૂધ ઘટી ગયું",
            "ਦੁੱਧ ਵਧਾਉਣਾ", "ਦੁੱਧ ਦਾ ਉਤਪਾਦਨ", "ਦੁੱਧ ਵਿਚ ਫੈਟ", "ਦੁੱਧ ਘਟ ਗਿਆ",
            "କ୍ଷୀର ବୃଦ୍ଧି", "କ୍ଷୀର ଉତ୍ପାଦନ", "କ୍ଷୀରରେ ଫ୍ୟାଟ୍",
            "دودھ بڑھانا", "دودھ کی پیداوار", "دودھ میں چکنائی",
            "दूध वाढवणे", "दूध उत्पादन", "दुधातील फॅट", "दूध कमी झाले"
        ]
    },
    "animal_profile": {
        "keywords": [
            "breed", "murrah", "gir", "sahiwal", "holstein", "jersey", "red sindhi", "tharparkar",
            "body condition score", "bcs", "cow age", "weight estimation", "parity", "lactation stage",
            "dry period", "pregnant cow", "heifer", "calf",
            "மாட்டின் இனம்", "நாட்டு மாடு", "கிர்", "சாஹிவால்", "ஜெர்சி", "எச்எப்", "சினை மாடு", "கன்றுக்குட்டி",
            "नस्ल", "गाय की नस्ल", "मुर्रा", "गीर", "साहीवाल", "जर्सी", "एचएफ", "गर्भवती गाय", "बछड़ा",
            "జాతి", "ఆవు జాతి", "ముర్రా", "గిర్", "సాహివాల్", "గర్భం",
            "ತಳಿ", "ಹಸುವಿನ ತಳಿ", "ಮುರ್ರಾ", "ಗಿರ್", "ಸಾಹಿವಾಲ್", "ಗರ್ಭಾವಸ್ಥೆ",
            "ഇനം", "പശുവിന്റെ ഇനം", "ഗിർ", "സാഹിവാൾ", "ഗർഭിണി പശു",
            "জাত", "গরুর জাত", "গির", "শাহিওয়াল", "জার্সি", "গাভীন",
            "জাতি", "গাভীৰ জাত", "গিৰ",
            "ઓલાદ", "ગાયની ઓલાદ", "ગીર", "સાહીવાલ", "સગર્ભા ગાય",
            "ਨਸਲ", "ਗਾਂ ਦੀ ਨਸਲ", "ਮੁਰ੍ਹਾ", "ਗੀਰ", "ਸਾਹੀਵਾਲ",
            "ଜାତି", "ଗାଈ ଜାତି", "ଗୀର", "ସାହିୱାଲ",
            "نسل", "گائے کی نسل", "ساہیوال", "گر", "حاملہ گائے",
            "जात", "गाईची जात", "गीर", "साहिवाल", "गाभण गाय"
        ]
    },
    "general_dairy": {
        "keywords": [
            "dairy farm", "shed design", "ventilation", "housing", "clean milk production", "drinking water",
            "heat stress", "summer management", "hygiene", "dairy profit", "farm management",
            "பண்ணை மேலாண்மை", "கொட்டகை", "சுத்தமான பால்", "குடிநீர்", "வெப்ப மேலாண்மை",
            "डेयरी फार्म", "बाड़ा निर्माण", "स्वच्छ दूध", "पीने का पानी", "गर्मी से बचाव", "पशु प्रबंधन",
            "డైరీ ఫారం", "పాక నిర్మాణం", "పరిశుభ్రమైన పాలు", "తాగునీరు",
            "ಡೈರಿ ಫಾರ್ಮ್", "ಕೊಟ್ಟಿಗೆ ನಿರ್ಮಾಣ", "ಸ್ವಚ್ಛ ಹಾಲು", "ಕುಡಿಯುವ ನೀರು",
            "ഡെയറി ഫാം", "തൊഴുത്ത് നിർമ്മാണം", "ശുദ്ധമായ പാൽ", "കുടിവെള്ളം",
            "ডেইরি ফার্ম", "গোয়ালঘর", "পরিচ্ছন্ন দুধ", "পানীয় জল",
            "দুগ্ধ পাম", "গোহালি নিৰ্মাণ",
            "ડેરી ફાર્મ", "તબેલો", "સ્વચ્છ દૂધ", "પીવાનું પાણી",
            "ਡੇਅਰੀ ਫਾਰਮ", "ਵਾੜਾ", "ਸਾਫ ਦੁੱਧ", "ਪੀਣ ਵਾਲਾ ਪਾਣੀ",
            "ଡାଏରୀ ଫାର୍ମ", "ଗୁହାଳ", "ପରିଷ୍କାର କ୍ଷୀର",
            "ڈیری فارم", "باڑے کا ڈیزائن", "صاف دودھ", "پینے کا پانی",
            "डेअरी फार्म", "गोठा व्यवस्थापन", "स्वच्छ दूध", "पिण्याचे पाणी"
        ]
    }
}


class IntentService:
    """Production Intent Classification Service."""

    def __init__(self):
        self.lexicon = INTENT_LEXICON

    def classify(
        self,
        text: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> IntentMatch:
        """
        Classifies user query into one of 9 domain intents.
        Uses lexical match, regex patterns, entity extraction, and conversational context.
        """
        clean_text = text.strip().lower()
        if not clean_text:
            return IntentMatch(intent="unknown", module="chat", confidence=0.0, matched_keywords=[])

        # 1. Direct Regex / Entity Checks
        # Silage query patterns
        if re.search(r"\b(silage|ferment|fqi|silo|pit|bunker)\b", clean_text) or "சைலேஜ்" in clean_text or "साइलेज" in clean_text or "సైలేజ్" in clean_text or "ಸೈಲೇಜ್" in clean_text:
            return IntentMatch(intent="silage_quality", module="silage", confidence=0.95, matched_keywords=["silage"])

        # Nutrition calculation indicators: e.g. "15 litre", "420 kg", "feed requirement", "feed formula", "kudukuthu enna feed"
        has_milk_metric = bool(re.search(r"\b\d+\s*(l|litre|liter|litres|kg)\b", clean_text) or "லிட்டர்" in clean_text or "लीटर" in clean_text or "లీటర్ల" in clean_text)
        has_feed_ask = bool(re.search(r"\b(feed|ration|diet|saapadu|kudukanum|khaye|khilaye|aahaaram|theetta)\b", clean_text) or "தீவனம்" in clean_text or "चारा" in clean_text or "மேత" in clean_text or "ಮೇವು" in clean_text)

        if has_milk_metric and has_feed_ask:
            return IntentMatch(intent="nutrition", module="nutrition", confidence=0.95, matched_keywords=["metric_and_feed_ask"])

        # 2. Score across all Intent Lexicons
        scores: Dict[str, Tuple[int, List[str]]] = {}
        for intent_name, intent_data in self.lexicon.items():
            matched = []
            for kw in intent_data["keywords"]:
                kw_clean = kw.lower()
                # If keyword contains non-ASCII (e.g. Indic scripts: Tamil, Hindi, etc.)
                if any(ord(c) > 127 for c in kw_clean):
                    if kw_clean in clean_text:
                        matched.append(kw)
                else:
                    # ASCII words
                    if " " in kw_clean:
                        if kw_clean in clean_text:
                            matched.append(kw)
                    else:
                        if re.search(rf"\b{re.escape(kw_clean)}\b", clean_text):
                            matched.append(kw)

            if matched:
                scores[intent_name] = (len(matched), matched)

        if scores:
            top_intent = max(scores, key=lambda k: scores[k][0])
            match_count, kw_list = scores[top_intent]
            confidence = min(0.95, 0.60 + 0.15 * match_count)

            # Map intent to responsible backend module
            module_map = {
                "greeting": "chat",
                "silage_quality": "silage",
                "nutrition": "nutrition",
                "feed": "feed",
                "cattle_health_general": "health",
                "milk_production": "milk_production",
                "animal_profile": "animal_profile",
                "general_dairy": "general_dairy",
                "unknown": "chat"
            }
            return IntentMatch(
                intent=top_intent,
                module=module_map.get(top_intent, "chat"),
                confidence=confidence,
                matched_keywords=kw_list
            )

        # 3. Contextual Inference from History
        if conversation_history:
            for msg in reversed(conversation_history[-4:]):
                past_intent = msg.get("intent")
                if past_intent and past_intent in ["nutrition", "feed", "silage_quality", "milk_production"]:
                    # If current query contains numbers/percentages, follow-up words, or is a concise clarification
                    has_digits_or_pct = bool(re.search(r"\d", clean_text) or "%" in clean_text)
                    is_short_clarification = len(clean_text) <= 40
                    has_followup_word = any(w in clean_text for w in ["feed", "it", "what", "how", "ration", "chara", "enna", "epdi", "kudukanum", "khaye", "fat", "weight", "kg", "litre", "liter", "கொழுப்பு", "फैट"])
                    
                    if has_digits_or_pct or is_short_clarification or has_followup_word:
                        target_module = "nutrition" if past_intent in ["nutrition", "feed"] else ("silage" if past_intent == "silage_quality" else past_intent)
                        target_intent = "nutrition" if past_intent in ["nutrition", "feed"] else past_intent
                        return IntentMatch(
                            intent=target_intent,
                            module=target_module,
                            confidence=0.88,
                            matched_keywords=["contextual_followup"]
                        )


        # 4. Fallback unknown
        return IntentMatch(intent="unknown", module="chat", confidence=0.30, matched_keywords=[])



# Global singleton intent service
intent_service = IntentService()
