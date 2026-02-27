TIPS = {
    "en": {
        "SAFE": "Looks safe. Always verify UPI ID before paying.",
        "RISKY": "Suspicious pattern found. Recheck recipient and never share OTP or UPI PIN.",
        "BLOCKED": "Transaction blocked for your safety. Avoid unknown QR codes and report scam calls.",
    },
    "hi": {
        "SAFE": "लेनदेन सुरक्षित लगता है। भुगतान से पहले UPI ID जांचें।",
        "RISKY": "संदिग्ध गतिविधि मिली। प्राप्तकर्ता जांचें और OTP या UPI PIN कभी साझा न करें।",
        "BLOCKED": "आपकी सुरक्षा के लिए लेनदेन रोका गया। अज्ञात QR और फर्जी कॉल से बचें।",
    },
    "te": {
        "SAFE": "లావాదేవీ సురక్షితంగా కనిపిస్తోంది. చెల్లింపుకు ముందు UPI ID నిర్ధారించండి.",
        "RISKY": "అనుమానాస్పద నమూనా గుర్తించబడింది. OTP లేదా UPI PIN ఎప్పుడూ పంచుకోవద్దు.",
        "BLOCKED": "మీ భద్రత కోసం లావాదేవీ నిలిపివేయబడింది. తెలియని QR కోడ్‌లను స్కాన్ చేయవద్దు.",
    },
}


def get_tip(language: str, verdict: str) -> str:
    normalized = (language or "en").strip().lower()
    aliases = {
        "english": "en",
        "hindi": "hi",
        "telugu": "te",
    }
    lang_key = aliases.get(normalized, normalized)
    lang_bucket = TIPS.get(lang_key, TIPS["en"])
    return lang_bucket.get(verdict, TIPS["en"]["RISKY"])
