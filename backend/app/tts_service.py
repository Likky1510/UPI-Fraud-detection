from __future__ import annotations

from dataclasses import dataclass

from .schemas import LanguageCode


LANGUAGE_CODE_MAP = {
    "en": "en-IN",
    "hi": "hi-IN",
    "te": "te-IN",
}

VOICE_NAME_CANDIDATES = {
    "en": ["en-IN-Wavenet-A", "en-IN-Standard-A"],
    "hi": ["hi-IN-Wavenet-A", "hi-IN-Standard-A"],
    "te": ["te-IN-Standard-A"],
}


@dataclass
class CloudTTSService:
    client: object | None = None

    def _get_client(self):
        if self.client is not None:
            return self.client

        try:
            from google.cloud import texttospeech  # type: ignore
        except Exception as exc:  # pragma: no cover - runtime dependency guard
            raise RuntimeError("google-cloud-texttospeech is not installed") from exc

        self.client = texttospeech.TextToSpeechClient()
        return self.client

    def synthesize_mp3_with_meta(self, text: str, language: LanguageCode) -> tuple[bytes, dict]:
        clean_text = (text or "").strip()
        if not clean_text:
            raise ValueError("Text is empty")
        if len(clean_text) > 4000:
            clean_text = clean_text[:4000]

        client = self._get_client()
        from google.cloud import texttospeech  # type: ignore

        synthesis_input = texttospeech.SynthesisInput(text=clean_text)
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=0.92,
            pitch=1.5,
        )
        language_code = LANGUAGE_CODE_MAP.get(language, "en-IN")
        candidates = VOICE_NAME_CANDIDATES.get(language, [])
        last_exc: Exception | None = None

        for voice_name in candidates:
            try:
                voice = texttospeech.VoiceSelectionParams(
                    language_code=language_code,
                    name=voice_name,
                    ssml_gender=texttospeech.SsmlVoiceGender.FEMALE,
                )
                response = client.synthesize_speech(
                    input=synthesis_input,
                    voice=voice,
                    audio_config=audio_config,
                )
                return bytes(response.audio_content), {
                    "language": language,
                    "language_code": language_code,
                    "selected_voice_name": voice_name,
                    "selection_mode": "named_voice",
                }
            except Exception as exc:  # pragma: no cover - network/service errors
                last_exc = exc

        # Fallback to language+gender when specific voice names are unavailable.
        try:
            voice = texttospeech.VoiceSelectionParams(
                language_code=language_code,
                ssml_gender=texttospeech.SsmlVoiceGender.FEMALE,
            )
            response = client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config,
            )
            return bytes(response.audio_content), {
                "language": language,
                "language_code": language_code,
                "selected_voice_name": None,
                "selection_mode": "language_gender_fallback",
            }
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Unable to synthesize TTS audio") from (last_exc or exc)

    def synthesize_mp3(self, text: str, language: LanguageCode) -> bytes:
        audio, _meta = self.synthesize_mp3_with_meta(text, language)
        return audio

    def debug_voice_selection(self, language: LanguageCode) -> dict:
        # Use a tiny synthesis request to reveal the exact path/voice resolution used at runtime.
        _audio, meta = self.synthesize_mp3_with_meta("UPI Sentinel voice check", language)
        return meta
