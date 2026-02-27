from __future__ import annotations

from dataclasses import dataclass

from .schemas import LanguageCode


LANGUAGE_CODE_MAP = {
    "en": "en-IN",
    "hi": "hi-IN",
    "te": "te-IN",
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

    def synthesize_mp3(self, text: str, language: LanguageCode) -> bytes:
        clean_text = (text or "").strip()
        if not clean_text:
            raise ValueError("Text is empty")
        if len(clean_text) > 4000:
            clean_text = clean_text[:4000]

        client = self._get_client()
        from google.cloud import texttospeech  # type: ignore

        synthesis_input = texttospeech.SynthesisInput(text=clean_text)
        voice = texttospeech.VoiceSelectionParams(
            language_code=LANGUAGE_CODE_MAP.get(language, "en-IN"),
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE,
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=0.92,
            pitch=1.5,
        )

        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config,
        )
        return bytes(response.audio_content)
