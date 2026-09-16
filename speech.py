
import whisper
from gtts import gTTS


def load_speech_model():
    return whisper.load_model("base")


def transcribe_audio(model, audio_file):

    with open("temp_audio.wav", "wb") as f:
        f.write(audio_file.getbuffer())

    result = model.transcribe("temp_audio.wav")

    return result["text"].strip()


def text_to_speech(text, filename="ai_response.mp3"):

    tts = gTTS(
        text=text,
        lang="en"
    )

    tts.save(filename)

    return filename
