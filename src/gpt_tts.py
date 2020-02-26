import pygame
from src.really_simple_gpt import ReallySimpleGPT
from gtts import gTTS as TextToSpeech
import time


def say_sentences(text_to_speak="Hello world!"):
    speech = TextToSpeech(text=text_to_speak, lang=ReallySimpleGPT.language)
    seconds = int(time.time())
    filename = f"tmp/{seconds}.mp3"
    speech.save(filename)
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()


if __name__ == "__main__":
    pygame.mixer.init()
    really_simple_gpt = ReallySimpleGPT()
    while True:
        generated_text = really_simple_gpt.generate()
        print(generated_text)
        say_sentences(generated_text)
        really_simple_gpt.fine_tune()
