import os
import pygame
import sys
import time
from gtts import gTTS as TextToSpeech
from src.really_simple_gpt import ReallySimpleGPT


if __name__ == "__main__":
    try:
        run_continuously = "--continuous" in sys.argv or "-c" in sys.argv
        should_fine_tune = "--fine-tune" in sys.argv or "-ft" in sys.argv
        should_generate = "--generate" in sys.argv or "-g" in sys.argv
        should_speak = "--speak" in sys.argv or "-s" in sys.argv
        pygame.mixer.init()
        really_simple_gpt = ReallySimpleGPT()
        first_run = True
        while first_run or run_continuously:
            first_run = False
            if should_generate:
                generated_text = really_simple_gpt.generate()
                print(generated_text)
                if should_speak:
                    speech = TextToSpeech(text=generated_text, lang=ReallySimpleGPT.language)
                    seconds = int(time.time())
                    filename = f"tmp/{seconds}.mp3"
                    speech.save(filename)
                    pygame.mixer.music.load(filename)
                    pygame.mixer.music.play()
                    os.remove(filename)
            if should_fine_tune:
                steps = 1 if run_continuously else None
                really_simple_gpt.fine_tune(steps=steps)
    except KeyboardInterrupt:
        pass
