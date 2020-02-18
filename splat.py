import re as regexp
import gpt_2_simple as gpt2
import os
import requests
from gtts import gTTS as TextToSpeech
import pygame
import time


class UnicornAI:
    language = "en"
    models_available = {
        "small": "124M",
        "medium": "355M",
        "large": "774M",
        "huge": "1558M",
    }
    tunings_available = {
        "shakespeare": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        "fortune-cookies": "https://raw.githubusercontent.com/reggi/fortune-cookie/master/fortune-cookies.txt",
        "frankenstein": "https://www.gutenberg.org/files/84/84-0.txt"
    }

    def __init__(self, model="medium", tuning="frankenstein", starter_text="Last Friday I", log_function=None):
        self.run_name = f"{tuning}-{model}"
        self.prefix = starter_text
        self.model = UnicornAI.models_available[model]
        self.brain = None
        if not os.path.isdir(os.path.join("models", self.model)):
            if log_function:
                log_function(f"Downloading {model} model...")
            gpt2.download_gpt2(model_name=self.model)
        self.tuning_path = os.path.join("tunings", f"{tuning}.txt")
        if not os.path.isfile(self.tuning_path):
            if log_function:
                log_function(f"Downloading {tuning} library...")
            tuning_url = UnicornAI.tunings_available[tuning]
            tuning_text = requests.get(tuning_url).text
            with open(self.tuning_path, "w") as file:
                file.write(tuning_text)
        if log_function:
            log_function("Starting brain...")
        self.reset_session()
        if log_function:
            log_function("Starting voice...")
        pygame.mixer.init()

    @staticmethod
    def say_sentences(text_to_speak):
        complete_sentences = regexp.split("[.!?]", text_to_speak)[:-1]
        complete_sentences = ". ".join(complete_sentences)
        speech = TextToSpeech(text=complete_sentences, lang=UnicornAI.language)
        seconds = time.time()
        filename = f"tmp/{seconds}.mp3"
        speech.save(filename)
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()

    def generate_text(self, prefix=None):
        return gpt2.generate(
            self.brain,
            prefix=prefix or self.prefix, length=100, run_name=self.run_name, return_as_list=True
        )[0]

    def reset_session(self):
        if self.brain is not None:
            gpt2.reset_session(self.brain)
        self.brain = gpt2.start_tf_sess()
        gpt2.finetune(
            self.brain, self.tuning_path,
            model_name=self.model, steps=1, overwrite=True, run_name=self.run_name
        )


if __name__ == "__main__":
    unicorn = UnicornAI(log_function=lambda x: print(x))
    while True:
        print(f"Generating text using {unicorn.run_name}...")
        generated_text = unicorn.generate_text()
        print("Speaking text...")
        print(generated_text)
        unicorn.say_sentences(generated_text)
        print("Resetting session...")
        unicorn.reset_session()
