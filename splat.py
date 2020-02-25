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
        "fortune-cookies": "https://raw.githubusercontent.com/reggi/fortune-cookie/master/fortune-cookies.txt",
        "frankenstein": "https://www.gutenberg.org/files/84/84-0.txt",
        "pride": "https://www.gutenberg.org/files/1342/1342-0.txt",
        "shakespeare": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    }

    def __init__(self, model="small", tuning="frankenstein", log_function=lambda x: print(x)):
        self.run_name = f"{tuning}-{model}"
        self.model = UnicornAI.models_available[model]
        self.brain = None
        self.tuning_path = os.path.join("tunings", f"{tuning}.txt")
        if not os.path.isdir(os.path.join("models", self.model)):
            log_function(f"Downloading {model} model...")
            gpt2.download_gpt2(model_name=self.model)
        if not os.path.isfile(self.tuning_path):
            log_function(f"Downloading {tuning} library...")
            tuning_url = UnicornAI.tunings_available[tuning]
            tuning_text = requests.get(tuning_url).text
            with open(self.tuning_path, "w") as file:
                file.write(tuning_text)
        log_function("Starting brain...")
        self.reset_session()
        log_function("Starting voice...")
        pygame.mixer.init()

    @staticmethod
    def say_sentences(text_to_speak):
        speech = TextToSpeech(text=text_to_speak, lang=UnicornAI.language)
        seconds = int(time.time())
        filename = f"tmp/{seconds}.mp3"
        speech.save(filename)
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()

    def generate_text(self, prefix="Last Friday I", length=50):
        return gpt2.generate(self.brain, prefix=prefix, length=length, run_name=self.run_name, return_as_list=True)[0]

    def reset_session(self):
        if self.brain is not None:
            gpt2.reset_session(self.brain)
        self.brain = gpt2.start_tf_sess()
        gpt2.finetune(
            self.brain, self.tuning_path,
            model_name=self.model, steps=1, overwrite=True, run_name=self.run_name
        )


if __name__ == "__main__":
    unicorn = UnicornAI()
    while True:
        print(f"Generating text using {unicorn.run_name}...")
        generated_text = unicorn.generate_text()
        print("Speaking text...")
        print(generated_text)
        unicorn.say_sentences(generated_text)
        print("Resetting session...")
        unicorn.reset_session()
