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
        self.tuning_description = tuning
        self.model_description = model
        self.log_function = log_function
        self.download_model()
        self.download_tuning()
        log_function("Starting voice...")
        pygame.mixer.init()

    @property
    def model_name(self):
        return UnicornAI.models_available[self.model_description]

    @property
    def tuning_path(self):
        return os.path.join("tunings", f"{self.tuning_description}.txt")

    @property
    def run_name(self):
        return f"{self.tuning_description}-{self.model_description}"

    def download_model(self):
        if not os.path.isdir(os.path.join("models", self.model_name)):
            self.log_function(f"* Downloading {self.model_description} model...")
            gpt2.download_gpt2(model_name=self.model_name)

    def download_tuning(self):
        if not os.path.isfile(self.tuning_path):
            self.log_function(f"* Downloading {self.tuning_description} library...")
            tuning_url = UnicornAI.tunings_available[self.tuning_description]
            tuning_text = requests.get(tuning_url).text
            with open(self.tuning_path, "w") as file:
                file.write(tuning_text)

    def say_sentences(self, text_to_speak="Hello world!"):
        self.log_function("* Speaking...")
        self.log_function("\"\"\"")
        self.log_function(text_to_speak)
        self.log_function("\"\"\"")
        speech = TextToSpeech(text=text_to_speak, lang=UnicornAI.language)
        seconds = int(time.time())
        filename = f"tmp/{seconds}.mp3"
        speech.save(filename)
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()

    def generate(self, prefix="Last Friday I", length=50):
        self.log_function("* Generating text...")
        self.log_function(f"  * {self.model_description} language model tuned towards {self.tuning_description}.")
        brain = gpt2.start_tf_sess()
        run_name = self.run_name if os.path.isdir(os.path.join("checkpoint", self.run_name)) else None
        model_name = None if os.path.isdir(os.path.join("checkpoint", self.run_name)) else self.model_name
        gpt2.load_gpt2(brain, run_name=run_name, model_name=model_name)
        text = gpt2.generate(
            brain,
            prefix=prefix, length=length, run_name=run_name, return_as_list=True, model_name=self.model_name,
        )[0]
        gpt2.reset_session(brain)
        return text

    def fine_tune(self, steps=1):
        self.log_function(f"* Fine tuning towards {self.tuning_description}...")
        brain = gpt2.start_tf_sess()
        gpt2.finetune(
            brain, self.tuning_path,
            model_name=self.model_name, steps=steps, run_name=self.run_name
        )
        gpt2.reset_session(brain)


if __name__ == "__main__":
    unicorn = UnicornAI()
    generated_text = unicorn.generate()
    unicorn.say_sentences(generated_text)
    unicorn.fine_tune()
