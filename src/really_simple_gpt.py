import gpt_2_simple as gpt2
import os
import requests


class ReallySimpleGPT:
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

    @property
    def model_name(self):
        return ReallySimpleGPT.models_available[self.model_description]

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
            tuning_url = ReallySimpleGPT.tunings_available[self.tuning_description]
            tuning_text = requests.get(tuning_url).text
            with open(self.tuning_path, "w") as file:
                file.write(tuning_text)

    def generate(self, prefix="I once", length=100):
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

    def fine_tune(self, steps=100):
        self.log_function(f"* Fine tuning towards {self.tuning_description}...")
        brain = gpt2.start_tf_sess()
        gpt2.finetune(
            brain, self.tuning_path,
            model_name=self.model_name, steps=steps, run_name=self.run_name, overwrite=True
        )
        gpt2.reset_session(brain)
