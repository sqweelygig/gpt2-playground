import gpt_2_simple as gpt2
import os
import requests
import pyttsx3 as text_to_speech

language = "en"
model_name = "medium"
tuning_name = "frankenstein"
starter_text = "Last Friday I"

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

model_dir = models_available[model_name]
if not os.path.isdir(os.path.join("models", model_dir)):
    print(f"Downloading {model_name} model...")
    gpt2.download_gpt2(model_name=model_dir)

tuning_file = os.path.join("tunings", f"{tuning_name}.txt")
if not os.path.isfile(tuning_file):
    print(f"Downloading {tuning_name} library...")
    tuning_url = tunings_available[tuning_name]
    data = requests.get(tuning_url)
    with open(tuning_file, 'w') as f:
        f.write(data.text)


def say_sentences(text):
    voice = text_to_speech.init()
    complete_sentences = text.split(".")[:-1]
    text_to_speak = ". ".join(complete_sentences)
    voice.say(text_to_speak)
    voice.runAndWait()


while True:
    sess = gpt2.start_tf_sess()
    run_name = f"{tuning_name}-{model_name}"
    print(f"Tuning {run_name}...")
    gpt2.finetune(sess, tuning_file, model_name=model_dir, steps=1, overwrite=True, run_name=run_name)
    print(f"Generating text using {run_name}...")
    generated_text = gpt2.generate(sess, prefix=starter_text, length=128, run_name=run_name, return_as_list=True)[0]
    print(generated_text)
    print(f"Speaking text...")
    say_sentences(generated_text)
    gpt2.reset_session(sess)
