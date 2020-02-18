import gpt_2_simple as gpt2
import os
import requests

tuning_name = "shakespeare"
model_name = "medium"
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
}

model_dir = models_available[model_name]
if not os.path.isdir(os.path.join("models", model_dir)):
    print(f"Downloading {model_name} model... ({model_dir})")
    gpt2.download_gpt2(model_name=model_dir)

tuning_file = os.path.join("tunings", f"{tuning_name}.txt")
if not os.path.isfile(tuning_file):
    tuning_url = tunings_available[tuning_name]
    data = requests.get(tuning_url)
    with open(tuning_file, 'w') as f:
        f.write(data.text)

while True:
    sess = gpt2.start_tf_sess()
    run_name = f"{tuning_name}-{model_name}"
    gpt2.finetune(sess, tuning_file, model_name=model_dir, steps=1, overwrite=True, run_name=run_name)
    gpt2.generate(sess, prefix=starter_text, length=31, run_name=run_name)
    gpt2.reset_session(sess)
