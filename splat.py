import gpt_2_simple as gpt2
import os
import requests

models_available = ["124M", "355M", "774M", "1.5B"]
tunings_available = {
    "shakespeare": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
}

model_name = models_available[0]
tuning_name = "shakespeare"

if not os.path.isdir(os.path.join("models", model_name)):
    print("Downloading {model_name} model...".format(model_name=model_name))
    gpt2.download_gpt2(model_name=model_name)

file_name = os.path.join("tunings", "{tuning_name}.txt".format(tuning_name=tuning_name))
if not os.path.isfile(file_name):
    url = tunings_available[tuning_name]
    data = requests.get(url)
    with open(file_name, 'w') as f:
        f.write(data.text)

while True:
    sess = gpt2.start_tf_sess()
    gpt2.finetune(sess, file_name, model_name=model_name, steps=1, overwrite=True, run_name=tuning_name)
    gpt2.generate(sess, prefix="Last Friday I", length=31, run_name=tuning_name)
    gpt2.reset_session(sess)
