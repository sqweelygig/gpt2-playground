from src.really_simple_gpt import ReallySimpleGPT
import flask

app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route("/", methods=["GET"])
def home():
    really_simple_gpt = ReallySimpleGPT()
    return really_simple_gpt.generate()


app.run()
