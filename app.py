from flask import Flask, render_template, request
import config
from utils.preprocessing import preprocess_input
from utils.model_engine import run_model
from utils.llm_engine import run_llm

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None
    response = None

    if request.method == "POST":
        text_input = request.form.get("text_input")
        number_input = request.form.get("number_input")

        data = {"text": text_input, "score": number_input}
        processed = preprocess_input(data)

        if config.USE_CLASSIFIER:
            prediction, confidence = run_model(processed)
        elif config.USE_LLM:
            response = run_llm(text_input)

    return render_template("index.html",
                           prediction=prediction,
                           confidence=confidence,
                           response=response)

if __name__ == "__main__":
    app.run(debug=True)
