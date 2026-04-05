import os
import joblib
import numpy as np
from flask import Flask, render_template, request


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

app = Flask(__name__)


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "model.pkl not found. Run 'python train_model.py' first."
        )
    return joblib.load(MODEL_PATH)


model = load_model()


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None
    sqft_value = ""

    if request.method == "POST":
        sqft_value = request.form.get("sqft", "").strip()
        try:
            sqft = float(sqft_value)
            if sqft <= 0:
                raise ValueError("Square feet must be greater than zero.")

            predicted_price = model.predict(np.array([[sqft]]))[0]
            prediction = f"${predicted_price:,.2f}"
        except ValueError as exc:
            error = str(exc) if str(exc) else "Please enter a valid number."

    return render_template(
        "index.html", prediction=prediction, error=error, sqft_value=sqft_value
    )


if __name__ == "__main__":
    app.run(debug=True)