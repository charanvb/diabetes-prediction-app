from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model and scaler
model = pickle.load(open("diabetes_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get values from form
        features = [
            float(request.form["Pregnancies"]),
            float(request.form["Glucose"]),
            float(request.form["BloodPressure"]),
            float(request.form["SkinThickness"]),
            float(request.form["Insulin"]),
            float(request.form["BMI"]),
            float(request.form["DiabetesPedigreeFunction"]),
            float(request.form["Age"])
        ]

        # Scale input
        final_features = scaler.transform([features])

        # Predict class and probability
        prediction = model.predict(final_features)[0]
        probability = model.predict_proba(final_features)[0][1]
        probability_percent = round(probability * 100, 2)

        # Decide result
        if prediction == 1:
            result = "Diabetic"
            result_class = "positive"
        else:
            result = "Not Diabetic"
            result_class = "negative"

        return render_template(
            "index.html",
            prediction_text=f"Result: {result}",
            result_class=result_class
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text="Error in input values"
        )

if __name__ == "__main__":
    import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

