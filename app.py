from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

# Load trained pipeline
with open("pipe.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect form inputs
        features = [x for x in request.form.values()]
        
        # Convert to dataframe (ensure same feature names as training)
        # Example: change the list of column names to match your training dataset
        col_names = ["age", "sex", "bmi", "children", "smoker", "region"]
        input_df = pd.DataFrame([features], columns=col_names)

        # Predict using pipeline
        prediction = model.predict(input_df)[0]

        return render_template(
            "index.html",
            prediction_text=f"Predicted Health Insurance Premium: ${prediction:.2f}",
        )
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)
