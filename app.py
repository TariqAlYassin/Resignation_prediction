from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Page 1
@app.route('/')
def index():
    return render_template("index.html")

@app.route("/prediction", methods=["POST"])  # Corrected "Post" to "POST"
def prediction():
    if request.method =="POST":
        years_spent = int(request.form["years_spent"])  # Use square brackets
        satisfaction_level = float(request.form["satisfaction_level"])
        average_monthly_hours = int(request.form["average_monthly_hours"])  # Convert to int
        number_project = int(request.form["number_project"])  # Convert to int
        last_evaluation = float(request.form["last_evaluation"])
        technical = int(request.form["technical"])  # Convert to int if applicable
        sales = int(request.form["sales"])  # Convert to int if applicable
        management = int(request.form["management"])  # Convert to int if applicable
        promotion_last_5years = int(request.form["promo"])  # Convert to int if applicable

        df = pd.DataFrame({
            "time_spend_company": [years_spent],  # Wrap in list to create a DataFrame
            "satisfaction_level": [satisfaction_level],
            "average_montly_hours": [average_monthly_hours],
            "number_project": [number_project],
            "last_evaluation": [last_evaluation],
            "technical": [technical],
            "sales": [sales],
            "management": [management],  # Corrected column name
            "promotion_last_5years": [promotion_last_5years]
        })
    
    with open("model_resignation", "rb") as file:
        model_info = joblib.load(file)
        model = model_info["model"]
        threshold = model_info["threshold"]

    y_prob = model.predict_proba(df)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    
    if y_pred == 1:
        message = "The employee tends to leave the company"
    else:
        message = "The employee tends to stay at the company"
    
    return render_template("prediction.html", message=message)

if __name__ == '__main__':
    app.run(debug=True)
