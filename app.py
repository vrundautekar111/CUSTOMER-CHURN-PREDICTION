from flask import (
    Flask,
    render_template,
    request
)

import pandas as pd
import joblib
import os


app = Flask(__name__)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

MODEL_PATH = "model/churn_model.pkl"

model = joblib.load(
    MODEL_PATH
)

print(
    "Churn model loaded successfully!"
)


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        gender = request.form["gender"]

        senior_citizen = int(
            request.form["senior_citizen"]
        )

        partner = request.form["partner"]

        dependents = request.form["dependents"]

        tenure = float(
            request.form["tenure"]
        )

        phone_service = request.form["phone_service"]

        internet_service = request.form[
            "internet_service"
        ]

        contract = request.form["contract"]

        payment_method = request.form[
            "payment_method"
        ]

        monthly_charges = float(
            request.form["monthly_charges"]
        )

        total_charges = float(
            request.form["total_charges"]
        )


        # Create DataFrame

        customer = pd.DataFrame(
            {
                "gender": [gender],

                "senior_citizen": [
                    senior_citizen
                ],

                "partner": [partner],

                "dependents": [
                    dependents
                ],

                "tenure": [tenure],

                "phone_service": [
                    phone_service
                ],

                "internet_service": [
                    internet_service
                ],

                "contract": [
                    contract
                ],

                "payment_method": [
                    payment_method
                ],

                "monthly_charges": [
                    monthly_charges
                ],

                "total_charges": [
                    total_charges
                ]
            }
        )


        # Prediction

        prediction = model.predict(
            customer
        )[0]

        probability = model.predict_proba(
            customer
        )[0][1]


        probability_percentage = (
            probability * 100
        )


        # Result

        if prediction == 1:

            result = "High Risk of Churn"

            recommendation = (
                "This customer is likely to leave. "
                "Consider offering a discount, "
                "loyalty benefit, or longer-term contract."
            )

        else:

            result = "Low Risk of Churn"

            recommendation = (
                "This customer is unlikely to leave. "
                "Continue providing good service and "
                "customer engagement."
            )


        return render_template(
            "result.html",

            prediction=result,

            probability=round(
                probability_percentage,
                2
            ),

            recommendation=recommendation
        )


    except Exception as e:

        return render_template(
            "result.html",

            prediction="Prediction Error",

            probability=0,

            recommendation=str(e)
        )


# --------------------------------------------------
# ANALYSIS PAGE
# --------------------------------------------------

@app.route("/analysis")
def analysis():

    dataset_path = (
        "dataset/customer_churn.csv"
    )

    df = pd.read_csv(
        dataset_path
    )


    total_customers = len(df)


    churned_customers = len(
        df[
            df["churn"] == "Yes"
        ]
    )


    churn_rate = (
        churned_customers
        / total_customers
        * 100
    )


    retained_customers = (
        total_customers
        - churned_customers
    )


    return render_template(
        "analysis.html",

        total_customers=total_customers,

        churned_customers=churned_customers,

        retained_customers=retained_customers,

        churn_rate=round(
            churn_rate,
            2
        )
    )


# --------------------------------------------------
# RUN APPLICATION
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        debug=True,
        port=5000
    )