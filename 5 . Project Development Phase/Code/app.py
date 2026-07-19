from flask import Flask, render_template, request, redirect, url_for
import joblib
import pandas as pd
import numpy as np
import traceback
import random
import datetime

# -----------------------------
# Create Flask App
# -----------------------------
app = Flask(__name__)

# -----------------------------
# Load Trained Model and Preprocessing Elements
# -----------------------------
try:
    model = joblib.load("models/model.pkl")
    encoders = joblib.load("models/encoder.pkl")
    scaler = joblib.load("models/scaler.pkl")
    print("[SUCCESS] Model, Encoders, and Scaler Loaded Successfully")
except Exception as e:
    print("[ERROR] Error loading model assets:", str(e))
    traceback.print_exc()

# -----------------------------
# Home Page
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")

# -----------------------------
# Prediction Page (Form & Engine)
# -----------------------------
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            # 1. Parse Input Form Data
            applicant_name = request.form.get("applicant_name", "Applicant")
            age = int(request.form.get("age", 35))
            loan_type = request.form.get("loan_type", "Home Loan")
            gender = request.form.get("gender", "Male")
            married = request.form.get("married", "No")
            dependents = request.form.get("dependents", "0")
            education = request.form.get("education", "Graduate")
            self_employed = request.form.get("self_employed", "No")
            applicant_income = float(request.form.get("applicant_income", 0))
            coapplicant_income = float(request.form.get("coapplicant_income", 0))
            loan_amount = float(request.form.get("loan_amount", 0))
            loan_term = float(request.form.get("loan_term", 360))
            credit_history_str = request.form.get("credit_history", "Good")
            property_area = request.form.get("property_area", "Urban")

            # Map credit history string to numeric value (1.0 or 0.0)
            credit_history_val = 1.0 if credit_history_str in ["Good", "Yes", "1", "1.0"] else 0.0

            # Scale inputs logically if they represent full amounts instead of dataset standards
            applicant_income_val = applicant_income
            if applicant_income_val > 15000:
                # Convert annual to monthly
                applicant_income_val = applicant_income_val / 12.0

            coapplicant_income_val = coapplicant_income
            if coapplicant_income_val > 15000:
                # Convert annual to monthly
                coapplicant_income_val = coapplicant_income_val / 12.0

            loan_amount_val = loan_amount
            if loan_amount_val >= 1000:
                # Convert full dollars to thousands
                loan_amount_val = loan_amount_val / 1000.0

            # Standardize loan term in months (if entered in years)
            loan_term_val = loan_term
            if loan_term <= 30:
                loan_term_val = loan_term * 12

            # 2. Label Encoding
            gender_encoded = encoders["Gender"].transform([gender])[0]
            married_encoded = encoders["Married"].transform([married])[0]
            dependents_encoded = encoders["Dependents"].transform([dependents])[0]
            education_encoded = encoders["Education"].transform([education])[0]
            self_employed_encoded = encoders["Self_Employed"].transform([self_employed])[0]
            property_area_encoded = encoders["Property_Area"].transform([property_area])[0]

            # 3. Assemble features in a pandas DataFrame
            feature_names = [
                "Gender", "Married", "Dependents", "Education", "Self_Employed",
                "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term",
                "Credit_History", "Property_Area"
            ]
            
            features_df = pd.DataFrame([[
                gender_encoded,
                married_encoded,
                dependents_encoded,
                education_encoded,
                self_employed_encoded,
                applicant_income_val,
                coapplicant_income_val,
                loan_amount_val,
                loan_term_val,
                credit_history_val,
                property_area_encoded
            ]], columns=feature_names)

            # 4. Handle model scaling if best model is KNN
            if type(model).__name__ == "KNeighborsClassifier":
                features_to_predict = scaler.transform(features_df)
            else:
                features_to_predict = features_df

            # 5. Run Model Prediction and Probabilities
            prediction = model.predict(features_to_predict)[0]
            probabilities = model.predict_proba(features_to_predict)[0]

            # 6. Calculate Metrics
            # Target encoders['Loan_Status'] maps index 0 to 'N' and index 1 to 'Y'
            eligibility_pct = round(probabilities[1] * 100, 2)
            confidence = round(max(probabilities) * 100, 2)
            prob_reject = probabilities[0]

            # 7. Determine Dynamic Risk Level
            if prediction == 0 or credit_history_val == 0 or prob_reject >= 0.5:
                risk_level = "High Risk"
                risk_class = "danger"
            elif prob_reject >= 0.25:
                risk_level = "Medium Risk"
                risk_class = "warning"
            else:
                risk_level = "Low Risk"
                risk_class = "success"

            # 8. Generate Reasons
            reasons = []
            if credit_history_val == 1.0:
                reasons.append("Good Credit History: The applicant has a reliable track record of repayment.")
            else:
                reasons.append("Bad Credit History: No credit record or previous delinquencies detected.")

            total_income = applicant_income_val + coapplicant_income_val
            if total_income >= 5000:
                reasons.append("Stable Income: Monthly household income is sufficient for standard payments.")
            elif total_income < 2500:
                reasons.append("Low Income: Household income is near low-tier brackets for this loan size.")
            else:
                reasons.append("Moderate Income: Monthly household income covers minimal living standards.")

            if total_income > 0:
                annual_income = total_income * 12
                loan_to_income = (loan_amount_val * 1000) / annual_income
                if loan_to_income > 4.5:
                    reasons.append(f"High Loan Amount: Requested loan is {loan_to_income:.1f}x the annual household earnings.")
                else:
                    reasons.append(f"Stable Debt Burden: Loan requested size is aligned with annual earnings.")

            if coapplicant_income_val > 0:
                reasons.append(f"Co-applicant Support: Co-applicant's monthly contribution of ${coapplicant_income:,.2f} strengthens the application.")

            if loan_term_val <= 180:
                reasons.append(f"Short Exposure Term: Repayment term of {int(loan_term_val)} months reduces overall risk duration.")
            elif loan_term_val > 360:
                reasons.append(f"Extended Repayment Term: Term length exceeds standard brackets, increasing long-term default exposure.")

            # 9. Formulate Recommendation Cards
            recommendations = []
            status_label = "Approved" if prediction == 1 else "Rejected"
            if status_label == "Approved":
                recommendations.append({
                    "title": "Approve Loan Request",
                    "text": "The applicant meets institutional thresholds. Recommended to proceed with standard processing.",
                    "icon": "fa-circle-check"
                })
                if credit_history_val == 1.0 and total_income >= 6000:
                    recommendations.append({
                        "title": "Qualifies for Prime Rates",
                        "text": "Given the exceptional credit history and high income, the applicant is eligible for prime lending rates.",
                        "icon": "fa-award"
                    })
                else:
                    recommendations.append({
                        "title": "Perform Income Audit",
                        "text": "Ensure thorough audit of paystubs and tax documents before final payout.",
                        "icon": "fa-clipboard-check"
                    })
            else:
                recommendations.append({
                    "title": "Improve Credit Rating",
                    "text": "Advise the client to clear outstanding debts and maintain timely repayments to rebuild credit.",
                    "icon": "fa-chart-line"
                })
                if credit_history_val == 0:
                    recommendations.append({
                        "title": "Demand Extra Collateral",
                        "text": "Require co-signers or additional physical collateral to secure the debt.",
                        "icon": "fa-vault"
                    })
                recommendations.append({
                    "title": "Adjust Loan Terms",
                    "text": "Recommend applying for a smaller loan size or extending the term to lower monthly debt payments.",
                    "icon": "fa-calculator"
                })

            # Formatting inputs for review layout
            inputs_display = {
                "applicant_name": applicant_name,
                "age": age,
                "loan_type": loan_type,
                "gender": gender,
                "married": married,
                "dependents": dependents,
                "education": education,
                "self_employed": self_employed,
                "applicant_income": f"${applicant_income:,.2f}",
                "coapplicant_income": f"${coapplicant_income:,.2f}",
                "loan_amount": f"${loan_amount:,.2f}",
                "loan_term": f"{int(loan_term_val)} Months",
                "credit_history": "Good" if credit_history_val == 1.0 else "Bad",
                "property_area": property_area
            }

            # Generate Applicant ID and map Result Status
            applicant_id = f"LP-{random.randint(100000, 999999)}"
            if status_label == "Approved" and risk_level == "Medium Risk":
                prediction_result = "Moderate Risk"
            elif status_label == "Approved":
                prediction_result = "Approved"
            else:
                prediction_result = "Rejected"
            
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Render Results Page with prediction data
            return render_template(
                "result.html",
                status=status_label,
                prediction_result=prediction_result,
                applicant_id=applicant_id,
                current_time=current_time,
                eligibility_pct=eligibility_pct,
                confidence=confidence,
                risk_level=risk_level,
                risk_class=risk_class,
                reasons=reasons,
                recommendations=recommendations,
                inputs=inputs_display
            )
        except Exception as e:
            traceback.print_exc()
            return render_template("predict.html", error=f"Prediction failed: {str(e)}")

    # GET request: render empty form page
    return render_template("predict.html")

# -----------------------------
# Result Page Redirector (if direct GET)
# -----------------------------
@app.route("/result")
def result():
    return redirect(url_for("predict"))

# -----------------------------
# Project Info Page (Redirect to About)
# -----------------------------
@app.route("/project")
def project():
    return redirect(url_for("about"))

# -----------------------------
# About Page (Merged Documentation)
# -----------------------------
@app.route("/about")
def about():
    return render_template("about.html")

# -----------------------------
# Dedicated Applicant Profile Page
# -----------------------------
@app.route("/profile")
def profile():
    return render_template("profile.html")

# -----------------------------
# History Page
# -----------------------------
@app.route("/history")
def history():
    return render_template("history.html")

# -----------------------------
# Run Flask App
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)