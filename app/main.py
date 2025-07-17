import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from pydantic import BaseModel
from catboost import CatBoostClassifier
import pandas as pd
import joblib
import uvicorn
import streamlit as st
import re
from components.form import get_user_input_form  # ✅ Correct import

import nest_asyncio

# ---------------------------- MODEL SETUP ----------------------------
MODEL_PATH = "models/catboost_model.pkl"
FEATURE_NAMES_PATH = "models/catboost_features.pkl"

model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURE_NAMES_PATH)

# ---------------------------- FASTAPI SETUP ----------------------------
app = FastAPI(title="Credit Risk Predictor")
nest_asyncio.apply()  # ✅ Needed when running FastAPI inside same thread

class UserData(BaseModel):
    data: dict

@app.get("/")
def root():
    return {"message": "✅ FastAPI is running!"}

@app.post("/predict")
def predict(user_data: UserData):
    try:
        input_data = user_data.data
        df = pd.DataFrame([input_data])
        df.columns = df.columns.str.replace(r"[^\w]", "_", regex=True)

        for col in feature_names:
            if col not in df.columns:
                df[col] = 0

        df = df[feature_names]
        pred = model.predict(df)[0]
        prob = model.predict_proba(df)[0][1]

        return {
            "prediction": int(pred),
            "probability": round(float(prob), 4)
        }

    except Exception as e:
        return {"error": str(e)}

# ---------------------------- STREAMLIT APP ----------------------------
def run_streamlit_app():
    st.set_page_config(page_title="Credit Risk Prediction", layout="wide")
    st.title("💳 Credit Risk Prediction App")

    # ✅ Correct function name
    user_input = get_user_input_form()

    if st.button("Predict Credit Risk"):
        input_dict = {feature: 0 for feature in feature_names}
        raw_input = user_input.to_dict(orient="records")[0]
        sanitized_input = {
            re.sub(r"[^\w]", "_", k): v for k, v in raw_input.items()
        }
        input_dict.update(sanitized_input)

        user_df = pd.DataFrame([input_dict], columns=feature_names)
        user_df = user_df.reindex(columns=feature_names, fill_value=0)

        prediction = model.predict(user_df)[0]
        probability = model.predict_proba(user_df)[0][1]

        st.success(f"✅ Prediction: {'High Risk' if prediction == 1 else 'Low Risk'}")
        st.info(f"📊 Probability of Default: {probability:.2%}")

# ---------------------------- ENTRY POINT ----------------------------
if __name__ == "__main__":
    if "serve" in sys.argv:
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    else:
        run_streamlit_app()
