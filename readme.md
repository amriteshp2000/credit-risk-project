# 📈 Credit Risk Prediction Project

A robust, production-ready machine learning project for predicting credit risk using real-world financial application data from the [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/) competition.

This project uses:

* Machine learning models (Logistic Regression, XGBoost, LightGBM, CatBoost)
* FastAPI for prediction API
* Streamlit for interactive UI
* Docker support for both FastAPI and Streamlit services

---

## 🔍 Project Structure Overview

```
credit-risk-project/
├── app/                      # FastAPI + Streamlit application logic
│   ├── components/           # Streamlit UI forms
│   │   └── main.py           # Streamlit form layout
│   └── main.py               # Unified entry point for API/UI
├── data/                     # Dataset
│   ├── raw/                  # Raw data (as downloaded)
│   └── processed/            # Processed feature & target data for modeling
├── models/                   # Trained ML models (pkl format)
├── notebooks/                # Exploratory & modeling notebooks
│   ├── preprocessing.ipynb   # Preprocessing logic (Top Kaggle approach)
│   ├── modeling.ipynb        # Training + evaluation of models
│   ├── shap_analysis.ipynb   # Model interpretability
├── src/                      # Python scripts for modular ML pipeline
│   ├── data_loader.py        # Data loading helpers
│   ├── feature_engineering.py# Feature creation logic
│   ├── model.py              # Model training + saving
│   └── utils.py              # General utilities
├── .dockerignore
├── Dockerfile.fastapi        # Dockerfile for FastAPI API server
├── Dockerfile.streamlit      # Dockerfile for Streamlit UI server
├── requirements.txt          # Python dependencies
├── README.md                 # Project overview & usage
```

---

## 🧰 Key Components

### `/models/`

Serialized trained models:

* `catboost_model.pkl`, `xgboost_model.pkl`, etc.
* `catboost_features.pkl`: stores the feature list used for CatBoost prediction.

### `/app/main.py`

* Unified script for running FastAPI (`uvicorn`) or Streamlit based on execution mode.
* `/predict` route for prediction API.
* Streamlit app renders form-based UI.

### `/app/components/main.py`

* Renders form inputs via `get_user_input_form()` function for Streamlit UI.

### `/src/`

* `feature_engineering.py`: domain-relevant feature generation.
* `data_loader.py`: read data from CSV/SQL.
* `model.py`: training and saving of models.
* `utils.py`: reusable helper functions.

### `/notebooks/`

* Exploratory and modeling notebooks with Kaggle-level insights.

---

## 🐳 Dockerized Setup

### Step 1: Build Docker Images

```bash
# Build FastAPI container
docker build -f Dockerfile.fastapi -t credit-risk-api .

# Build Streamlit container
docker build -f Dockerfile.streamlit -t credit-risk-ui .
```

### Step 2: Run Docker Containers

```bash
# Run FastAPI on port 8000
docker run -p 8000:8000 credit-risk-api

# Run Streamlit on port 8501
docker run -p 8501:8501 credit-risk-ui
```

### Access Points

* FastAPI Swagger: [http://localhost:8000/docs](http://localhost:8000/docs)
* Streamlit App: [http://localhost:8501](http://localhost:8501)

---

## 🧪 Local Development Setup (No Docker)

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/credit-risk-project.git
cd credit-risk-project
```

### 2. Setup Python Environment

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the App

```bash
# Run FastAPI server
python app/main.py serve

# Run Streamlit app
python app/main.py
```

---

## 🔐 Tips for Production Use

* Use `.env` to store sensitive keys and secrets
* Add authentication to API endpoints
* Deploy behind Nginx with HTTPS (e.g. using Docker Compose)
* Integrate CI/CD with GitHub Actions

---

## 🙏 Credits

* Dataset: Home Credit Default Risk - Kaggle
* Inspired by top Kaggle solutions
* Libraries: FastAPI, Streamlit, XGBoost, LightGBM, CatBoost, SHAP

---
