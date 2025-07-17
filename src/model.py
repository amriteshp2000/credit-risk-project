# modeling.py

import joblib
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def get_catboost_model():
    return CatBoostClassifier(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        loss_function='Logloss',
        eval_metric='AUC',
        random_state=42,
        verbose=100,
        early_stopping_rounds=100
    )


def get_logistic_model():
    return LogisticRegression(
        max_iter=10000,
        random_state=42
    )


def get_xgboost_model():
    return XGBClassifier(
        use_label_encoder=False,
        eval_metric='auc',
        random_state=42
    )


def get_lightgbm_model():
    return LGBMClassifier(
        random_state=42
    )


def load_pretrained_models(model_dir="../models"):
    models = {}
    models["catboost"] = joblib.load(f"{model_dir}/catboost_model.pkl")
    models["logistic"] = joblib.load(f"{model_dir}/logistic_model.pkl")
    models["xgboost"] = joblib.load(f"{model_dir}/xgboost_model.pkl")
    models["lightgbm"] = joblib.load(f"{model_dir}/lightgbm_model.pkl")
    models["blended"] = joblib.load(f"{model_dir}/blended_model.pkl")
    return models


