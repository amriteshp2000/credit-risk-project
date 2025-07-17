import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def engineer_application_features(df):
    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365
    df["EMPLOYMENT_YEARS"] = -df["DAYS_EMPLOYED"] / 365
    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["GOODS_CREDIT_RATIO"] = df["AMT_GOODS_PRICE"] / df["AMT_CREDIT"]
    df["PAYMENT_LENGTH"] = df["AMT_CREDIT"] / df["AMT_ANNUITY"]

    for col in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]:
        df[col + "_imp"] = df[col].fillna(df[col].median())

    ext_cols = ["EXT_SOURCE_1_imp", "EXT_SOURCE_2_imp", "EXT_SOURCE_3_imp"]
    df["EXT_SOURCES_MEAN"] = df[ext_cols].mean(axis=1)
    df["EXT_SOURCES_STD"] = df[ext_cols].std(axis=1)
    df["EXT_SOURCES_PROD"] = df[ext_cols].prod(axis=1)

    le = LabelEncoder()
    for col in df.select_dtypes("object").columns:
        if df[col].nunique() == 2:
            df[col] = le.fit_transform(df[col])
    df = pd.get_dummies(df, drop_first=True)

    return df

def merge_bureau_features(app_df, bureau_df):
    bb_agg = bureau_df.groupby("SK_ID_CURR").agg({
        "DAYS_CREDIT": "min",
        "DAYS_CREDIT_ENDDATE": "max",
        "CREDIT_DAY_OVERDUE": "mean",
        "AMT_CREDIT_SUM": "sum",
        "BB_STATUS_MEAN": "mean",
        "BB_COUNT": "sum"
    })
    return app_df.join(bb_agg, on="SK_ID_CURR")

def merge_previous_application_features(app_df, prev_df):
    prev_agg = prev_df.groupby("SK_ID_CURR").agg({
        "AMT_APPLICATION": "mean",
        "AMT_CREDIT": "mean",
        "AMT_DOWN_PAYMENT": "mean",
        "DAYS_DECISION": "min",
        "CNT_PAYMENT": "sum",
        "NAME_CONTRACT_TYPE": "nunique"
    })
    return app_df.join(prev_agg, on="SK_ID_CURR")
