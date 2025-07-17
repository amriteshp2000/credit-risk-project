import pandas as pd

def load_application_data(path="../data/raw/application_train.csv"):
    df = pd.read_csv(path)
    df = df[df["CODE_GENDER"] != "XNA"]
    df["DAYS_EMPLOYED"].replace(365243, pd.NA, inplace=True)
    return df

def load_bureau_data(bureau_path="../data/raw/bureau.csv", bb_path="../data/raw/bureau_balance.csv"):
    bureau = pd.read_csv(bureau_path)
    bb = pd.read_csv(bb_path)

    status_map = {
        '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5,
        'C': 0, 'X': 0
    }
    bb["STATUS"] = bb["STATUS"].map(status_map)
    bb_agg = bb.groupby("SK_ID_BUREAU")["STATUS"].agg(['mean', 'size']).rename(
        columns={"mean": "BB_STATUS_MEAN", "size": "BB_COUNT"}
    )
    bureau = bureau.join(bb_agg, on="SK_ID_BUREAU")
    return bureau

def load_previous_application_data(path="../data/raw/previous_application.csv"):
    return pd.read_csv(path)

# Add more loaders here for POS_CASH, installments, credit_card if needed
