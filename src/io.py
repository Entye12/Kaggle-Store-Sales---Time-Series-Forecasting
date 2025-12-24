import pandas as pd
from pathlib import Path
import sys

ROOT = Path.cwd()
if (ROOT / "src").exists() is False and (ROOT.parent / "src").exists():
    ROOT = ROOT.parent

sys.path.insert(0, str(ROOT))


def load_train(path= ROOT / "Data/train.csv"):
    df = pd.read_csv(
        path,
        usecols=["store_nbr", "family", "date", "sales", "onpromotion"],
        dtype={
            "store_nbr": "category",
            "family": "category",
            "sales": "float32",
            "onpromotion": "uint32",
        },
        parse_dates=["date"],
    )
    df["date"] = df["date"].dt.to_period("D")
    return df.set_index(["store_nbr", "family", "date"]).sort_index()


def load_test(path=ROOT / "Data/test.csv"):
    df = pd.read_csv(
        path,
        usecols=["store_nbr", "family", "date", "onpromotion"],
        dtype={
            "store_nbr": "category",
            "family": "category",
            "onpromotion": "uint32",
        },
        parse_dates=["date"],
    )
    df["date"] = df["date"].dt.to_period("D")
    return df.set_index(["store_nbr", "family", "date"]).sort_index()


def load_oil(path=ROOT / "Data/oil.csv"):
    oil = pd.read_csv(path, parse_dates=["date"])
    oil["date"] = oil["date"].dt.to_period("D")
    oil = oil.set_index("date").sort_index()

    
    full_idx = pd.period_range(oil.index.min(), oil.index.max(), freq="D")
    oil = oil.reindex(full_idx)

    oil["dcoilwtico"] = oil["dcoilwtico"].interpolate().ffill().bfill()
    return oil



def load_holidays(path= ROOT / "Data/holidays_events.csv"):
    hol = pd.read_csv(path, parse_dates=["date"])
    hol["date"] = hol["date"].dt.to_period("D")
    return hol.set_index("date").sort_index()



def load_stores(path=ROOT / "Data/stores.csv"):
    df = pd.read_csv(path)
    df["store_nbr"] = df["store_nbr"].astype("category")
    for c in ["city", "state", "type"]:
        if c in df.columns:
            df[c] = df[c].astype("category")
    return df

