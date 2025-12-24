import pandas as pd
from src.utils import *

# ---------------------------------------------------------------------
# Calendar features
# ---------------------------------------------------------------------

def add_calendar_features(df, date_level="date"):
    """
    Add basic calendar features based on the date index level.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame indexed by MultiIndex including `date_level`.
    date_level : str
        Name of the date level in the index.

    Returns
    -------
    pd.DataFrame
        DataFrame with added calendar features.
    """
    dates = df.index.get_level_values(date_level)

    if isinstance(dates, pd.PeriodIndex):
        dates_dt = dates.to_timestamp()
    else:
        dates_dt = pd.DatetimeIndex(dates)

    out = df.copy()
    out["dayofweek"] = dates_dt.dayofweek.astype("int8")
    out["week"] = dates_dt.isocalendar().week.astype("int16").to_numpy()
    out["month"] = dates_dt.month.astype("int8")
    out["day"] = dates_dt.day.astype("int8")
    out["is_weekend"] = (out["dayofweek"] >= 5).astype("int8")

    # Payday: 15th or last day of month
    last_day = dates_dt.days_in_month.astype("int8")
    out["is_payday"] = ((out["day"] == 15) | (out["day"] == last_day)).astype("int8")

    return out


# ---------------------------------------------------------------------
# Oil
# ---------------------------------------------------------------------

def join_oil(df, oil, date_level="date"):
    """
    Join oil prices to a MultiIndex DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Indexed by MultiIndex including `date_level`.
    oil : pd.DataFrame
        Indexed by date (PeriodIndex), column `dcoilwtico`.
    date_level : str
        Name of the date level in df.

    Returns
    -------
    pd.DataFrame
        DataFrame with oil price column added.
    """
    out = df.copy()
    dates = out.index.get_level_values(date_level)
    out["oil"] = oil.loc[dates, "dcoilwtico"].to_numpy()
    return out


# ---------------------------------------------------------------------
# Holidays
# ---------------------------------------------------------------------

def build_holiday_features(holidays, start=None, end=None):
    """
    Build daily holiday indicator features.

    Parameters
    ----------
    holidays : pd.DataFrame
        holidays_events.csv loaded and indexed by date.
    start, end : optional
        Date range to filter holidays.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by date with column 'is_holiday'.
    """
    hol = holidays.copy()

    # Keep only observed holidays
    hol = hol[hol["transferred"] == False]

    # Binary indicator
    hol["is_holiday"] = 1

    # One row per date
    hol = hol.groupby(level=0)["is_holiday"].max().to_frame()

    # Restrict time range
    if start is not None:
        hol = hol.loc[start:]
    if end is not None:
        hol = hol.loc[:end]

    return hol



def join_holidays(df, holiday_features, date_level="date"):
    """
    Join holiday indicators to a MultiIndex DataFrame.

    Missing dates are filled with 0 (non-holiday).

    Parameters
    ----------
    df : pd.DataFrame
        Indexed by MultiIndex including date.
    holiday_features : pd.DataFrame
        Indexed by date.
    date_level : str

    Returns
    -------
    pd.DataFrame
    """
    out = df.copy()
    dates = out.index.get_level_values(date_level)
    out["is_holiday"] = holiday_features.reindex(dates, fill_value=0).to_numpy()
    return out



# ---------------------------------------------------------------------
# Stores
# ---------------------------------------------------------------------


def join_stores(df, stores):
    """
    Join stores.csv metadata on store_nbr.
    df is indexed by MultiIndex including store_nbr.
    """
    out = df.copy()
    store = out.index.get_level_values("store_nbr")
    out = out.reset_index()  # bring store_nbr into columns for merge
    out = out.merge(stores, on="store_nbr", how="left")
    out = out.set_index(["date", "store_nbr", "family"]).sort_index()
    return out



# ---------------------------------------------------------------------
# Feature Builder
# ---------------------------------------------------------------------


class FeatureBuilder:
    def __init__(self, oil, holidays, stores, y_lags=(1, 8), promo_lags=(1, 7), alpha=20.0):
        self.oil = oil
        self.holidays = holidays
        self.stores = stores
        self.y_lags = list(y_lags)
        self.promo_lags = list(promo_lags)
        self.alpha = float(alpha)

        # fitted state
        self.global_mean_ = None
        self.store_smooth_ = None
        self.family_smooth_ = None
        self.cat_levels_ = {}

    def _base(self, df):
        # expects df indexed by (date, store_nbr, family)

        # ------------------------------------------------------------
        # Base data : oil, holidays dummies, calendar features and stores.
        # ------------------------------------------------------------
        X = df[["onpromotion"]].copy()
        X = add_calendar_features(X, date_level="date")
        X = join_oil(X, self.oil, date_level="date")

        dates = X.index.get_level_values("date")
        holiday_feats = build_holiday_features(self.holidays, start=dates.min(), end=dates.max())
        X = join_holidays(X, holiday_feats, date_level="date")

        X = join_stores(X, self.stores)
        X.index = X.index.set_names(["date", "store_nbr", "family"])
        return X

    def fit(self, train_df):
        # train_df contains sales + onpromotion and is indexed (date, store_nbr, family)

        # ------------------------------------------------------------
        # Add dynamic features: sales lags + rolling stats + promo dynamics
        # ------------------------------------------------------------
        ylog = np.log1p(train_df["sales"]).rename("ylog")

        # fit smoothed target stats (train-only!)
        gm = ylog.mean()
        self.global_mean_ = gm

        store_stats = ylog.groupby(level="store_nbr").agg(["mean", "count"])
        fam_stats   = ylog.groupby(level="family").agg(["mean", "count"])
        a = self.alpha

        self.store_smooth_ = (store_stats["mean"] * store_stats["count"] + gm * a) / (store_stats["count"] + a)
        self.family_smooth_ = (fam_stats["mean"] * fam_stats["count"] + gm * a) / (fam_stats["count"] + a)

        # store category levels for stable dtype across splits (optional but helps)
        self.cat_levels_["store_nbr"] = train_df.index.get_level_values("store_nbr").unique()
        self.cat_levels_["family"] = train_df.index.get_level_values("family").unique()

        return self

    def transform(self, df, *, ylog_history=None, promo_all=None):
        """
        df: dataframe indexed (date, store_nbr, family) with at least onpromotion.
        ylog_history: Series indexed (date, store_nbr, family) containing log1p(sales) for lag features.
                      For train/valid this can be ylog from the same split (or full train with careful slicing).
                      For test this should be full train ylog so lags come from last observed sales.
        promo_all: Series indexed (date, store_nbr, family) of onpromotion across train+test if you want future-known promo lags.
        """
        X = self._base(df)

        # stable categorical columns (optional)
        X["store_nbr_cat"] = X.index.get_level_values("store_nbr").astype("category")
        X["family_cat"] = X.index.get_level_values("family").astype("category")

        # handle store metadata categories (your pattern)
        for c in ["city", "state", "type", "cluster"]:
            if c in X.columns:
                X[c] = X[c].astype("category").cat.add_categories("UNK").fillna("UNK")

        # target stats (from fit)
        store_idx = X.index.get_level_values("store_nbr")
        fam_idx = X.index.get_level_values("family")
        X["store_avg"] = store_idx.map(self.store_smooth_).astype("float32").fillna(self.global_mean_)
        X["family_avg"] = fam_idx.map(self.family_smooth_).astype("float32").fillna(self.global_mean_)

        # sales lags (must come from ylog_history; for test use train history)
        if ylog_history is not None:
            ylog_history = ylog_history.rename("ylog").reorder_levels(["date","store_nbr","family"]).sort_index()
            X = X.join(make_group_lags(ylog_history, lags=self.y_lags, name="ylog"))

        # promo lags (ok to use train+test because promo is known in advance)
        if promo_all is not None:
            promo_all = promo_all.rename("promo").reorder_levels(["date","store_nbr","family"]).sort_index()
            promo_lags = make_group_lags(promo_all, lags=self.promo_lags, name="promo")
            X = X.join(promo_lags.reindex(X.index))
            promo_cols = [c for c in X.columns if "promo_" in c]
            X[promo_cols] = X[promo_cols].fillna(0)

        return X

