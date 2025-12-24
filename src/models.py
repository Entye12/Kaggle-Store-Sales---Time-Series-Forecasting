import numpy as np
import pandas as pd

class HybridModel:
    """
    Two-stage model:
      1) model1 predicts y_wide (multioutput), indexed by date
      2) model2 predicts residuals on long format (date, series_id)
    Supports log1p training for RMSLE alignment.
    """

    def __init__(self, model1, model2, transform="log1p", clip=True):
        self.model1 = model1
        self.model2 = model2
        self.transform = transform
        self.clip = clip

        self.y_columns_ = None
        self.fitted_ = False

    def _transform_y(self, y):
        if self.transform is None:
            return y
        if self.transform == "log1p":
            return np.log1p(y)
        raise ValueError(f"Unknown transform: {self.transform}")

    def _inverse_transform_y(self, y):
        if self.transform is None:
            return y
        if self.transform == "log1p":
            return np.expm1(y)
        raise ValueError(f"Unknown transform: {self.transform}")

    @staticmethod
    def _to_long(df_wide,level):
        # (date x series) -> (date, series) long series
        return df_wide.stack(level = level)
    
    @staticmethod
    def _to_wide(s_long, columns):
        # s_long index: (date, store_nbr, family)
        wide = s_long.unstack(level=["store_nbr", "family"])
        return wide.reindex(columns=columns)

    def fit(self, X1, X2, y_wide):
        """
        X1: indexed by date (PeriodIndex or DatetimeIndex)
        X2: indexed by (date, series_id) MultiIndex matching y_wide.stack()
        y_wide: DataFrame indexed by date, columns=series_id
        """
        self.y_columns_ = y_wide.columns

        y1 = self._transform_y(y_wide)

        # Stage 1: fit on wide y
        self.model1.fit(X1, y1)
        y1_fit = pd.DataFrame(
            self.model1.predict(X1),
            index=X1.index,
            columns=self.y_columns_
        )

        # Residuals in transformed space
        resid_wide = y1 - y1_fit
        resid_long = self._to_long(resid_wide, level=[0,1])
        resid_long.index = resid_long.index.set_names(["date","store_nbr","family"])

        # Align X2 to residuals
        common_idx = resid_long.index.intersection(X2.index)
        resid_long = resid_long.loc[common_idx]
        X2_aligned = X2.loc[common_idx]

        resid_long = resid_long.sort_index()
        X2_aligned = X2_aligned.sort_index()
    

        # Stage 2: fit on residuals
        self.model2.fit(X2_aligned, resid_long)

        self.fitted_ = True
        return self

    def predict(self, X1, X2):
        if not self.fitted_:
            raise RuntimeError("Call fit() before predict().")

        # Stage 1 prediction (wide, transformed)
        y1_pred = pd.DataFrame(
            self.model1.predict(X1),
            index=X1.index,
            columns=self.y_columns_
        )

        # Stage 2 prediction (long residuals)
        # Ensure X2 index corresponds to (date, series_id)
        resid2_pred = pd.Series(self.model2.predict(X2), index=X2.index)
        resid2_wide = self._to_wide(resid2_pred, self.y_columns_)

        # Combine in transformed space, inverse transform
        y_pred = self._inverse_transform_y(y1_pred + resid2_wide)

        if self.clip:
            y_pred = y_pred.clip(lower=0.0)

        return y_pred
