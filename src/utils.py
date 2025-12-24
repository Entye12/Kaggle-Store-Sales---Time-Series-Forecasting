import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def _as_list(x, start=1):
    if isinstance(x, int):
        return list(range(start, x + start))
    return list(x)


def make_lags(ts, lags, lead_time=1, name='y'):
    lags = _as_list(lags, start=lead_time)
    return pd.concat({f'{name}_lag_{i}': ts.shift(i) for i in lags}, axis=1)

def make_leads(ts, leads, name='y'):
    leads = _as_list(leads, start=1)
    return pd.concat({f'{name}_lead_{i}': ts.shift(-i) for i in reversed(leads)}, axis=1)


def make_group_lags(ts, lags, group_levels=None, date_level='date', lead_time=1, name='y'):
    """
    Create lag features within each group (e.g., store_nbr, family).

    ts: pd.Series indexed by MultiIndex including date_level.
    group_levels: levels to group by (defaults to all levels except date_level).
    """
    if not isinstance(ts.index, pd.MultiIndex):
        return make_lags(ts, lags, lead_time=lead_time, name=name)

    lags = _as_list(lags, start=lead_time)

    if group_levels is None:
        group_levels = [lvl for lvl in ts.index.names if lvl != date_level]

    g = ts.groupby(level=group_levels, sort=False)
    out = {f'{name}_lag_{i}': g.shift(i) for i in lags}
    return pd.concat(out, axis=1)

def make_group_leads(ts, leads, group_levels=None, date_level='date', name='y'):
    """
    Create lead features within each group (valid for known future exogenous variables).
    """
    if not isinstance(ts.index, pd.MultiIndex):
        return make_leads(ts, leads, name=name)

    leads = _as_list(leads, start=1)

    if group_levels is None:
        group_levels = [lvl for lvl in ts.index.names if lvl != date_level]

    g = ts.groupby(level=group_levels, sort=False)
    out = {f'{name}_lead_{i}': g.shift(-i) for i in reversed(leads)}
    return pd.concat(out, axis=1)

import pandas as pd

def make_group_rolling(ts, windows, group_levels=None, date_level="date", shift=1, name="y"):
    """
    Rolling stats within group. shift=1 avoids using current-day target.
    Works robustly with MultiIndex and avoids duplicate level-name issues.
    """
    if not isinstance(ts.index, pd.MultiIndex):
        windows = [windows] if isinstance(windows, int) else list(windows)
        X = pd.DataFrame(index=ts.index)
        for w in windows:
            X[f"{name}_roll_mean_{w}"] = ts.shift(shift).rolling(w).mean()
            X[f"{name}_roll_std_{w}"]  = ts.shift(shift).rolling(w).std()
        return X

    windows = [windows] if isinstance(windows, int) else list(windows)

    if group_levels is None:
        group_levels = [lvl for lvl in ts.index.names if lvl != date_level]

    # shift within each group
    shifted = ts.groupby(level=group_levels, sort=False).shift(shift)

    out = {}
    for w in windows:
        r = shifted.groupby(level=group_levels, sort=False).rolling(w)

        mean = r.mean()
        std  = r.std()

        # After groupby+rolling, index becomes (group_levels..., original_index...)
        # Drop the prepended group_levels by position (0..len(group_levels)-1)
        drop_n = len(group_levels)
        mean.index = mean.index.droplevel(list(range(drop_n)))
        std.index  = std.index.droplevel(list(range(drop_n)))

        out[f"{name}_roll_mean_{w}"] = mean
        out[f"{name}_roll_std_{w}"]  = std

    return pd.concat(out, axis=1)


def make_multistep_target(ts, steps, reverse=False, group_levels=None, date_level='date', name='y'):
    """
    Multi-step target within each group (for direct forecasting).
    """
    shifts = list(reversed(range(steps))) if reverse else list(range(steps))
    if isinstance(ts.index, pd.MultiIndex):
        if group_levels is None:
            group_levels = [lvl for lvl in ts.index.names if lvl != date_level]
        g = ts.groupby(level=group_levels, sort=False)
        return pd.concat({f'{name}_step_{i+1}': g.shift(-i) for i in shifts}, axis=1)
    else:
        return pd.concat({f'{name}_step_{i+1}': ts.shift(-i) for i in shifts}, axis=1)
    


def create_multistep_example(n, steps, lags, lead_time=1):
    ts = pd.Series(
        np.arange(n),
        index=pd.period_range(start='2010', freq='A', periods=n, name='Year'),
        dtype=pd.Int8Dtype,
    )
    X = make_lags(ts, lags, lead_time)
    y = make_multistep_target(ts, steps, reverse=True)
    data = pd.concat({'Targets': y, 'Features': X}, axis=1)
    data = data.style.set_properties(['Targets'], **{'background-color': 'LavenderBlush'}) \
                     .set_properties(['Features'], **{'background-color': 'Lavender'})
    return data


def load_multistep_data():
    df1 = create_multistep_example(10, steps=1, lags=3, lead_time=1)
    df2 = create_multistep_example(10, steps=3, lags=4, lead_time=2)
    df3 = create_multistep_example(10, steps=3, lags=4, lead_time=1)
    return [df1, df2, df3]
