#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

from sklearn.metrics import mean_squared_error

import statsmodels.api as sm

import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import plotly.express as px

def train_test_split_ols(
    df,
    target_col='plaque_assay',
    additional_feature=None,
    n_splits=3,
    n_lags=3,
    start_index=0,
    plot=False
):
    def _add_lags(data, val, n_lags):
        for i in range(1, n_lags + 1):
            data[f'lag{i}_{val}'] = data[val].shift(i)
        return data

    def _prepare_data(df, train_idx, test_idx, with_additional=False):
        full = df.copy()

        # For plotting purposes
        train_raw = df.iloc[train_idx].copy()
        test_raw = df.iloc[test_idx].copy()
        
        # Add lags based on full data
        full = _add_lags(full, val=target_col, n_lags=n_lags)
        
        if with_additional and additional_feature:
            full = _add_lags(full, val=additional_feature, n_lags=n_lags)
            
        # fill NaN with mean of the column
        full = full.fillna(full.select_dtypes(include='number').mean())

        train = full.iloc[train_idx]
        test = full.iloc[test_idx]

        features = [f'lag{i}_{target_col}' for i in range(1, n_lags + 1)]
        if with_additional and additional_feature:
            features += [f'lag{i}_{additional_feature}' for i in range(1, n_lags + 1)]

        X_train = sm.add_constant(train[features], has_constant='add')
        X_test = sm.add_constant(test[features], has_constant='add')

        return X_train, train[target_col], X_test, test[target_col], test, train_raw, test_raw


    df = df[start_index:].sort_index()
    tss = TimeSeriesSplit(n_splits=n_splits)
    
    if plot:
        fig, axs = plt.subplots(n_splits, 2, figsize=(20, n_splits * 1.5), sharex=True)
    
    preds, scores = [], []

    final_rmses = []
    y_pred_values = []
    x_pred_values = []
    
    for col, use_additional in enumerate([False, True]):
        for fold, (train_idx, val_idx) in enumerate(tss.split(df)):
            _prepare_data(df, train_idx, val_idx, with_additional=False)
            X_train, y_train, X_test, y_test, test_filtered, train_raw, test_raw = _prepare_data(df, train_idx, val_idx, with_additional=use_additional)

            model = sm.OLS(y_train, X_train)
            results = model.fit()
            try:
                ypred = results.predict(X_test)
            except ValueError as e:
                print(f"Error in fold {fold} with additional feature {additional_feature}: {e}")
                print(f"Train columns: {X_train.columns}")
                print(f"Test columns: {X_test.columns}")
                print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
                print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
                continue
            preds.append(ypred)
            score = np.sqrt(mean_squared_error(y_test, ypred))
            scores.append(score)

            if plot:
                train_raw[target_col].plot(ax=axs[fold, col], label='Training Set', marker='o', linewidth=3)
                test_raw[target_col].plot(ax=axs[fold, col], label='Test Set', marker='D', linewidth=3)
                pd.Series(ypred, index=test_filtered.index).plot(ax=axs[fold, col], label='Predicted', marker='x', linewidth=3)


                axs[fold, col].set_ylabel("log10 PFU", fontsize=10)
                axs[fold, col].set_title(f'Data Train/Test Split Fold {fold} (RMSE: {round(score, 4)})')
                
            if fold == n_splits - 1:
                final_rmses.append(score)
                y_pred_values.append(ypred)
                x_pred_values.append(test_filtered.index)

    if plot:
        fig.suptitle(f'OLS Time Series Prediction of {target_col}\nwithout (left) and with (right) {additional_feature} as a feature (lag={n_lags})')
        plt.tight_layout()
        plt.show()

    return final_rmses, x_pred_values, y_pred_values

import statsmodels.api as sm

def add_lags(data, val, n_lags):
    for i in range(1, n_lags + 1):
        data[f'lag{i}_{val}'] = data[val].shift(i)
    return data

def prepare_data(full, train_idx, test_idx, target_col, n_lags, additional_feature=None, with_additional=False):
    train = full.iloc[train_idx].copy()
    test = full.iloc[test_idx].copy()
    
    means = train.select_dtypes(include="number").mean()
    train_df = train.fillna(means)
    test_df  = test.fillna(means)

    features = [f'lag{i}_{target_col}' for i in range(1, n_lags + 1)]
    if with_additional and additional_feature:
        features += [f'lag{i}_{additional_feature}' for i in range(1, n_lags + 1)]

    X_train = train[features]
    X_test = test[features]

    y_train = train[target_col]
    y_test = test[target_col]
    
    X_train = sm.add_constant(train[features], has_constant='add')
    X_test = sm.add_constant(test[features], has_constant='add')

    return X_train, y_train, X_test, y_test, train, test


def iterative_ols_comparison(
    df,
    target_col='plaque_assay',
    additional_feature=None,
    n_forecast=3,
    step=1,
    n_lags=3,
    start_index=0,
    plot=False
):
    df = df[start_index:].sort_index()

    full = df.copy()
    full = add_lags(full, val=target_col, n_lags=n_lags)
    if additional_feature:
        full = add_lags(full, val=additional_feature, n_lags=n_lags)
    full = full.fillna(full.select_dtypes(include='number').mean())
    
    if plot:
        fig, axs = plt.subplots(ncols=2, figsize=(20, 8))
        axs = axs.flatten()

    all_rmses = []
    pred_values = []
    
    for col, use_additional in enumerate([False, True]):
        preds, scores = [], []
        
        for fold in range(n_forecast):
            min_train_idx = 0
            train_idx = np.arange(min_train_idx, len(df) - (n_forecast - fold) * step)
            # train_idx = np.arange(0, len(df) - (n_forecast - fold) * step)
            val_idx = np.arange(len(df) - (n_forecast - fold) * step, len(df) - (n_forecast - fold - 1) * step)
            if len(val_idx) == 0 or len(train_idx) == 0:
                break
            
            X_train, y_train, X_test, y_test, train, test = prepare_data(
                full, train_idx, val_idx, target_col, n_lags, additional_feature, with_additional=use_additional
            )

            model = sm.OLS(y_train, X_train)
            results = model.fit()
            ypred = results.predict(X_test)

            preds.append(ypred)
            score = np.sqrt(mean_squared_error(y_test, ypred))
            scores.append(score)
                
            if fold == n_forecast - 1:
                real_data = pd.concat([train[target_col],test[target_col]])
                prediction_data = pd.concat(preds).sort_index()
                total_rmse = np.sqrt(mean_squared_error(real_data.iloc[-n_forecast:].copy(), prediction_data))
                all_rmses.append(scores)
                pred_values.append(preds)
                
                if plot:
                    real_data.plot(ax=axs[col], label='Training Set', marker='o', linewidth=3)
                    axs[col].plot(prediction_data, marker='D', linewidth=3, label='Predicted values')
                    axs[col].set_ylabel("log10 PFU", fontsize=10)
                    axs[col].set_title(f'Data Train/Test Split Fold {fold} (RMSE: {round(total_rmse, 4)})')
                

    if plot:
        fig.suptitle(f'OLS Time Series Prediction of {target_col}\nwithout (left) and with (right) {additional_feature} as a feature (lag={n_lags})')
        plt.tight_layout()
        plt.show()

    return all_rmses, pred_values

def iterative_ols(
    df,
    target_col='plaque_assay',
    additional_feature=None,
    n_forecast=3,
    step=1,
    n_lags=3,
    start_index=0,
    plot=None
):
    """
    Walk‐forward or block OLS forecasting.
    - If step=1, does one‐step‐ahead forecasting, retraining each day on true data.
    - If step>1, trains once on the “train” block and forecasts `step` points in one shot,
        feeding back its own predictions for intra‐block lags.

    Returns:
      all_rmses: list of floats (one RMSE per fold)
      pred_values: list of pd.Series (each fold’s predictions; last entry is final fold’s series)
      total_rmse: float or None (RMSE of the final fold)
      fig: Plotly Figure or None (if plotting was requested)
    """
    # 1) Truncate & sort
    df = df.iloc[start_index:].sort_index()
    if df.empty:
        return [], [], None, (go.Figure() if plot else None)

    # 2) Determine how many folds (ceiling) so that we cover all n_forecast days
    n_folds = math.ceil(n_forecast / step)

    all_rmses = []
    pred_values = []
    use_additional = (additional_feature is not None)

    # 3) Prepare figure if requested
    if plot is None:
        fig = go.Figure()
    else:
        fig = plot

    # 4) Loop over folds
    for fold in range(n_folds):
        # (a) How many days remain to forecast?
        already_predicted = fold * step
        remaining = n_forecast - already_predicted
        if remaining <= 0:
            break

        # (b) Compute train_end, train_idx, test_idx
        train_end  = len(df) - remaining
        train_idx  = np.arange(0, train_end)
        test_size  = min(step, remaining)
        test_start = train_end
        test_end   = train_end + test_size
        test_idx   = np.arange(test_start, test_end)

        if train_idx.size == 0 or test_idx.size == 0:
            break

        # (c) Build true history up to train_end - 1
        history_target = df[target_col].iloc[train_idx].copy()
        if use_additional:
            history_add = df[additional_feature].iloc[train_idx].copy()
        else:
            history_add = None

        # (d) Construct training design matrix (lags)
        records = []
        y_train = []
        valid_indices = history_target.index[n_lags:]  # only rows having all n_lags available
        for i, idx_label in enumerate(valid_indices):
            row = {}
            # target lags
            for lag_i in range(1, n_lags + 1):
                row[f'lag{lag_i}_{target_col}'] = history_target.iloc[i + (n_lags - lag_i)]
            # additional_feature lags
            if use_additional:
                for lag_i in range(1, n_lags + 1):
                    row[f'lag{lag_i}_{additional_feature}'] = history_add.iloc[i + (n_lags - lag_i)]
            records.append(row)
            y_train.append(history_target.loc[idx_label])

        if not records:
            # Not enough data to build even one lag row
            break

        X_train_df = pd.DataFrame(records, index=valid_indices)
        X_train = sm.add_constant(X_train_df, has_constant='add')
        y_train = pd.Series(y_train, index=valid_indices)

        # (e) Fit OLS on the training block
        model = sm.OLS(y_train, X_train)
        results = model.fit()

        # (f) Multi‐step forecast within this fold
        preds_block = []
        test_indices = df.index[test_idx]  # actual index labels for the block
        hist_targ = history_target.copy()
        if use_additional:
            hist_add = history_add.copy()

        for h in range(test_size):
            idx_to_predict = test_indices[h]
            row = {}
            # target lags from hist_targ
            if len(hist_targ) < n_lags:
                raise ValueError(
                    f"Not enough lag history to forecast fold={fold}, step={h}."
                )
            for lag_i in range(1, n_lags + 1):
                row[f'lag{lag_i}_{target_col}'] = hist_targ.iloc[-lag_i]
            # additional_feature lags (we assume true future is known)
            if use_additional:
                for lag_i in range(1, n_lags + 1):
                    prev_loc = df.index.get_loc(idx_to_predict) - lag_i
                    lag_label = df.index[prev_loc]
                    row[f'lag{lag_i}_{additional_feature}'] = df.loc[lag_label, additional_feature]

            X_next = pd.DataFrame([row], index=[idx_to_predict])
            X_next = sm.add_constant(X_next, has_constant='add')
            yhat = results.predict(X_next).iloc[0]
            preds_block.append(yhat)

            # Instead of hist_targ = hist_targ.append(...), use loc:
            hist_targ.loc[idx_to_predict] = yhat
            if use_additional:
                hist_add.loc[idx_to_predict] = df.loc[idx_to_predict, additional_feature]

        # (g) Compute RMSE for this fold
        true_vals = df[target_col].loc[test_indices].values
        pred_vals = np.array(preds_block)
        fold_rmse = np.sqrt(mean_squared_error(true_vals, pred_vals))

        all_rmses.append(fold_rmse)
        pred_series = pd.Series(preds_block, index=test_indices)
        pred_values.append(pred_series)

        # (h) Plot real vs. predicted if last fold and plotting requested
        if (plot is not None):
            real_idx  = history_target.index.append(test_indices)
            real_vals = pd.concat([history_target, df[target_col].loc[test_indices]])
            if (fold == n_folds - 1):
                preds = [p.values[0] for p in pred_values]
                tps = [p.index[0] for p in pred_values]
                fig.add_trace(go.Scatter(
                    x=real_idx,
                    y=real_vals.values,
                    mode='lines+markers',
                    name='Real data',
                    line=dict(color='black', width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=tps,
                    y=preds,
                    mode='lines+markers',
                    name='Forecasted data',
                    line=dict(color='blue' if use_additional else 'orange', width=2)
                ))
                if use_additional:
                    title = f"Forecast with '{additional_feature}'"
                else:
                    title = "Forecast with restricted model"
                
                # total_rmse = np.sqrt(mean_squared_error(real_vals[tps].values, preds))
                title += f" n_forecast={n_forecast}, step={step}, n_lags={n_lags}"

                fig.update_layout(
                    title=title,
                    xaxis_title="Index",
                    yaxis_title=target_col
                )
                pred_traces = [trace for trace in fig.data if trace.name == 'Pred points']
                other_traces = [trace for trace in fig.data if trace.name != 'Pred points']
                # Put prediction traces on top
                fig.data = tuple(other_traces + pred_traces)
                
            fig.add_trace(go.Scatter(
                x=test_indices,
                y=pred_vals,
                mode='markers',
                name='Pred points',
                marker=dict(
                    color=list(range(len(test_indices))),
                    colorscale='Portland',
                    showscale=False,
                    size=8
                ),
                hovertemplate=(
                    "Index: %{x}<br>"
                    "Prediction: %{y}<br>"
                    f"Fold RMSE: {fold_rmse:.4f}<br>"
                ),
                showlegend=False
            ))

    # 5) Return final results
    total_rmse = np.mean(all_rmses) if all_rmses else None
    last_pred_series = pred_values if pred_values else None
    return all_rmses, last_pred_series, total_rmse, (fig if plot else None)


def iterative_ols_all_dips(
    summary_df,
    ts_data,
    target_col='plaque_assay',
    lags=[2, 3],
    n_forecast=20,
    step=1,
    start_index=0,
    plot=False
):
    import plotly.subplots as sp

    for lag in lags:
        for col in [f'pred_restr_lag_{lag}', f'pred_full_lag_{lag}',
                    f'rmse_restr_lag_{lag}', f'rmse_full_lag_{lag}',
                    f'total_rmse_restr_lag_{lag}', f'total_rmse_full_lag_{lag}']:
            summary_df[col] = None

        # Prepare restricted model once (common for all dips)
        restr_rmse, restr_pred_values, restr_total_rmse, _ = iterative_ols(
            ts_data,
            target_col=target_col,
            additional_feature=None,
            n_forecast=n_forecast,
            step=step,
            n_lags=lag,
            start_index=start_index,
            plot=None
        )

        for rowid, row in summary_df.iterrows():
            dip = row['key']  # each dip acts as an additional feature

            if plot:
                restr_rmse, restr_pred_values, restr_total_rmse, fig_restr = iterative_ols(
                    ts_data,
                    target_col=target_col,
                    additional_feature=None,
                    n_forecast=n_forecast,
                    step=step,
                    n_lags=lag,
                    start_index=start_index,
                    plot=go.Figure()
                )

            full_rmses, full_pred_values, full_total_rmse, fig_full = iterative_ols(
                ts_data,
                target_col=target_col,
                additional_feature=dip,
                n_forecast=n_forecast,
                step=step,
                n_lags=lag,
                start_index=start_index,
                plot=go.Figure() if plot else None
            )
            # Save values
            summary_df.at[rowid, f'rmse_restr_lag_{lag}'] = restr_rmse
            summary_df.at[rowid, f'pred_restr_lag_{lag}'] = restr_pred_values
            summary_df.at[rowid, f'total_rmse_restr_lag_{lag}'] = restr_total_rmse

            summary_df.at[rowid, f'rmse_full_lag_{lag}'] = full_rmses
            summary_df.at[rowid, f'pred_full_lag_{lag}'] = full_pred_values
            summary_df.at[rowid, f'total_rmse_full_lag_{lag}'] = full_total_rmse

            if plot:
                fig = sp.make_subplots(rows=1, cols=2, subplot_titles=(f"Restricted (RMSE={round(restr_total_rmse, 4)})",
                                                                    f"Full (RMSE={round(full_total_rmse, 4)})"))
                
                for trace in fig_restr.data:
                    fig.add_trace(trace, row=1, col=1)
                    
                for trace in fig_full.data:
                    fig.add_trace(trace, row=1, col=2)
                    
                fig.update_layout(title_text=f"Comparison for lag={lag} and dip={dip}"
                                )
                fig.show()

    return summary_df