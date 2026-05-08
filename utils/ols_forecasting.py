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
    def add_lags(data, val, n_lags):
        for i in range(1, n_lags + 1):
            data[f'lag{i}_{val}'] = data[val].shift(i)
        return data

    def prepare_data(df, train_idx, test_idx, with_additional=False):
        full = df.copy()

        # For plotting purposes
        train_raw = df.iloc[train_idx].copy()
        test_raw = df.iloc[test_idx].copy()
        
        # Add lags based on full data
        full = add_lags(full, val=target_col, n_lags=n_lags)
        
        if with_additional and additional_feature:
            full = add_lags(full, val=additional_feature, n_lags=n_lags)
            
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
            prepare_data(df, train_idx, val_idx, with_additional=False)
            X_train, y_train, X_test, y_test, test_filtered, train_raw, test_raw = prepare_data(df, train_idx, val_idx, with_additional=use_additional)

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
    train = full.iloc[train_idx]
    test = full.iloc[test_idx]

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
    df = df[start_index:].sort_index()
    full = df.copy()
    full = add_lags(full, val=target_col, n_lags=n_lags)
    
    if additional_feature:
        full = add_lags(full, val=additional_feature, n_lags=n_lags)
    
    full = full.fillna(full.select_dtypes(include='number').mean())

    all_rmses = []
    pred_values = []
    preds, scores = [], []
    use_additional = additional_feature is not None

    if plot is None:
        fig = go.Figure()
    else:
        fig = plot
    prediction_scatters = None
    
    n_forecast = int(n_forecast / step)
    
    for fold in range(n_forecast):
        train_idx = np.arange(0, len(df) - (n_forecast - fold) * step)
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
            real_data = pd.concat([train[target_col], test[target_col]])
            prediction_data = pd.concat(preds).sort_index()
            len_pred = len(prediction_data)
            total_rmse = np.sqrt(mean_squared_error(real_data.iloc[-len_pred:], prediction_data))
            all_rmses.append(scores)
            pred_values.append(prediction_data)
            if plot is not None:
                fig.add_trace(go.Scatter(
                    x=real_data.index, y=real_data.values,
                    mode='lines+markers', name='Real data',
                    line=dict(color='black', width=2)
                ))

                fig.add_trace(go.Scatter(
                    x=prediction_data.index, y=prediction_data.values,
                    mode='lines+markers',
                    name='Iteratively predicted data',
                    line=dict(color='blue' if use_additional else 'orange', width=2)
                ))
                for ypred, score in zip(preds, scores):
                    fig.add_trace(go.Scatter(
                        x=ypred.index, y=ypred, 
                        mode='markers', name='Predicted',
                        marker=dict(
                            color=ypred.index,
                            colorscale='Portland',
                            showscale=False,
                            size=8
                        ),
                        hovertemplate=f'RMSE: {round(score, 4)}<br>dpi: %{{x}}<br>prediction: %{{y}}',
                        showlegend=False)
                    )

                title = f"Iterative forecast with OLS"
                if additional_feature:
                    title += f" and {additional_feature}"
                title += f" (lag={n_lags}) - RMSE: {round(total_rmse, 4)}"
                fig.update_layout(title=title, yaxis_title="log10 PFU")

            return all_rmses, pred_values, total_rmse, fig

    return all_rmses, pred_values, total_rmse, fig if plot else None

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