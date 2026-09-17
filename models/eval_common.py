import pandas as pd
import numpy as np
import os
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import clone

def get_timeseries_cv_predictions(model, X, y, race_groups, n_splits=5):
    unique_races = race_groups.drop_duplicates().reset_index(drop=True)
    tss = TimeSeriesSplit(n_splits=n_splits)

    y_proba = np.full(len(X), np.nan)

    for train_race_idx, test_race_idx in tss.split(unique_races):
        train_races = set(unique_races.iloc[train_race_idx])
        test_races = set(unique_races.iloc[test_race_idx])

        train_mask = race_groups.isin(train_races).values
        test_mask = race_groups.isin(test_races).values

        model_clone = clone(model)
        model_clone.fit(X[train_mask], y[train_mask])
        y_proba[test_mask] = model_clone.predict_proba(X[test_mask])[:, 1]

    return y_proba

def get_race_aware_splits(race_groups, n_splits=5):
    unique_races = race_groups.drop_duplicates().reset_index(drop=True)
    tss = TimeSeriesSplit(n_splits=n_splits)

    splits = []
    for train_race_idx, test_race_idx in tss.split(unique_races):
        train_races = set(unique_races.iloc[train_race_idx])
        test_races = set(unique_races.iloc[test_race_idx])

        train_idx = np.where(race_groups.isin(train_races).values)[0]
        test_idx = np.where(race_groups.isin(test_races).values)[0]
        splits.append((train_idx, test_idx))

    return splits

def top3_per_race_precision(df, prob_col='PodiumProb', group_cols=['Year', 'RoundNumber']):
    correct = 0
    total_predictions = 0
    for _, race in df.groupby(group_cols):
        top3_predicted = race.nlargest(3, prob_col)
        actual = set(race[race['Podium'] == 1]['DriverId'])
        predicted = set(top3_predicted['DriverId'])
        correct += len(actual & predicted)
        total_predictions += 3
    return correct / total_predictions

def evaluate_model(df, model_name, prob_col='PodiumProb', label_col='Podium', group_cols=['Year', 'RoundNumber'], threshold=0.5):
    y_true = df[label_col]
    y_prob = df[prob_col]
    y_pred = (y_prob >= threshold).astype(int)

    results = {
        'model': model_name,
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_prob),
        'top3_per_race_acc': top3_per_race_precision(df, prob_col, group_cols)
    }

    return results

def find_best_threshold(df, prob_col='PodiumProb', label_col='Podium', thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.05, 0.55, 0.05)

    y_true = df[label_col]
    y_prob = df[prob_col]

    best_threshold = 0.5
    best_f1 = -1
    results = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        results.append({'threshold': t, 'precision': precision, 'recall': recall, 'f1': f1})

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    return best_threshold, pd.DataFrame(results)

def log_results(results_dict, path):
    row = pd.DataFrame([results_dict])
    file_exists= os.path.exists(path)
    row.to_csv(path, mode='a', header= not file_exists, index=False)