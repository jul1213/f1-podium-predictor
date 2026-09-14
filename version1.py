import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, GridSearchCV, cross_val_predict
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

RANDOM_STATE = 67
BEST_RF_THRESHOLD = 0.45

DATA_PATH = 'combined_races.csv'
RESULTS_PATH = 'v1_model_results.csv'

def get_cv_predictions(model, X, y, groups, n_splits=5):
    gkf = GroupKFold(n_splits)
    y_proba = cross_val_predict(model, X, y, cv=gkf.split(X, y, groups=groups), method='predict_proba')
    return y_proba[:, 1] # prob of podium

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

def log_results(results_dict, path=RESULTS_PATH):
    row = pd.DataFrame([results_dict])
    file_exists= os.path.exists(path)
    row.to_csv(path, mode='a', header= not file_exists, index=False)

def main():
    df = pd.read_csv(DATA_PATH)
    train = df[df['Year'] <= 2023].copy()
    test = df[df['Year'] >= 2024].copy()

    features = ['GridPosition', 'GridPenalty']

    before = len(train)
    train = train.dropna(subset=features).copy()
    print(f"Dropped {before - len(train)} rows with missing {features}")

    test = test.dropna(subset=features).copy()

    X_train = train[features]
    y_train = train['Podium']

    X_test = test[features]
    y_test = test['Podium']

    race_groups = train['Year'].astype(str) + '_' + train['RoundNumber'].astype(str)

    # trivial baseline (top 3 quali -> podium)
    train['PodiumProb'] = (train['QualifyingPosition'] <= 3).astype(int)
    trivial_results = evaluate_model(train, 'trivial_baseline')
    log_results(trivial_results)

    # trivial baseline 2 (top 3 starting -> podium)
    train['PodiumProb'] = ((train['GridPosition'] <= 3) & (train['GridPosition'] > 0)).astype(int)
    trivial_results2 = evaluate_model(train, 'trivial_baseline2')
    log_results(trivial_results2)

    # logistic regression baseline (grouped by race)
    base_logreg = LogisticRegression()
    train['PodiumProb'] = get_cv_predictions(base_logreg, X_train, y_train, race_groups)
    baseline_results = evaluate_model(train, 'baseline_logreg')
    log_results(baseline_results)

    # tune log reg (gridsearch cv)
    param_grid = [
        {
            'C' : np.logspace(-4, 4, 20),
            'class_weight': [None, 'balanced'],
        }
    ]
    gkf = GroupKFold(n_splits=5)
    clf = GridSearchCV(
        LogisticRegression(max_iter=1000, solver='lbfgs'),
        param_grid=param_grid,
        cv=gkf,
        scoring='f1',
        verbose=True,
        n_jobs=-1
    )
    clf.fit(X_train, y_train, groups=race_groups)
    print("Best params for log reg:", clf.best_params_)
    best_logreg = clf.best_estimator_
    train['PodiumProb'] = get_cv_predictions(best_logreg, X_train, y_train, race_groups)
    best_threshold, sweep_df = find_best_threshold(train)
    print(sweep_df)
    print("Best threshold for tuned log reg:", best_threshold)
    best_logreg_results = evaluate_model(train, 'best_logreg', threshold=best_threshold)
    log_results(best_logreg_results)

    # baseline random forest
    rf = RandomForestClassifier(random_state=RANDOM_STATE)
    train['PodiumProb'] = get_cv_predictions(rf, X_train, y_train, race_groups)
    baseline_rf_results = evaluate_model(train, 'baseline_rf')
    log_results(baseline_rf_results)

    # tune random forest
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 10, None],
        'min_samples_leaf': [1, 2, 4, 5, 10, 20],
        'class_weight': [None, 'balanced'],
    }

    gkf = GroupKFold(n_splits=5)
    rclf = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE),
        param_grid=param_grid,
        cv=gkf,
        scoring='f1',
        verbose=True,
        n_jobs=-1,
    )
    rclf.fit(X_train, y_train, groups=race_groups)
    print("Best params for rf:", rclf.best_params_)
    best_rf = rclf.best_estimator_
    train['PodiumProb'] = get_cv_predictions(best_rf, X_train, y_train, race_groups)
    best_threshold, sweep_df = find_best_threshold(train)
    print(sweep_df)
    print("Best threshold for tuned rf:", best_threshold)
    best_rf_results = evaluate_model(train, 'best_rf', threshold=best_threshold)
    log_results(best_rf_results)

    # test on best rf with best threshold 
    final_rf = rclf.best_estimator_

    test = test.copy()
    test['PodiumProb'] = final_rf.predict_proba(X_test)[: , 1]

    final_test_results = evaluate_model(test, 'best_rf_FINAL_TEST', threshold=BEST_RF_THRESHOLD)

    # --- diagnostics for final test result ---
    n_races = test[['Year', 'RoundNumber']].drop_duplicates().shape[0]
    print(f"\nTest set: {n_races} races, {len(test)} rows")

    pred_counts = test.groupby(['Year', 'RoundNumber']).apply(
        lambda r: (r['PodiumProb'] >= BEST_RF_THRESHOLD).sum()
    )
    print("\nPredicted podium count per race (should mostly be 3 if precision=recall=F1):")
    print(pred_counts.value_counts().sort_index())
    print("\nRaces where predicted count != 3:")
    print(pred_counts[pred_counts != 3])

    log_results(final_test_results)


if __name__ == '__main__':
    main()