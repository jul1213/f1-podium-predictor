import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, GridSearchCV, cross_val_predict

from eval_common import evaluate_model, find_best_threshold, log_results

RANDOM_STATE = 67
BEST_RF_THRESHOLD = 0.45

DATA_PATH = 'data/combined_races.csv'
RESULTS_PATH = 'results/v1_model_results.csv'

def get_cv_predictions(model, X, y, groups, n_splits=5):
    gkf = GroupKFold(n_splits)
    y_proba = cross_val_predict(model, X, y, cv=gkf.split(X, y, groups=groups), method='predict_proba')
    return y_proba[:, 1] # prob of podium

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
    log_results(trivial_results, RESULTS_PATH)

    # trivial baseline 2 (top 3 starting -> podium)
    train['PodiumProb'] = ((train['GridPosition'] <= 3) & (train['GridPosition'] > 0)).astype(int)
    trivial_results2 = evaluate_model(train, 'trivial_baseline2')
    log_results(trivial_results2, RESULTS_PATH)

    # logistic regression baseline (grouped by race)
    base_logreg = LogisticRegression()
    train['PodiumProb'] = get_cv_predictions(base_logreg, X_train, y_train, race_groups)
    baseline_results = evaluate_model(train, 'baseline_logreg')
    log_results(baseline_results, RESULTS_PATH)

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
    log_results(best_logreg_results, RESULTS_PATH)

    # baseline random forest
    rf = RandomForestClassifier(random_state=RANDOM_STATE)
    train['PodiumProb'] = get_cv_predictions(rf, X_train, y_train, race_groups)
    baseline_rf_results = evaluate_model(train, 'baseline_rf')
    log_results(baseline_rf_results, RESULTS_PATH)

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
    log_results(best_rf_results, RESULTS_PATH)

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

    log_results(final_test_results, RESULTS_PATH)


if __name__ == '__main__':
    main()