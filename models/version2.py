import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from eval_common import get_timeseries_cv_predictions, get_race_aware_splits, evaluate_model, find_best_threshold, log_results

DATA_PATH = "data/f1_v2.csv"
RESULTS_PATH = "results/v2_model_results.csv"
RANDOM_STATE = 67

# baseline hyperparam for rf
N_ESTIMATORS = 200
CLASS_WEIGHT = None
MAX_DEPTH = 10
MIN_SAMPLES_LEAF = 2

def main():
    df = pd.read_csv(DATA_PATH)

    train = df[df['Year'] <= 2023].copy()
    test = df[df['Year'] >= 2024].copy()

    features = [
        'QualifyingPosition', 'GapToPole', 'GridPosition', 'GridPenalty', 
        'IsSprintWeekend', 'SprintPoints', 
        'DriverStandingPoints', 'TeamStandingPoints', 
        'DriverAvgFinish_3', 'TeamAvgFinish_3', 'DriverAvgFinish_5', 'TeamAvgFinish_5'
    ]

    before = len(train)
    train = train.dropna(subset=features).copy()
    print(f"Dropped {before - len(train)} rows with missing {features}")

    X_train = train[features]
    y_train = train['Podium']

    before_test = len(test)
    test = test.dropna(subset=features).copy()
    print(f"Dropped {before_test - len(test)} test rows with missing {features}")

    X_test = test[features]
    y_test = test['Podium']

    race_groups = train['Year'].astype(str) + '_' + train['RoundNumber'].astype(str)

    # random forest baseline
    rf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        class_weight=CLASS_WEIGHT,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE
    )
    train['PodiumProb'] = get_timeseries_cv_predictions(rf, X_train, y_train, race_groups)
    train_eval = train.dropna(subset=['PodiumProb']).copy()

    best_threshold, sweep_df = find_best_threshold(train_eval)
    print(sweep_df)
    print("Best threshold for rf with timeseries cv:", best_threshold)

    # threshold 0.35 (from timeseries best)
    baseline_rf_v2_thresh035 = evaluate_model(train_eval, 'baseline_rf_v2_thresh035', threshold=0.35)
    log_results(baseline_rf_v2_thresh035, RESULTS_PATH)

    # own best threshold
    baseline_rf_v2_thresh_best = evaluate_model(train_eval, 'baseline_rf_v2_thresh_best', threshold=best_threshold)
    log_results(baseline_rf_v2_thresh_best, RESULTS_PATH)


    # random forest tuned
    race_aware_cv = get_race_aware_splits(race_groups)

    param_grid = {
        'n_estimators': [100, 200, 300, 400, 500, 600],
        'max_depth': [3, 5, 10, None],
        'min_samples_leaf': [1, 2, 4, 5, 10, 20],
        'class_weight': [None, 'balanced']
    }

    rclf = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE),
        param_grid=param_grid,
        cv=race_aware_cv,
        scoring='f1',
        verbose=True,
        n_jobs=-1,
    )
    rclf.fit(X_train, y_train)
    print("Best params for v2 rf:", rclf.best_params_)
    best_rf = rclf.best_estimator_

    train['PodiumProb'] = get_timeseries_cv_predictions(best_rf, X_train, y_train, race_groups=race_groups)
    train_eval = train.dropna(subset=['PodiumProb']).copy()

    new_best_threshold, sweep_df = find_best_threshold(train_eval)
    print(sweep_df)
    print("Best threshold for tuned rf:", new_best_threshold)

    tuned_rf_v2_thresh_baseline = evaluate_model(train_eval, 'tuned_rf_v2_thresh_baseline', threshold = best_threshold)
    log_results(tuned_rf_v2_thresh_baseline, RESULTS_PATH)

    tuned_rf_v2_thresh_best = evaluate_model(train_eval, 'tuned_rf_v2_thresh_best', threshold=new_best_threshold)
    log_results(tuned_rf_v2_thresh_best, RESULTS_PATH)

    # best rf and best thresh on test
    final_rf = rclf.best_estimator_

    test['PodiumProb'] = final_rf.predict_proba(X_test)[:, 1]

    final_test_results = evaluate_model(test, 'best_rf_v2_FINAL_TEST', threshold=new_best_threshold)
    log_results(final_test_results, RESULTS_PATH)

    # final diagnostics
    n_races = test[['Year', 'RoundNumber']].drop_duplicates().shape[0]
    print(f"\nFinal V2 test set: {n_races} races, {len(test)} rows")

    pred_counts = test.groupby(['Year', 'RoundNumber']).apply(
        lambda r: (r['PodiumProb'] >= new_best_threshold).sum(), include_groups=False
    )
    print("\nPredicted podium count per race (should mostly be 3):")
    print(pred_counts.value_counts().sort_index())
    print("\nRaces where predicted count != 3:")
    print(pred_counts[pred_counts != 3])

    print(f"\nFinal V2 results: {final_test_results}")



if __name__ == '__main__':
    main()