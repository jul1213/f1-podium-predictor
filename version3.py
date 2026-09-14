import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from eval_common import get_timeseries_cv_predictions, get_race_aware_splits, find_best_threshold, evaluate_model, log_results

DATA_PATH = "f1_v3.csv"
RESULTS_PATH = "v3_model_results.csv"
RANDOM_STATE = 67

# v2 best rf hyper params:
N_ESTIMATORS = 400
CLASS_WEIGHT = None
MAX_DEPTH = 10
MIN_SAMPLES_LEAF = 4

def run_experiment(df, features, run_name, threshold_carryover=0.35):   
    train = df[df['Year'] <= 2023].copy()
    test = df[df['Year'] >= 2024].copy()

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
    print(f"[{run_name}] Best baseline threshold", best_threshold)

    # threshold 0.35 (from v2 best)
    log_results(evaluate_model(train_eval, f'{run_name}_baseline_thresh_carryover', threshold=0.35), RESULTS_PATH)

    # own best threshold
    log_results(evaluate_model(train_eval, f'{run_name}_baseline_thresh_best', threshold=best_threshold), RESULTS_PATH)


    # tuned random forest
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
    print(f"[{run_name}] Best tuned params:", rclf.best_params_)
    best_rf = rclf.best_estimator_

    train['PodiumProb'] = get_timeseries_cv_predictions(best_rf, X_train, y_train, race_groups=race_groups)
    train_eval = train.dropna(subset=['PodiumProb']).copy()

    new_best_threshold, sweep_df = find_best_threshold(train_eval)
    print(sweep_df)
    print(f"[{run_name}] Best threshold:", new_best_threshold)

    log_results(evaluate_model(train_eval, f'{run_name}_tuned_thresh_baseline', threshold = best_threshold), RESULTS_PATH)
    log_results(evaluate_model(train_eval, f'{run_name}_tuned_thresh_best', threshold=new_best_threshold), RESULTS_PATH)


    # best rf and best thresh on test
    final_rf = rclf.best_estimator_

    test['PodiumProb'] = final_rf.predict_proba(X_test)[:, 1]

    final_test_results = evaluate_model(test, f'{run_name}_FINAL_TEST', threshold=new_best_threshold)
    log_results(final_test_results, RESULTS_PATH)

def main():
    df = pd.read_csv(DATA_PATH)

    features_track = [
        'TrackVolatility',
        'QualifyingPosition', 'GapToPole', 'GridPosition', 'GridPenalty', 
        'IsSprintWeekend', 'SprintPoints', 
        'DriverStandingPoints', 'TeamStandingPoints', 
        'DriverAvgFinish_3', 'TeamAvgFinish_3', 'DriverAvgFinish_5', 'TeamAvgFinish_5'
    ]

    features_roundNo = [
        'RoundNumber',
        'QualifyingPosition', 'GapToPole', 'GridPosition', 'GridPenalty', 
        'IsSprintWeekend', 'SprintPoints', 
        'DriverStandingPoints', 'TeamStandingPoints', 
        'DriverAvgFinish_3', 'TeamAvgFinish_3', 'DriverAvgFinish_5', 'TeamAvgFinish_5'
    ]

    features_combined = [
        'RoundNumber', 'TrackVolatility',
        'QualifyingPosition', 'GapToPole', 'GridPosition', 'GridPenalty', 
        'IsSprintWeekend', 'SprintPoints', 
        'DriverStandingPoints', 'TeamStandingPoints', 
        'DriverAvgFinish_3', 'TeamAvgFinish_3', 'DriverAvgFinish_5', 'TeamAvgFinish_5'
    ]

    run_experiment(df, features_track, 'v3_track')
    run_experiment(df, features_roundNo, 'v3_roundno')
    run_experiment(df, features_combined, 'v3_combined')

if __name__ == '__main__':
    main()