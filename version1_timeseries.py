import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from eval_common import get_timeseries_cv_predictions, evaluate_model, find_best_threshold, log_results

RESULTS_PATH = 'timeseries_model_results.csv'
RANDOM_STATE = 67

def main():
    df = pd.read_csv('combined_races.csv')

    # sort in order of year and round number - since time series split used
    df = df.sort_values(['Year', 'RoundNumber']).reset_index(drop=True)

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

    # rerun v1 (random forest) with time series split
    rf = RandomForestClassifier(
        n_estimators=200,
        class_weight=None,
        max_depth=10,
        min_samples_leaf=2,
        random_state=RANDOM_STATE
    )
    train['PodiumProb'] = get_timeseries_cv_predictions(rf, X_train, y_train, race_groups=race_groups)
    train_eval = train.dropna(subset=['PodiumProb']).copy()

    best_threshold, sweep_df = find_best_threshold(train_eval)
    print(sweep_df)
    print("Best threshold for rf with timeseries cv:", best_threshold)

    # using old best threshold in v1
    timeseries_rf_results_045 = evaluate_model(train_eval, 'timeseries_rf_045', threshold = 0.45)
    log_results(timeseries_rf_results_045, RESULTS_PATH)

    # at own best threshold for future ref
    timeseries_rf_results_best = evaluate_model(train_eval, 'timeseries_rf_best', threshold = best_threshold)
    log_results(timeseries_rf_results_best, RESULTS_PATH)

if __name__ == '__main__':
    main()