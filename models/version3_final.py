import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from eval_common import evaluate_model, log_results

RANDOM_STATE = 67
N_ESTIMATORS = 200
CLASS_WEIGHT = None
MAX_DEPTH = 10
MIN_SAMPLES_LEAF = 5
THRESH = 0.45
LOCKED_TEST_YEARS = (2024, 2025)

FEATURES = [
        'TrackVolatility',
        'QualifyingPosition', 'GapToPole', 'GridPosition', 'GridPenalty', 
        'IsSprintWeekend', 'SprintPoints', 
        'DriverStandingPoints', 'TeamStandingPoints', 
        'DriverAvgFinish_3', 'TeamAvgFinish_3', 'DriverAvgFinish_5', 'TeamAvgFinish_5'
    ]

def main():
    df = pd.read_csv('data/f1_v3.csv')

    train = df[df['Year'] <= 2023].dropna(subset=FEATURES).copy()
    test = df[df['Year'].isin(LOCKED_TEST_YEARS)].dropna(subset=FEATURES).copy()

    X_train = train[FEATURES]
    y_train = train['Podium']
    X_test = test[FEATURES]

    rf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        class_weight=CLASS_WEIGHT,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE
    )
    rf.fit(X_train, y_train)

    test['PodiumProb'] = rf.predict_proba(X_test)[:, 1]

    # shoudl only be run once - so skip if exists
    if not os.path.exists('results/v3_final_locked_results.csv'):
        final_results = evaluate_model(test, 'v3_track_LOCKED', threshold=THRESH)
        log_results(final_results, 'results/v3_final_locked_results.csv')

    # for dashboard: 2024 season + onwards
    inference_set = df[df['Year'] >= 2024].dropna(subset=FEATURES).copy()
    X_inference = inference_set[FEATURES]

    inference_set['PodiumProb'] = rf.predict_proba(X_inference)[:, 1]

    predictions_out = inference_set[['Year', 'RoundNumber', 'EventName', 'Abbreviation',
                              'GridPosition', 'ClassifiedPosition', 'Status', 'Podium', 'PodiumProb']].copy()
    predictions_out.to_csv('data/f1_v3_predictions.csv', index=False)


if __name__ == '__main__':
    main()