import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from eval_common import evaluate_model, log_results

RANDOM_STATE = 67
N_ESTIMATORS = 200
CLASS_WEIGHT = None
MAX_DEPTH = 10
MIN_SAMPLES_LEAF = 5
THRESH = 0.45

FEATURES = [
        'TrackVolatility',
        'QualifyingPosition', 'GapToPole', 'GridPosition', 'GridPenalty', 
        'IsSprintWeekend', 'SprintPoints', 
        'DriverStandingPoints', 'TeamStandingPoints', 
        'DriverAvgFinish_3', 'TeamAvgFinish_3', 'DriverAvgFinish_5', 'TeamAvgFinish_5'
    ]

def main():
    df = pd.read_csv('f1_v3.csv')

    train = df[df['Year'] <= 2023].dropna(subset=FEATURES).copy()
    test = df[df['Year'] >= 2024].dropna(subset=FEATURES).copy()

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

    final_results = evaluate_model(test, 'v3_track_LOCKED', threshold=THRESH)
    log_results(final_results, 'v3_final_locked_results.csv')

    predictions_out = test[['Year', 'RoundNumber', 'EventName', 'Abbreviation',
                              'GridPosition', 'ClassifiedPosition', 'Status', 'Podium', 'PodiumProb']].copy()
    predictions_out.to_csv('f1_v3_predictions.csv', index=False)


if __name__ == '__main__':
    main()