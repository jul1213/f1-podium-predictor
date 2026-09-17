import pandas as pd
import numpy as np

def compute_track_volatility(df):
    df = df.copy()

    # each race predictable or high no. of overtakes
    df = df.sort_values(['Year', 'RoundNumber'])
    race_level = (
        df.groupby(['Year', 'RoundNumber', 'Location'])
          .apply(lambda r: ((r['GridPosition'] <= 3) & (r['Podium'] == 1)).sum(), include_groups=False)
          .reset_index(name='Top3GridPodiumCount')
    )

    # historical average no. of overtakes (podium wise)
    race_level = race_level.sort_values(['Location', 'Year', 'RoundNumber'])
    race_level['TrackVolatility'] = (
        race_level.groupby('Location')['Top3GridPodiumCount']
                  .transform(lambda x: x.shift(1).expanding().mean())
    )

    # for cold start:
    global_avg = race_level['TrackVolatility'].mean()
    race_level['TrackVolatility'] = race_level['TrackVolatility'].fillna(global_avg)

    return df.merge(
        race_level[['Year', 'RoundNumber', 'Location', 'TrackVolatility']],
        on = ['Year', 'RoundNumber', 'Location'],
        how='left'
    )

def main():
    df = pd.read_csv('data/f1_v2.csv')
    track_df = compute_track_volatility(df)

    track_df.to_csv('data/f1_v3.csv', index=False)

if __name__ == '__main__':
    main()