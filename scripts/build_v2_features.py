import pandas as pd
import numpy as np

def fix_pitlane_starts(df):
    # originally grid position 0 means pit lane start
    # so change to last pos + 1 (for all grid pos == 0)
    df = df.copy()

    valid_grid = df['GridPosition'].replace(0, np.nan)
    race_max = valid_grid.groupby([df['Year'], df['RoundNumber']]).transform('max')

    df['GridPosition'] = np.where(
        df['GridPosition'] == 0,
        race_max + 1,
        df['GridPosition']
    )

    return df

def rolling_avg(group, col, window):
    return group[col].shift(1).rolling(window, min_periods=1).mean()

def main():
    races = pd.read_csv('data/combined_races.csv').rename(columns={'Points' : 'RacePoints'})
    sprints = pd.read_csv('data/combined_sprints.csv').rename(columns={
        'Points' : 'SprintPoints',
        'ClassifiedPosition' : 'SprintPosition'
    })

    df = pd.merge(
        races,
        sprints[['Year', 'RoundNumber', 'DriverId', 'SprintPosition', 'SprintPoints']],
        on=['Year', 'RoundNumber', 'DriverId'],
        how='left'
    )

    df['SprintPoints'] = df['SprintPoints'].fillna(0)
    df['IsSprintWeekend'] = df['SprintPosition'].notna()

    df = fix_pitlane_starts(df)
    df['GridPenalty'] = df["GridPosition"] - df['QualifyingPosition']

    df = df.sort_values(['Year', 'RoundNumber']).reset_index(drop=True)

    # standing reset per season 
    df['TotalPoints'] = df['RacePoints'] + df['SprintPoints']

    df['DriverStandingPoints'] = (
        df.groupby(['Year', 'DriverId'])['TotalPoints']
        .transform(lambda x: x.shift(1).cumsum())
    )

    df['TeamStandingPoints'] = (
        df.groupby(['Year', 'TeamId'])['TotalPoints']
        .transform(lambda x: x.shift(1).cumsum())
    )

    ## rolling
    drivers_per_year = df.groupby('Year')['DriverId'].transform('nunique')
    neutral_finish = drivers_per_year / 2

    for window in [3, 5]:
        df[f'DriverAvgFinish_{window}'] = (
            df.groupby(['Year', 'DriverId'], group_keys=False)
            .apply(lambda g, w=window: rolling_avg(g, 'ClassifiedPosition', w), include_groups=False)
        )
        df[f'DriverAvgFinish_{window}'] = df[f'DriverAvgFinish_{window}'].fillna(neutral_finish)

        df[f'TeamAvgFinish_{window}'] = (
            df.groupby(['Year', 'TeamId'], group_keys=False)
            .apply(lambda g, w=window: rolling_avg(g, 'ClassifiedPosition', w), include_groups=False)
        )
        df[f'TeamAvgFinish_{window}'] = df[f'TeamAvgFinish_{window}'].fillna(neutral_finish)


    # cold start handling:
    df['DriverStandingPoints'] = df['DriverStandingPoints'].fillna(0)
    df['TeamStandingPoints'] = df['TeamStandingPoints'].fillna(0)


    df = df[[
        'Year', 'RoundNumber', 'EventName', 'Location',
        'Abbreviation', 'DriverId', 'TeamId', 'QualifyingPosition', 'GapToPole', 'GridPosition', 'GridPenalty', 'ClassifiedPosition',
        'RacePoints', 'Status', 'Podium',
        'IsSprintWeekend', 'SprintPosition', 'SprintPoints',
        'TotalPoints', 'DriverStandingPoints', 'TeamStandingPoints',
        'DriverAvgFinish_3', 'TeamAvgFinish_3', 'DriverAvgFinish_5', 'TeamAvgFinish_5'
    ]]

    # save
    df.to_csv('data/f1_v2.csv', index=False)

if __name__ == '__main__':
    main()