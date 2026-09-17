import fastf1
import pandas as pd
import time
from datetime import datetime
import os
from pipeline_common import log_failure, get_done_set, append_race, load_with_retry

fastf1.Cache.enable_cache('f1_cache')

COMBINED_PATH = 'data/combined_races.csv'
FAILED_LOG_PATH = 'failed_races.csv'

def main():
    done = get_done_set(COMBINED_PATH)

    if os.path.exists(FAILED_LOG_PATH):
        os.remove(FAILED_LOG_PATH)

    expected = set()

    for year in range(2018, datetime.now().year + 1):
        schedule = fastf1.get_event_schedule(year)
        race_schedule = schedule[schedule['EventFormat'] != 'testing']

        for _, row in race_schedule.iterrows():
            round_number = row['RoundNumber']
            event_name = row['EventName']
            expected.add((year, round_number))

            # skip if alr exists
            if (year, round_number) in done:
                print(f"Skipping {year} R{round_number} ({event_name}) — already saved")
                continue

            print(f"Pulling {year} R{round_number} ({event_name})...")

            try:
                race = load_with_retry(year, round_number, 'R')
                quali = load_with_retry(year, round_number, 'Q')
            except fastf1.exceptions.RateLimitExceededError as e:
                print(f"\nHit the hard rate limit: {e}")
                exit(1)
            except RuntimeError as e:
                print(f"  SKIPPING this race entirely: {e}")
                log_failure(year, round_number, event_name, e, FAILED_LOG_PATH)
                continue  # move on, don't kill the whole run

            raceDF = race.results.loc[:, [
                'Abbreviation', 'DriverId', 'TeamId',
                'ClassifiedPosition', 'GridPosition', 'Status', 'Points'
            ]]
            qualiDF = quali.results.loc[:, ['DriverId', 'Position', 'Q1', 'Q2', 'Q3']].copy()
            qualiDF['BestQualiTime'] = qualiDF[['Q1', 'Q2', 'Q3']].min(axis=1)

            pole_time = qualiDF['BestQualiTime'].min()
            qualiDF['GapToPole'] = (qualiDF['BestQualiTime'] - pole_time).dt.total_seconds()

            df = pd.merge(raceDF, qualiDF, on='DriverId', how='inner')
            df.rename(columns={"Position": "QualifyingPosition"}, inplace=True)

            df['GridPosition'] = pd.to_numeric(df['GridPosition'], errors='coerce')
            df['QualifyingPosition'] = pd.to_numeric(df['QualifyingPosition'], errors='coerce')
            df['GridPenalty'] = df['GridPosition'] - df['QualifyingPosition']

            df['ClassifiedPosition'] = pd.to_numeric(df['ClassifiedPosition'], errors='coerce')
            df['Podium'] = (df['ClassifiedPosition'] <= 3).astype(int)

            df['Year'] = year
            df['RoundNumber'] = round_number
            df['EventName'] = event_name
            df['Location'] = row['Location']

            # reorder my cols + drop quali artifact
            df = df[[
                'Year', 'RoundNumber', 'EventName', 'Location',
                'Abbreviation', 'DriverId', 'TeamId',
                'QualifyingPosition', 'GapToPole', 'GridPosition', 'GridPenalty',
                'ClassifiedPosition', 'Points', 'Status', 'Podium'
            ]]

            append_race(df, COMBINED_PATH)

            time.sleep(1)

    # sanity check : did we get all races ?
    final_done = get_done_set(COMBINED_PATH)
    missing = expected - final_done
    
    if missing:
        print(f"WARNING: {len(missing)} races are missing from {COMBINED_PATH}:")
        for year, rnd in sorted(missing):
            print(f"  {year} round {rnd}")
        print(f"(Check {FAILED_LOG_PATH} for reasons, or just rerun this script — it will only retry the missing ones.)")
    else:
        print("All expected races are present. Nothing missing.")

if __name__ == '__main__':
    main()