import fastf1
import pandas as pd
import time
import datetime
import os
from pipeline_common import log_failure, get_done_set, append_race, load_with_retry

COMBINED_PATH = 'data/combined_sprints.csv'
FAILED_LOG_PATH = 'failed_sprints.csv'

fastf1.Cache.enable_cache('f1_cache')

def main():
    done = get_done_set(COMBINED_PATH)

    # start with clean failed log every run
    if os.path.exists(FAILED_LOG_PATH):
        os.remove(FAILED_LOG_PATH)

    expected = set()

    for year in range(2018, datetime.now().year + 1):
        schedule = fastf1.get_event_schedule(year)
        sprint_schedule = schedule[schedule['EventFormat'].isin(['sprint', 'sprint_shootout', 'sprint_qualifying'])]

        for _, row in sprint_schedule.iterrows():
            round_number = row['RoundNumber']
            event_name = row['EventName']
            expected.add((year, round_number))

            # skip if alr exists
            if (year, round_number) in done:
                print(f"Skipping {year} R{round_number} ({event_name}) — already saved")
                continue

            print(f"Pulling {year} R{round_number} ({event_name})...")

            try:
                sprint = load_with_retry(year, round_number, 'S')
            except fastf1.exceptions.RateLimitExceededError as e:
                print(f"\nHit the hard rate limit: {e}")
                exit(1)
            except RuntimeError as e:
                print(f"  SKIPPING this race entirely: {e}")
                log_failure(year, round_number, event_name, e, FAILED_LOG_PATH)
                continue  # move on, don't kill the whole run

            sprintDF = sprint.results.loc[:, [
                'Abbreviation', 'DriverId', 'TeamId', 
                'ClassifiedPosition', 'Points'
            ]]

            sprintDF['ClassifiedPosition'] = pd.to_numeric(sprintDF['ClassifiedPosition'], errors='coerce')

            sprintDF['Year'] = year
            sprintDF['RoundNumber'] = round_number
            sprintDF['EventName'] = event_name

            # reorder my cols + drop quali artifact
            sprintDF = sprintDF[[
                'Year', 'RoundNumber', 'EventName',
                'Abbreviation', 'DriverId', 'TeamId',
                'ClassifiedPosition', 'Points'
            ]]

            append_race(sprintDF, COMBINED_PATH)

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