import fastf1
import pandas as pd
import time
import os

fastf1.Cache.enable_cache('f1_cache')

def log_failure(year, round_number, event_name, reason, path):
    """Record a race that failed to load, so we can check later."""
    file_exists = os.path.exists(path)
    row = pd.DataFrame([{
        'Year': year,
        'RoundNumber': round_number,
        'EventName': event_name,
        'Reason': str(reason),
        'Timestamp': pd.Timestamp.now()
    }])
    row.to_csv(path, mode='a', header=not file_exists, index=False)


def get_done_set(path):
    if not os.path.exists(path):
        return set()
    existing = pd.read_csv(path, usecols=['Year', 'RoundNumber'])
    return set(zip(existing['Year'], existing['RoundNumber']))


def append_race(df, path):
    file_exists = os.path.exists(path)
    df.to_csv(path, mode='a', header=not file_exists, index=False)


''' load session, if rate limited propagate to caller immediately, 
otherwise if failed retry max 3 more times '''
def load_with_retry(year, round_number, session_type, max_retries=3):
    for attempt in range(max_retries):
        try:
            session = fastf1.get_session(year, round_number, session_type)
            session.load(laps=False, telemetry=False, weather=False, messages=False)
            return session
        except fastf1.exceptions.RateLimitExceededError:
            raise  # don't retry
        except Exception as e:
            wait = 2 ** attempt  # 1, 2, 4 seconds
            time.sleep(wait)
    raise RuntimeError(f"Gave up on {year} round {round_number} {session_type} after {max_retries} attempts")