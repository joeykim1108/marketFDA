# ==============================================================================
# DATA COLLECTION SCRIPT (ALPACA API - ROBUST VERSION 2)
# Purpose: Fetch several years of minute-by-minute data for a given stock ticker.
# Features:
# - Uses the get_bars_iter() method to correctly handle API pagination.
# ==============================================================================

import pandas as pd
import alpaca_trade_api as tradeapi
from datetime import datetime, date, timedelta
import time
import os
from tqdm import tqdm
import pandas_market_calendars as mcal
from dotenv import load_dotenv

load_dotenv()

print("--- SCRIPT START (ALPACA API - ROBUST VERSION) ---")

# --- Configuration ---
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_API_SECRET")
BASE_URL = "https://paper-api.alpaca.markets"

if not API_KEY or not API_SECRET:
    print("*** ERROR: Alpaca API keys not found. Please set them in your .env file. ***")
    exit()

TICKER_SYMBOL = 'SPY'
OUTPUT_FILENAME = f"{TICKER_SYMBOL.upper()}_data_alpaca.csv"
TIMEZONE = 'America/New_York'

# --- Define the overall date range ---
overall_end_date = date.today()
overall_start_date = overall_end_date - timedelta(days=365 * 5)

print(f"Configuration set for {TICKER_SYMBOL}.")
print(f"Target Date Range: {overall_start_date} to {overall_end_date}.")

# --- Get a list of actual trading days ---
nyse = mcal.get_calendar('NYSE')
trading_days = nyse.schedule(start_date=overall_start_date, end_date=overall_end_date).index.date
print(f"Found {len(trading_days)} trading days in the specified range.")

# --- FAULT-TOLERANCE: Check if a partial file exists and resume ---
if os.path.exists(OUTPUT_FILENAME):
    print(f"\nPartial file '{OUTPUT_FILENAME}' found. Resuming download.")
    existing_df = pd.read_csv(OUTPUT_FILENAME, index_col=0, parse_dates=True)
    if not existing_df.empty:
        last_date_in_file = existing_df.index.max().date()
        trading_days_to_fetch = trading_days[trading_days > last_date_in_file]
        print(f"Resuming from date: {trading_days_to_fetch[0] if trading_days_to_fetch.size > 0 else 'N/A'}")
    else:
        trading_days_to_fetch = trading_days
else:
    print(f"\nNo partial file found. Starting new download.")
    existing_df = pd.DataFrame()
    trading_days_to_fetch = trading_days

# --- Initialize the Alpaca Client ---
try:
    api = tradeapi.REST(API_KEY, API_SECRET, api_version='v2')
    print("Alpaca client initialized successfully.")
except Exception as e:
    print(f"*** ERROR: Could not initialize Alpaca client. Check your API Keys. Error: {e} ***")
    exit()

# --- Loop by Trading Day with Progress Bar ---
print("\n--- Starting robust data download from Alpaca... ---")

if trading_days_to_fetch.size == 0:
    print("Data is already up to date. Exiting.")
    exit()

all_bars_raw = []
if not existing_df.empty:
    existing_df.reset_index(inplace=True)
    existing_df.rename(columns={'timestamp': 't', 'Open': 'o', 'High': 'h', 'Low': 'l', 'Close': 'c', 'Volume': 'v'}, inplace=True)
    all_bars_raw = existing_df.to_dict('records')


with tqdm(total=len(trading_days_to_fetch), desc="Fetching Data") as pbar:
    for day in trading_days_to_fetch:
        # Define the start and end times for this specific day
        start_dt = pd.Timestamp(f'{day} 08:00', tz=TIMEZONE).isoformat() # Widen time window slightly
        end_dt = pd.Timestamp(f'{day} 18:00', tz=TIMEZONE).isoformat()

        try:
            # ***** THIS IS THE CORRECTED LOGIC *****
            # get_bars_iter returns a generator that handles pagination for you.
            bars_iterator = api.get_bars_iter(
                TICKER_SYMBOL,
                '1Min',
                start=start_dt,
                end=end_dt
            )
            # We convert the iterator to a list and extend our main list
            all_bars_raw.extend(list(bars_iterator))

        except Exception as e:
            print(f"\n*** WARNING: Could not download chunk for {day}. Error: {e}. Skipping. ***")

        # --- SAVE PROGRESS INCREMENTALLY ---
        if all_bars_raw:
            # We convert the raw bar objects to a list of dictionaries for pandas
            all_bars_dicts = [bar._raw for bar in all_bars_raw]

            temp_save_df = pd.DataFrame(all_bars_dicts)
            temp_save_df['timestamp'] = pd.to_datetime(temp_save_df['t'])
            temp_save_df = temp_save_df.set_index('timestamp')
            
            temp_save_df = temp_save_df[['o', 'h', 'l', 'c', 'v']]
            temp_save_df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            # Remove potential duplicates from overlapping saves before saving
            temp_save_df = temp_save_df[~temp_save_df.index.duplicated(keep='last')]
            
            temp_save_df.to_csv(OUTPUT_FILENAME)

        pbar.update(1)
        # Polite pause, though less critical with Alpaca's higher rate limits
        time.sleep(0.4)

print("\n--- All days downloaded. Final processing... ---")

# Final check and printout
if os.path.exists(OUTPUT_FILENAME):
    final_df = pd.read_csv(OUTPUT_FILENAME, index_col=0, parse_dates=True)
    if not final_df.empty:
        # Final sort to ensure data is chronological
        final_df.sort_index(inplace=True)
        final_df.to_csv(OUTPUT_FILENAME)
        
        print("\nFinal details:")
        print(f"Total rows: {len(final_df)}")
        print(f"Date Range: {final_df.index.min()} to {final_df.index.max()}")
    else:
        print("\nNo data was downloaded.")
else:
    print("\nNo data was downloaded.")

print("\n--- SCRIPT COMPLETE ---")