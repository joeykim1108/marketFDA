# ==============================================================================
# DATA COLLECTION SCRIPT (ALPACA API - ROBUST VERSION)
# Purpose: Fetch several years of minute-by-minute data for a given stock ticker.
# Features:
# - Uses Alpaca's free market data API.
# - Loops over actual trading days to be highly efficient.
# - Handles API pagination to get all data for a day.
# - Saves progress incrementally and resumes from the last downloaded date.
# ==============================================================================

import pandas as pd
import alpaca_trade_api as tradeapi
from datetime import datetime, date, timedelta
import time
import os
from tqdm import tqdm
import pandas_market_calendars as mcal # Import library for trading days
from dotenv import load_dotenv

load_dotenv()

print("--- SCRIPT START (ALPACA API - ROBUST VERSION) ---")

# --- Configuration ---
# Note: For free data, you just need keys. No need for a funded account.
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_API_SECRET")
BASE_URL = "https://paper-api.alpaca.markets"

if not API_KEY or not API_SECRET:
    print("*** ERROR: Alpaca API keys not found. Please set them in your .env file. ***")
    exit()

TICKER_SYMBOL = 'QQQ' # Switched to QQQ as planned
OUTPUT_FILENAME = f"{TICKER_SYMBOL.upper()}_data_alpaca.csv"
TIMEZONE = 'America/New_York'

# --- Define the overall date range ---
# Let's get 5 years of data to enable robust backtesting
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
    last_date_in_file = existing_df.index.max().date()
    # Filter trading_days to only include dates after the last downloaded date
    trading_days_to_fetch = [d for d in trading_days if d > last_date_in_file]
    print(f"Resuming from date: {trading_days_to_fetch[0] if trading_days_to_fetch else 'N/A'}")
else:
    print(f"\nNo partial file found. Starting new download.")
    existing_df = pd.DataFrame()
    trading_days_to_fetch = trading_days

# --- Initialize the Alpaca Client ---
try:
    api = tradeapi.REST(API_KEY, API_SECRET, base_url=BASE_URL, api_version='v2')
    print("Alpaca client initialized successfully.")
except Exception as e:
    print(f"*** ERROR: Could not initialize Alpaca client. Check your API Keys. Error: {e} ***")
    exit()

# --- Loop by Trading Day with Progress Bar ---
print("\n--- Starting robust data download from Alpaca... ---")

if not trading_days_to_fetch:
    print("Data is already up to date. Exiting.")
    exit()

all_bars = []
if not existing_df.empty:
    # Important: Convert existing data to a list of bars to append to
    existing_df.reset_index(inplace=True)
    # Rename columns to match Alpaca's format temporarily
    existing_df.rename(columns={'timestamp': 't', 'Open': 'o', 'High': 'h', 'Low': 'l', 'Close': 'c', 'Volume': 'v'}, inplace=True)
    all_bars = existing_df.to_dict('records')


with tqdm(total=len(trading_days_to_fetch), desc="Fetching Data") as pbar:
    for day in trading_days_to_fetch:
        # Define the start and end times for the API call with timezone
        start_dt = pd.Timestamp(f'{day} 00:00', tz=TIMEZONE).isoformat()
        end_dt = pd.Timestamp(f'{day} 23:59', tz=TIMEZONE).isoformat()
        
        page_token = None
        try:
            while True:
                response = api.get_bars(
                    TICKER_SYMBOL,
                    '1Min',
                    start=start_dt,
                    end=end_dt,
                    limit=10000, # Max limit per request
                    page_token=page_token
                )._raw # Use ._raw to get a list of dicts directly

                if response:
                    all_bars.extend(response)
                    # Check if there's a next page
                    last_bar_time = pd.to_datetime(response[-1]['t'])
                    page_token = response[-1]['t'].isoformat() # This is not a real page token, but a way to paginate
                    if len(response) < 10000:
                        break # We've received the last page
                else:
                    break # No more data for this day

        except Exception as e:
            print(f"\n*** WARNING: Could not download chunk for {day}. Error: {e}. Skipping. ***")

        # --- SAVE PROGRESS INCREMENTALLY ---
        if all_bars:
            temp_save_df = pd.DataFrame(all_bars)
            # CRITICAL: Alpaca's timestamp 't' is already timezone-aware
            temp_save_df['timestamp'] = pd.to_datetime(temp_save_df['t'])
            temp_save_df = temp_save_df.set_index('timestamp')
            
            # Select and rename columns to your desired final format
            temp_save_df = temp_save_df[['o', 'h', 'l', 'c', 'v']]
            temp_save_df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            # Remove potential duplicates from overlapping saves
            temp_save_df = temp_save_df[~temp_save_df.index.duplicated(keep='first')]
            
            temp_save_df.to_csv(OUTPUT_FILENAME)

        pbar.update(1)
        # Polite pause, though less critical with Alpaca's higher rate limits
        time.sleep(0.5)

print("\n--- All days downloaded. Final processing... ---")

# --- Final Data Cleanup and Save ---
if all_bars:
    final_df = pd.read_csv(OUTPUT_FILENAME, index_col=0, parse_dates=True)
    print("\nFinal details:")
    print(f"Total rows: {len(final_df)}")
    print(f"Date Range: {final_df.index.min()} to {final_df.index.max()}")
else:
    print("\nNo data was downloaded.")

print("\n--- SCRIPT COMPLETE ---")