# ==============================================================================
# FINAL DATA COLLECTION SCRIPT (POLYGON.IO - FAULT-TOLERANT VERSION)
# Purpose: Fetch one full year of minute-by-minute data for a given stock ticker.
# Features:
# - Robust date-looping to avoid pagination issues.
# - Rate-limit compliant with an optimized pause.
# - Saves progress incrementally to survive interruptions.
# - Resumes from the last downloaded date if the script is restarted.
# - Includes a TQDM progress bar for user feedback.
# ==============================================================================

import pandas as pd
from polygon import RESTClient
from datetime import date, timedelta
import time
import os
from tqdm import tqdm  # Import the progress bar library

print("--- SCRIPT START (POLYGON.IO - FAULT-TOLERANT VERSION) ---")

# --- Configuration ---
# PASTE YOUR POLYGON API KEY HERE
API_KEY = "Pg7L8F9eKfqzQioboCI4k_y0k_Ot3uqI"

TICKER_SYMBOL = 'QQQ'
OUTPUT_FILENAME = f"{TICKER_SYMBOL.upper()}_data.csv"
CHUNK_DAYS = 2  # Fetch data in 2-day chunks to stay well under the 50k limit

# --- Define the overall date range for one full year ---
overall_end_date = date.today()
overall_start_date = overall_end_date - timedelta(days=365)

print(f"Configuration set for {TICKER_SYMBOL}.")
print(f"Target Date Range: {overall_start_date} to {overall_end_date}.")

# --- FAULT-TOLERANCE: Check if a partial file exists and resume ---
if os.path.exists(OUTPUT_FILENAME):
    print(f"\nPartial file '{OUTPUT_FILENAME}' found. Resuming download.")
    existing_df = pd.read_csv(OUTPUT_FILENAME, index_col=0, parse_dates=True)
    # Get the last timestamp from the file and start from the next day
    last_date_in_file = existing_df.index.max().date()
    current_start_date = last_date_in_file + timedelta(days=1)
    print(f"Resuming from date: {current_start_date}")
else:
    print(f"\nNo partial file found. Starting new download.")
    existing_df = pd.DataFrame() # Start with an empty dataframe
    current_start_date = overall_start_date

# --- Initialize the Polygon Client ---
try:
    # Disable internal retries to manually control flow
    client = RESTClient(API_KEY, retries=0, trace=False)
    print("Polygon client initialized successfully.")
except Exception as e:
    print(f"*** ERROR: Could not initialize Polygon client. Check your API Key. Error: {e} ***")
    exit()

# --- Brute Force Loop by Date with Progress Bar ---
print("\n--- Starting robust data download from Polygon.io... ---")

# Calculate the total number of days to iterate over for the progress bar
total_days_to_fetch = (overall_end_date - current_start_date).days
if total_days_to_fetch <= 0:
    print("Data is already up to date. Exiting.")
    exit()

# Initialize the progress bar
with tqdm(total=total_days_to_fetch, desc="Fetching Data") as pbar:
    while current_start_date <= overall_end_date:
        current_end_date = current_start_date + timedelta(days=CHUNK_DAYS - 1)
        if current_end_date > overall_end_date:
            current_end_date = overall_end_date
        
        try:
            response = client.get_aggs(
                ticker=TICKER_SYMBOL, multiplier=1, timespan="minute",
                from_=current_start_date, to=current_end_date, limit=50000
            )
            
            if response:
                # Convert the new chunk to a DataFrame
                new_chunk_df = pd.DataFrame(response)
                # Append the new chunk to our main DataFrame in memory
                if not new_chunk_df.empty:
                    existing_df = pd.concat([existing_df, new_chunk_df])
            
        except Exception as e:
            print(f"\n*** WARNING: Could not download chunk for {current_start_date}. Error: {e}. Skipping. ***")
        
        # --- SAVE PROGRESS INCREMENTALLY ---
        if not existing_df.empty:
            # We must first convert the timestamp column to save correctly
            temp_save_df = existing_df.copy()
            if 'timestamp' in temp_save_df.columns:
                 temp_save_df['timestamp'] = pd.to_datetime(temp_save_df['timestamp'], unit='ms', utc=True)
                 temp_save_df = temp_save_df.set_index('timestamp')
                 temp_save_df = temp_save_df[['open', 'high', 'low', 'close', 'volume']]
                 temp_save_df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                 temp_save_df.to_csv(OUTPUT_FILENAME)

        # Update the progress bar by the number of days in the chunk
        days_in_chunk = (current_end_date - current_start_date).days + 1
        pbar.update(days_in_chunk)
        
        # Move to the next chunk
        current_start_date += timedelta(days=CHUNK_DAYS)
        
        # Polite pause to respect the rate limit
        time.sleep(12.2)

print("\n--- All chunks downloaded. Final processing... ---")

# --- Final Data Cleanup and Save ---
# This block ensures the final CSV is clean and correctly formatted
if not existing_df.empty:
    # Drop duplicates just in case there was an overlap from restarting
    existing_df.drop_duplicates(subset=['timestamp'], inplace=True)
    
    # Perform the final data type conversions and save
    stock_data = existing_df.copy()
    stock_data['timestamp'] = pd.to_datetime(stock_data['timestamp'], unit='ms', utc=True)
    stock_data = stock_data.set_index('timestamp')
    stock_data = stock_data[['open', 'high', 'low', 'close', 'volume']]
    stock_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    print(f"\n--> ATTEMPTING FINAL SAVE to file: '{OUTPUT_FILENAME}'...")
    stock_data.to_csv(OUTPUT_FILENAME)
    print(f"--- SUCCESS! File '{OUTPUT_FILENAME}' has been saved. ---")

    print("\nFinal details:")
    print(f"Total rows: {len(stock_data)}")
    print(f"Date Range: {stock_data.index.min()} to {stock_data.index.max()}")
else:
    print("\nNo data was downloaded.")

print("\n--- SCRIPT COMPLETE ---")