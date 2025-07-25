# ==============================================================================
# SCRIPT: 02_analysis_and_backtest.py
# PURPOSE:
# 1. Load the full 5-year dataset.
# 2. Split data into a fixed training and testing period (Anchored Walk-Forward).
# 3. Engineer features and train models ONLY on the training data.
# 4. Generate out-of-sample predictions for the entire test period.
# 5. Run a realistic, cost-based backtest using vectorbt.
# ==============================================================================

import pandas as pd
import skfda
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from xgboost import XGBRegressor, XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import vectorbt as vbt # The backtesting library

print("--- SCRIPT START: ANALYSIS AND BACKTESTING ---")


# --- PHASE 1: LOAD AND PREPARE DATA ---
print("\n--- PHASE 1: Loading and Preparing Data ---")

# --- Configuration ---
TICKER_SYMBOL = 'SPY'
INPUT_FILENAME = f"{TICKER_SYMBOL.upper()}_data_alpaca.csv"
TRADING_START_TIME = '09:30'
TRADING_END_TIME = '16:00'
TIMEZONE = 'America/New_York'
EXPECTED_MINUTES = 391 # 9:30 to 16:00 inclusive is 391 minutes
MINIMUM_DATA_POINTS_THRESHOLD = int(EXPECTED_MINUTES * 0.9)
MORNING_PERIOD_MINUTES = 181 # Up to 12:30 PM

try:
    stock_data = pd.read_csv(INPUT_FILENAME, index_col=0, parse_dates=True)
    # Ensure the index is timezone-aware
    if stock_data.index.tz is None:
        stock_data.index = stock_data.index.tz_localize('UTC').tz_convert(TIMEZONE)
    else:
        stock_data.index = stock_data.index.tz_convert(TIMEZONE)
    print(f"Successfully loaded {len(stock_data)} rows from '{INPUT_FILENAME}'.")
except FileNotFoundError:
    print(f"*** ERROR: The file '{INPUT_FILENAME}' was not found. ***")
    print("Please run '01_data_collection_alpaca.py' first.")
    exit()


# --- PHASE 2: ANCHORED TIME-SERIES SPLIT ---
# This is the core of a rigorous backtest. We define a hard cutoff.
# Everything before is history to learn from. Everything after is the "future".
print("\n--- PHASE 2: Splitting Data into Training and Test Sets ---")

TRAIN_END_DATE = '2023-12-31'
print(f"Training data will end on: {TRAIN_END_DATE}")

# Filter the data for valid trading hours
stock_data = stock_data.between_time(TRADING_START_TIME, TRADING_END_TIME)

# Split into training and test sets based on the date
train_df = stock_data.loc[stock_data.index.date < pd.to_datetime(TRAIN_END_DATE).date()]
test_df = stock_data.loc[stock_data.index.date >= pd.to_datetime(TRAIN_END_DATE).date()]

print(f"Training set: {train_df.index.min().date()} to {train_df.index.max().date()} ({len(train_df.index.normalize().unique())} days)")
print(f"Test set:     {test_df.index.min().date()} to {test_df.index.max().date()} ({len(test_df.index.normalize().unique())} days)")

if train_df.empty or test_df.empty:
    print("*** ERROR: Not enough data to perform a train/test split. Check your date range. ***")
    exit()


# --- PHASE 3: PROCESS DATA & ENGINEER FEATURES (FUNCTION) ---
# We create a function to do this so we can apply the same logic to both train and test sets.
print("\n--- PHASE 3: Engineering Features ---")

def process_data_into_curves_and_features(df):
    """
    Takes a dataframe of stock data and processes it into functional curves
    and a list of corresponding scalar features for each valid day.
    """
    unique_days = df.index.normalize().unique()
    daily_curves = []
    scalar_features_list = []
    valid_day_labels = [] # To keep track of which days we didn't skip
    grid_points = np.arange(EXPECTED_MINUTES)
    
    print(f"Processing {len(unique_days)} days...")
    for day in tqdm(unique_days):
        day_str = day.strftime('%Y-%m-%d')
        daily_prices_df = df.loc[day_str]
        
        if len(daily_prices_df) < MINIMUM_DATA_POINTS_THRESHOLD:
            continue

        # --- Resampling and Normalization ---
        target_index = pd.date_range(start=f"{day_str} {TRADING_START_TIME}", end=f"{day_str} {TRADING_END_TIME}", freq='1min', tz=TIMEZONE)
        resampled_day = daily_prices_df.reindex(target_index).interpolate(method='linear').ffill().bfill()
        
        curve_data = resampled_day['Close'].values
        curve_zeroed = curve_data - curve_data[0]
        curve_std = np.std(curve_zeroed)
        if curve_std > 1e-6:
            curve_normalized = curve_zeroed / curve_std
        else:
            curve_normalized = curve_zeroed
            
        daily_curves.append(curve_normalized)
        valid_day_labels.append(day) # Use the datetime object as the label

        # --- Feature Engineering ---
        # Overnight Gap
        try:
            current_day_start_loc = stock_data.index.get_loc(daily_prices_df.index[0])
            if current_day_start_loc > 0:
                previous_close = stock_data.iloc[current_day_start_loc - 1]['Close']
                current_open = daily_prices_df.iloc[0]['Open'] # Use the actual open price
                overnight_gap = (current_open - previous_close) / previous_close
            else:
                overnight_gap = 0
        except Exception:
            overnight_gap = 0
        
        # Morning Curve Features
        morning_curve = curve_normalized[:MORNING_PERIOD_MINUTES]
        morning_return = morning_curve[-1]
        volatility = np.std(morning_curve)
        slope = np.polyfit(np.arange(len(morning_curve)), morning_curve, 1)[0]
        
        scalar_features_list.append([
            volatility,
            morning_return,
            overnight_gap,
            slope
        ])
        
    fd_object = skfda.FDataGrid(data_matrix=daily_curves, grid_points=grid_points)
    
    return fd_object, np.array(scalar_features_list), valid_day_labels

# Process both datasets
train_fd, train_scalar, train_labels = process_data_into_curves_and_features(train_df)
test_fd, test_scalar, test_labels = process_data_into_curves_and_features(test_df)


# --- PHASE 4: FIT MODELS ON *TRAINING* DATA ONLY ---
print("\n--- PHASE 4: Fitting FPCA and ML Models on Training Data ---")

# --- 4a. Fit FPCA on Training Data ---
N_COMPONENTS = 5 # Let's use a few more components
fpca = skfda.preprocessing.dim_reduction.FPCA(n_components=N_COMPONENTS)
fpca.fit(train_fd)
# Now, transform the training data using the fitted FPCA
train_fpca_scores = fpca.transform(train_fd)

# --- 4b. Combine Features for Training ---
# It's good practice to scale scalar features
scalar_scaler = StandardScaler()
train_scalar_scaled = scalar_scaler.fit_transform(train_scalar)
# Combine FPCA scores and scaled scalar features
X_train = np.hstack((train_fpca_scores, train_scalar_scaled))

# --- 4c. Create Target Variables for Training ---
# Regression Target: The final normalized price of the day
y_train_reg = train_fd.data_matrix[:, -1].flatten()

# Classification Target: Did the afternoon go up or down?
afternoon_returns_train = train_fd.data_matrix[:, -1] - train_fd.data_matrix[:, MORNING_PERIOD_MINUTES - 1]
y_train_class = (afternoon_returns_train > 0).astype(int)

# --- 4d. Train the Machine Learning Models ---
print("Training prediction models...")
# We'll use the classification model for our backtest signal
classifier = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', max_depth=5)
classifier.fit(X_train, y_train_class)

# We can also train the regression model for analysis if needed
# regressor = XGBRegressor(n_estimators=100, random_state=42, early_stopping_rounds=10)
# regressor.fit(X_train, y_train_reg, eval_set=[(X_train, y_train_reg)], verbose=False)

print("Models trained successfully.")


# --- PHASE 5: GENERATE OUT-OF-SAMPLE PREDICTIONS ---
print("\n--- PHASE 5: Generating Predictions for the Test Period ---")

# Transform the test data using the scalers and FPCA fitted on the TRAIN data
test_fpca_scores = fpca.transform(test_fd)
test_scalar_scaled = scalar_scaler.transform(test_scalar)
X_test = np.hstack((test_fpca_scores, test_scalar_scaled))

# Generate predictions using the trained classifier
oos_predictions = classifier.predict(X_test) # OOS = Out-of-Sample


# --- PHASE 6: RUN VECTORBT BACKTEST ---
print("\n--- PHASE 6: Running Realistic Backtest on Test Period Predictions ---")

# Create a pandas Series for our predictions, indexed by the valid test days
# This alignment is CRITICAL for vectorbt
signals = pd.Series(oos_predictions, index=pd.to_datetime(test_labels))

# Create the entry/exit signals
# Signal: 1 = UP, 0 = DOWN. We will go long on UP, and go flat (sell) on DOWN.
entries = (signals == 1)
exits = (signals == 0)

# Get the price data for the backtest (use the close price at 12:30 PM to enter)
price_at_entry = test_df.between_time("12:30", "12:30")['Close']
# Align the price series with our signals
price_at_entry = price_at_entry.reindex(signals.index, method='ffill')

if price_at_entry.isnull().any():
    print("*** WARNING: Could not align all signals with entry prices. Some trades may be missed. ***")
    price_at_entry.ffill(inplace=True) # Forward fill any gaps

# Run the backtest with realistic costs
portfolio = vbt.Portfolio.from_signals(
    close=price_at_entry,  # Use the 12:30 price as our entry/exit point
    entries=entries,
    exits=exits,
    freq='D',             # Daily signals
    init_cash=100000,     # Starting capital
    fees=0.0005,          # 0.05% for commissions and slippage (a reasonable estimate)
    slippage=0.0005
)

# --- Display Results ---
print("\n\n--- BACKTEST RESULTS ---")
print(portfolio.stats())

print("\nPlotting equity curve and drawdowns...")
fig = portfolio.plot(title=f'Backtest Performance for {TICKER_SYMBOL}')
fig.show()

print("\n--- SCRIPT COMPLETE ---")