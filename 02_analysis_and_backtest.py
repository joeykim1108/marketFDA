# ==============================================================================
# SCRIPT: 02_analysis_and_backtest.py (VERSION 5.3 - DEFINITIVE LOGIC FIX)
# ==============================================================================
import pandas as pd
import skfda
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import vectorbt as vbt
from tqdm import tqdm
import xgboost as xgb

print("--- SCRIPT START: ANALYSIS AND BACKTESTING (V5.3 - DEFINITIVE LOGIC FIX) ---")

# --- HELPER FUNCTION ---
def calculate_advanced_features(morning_df):
    features = {}
    price_curve = morning_df['Close']
    running_max = price_curve.cummax()
    drawdown = (price_curve - running_max) / running_max
    features['max_drawdown'] = drawdown.min() if not drawdown.empty else 0
    if morning_df['Volume'].sum() > 0:
        vwap = (morning_df['Close'] * morning_df['Volume']).sum() / morning_df['Volume'].sum()
        features['vwap_deviation'] = (price_curve.iloc[-1] - vwap) / vwap if vwap > 0 else 0
    else:
        features['vwap_deviation'] = 0
    rolling_vol = price_curve.rolling(window=10).std().dropna()
    features['vol_of_vol'] = rolling_vol.std() if not rolling_vol.empty else 0
    return features


# --- PHASE 1: DATA PREPARATION ---
print("\n--- PHASE 1: Loading and Preparing Data ---")
TICKER_SYMBOL = 'QQQ'
INPUT_FILENAME = f"{TICKER_SYMBOL.upper()}_data_alpaca.csv"
TRADING_START_TIME = '09:30'; TRADING_END_TIME = '16:00'
TIMEZONE = 'America/New_York'; EXPECTED_MINUTES = 391
MORNING_PERIOD_MINUTES = 150
MINIMUM_DATA_POINTS_THRESHOLD = int(EXPECTED_MINUTES * 0.9)
PROBABILITY_THRESHOLD = 0.6
REGIME_FILTER_MA_WINDOW = 200

try:
    stock_data = pd.read_csv(INPUT_FILENAME, index_col=0, parse_dates=True)
    if stock_data.index.tz is None:
        stock_data.index = stock_data.index.tz_localize('UTC').tz_convert(TIMEZONE)
    else:
        stock_data.index = stock_data.index.tz_convert(TIMEZONE)
    print(f"Successfully loaded {len(stock_data)} rows from '{INPUT_FILENAME}'.")
except FileNotFoundError:
    print(f"*** ERROR: The file '{INPUT_FILENAME}' was not found. ***"); exit()

# --- DECOUPLED DAILY FEATURE ENGINEERING ---
ma_col_name = f'ma_{REGIME_FILTER_MA_WINDOW}'
daily_features_df = stock_data['Close'].resample('D').last().to_frame(name='Close')
daily_features_df.dropna(inplace=True)
daily_features_df[ma_col_name] = daily_features_df['Close'].rolling(window=REGIME_FILTER_MA_WINDOW).mean()
daily_features_df['prev_close'] = daily_features_df['Close'].shift(1)
daily_features_df.dropna(inplace=True)
print(f"Daily features lookup table created with {len(daily_features_df)} valid trading days.")
valid_dates = daily_features_df.index
stock_data = stock_data[stock_data.index.normalize().isin(valid_dates)]
print(f"Minute-level data filtered to {len(stock_data)} rows matching valid daily features.")
stock_data_filtered = stock_data.between_time(TRADING_START_TIME, TRADING_END_TIME)


# --- PHASE 2: ANCHORED TIME-SERIES SPLIT ---
print("\n--- PHASE 2: Splitting Data into Training and Test Sets ---")
TEST_START_DATE = pd.to_datetime('2024-01-01')
TRAIN_END_DATE = TEST_START_DATE - pd.Timedelta(days=1)

TRAINING_LOOKBACK_YEARS = 5

TRAIN_START_DATE = TRAIN_END_DATE - pd.DateOffset(years=TRAINING_LOOKBACK_YEARS)
train_df = stock_data_filtered.loc[(stock_data_filtered.index.date >= TRAIN_START_DATE.date()) &
                                   (stock_data_filtered.index.date <= TRAIN_END_DATE.date())]
test_df = stock_data_filtered.loc[stock_data_filtered.index.date >= TEST_START_DATE.date()]
if train_df.empty or test_df.empty:
    print("*** CRITICAL ERROR: Training or Test DataFrame is empty after splitting. Check dates and data. ***")
    exit()
print(f"Training set: {train_df.index.min().date()} to {train_df.index.max().date()}")
print(f"Test set:     {test_df.index.min().date()} to {test_df.index.max().date()}")


# --- PHASE 3: FEATURE ENGINEERING ---
def process_data_into_curves_and_features(df, morning_split_minute, daily_lookup_df):
    unique_days = df.index.normalize().unique()
    morning_curves, scalar_features_list, afternoon_returns_list, valid_day_labels = [], [], [], []
    grid_points = np.arange(morning_split_minute)
    print(f"Processing {len(unique_days)} days...")
    for day in tqdm(unique_days):
        day_str = day.strftime('%Y-%m-%d')
        daily_prices_df = df.loc[day_str]
        if len(daily_prices_df) < MINIMUM_DATA_POINTS_THRESHOLD: continue
        try:
            day_features = daily_lookup_df.loc[day]
            ma_val = day_features[ma_col_name]
            prev_close = day_features['prev_close']
        except KeyError:
            continue
        target_index = pd.date_range(start=f"{day_str} {TRADING_START_TIME}", end=f"{day_str} {TRADING_END_TIME}", freq='1min', tz=TIMEZONE)
        resampled_day = daily_prices_df.reindex(target_index).interpolate(method='linear').ffill().bfill()
        morning_df = resampled_day.iloc[:morning_split_minute]
        morning_curve_raw = morning_df['Close'].values
        morning_curve_zeroed = morning_curve_raw - morning_curve_raw[0]
        morning_curve_std = np.std(morning_curve_zeroed)
        morning_curve_normalized = morning_curve_zeroed / morning_curve_std if morning_curve_std > 1e-6 else morning_curve_zeroed
        open_price = resampled_day['Open'].iloc[0]
        overnight_gap = (open_price - prev_close) / prev_close
        is_uptrend = 1 if open_price > ma_val else 0
        morning_return = (morning_curve_raw[-1] - morning_curve_raw[0]) / morning_curve_raw[0] if morning_curve_raw[0] > 0 else 0
        volatility = np.std(morning_curve_normalized)
        slope = np.polyfit(np.arange(len(morning_curve_normalized)), morning_curve_normalized, 1)[0] if len(morning_curve_normalized) > 1 else 0
        adv_features = calculate_advanced_features(morning_df)
        afternoon_start_price = resampled_day['Close'].iloc[morning_split_minute - 1]
        afternoon_end_price = resampled_day['Close'].iloc[-1]
        afternoon_return = afternoon_end_price - afternoon_start_price
        morning_curves.append(morning_curve_normalized)
        valid_day_labels.append(day)
        afternoon_returns_list.append(afternoon_return)
        scalar_features_list.append([
            volatility, morning_return, overnight_gap, slope,
            adv_features['max_drawdown'], adv_features['vwap_deviation'],
            adv_features['vol_of_vol'], is_uptrend
        ])
        
    if not morning_curves:
        return None, np.array([]), np.array([]), []
    fd_object = skfda.FDataGrid(data_matrix=morning_curves, grid_points=grid_points)
    return fd_object, np.array(scalar_features_list), np.array(afternoon_returns_list), valid_day_labels

train_fd_morning, train_scalar, train_afternoon_returns, train_labels = process_data_into_curves_and_features(train_df, MORNING_PERIOD_MINUTES, daily_features_df)
test_fd_morning, test_scalar, test_afternoon_returns, test_labels = process_data_into_curves_and_features(test_df, MORNING_PERIOD_MINUTES, daily_features_df)
if train_fd_morning is None or test_fd_morning is None:
    print("*** CRITICAL ERROR: Feature engineering resulted in no valid data. Check input DataFrames and processing logic. ***")
    exit()


# --- PHASES 4 & 5: PREDICTION AND BACKTESTING ---
print("\n--- PHASES 4 & 5: Fitting, Predicting, and Backtesting (with Confidence & Regime Filter) ---")
N_COMPONENTS = 5

fpca = skfda.preprocessing.dim_reduction.FPCA(n_components=N_COMPONENTS)
train_fpca_scores = fpca.fit_transform(train_fd_morning)
scalar_scaler = StandardScaler()
train_scalar_scaled = scalar_scaler.fit_transform(train_scalar)
X_train = np.hstack((train_fpca_scores, train_scalar_scaled))
y_train_class = (train_afternoon_returns > 0).astype(int).ravel()



print("Training classifier...")
classifier = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', max_depth=5, n_jobs=-1)
classifier.fit(X_train, y_train_class)

# print("Training XGBoost classifier...")
# scale_pos_weight = np.sum(y_train_class == 0) / np.sum(y_train_class == 1)
# classifier = xgb.XGBClassifier(
#     n_estimators=100,
#     random_state=42,
#     max_depth=5,
#     use_label_encoder=False,
#     eval_metric='logloss',
#     scale_pos_weight=scale_pos_weight, # Use this instead of class_weight
#     n_jobs=-1
# )
# classifier.fit(X_train, y_train_class)

print("Generating out-of-sample predictions with confidence scoring...")
test_fpca_scores = fpca.transform(test_fd_morning)
test_scalar_scaled = scalar_scaler.transform(test_scalar)
X_test = np.hstack((test_fpca_scores, test_scalar_scaled))
oos_probabilities = classifier.predict_proba(X_test)[:, 1]
signals_from_prob = pd.Series(oos_probabilities > PROBABILITY_THRESHOLD, index=pd.to_datetime(test_labels))
daily_test_data_regime = test_df.resample('D').first()
all_test_days_index = pd.date_range(start=test_df.index.min().normalize(), end=test_df.index.max().normalize(), freq='D', tz=TIMEZONE)
daily_test_data_regime = daily_test_data_regime.reindex(all_test_days_index, method=None)
daily_test_data_regime[ma_col_name] = daily_test_data_regime.index.map(daily_features_df[ma_col_name])
daily_test_data_regime.ffill(inplace=True)
regime_is_up = daily_test_data_regime['Open'] > daily_test_data_regime[ma_col_name]
regime_filter = regime_is_up.reindex(signals_from_prob.index, method='ffill')
entries_boolean = signals_from_prob & regime_filter
print(f"Generated {entries_boolean.sum()} trade signals after applying a {PROBABILITY_THRESHOLD*100}% confidence threshold and a {REGIME_FILTER_MA_WINDOW}-day MA regime filter.")

print("Running backtest with new filtered signals...")
if entries_boolean.sum() > 0:
    # 1. Get the intraday price data for the test period. This will be the reference for our backtest.
    price = test_df['Close']

    # 2. Create the boolean entry signals, aligned to the minute-by-minute price index.
    #    The signal should be True ONLY at the exact minute of entry.
    entry_time = (pd.to_datetime(TRADING_START_TIME) + pd.Timedelta(minutes=MORNING_PERIOD_MINUTES)).time()
    
    # Create a Series of entry timestamps
    entry_timestamps = entries_boolean[entries_boolean].index.to_series().apply(
        lambda d: pd.Timestamp(f"{d.strftime('%Y-%m-%d')} {entry_time}", tz=TIMEZONE)
    )
    
    # Create the boolean entry series
    entries = pd.Series(False, index=price.index)
    entries.loc[entry_timestamps] = True

    # 3. Create boolean exit signals. The signal should be True at the close of the entry day.
    #    We can find the last timestamp for each day where we have an entry.
    exit_timestamps = entry_timestamps.apply(lambda ts: ts.normalize() + pd.Timedelta(hours=16)) # 4 PM exit
    
    # Create the boolean exit series
    exits = pd.Series(False, index=price.index)
    exits.loc[exit_timestamps] = True

    # 4. Run the backtest using the minute-level data and boolean signals.
    #    vectorbt will automatically use the price at the timestamp where the signal is True.
    portfolio = vbt.Portfolio.from_signals(
        close=price,  # Use the minute-level closing prices
        entries=entries,
        exits=exits,
        freq='1min',    # Set the frequency to 1 minute
        init_cash=100_000,
        fees=0,
        slippage=0.0002
    )

    print("\n\n--- BACKTEST RESULTS (WITH CONFIDENCE & REGIME FILTER) ---")
    print(portfolio.stats())
    print("\nPlotting equity curve...")
    fig_backtest = portfolio.plot(title=f'Backtest for {TICKER_SYMBOL} with Sell-at-Close Strategy')
    fig_backtest.show()

import matplotlib.pyplot as plt



print("\n--- SCRIPT COMPLETE ---")