# ==============================================================================
# SCRIPT: 02_analysis_and_backtest.py (VERSION 4.0 - FINAL TIMEZONE FIX)
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

print("--- SCRIPT START: ANALYSIS AND BACKTESTING (V4.0 - FINAL TIMEZONE FIX) ---")

# --- HELPER FUNCTION FOR ADVANCED FEATURES ---
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


# --- PHASE 1: LOAD AND PREPARE DATA ---
print("\n--- PHASE 1: Loading and Preparing Data ---")
TICKER_SYMBOL = 'QQQ'
INPUT_FILENAME = f"{TICKER_SYMBOL.upper()}_data_alpaca.csv"
TRADING_START_TIME = '09:30'; TRADING_END_TIME = '16:00'
TIMEZONE = 'America/New_York'; EXPECTED_MINUTES = 391
MORNING_PERIOD_MINUTES = 180
MINIMUM_DATA_POINTS_THRESHOLD = int(EXPECTED_MINUTES * 0.9)
PROBABILITY_THRESHOLD = 0.55
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

# 1. Create a clean daily DataFrame. The index is automatically timezone-aware.
daily_features_df = stock_data['Close'].resample('D').last().to_frame(name='Close')
daily_features_df.dropna(inplace=True)

# 2. Calculate features.
daily_features_df[ma_col_name] = daily_features_df['Close'].rolling(window=REGIME_FILTER_MA_WINDOW).mean()
daily_features_df['prev_close'] = daily_features_df['Close'].shift(1)

# 3. Drop all rows that have NaN values from the rolling/shifting operations.
daily_features_df.dropna(inplace=True)
print(f"Daily features lookup table created with {len(daily_features_df)} valid trading days.")

# 4. Filter the main minute-level data to only include days for which we have valid daily features.
valid_dates = daily_features_df.index
stock_data = stock_data[stock_data.index.normalize().isin(valid_dates)]
print(f"Minute-level data filtered to {len(stock_data)} rows matching valid daily features.")

# 5. Filter for market hours.
stock_data_filtered = stock_data.between_time(TRADING_START_TIME, TRADING_END_TIME)


# --- PHASE 2: ANCHORED TIME-SERIES SPLIT ---
print("\n--- PHASE 2: Splitting Data into Training and Test Sets ---")
TRAIN_END_DATE = '2023-12-31'
train_df = stock_data_filtered.loc[stock_data_filtered.index.date < pd.to_datetime(TRAIN_END_DATE).date()]
test_df = stock_data_filtered.loc[stock_data_filtered.index.date >= pd.to_datetime(TRAIN_END_DATE).date()]

if train_df.empty or test_df.empty:
    print("*** CRITICAL ERROR: Training or Test DataFrame is empty after splitting. Check dates and data. ***")
    exit()

print(f"Training set: {train_df.index.min().date()} to {train_df.index.max().date()}")
print(f"Test set:     {test_df.index.min().date()} to {test_df.index.max().date()}")


# --- PHASE 3: PROCESS DATA & ENGINEER FEATURES (FUNCTION) ---
print("\n--- PHASE 3: Engineering Features ---")
def process_data_into_curves_and_features(df, morning_split_minute, daily_lookup_df):
    unique_days = df.index.normalize().unique()
    daily_curves, scalar_features_list, valid_day_labels = [], [], []
    grid_points = np.arange(EXPECTED_MINUTES)
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

        curve_data = resampled_day['Close'].values
        curve_zeroed = curve_data - curve_data[0]
        curve_std = np.std(curve_zeroed)
        curve_normalized = curve_zeroed / curve_std if curve_std > 1e-6 else curve_zeroed
        daily_curves.append(curve_normalized)
        valid_day_labels.append(day)
        
        open_price = resampled_day['Open'].iloc[0]
        overnight_gap = (open_price - prev_close) / prev_close
        is_uptrend = 1 if open_price > ma_val else 0
        
        morning_curve = curve_normalized[:morning_split_minute]
        morning_return = morning_curve[-1] if len(morning_curve) > 0 else 0
        volatility = np.std(morning_curve) if len(morning_curve) > 0 else 0
        slope = np.polyfit(np.arange(len(morning_curve)), morning_curve, 1)[0] if len(morning_curve) > 1 else 0
        morning_df = resampled_day.iloc[:morning_split_minute]
        adv_features = calculate_advanced_features(morning_df)

        scalar_features_list.append([
            volatility, morning_return, overnight_gap, slope,
            adv_features['max_drawdown'], adv_features['vwap_deviation'],
            adv_features['vol_of_vol'], is_uptrend
        ])

    if not daily_curves:
        return None, np.array([]), []
    fd_object = skfda.FDataGrid(data_matrix=daily_curves, grid_points=grid_points)
    return fd_object, np.array(scalar_features_list), valid_day_labels

train_fd, train_scalar, train_labels = process_data_into_curves_and_features(train_df, MORNING_PERIOD_MINUTES, daily_features_df)
test_fd, test_scalar, test_labels = process_data_into_curves_and_features(test_df, MORNING_PERIOD_MINUTES, daily_features_df)

if train_fd is None or test_fd is None:
    print("*** CRITICAL ERROR: Feature engineering resulted in no valid data. Check input DataFrames and processing logic. ***")
    exit()

# --- PHASES 4 & 5: Fitting, Predicting, and Backtesting ---
print("\n--- PHASES 4 & 5: Fitting, Predicting, and Backtesting (with Confidence & Regime Filter) ---")
N_COMPONENTS = 5
fpca = skfda.preprocessing.dim_reduction.FPCA(n_components=N_COMPONENTS)
train_fpca_scores = fpca.fit_transform(train_fd)
scalar_scaler = StandardScaler()
train_scalar_scaled = scalar_scaler.fit_transform(train_scalar)
X_train = np.hstack((train_fpca_scores, train_scalar_scaled))
afternoon_returns_train = train_fd.data_matrix[:, -1] - train_fd.data_matrix[:, MORNING_PERIOD_MINUTES - 1]
y_train_class = (afternoon_returns_train > 0).astype(int).ravel()
print("Training classifier...")
classifier = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', max_depth=5, n_jobs=-1)
classifier.fit(X_train, y_train_class)
scalar_feature_names = [
    'Volatility', 'MorningReturn', 'OvernightGap', 'Slope',
    'MaxDrawdown', 'VWAP_Dev', 'VolOfVol', f'IsUptrend_MA{REGIME_FILTER_MA_WINDOW}'
]
feature_names = [f'PC_{i+1}' for i in range(N_COMPONENTS)] + scalar_feature_names
importances = classifier.feature_importances_
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)
fig_importance = px.bar(importance_df, x='Importance', y='Feature', orientation='h', title=f'Feature Importance ({TICKER_SYMBOL})')
fig_importance.update_layout(yaxis={'categoryorder':'total ascending'})
print("Generating out-of-sample predictions with confidence scoring...")
test_fpca_scores = fpca.transform(test_fd)
test_scalar_scaled = scalar_scaler.transform(test_scalar)
X_test = np.hstack((test_fpca_scores, test_scalar_scaled))
oos_probabilities = classifier.predict_proba(X_test)[:, 1]
signals_from_prob = pd.Series(oos_probabilities > PROBABILITY_THRESHOLD, index=pd.to_datetime(test_labels))
daily_test_data = test_df.resample('D').first()

### --- DEFINITIVE FIX FOR TIMEZONE ERROR --- ###
# Get all calendar days in the test set range, preserving the timezone.
all_test_days_index = pd.date_range(start=test_df.index.min().normalize(), end=test_df.index.max().normalize(), freq='D', tz=TIMEZONE)

# Reindex daily_test_data to this complete, timezone-aware calendar index.
daily_test_data = daily_test_data.reindex(all_test_days_index, method=None) # No ffill needed yet

# Now get the MA values using a clean map. Both indices are aware.
daily_test_data[ma_col_name] = daily_test_data.index.map(daily_features_df[ma_col_name])
daily_test_data.ffill(inplace=True) # Forward-fill values over non-trading days (weekends)

regime_is_up = daily_test_data['Open'] > daily_test_data[ma_col_name]

# Align the daily filter with the signal index (which only contains trading days)
regime_filter = regime_is_up.reindex(signals_from_prob.index, method='ffill')
### --- END OF FIX --- ###

entries = signals_from_prob & regime_filter
exits = ~entries

print(f"Generated {entries.sum()} trade signals after applying a {PROBABILITY_THRESHOLD*100}% confidence threshold and a {REGIME_FILTER_MA_WINDOW}-day MA regime filter.")

### --- DEBUGGING ANALYSIS SECTION --- ###
print("\n--- Running Debugging Analysis ---")
if entries.sum() == 0:
    print("!!! No trades were generated. Analyzing filters independently. !!!")
    print(f"Total days where Regime Filter was TRUE: {regime_filter.sum()} / {len(regime_filter)}")
    print(f"Total days where Model Confidence > {PROBABILITY_THRESHOLD*100}%: {signals_from_prob.sum()} / {len(signals_from_prob)}")
    fig_debug = go.Figure()
    fig_debug.add_trace(go.Scatter(x=pd.to_datetime(test_labels), y=oos_probabilities, mode='lines', name='Model Confidence (Prob of UP)'))
    fig_debug.add_trace(go.Scatter(x=pd.to_datetime(test_labels), y=[PROBABILITY_THRESHOLD] * len(test_labels), mode='lines', name=f'Confidence Threshold ({PROBABILITY_THRESHOLD*100}%)', line=dict(color='red', dash='dash')))
    regime_up_dates = regime_filter[regime_filter].index
    for day in regime_up_dates:
        fig_debug.add_vrect(x0=day, x1=day + pd.Timedelta(days=1), fillcolor="green", opacity=0.15, layer="below", line_width=0)
    fig_debug.update_layout(title='Diagnostic: Model Confidence vs. Regime Filter', xaxis_title='Date', yaxis_title='Prediction Probability', yaxis_range=[0, 1])
    fig_debug.show()

### --- END DEBUGGING ANALYSIS SECTION --- ###

print("Running backtest with new filtered signals...")
if entries.sum() > 0:
    entry_time_str = (pd.to_datetime(TRADING_START_TIME) + pd.Timedelta(minutes=MORNING_PERIOD_MINUTES)).strftime('%H:%M')
    price_at_entry = test_df.between_time(entry_time_str, entry_time_str)['Close']
    price_at_entry = price_at_entry.reindex(entries.index, method='ffill')
    if price_at_entry.isnull().any():
        price_at_entry.ffill(inplace=True); price_at_entry.bfill(inplace=True)
    portfolio = vbt.Portfolio.from_signals(close=price_at_entry, entries=entries, exits=exits, freq='D', init_cash=100000, fees=0.0012, slippage=0.0012)
    print("\n\n--- BACKTEST RESULTS (WITH CONFIDENCE & REGIME FILTER) ---")
    print(portfolio.stats())
    print("\nPlotting equity curve and feature importance plot...")
    fig_backtest = portfolio.plot(title=f'Backtest for {TICKER_SYMBOL} with Confidence & Regime Filter')
    fig_backtest.show()
    fig_importance.show()
else:
    print("\n--- No trades to backtest. Skipping portfolio analysis. ---")

print("\n--- SCRIPT COMPLETE ---")