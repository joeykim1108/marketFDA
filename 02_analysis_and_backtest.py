# ==============================================================================
# SCRIPT: 02_analysis_and_backtest.py (with Advanced Feature Engineering)
# ==============================================================================
import pandas as pd
import skfda
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import vectorbt as vbt
from tqdm import tqdm  # For progress bars

print("--- SCRIPT START: ANALYSIS AND BACKTESTING ---")


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
TICKER_SYMBOL = 'SPY'
INPUT_FILENAME = f"{TICKER_SYMBOL.upper()}_data_alpaca.csv"
TRADING_START_TIME = '09:30'; TRADING_END_TIME = '16:00'
TIMEZONE = 'America/New_York'; EXPECTED_MINUTES = 391

MORNING_PERIOD_MINUTES = 110

MINIMUM_DATA_POINTS_THRESHOLD = int(EXPECTED_MINUTES * 0.9)

try:
    stock_data = pd.read_csv(INPUT_FILENAME, index_col=0, parse_dates=True)
    if stock_data.index.tz is None:
        stock_data.index = stock_data.index.tz_localize('UTC').tz_convert(TIMEZONE)
    else:
        stock_data.index = stock_data.index.tz_convert(TIMEZONE)
    print(f"Successfully loaded {len(stock_data)} rows from '{INPUT_FILENAME}'.")
except FileNotFoundError:
    print(f"*** ERROR: The file '{INPUT_FILENAME}' was not found. ***"); exit()

daily_closes = stock_data['Close'].resample('D').last()
stock_data['ma_50'] = daily_closes.rolling(window=50).mean().reindex(stock_data.index, method='ffill')


# --- PHASE 2: ANCHORED TIME-SERIES SPLIT ---
print("\n--- PHASE 2: Splitting Data into Training and Test Sets ---")
TRAIN_END_DATE = '2023-12-31'
stock_data_filtered = stock_data.between_time(TRADING_START_TIME, TRADING_END_TIME)
train_df = stock_data_filtered.loc[stock_data_filtered.index.date < pd.to_datetime(TRAIN_END_DATE).date()]
test_df = stock_data_filtered.loc[stock_data_filtered.index.date >= pd.to_datetime(TRAIN_END_DATE).date()]
print(f"Training set: {train_df.index.min().date()} to {train_df.index.max().date()}")
print(f"Test set:     {test_df.index.min().date()} to {test_df.index.max().date()}")


# --- PHASE 3: PROCESS DATA & ENGINEER FEATURES (FUNCTION) ---
print("\n--- PHASE 3: Engineering Features ---")

def process_data_into_curves_and_features(df, morning_split_minute):
    unique_days = df.index.normalize().unique()
    daily_curves, scalar_features_list, valid_day_labels, curve_colors = [], [], [], []
    grid_points = np.arange(EXPECTED_MINUTES)
    
    print(f"Processing {len(unique_days)} days...")
    for day in tqdm(unique_days):
        day_str = day.strftime('%Y-%m-%d')
        daily_prices_df = df.loc[day_str]
        
        if len(daily_prices_df) < MINIMUM_DATA_POINTS_THRESHOLD: continue

        target_index = pd.date_range(start=f"{day_str} {TRADING_START_TIME}", end=f"{day_str} {TRADING_END_TIME}", freq='1min', tz=TIMEZONE)
        resampled_day = daily_prices_df.reindex(target_index).interpolate(method='linear').ffill().bfill()
        
        price_at_split = resampled_day['Close'].iloc[morning_split_minute - 1]
        final_price = resampled_day['Close'].iloc[-1]
        curve_colors.append('green' if final_price > price_at_split else 'red')

        curve_data = resampled_day['Close'].values
        curve_zeroed = curve_data - curve_data[0]
        curve_std = np.std(curve_zeroed)
        curve_normalized = curve_zeroed / curve_std if curve_std > 1e-6 else curve_zeroed
            
        daily_curves.append(curve_normalized)
        valid_day_labels.append(day)

        try:
            loc = stock_data.index.get_loc(daily_prices_df.index[0])
            prev_close = stock_data.iloc[loc - 1]['Close']
            open_price = daily_prices_df.iloc[0]['Open']
            overnight_gap = (open_price - prev_close) / prev_close
        except Exception:
            overnight_gap = 0
        
        morning_curve = curve_normalized[:morning_split_minute]
        morning_return = morning_curve[-1] if len(morning_curve) > 0 else 0
        volatility = np.std(morning_curve) if len(morning_curve) > 0 else 0
        slope = np.polyfit(np.arange(len(morning_curve)), morning_curve, 1)[0] if len(morning_curve) > 1 else 0
        
        morning_df = resampled_day.iloc[:morning_split_minute]
        adv_features = calculate_advanced_features(morning_df)

        ma50 = resampled_day['ma_50'].iloc[0]
        open_price = resampled_day['Open'].iloc[0]
        is_uptrend = 1 if open_price > ma50 else 0
        
        scalar_features_list.append([
            volatility, morning_return, overnight_gap, slope,
            adv_features['max_drawdown'], adv_features['vwap_deviation'], 
            adv_features['vol_of_vol'], is_uptrend
        ])
        
    fd_object = skfda.FDataGrid(data_matrix=daily_curves, grid_points=grid_points)
    
    return fd_object, np.array(scalar_features_list), valid_day_labels, curve_colors

train_fd, train_scalar, train_labels, train_curve_colors = process_data_into_curves_and_features(train_df, MORNING_PERIOD_MINUTES)
test_fd, test_scalar, test_labels, _ = process_data_into_curves_and_features(test_df, MORNING_PERIOD_MINUTES)


# --- (Phases 4 & 5 are condensed as they are mostly unchanged) ---
print("\n--- PHASES 4 & 5: Fitting, Predicting, and Backtesting ---")
N_COMPONENTS = 5
fpca = skfda.preprocessing.dim_reduction.FPCA(n_components=N_COMPONENTS)
train_fpca_scores = fpca.fit_transform(train_fd)
scalar_scaler = StandardScaler()
train_scalar_scaled = scalar_scaler.fit_transform(train_scalar)
X_train = np.hstack((train_fpca_scores, train_scalar_scaled))
afternoon_returns_train = train_fd.data_matrix[:, -1] - train_fd.data_matrix[:, MORNING_PERIOD_MINUTES - 1]
y_train_class = (afternoon_returns_train > 0).astype(int).ravel()
print("Training classifier with new advanced features...")
classifier = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', max_depth=5)
classifier.fit(X_train, y_train_class)

scalar_feature_names = [
    'Volatility', 'MorningReturn', 'OvernightGap', 'Slope', 
    'MaxDrawdown', 'VWAP_Dev', 'VolOfVol', 'IsUptrend'
]
feature_names = [f'PC_{i+1}' for i in range(N_COMPONENTS)] + scalar_feature_names
importances = classifier.feature_importances_
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)
fig_importance = px.bar(importance_df, x='Importance', y='Feature', orientation='h', title=f'Feature Importance with Advanced Features')
fig_importance.update_layout(yaxis={'categoryorder':'total ascending'})

print("Generating out-of-sample predictions...")
test_fpca_scores = fpca.transform(test_fd)
test_scalar_scaled = scalar_scaler.transform(test_scalar)
X_test = np.hstack((test_fpca_scores, test_scalar_scaled))
oos_predictions = classifier.predict(X_test)

print("Running backtest...")
entry_time_str = (pd.to_datetime(TRADING_START_TIME) + pd.Timedelta(minutes=MORNING_PERIOD_MINUTES)).strftime('%H:%M')
signals = pd.Series(oos_predictions, index=pd.to_datetime(test_labels))
entries = (signals == 1); exits = (signals == 0)

# ***** THIS IS THE CORRECTED LINE *****
price_at_entry = test_df.between_time(entry_time_str, entry_time_str)['Close']

price_at_entry = price_at_entry.reindex(signals.index, method='ffill')
if price_at_entry.isnull().any():
    price_at_entry.ffill(inplace=True); price_at_entry.bfill(inplace=True)
portfolio = vbt.Portfolio.from_signals(close=price_at_entry, entries=entries, exits=exits, freq='D', init_cash=100000, fees=0.001, slippage=0.001)

print("\n\n--- BACKTEST RESULTS (WITH ADVANCED FEATURES) ---")
print(portfolio.stats())

print("\nPlotting equity curve and new feature importance plot...")
fig_backtest = portfolio.plot(title=f'Backtest with Advanced Features')
fig_backtest.show()
fig_importance.show()

print("\n--- SCRIPT COMPLETE ---")