# SCRIPT 2: 02_data_processing.py (Definitive Plotting Version 2)
# PURPOSE: Manually plot the functional data curves to guarantee success.

import pandas as pd
import skfda
import numpy as np
import plotly.graph_objects as go # We need this for manual plotting
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression, LogisticRegression # <-- ADD LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier # <-- ADD RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report # <-- ADD new metrics
import pandas as pd
from xgboost import XGBRegressor, XGBClassifier

# --- Configuration ---
TICKER_SYMBOL = 'SPY'
INPUT_FILENAME = f"{TICKER_SYMBOL.upper()}_data.csv"
TRADING_START_TIME = '09:30'
TRADING_END_TIME = '16:00'
TIMEZONE = 'America/New_York'
EXPECTED_MINUTES = 391
MINIMUM_DATA_POINTS_THRESHOLD = int(EXPECTED_MINUTES * 0.9)
MORNING_PERIOD_MINUTES = 181

# --- 1. Load Clean Data ---
print(f"--- Loading and Preparing Data from '{INPUT_FILENAME}' ---")
try:
    stock_data = pd.read_csv(INPUT_FILENAME, index_col=0, parse_dates=True)
except FileNotFoundError:
    print(f"*** ERROR: The file '{INPUT_FILENAME}' was not found. ***")
    exit()
# --- NEW: Load Market Data for Context ---
MARKET_FILENAME = 'SPY_data.csv'
print(f"--- Loading Market Context Data from '{MARKET_FILENAME}' ---")
try:
    market_data = pd.read_csv(MARKET_FILENAME, index_col=0, parse_dates=True)
    # The market data needs the same initial timezone conversion
    market_data.index = market_data.index.tz_convert(TIMEZONE)
except FileNotFoundError:
    print(f"*** ERROR: The market data file '{MARKET_FILENAME}' was not found. ***")
    print("Please run '01_data_collection.py' for ticker 'SPY' first.")
    exit()

# --- 2. Pre-process the Time Series ---
print(f"Converting index to timezone: {TIMEZONE}")
stock_data.index = stock_data.index.tz_convert(TIMEZONE)
stock_data = stock_data.between_time(TRADING_START_TIME, TRADING_END_TIME)
unique_days = stock_data.index.normalize().unique()
print(f"Found {len(unique_days)} unique trading days to process.")

# --- 3. Resample and Interpolate into Functional Curves ---
print("\n--- Resampling Daily Curves & Engineering Advanced Features ---")
daily_curves = []
# NEW: Create a list to hold our engineered scalar features for each valid day
engineered_features_list = []
valid_day_labels = [] # To keep track of which days we didn't skip
grid_points = np.arange(EXPECTED_MINUTES)

for day in unique_days:
    day_str = day.strftime('%Y-%m-%d')
    daily_prices_df = stock_data.loc[day_str]
    
    if len(daily_prices_df) < MINIMUM_DATA_POINTS_THRESHOLD:
        print(f"Skipping {day_str}: Too short ({len(daily_prices_df)} points).")
        continue

    # --- Feature B: Calculate Overnight Gap ---
    try:
        # Get the row number for the start of the current day
        current_day_start_loc = stock_data.index.get_loc(daily_prices_df.index[0])
        if current_day_start_loc > 0:
            # Get the closing price of the previous trading day
            previous_close = stock_data.iloc[current_day_start_loc - 1]['Close']
            current_open = daily_prices_df.iloc[0]['Close']
            overnight_gap = (current_open - previous_close) / previous_close
        else:
            overnight_gap = 0 # No previous day for the first day in the dataset
    except Exception as e:
        print(f"Warning: Could not calculate overnight gap for {day_str}. Setting to 0. Error: {e}")
        overnight_gap = 0


    # --- Standard Resampling and Normalization (from your original code) ---
    target_index = pd.date_range(
        start=f"{day_str} {TRADING_START_TIME}", 
        end=f"{day_str} {TRADING_END_TIME}", 
        freq='1min', 
        tz=TIMEZONE
    )
    resampled_day = daily_prices_df.reindex(target_index).interpolate(method='linear').ffill().bfill()
    
    curve_data = resampled_day['Close'].values
    curve_zeroed = curve_data - curve_data[0]
    curve_std = np.std(curve_zeroed)
    if curve_std > 1e-6:
        curve_normalized = curve_zeroed / curve_std
    else:
        curve_normalized = curve_zeroed # Avoid division by zero for flat days
        
    daily_curves.append(curve_normalized)
    valid_day_labels.append(day_str) # Keep track of the day

    # --- Feature A: Calculate Market Context (SPY) Features ---
    try:
        # Process SPY data for the same day
        spy_day_df = market_data.loc[day_str]
        spy_resampled = spy_day_df.reindex(target_index).interpolate(method='linear').ffill().bfill()
        
        spy_curve_data = spy_resampled['Close'].values
        spy_curve_zeroed = spy_curve_data - spy_curve_data[0]
        spy_curve_std = np.std(spy_curve_zeroed)
        spy_normalized = spy_curve_zeroed / spy_curve_std if spy_curve_std > 1e-6 else spy_curve_zeroed
        
        spy_morning_curve = spy_normalized[:MORNING_PERIOD_MINUTES]
        spy_morning_return = spy_morning_curve[-1]
        spy_morning_volatility = np.std(spy_morning_curve)
    except KeyError:
        # Handle case where SPY has no data for this day (e.g., market holiday for SPY but not the stock)
        print(f"Warning: No SPY data for {day_str}. Using 0 for market features.")
        spy_morning_return = 0
        spy_morning_volatility = 0
        

    # --- Feature C: Calculate Advanced Scalar Features ---
    morning_curve = curve_normalized[:MORNING_PERIOD_MINUTES]
    
    # Original scalar features
    volatility = np.std(morning_curve)
    morning_return = morning_curve[-1]

    # New scalar features
    morning_grid_points = np.arange(len(morning_curve))
    slope = np.polyfit(morning_grid_points, morning_curve, 1)[0]
    
    # Calculate Max Drawdown
    running_max = np.maximum.accumulate(morning_curve)
    drawdown = (morning_curve - running_max) / (running_max + 1e-9) # as percentage
    max_drawdown = np.min(drawdown)


    # --- Combine all engineered features for this day ---
    engineered_features_list.append([
        volatility, 
        morning_return,
        overnight_gap,
        slope,
        max_drawdown,
        spy_morning_return,
        spy_morning_volatility
    ])

# --- End of Modified Loop ---

if not daily_curves:
    print("\n*** ERROR: No valid trading days could be processed. ***")
    exit()

fd_object = skfda.FDataGrid(
    data_matrix=daily_curves,
    grid_points=grid_points,
)
print(f"\nSuccessfully created a functional data object with {fd_object.n_samples} resampled curves.")

# --- 4. Visualize the Functional Data (Manual Method) ---
print("\n--- Manually Building Plot from Functional Data ---")

# 1. Create an empty figure object
fig = go.Figure()

# 2. Loop through each curve in our functional data object
for i in range(fd_object.n_samples):
    # Add a new line (trace) to the figure for each day
    fig.add_trace(go.Scatter(
        x=fd_object.grid_points[0],  # The x-axis (minutes)
        y=fd_object.data_matrix[i].flatten(), # The y-axis (price for day i)
        mode='lines',
        line=dict(width=1), # Make the lines thin
        showlegend=False
    ))

# 3. Customize the layout of the entire figure
fig.update_layout(
    title=f'Resampled Intraday Price Curves for SPY ({fd_object.n_samples} Trading Days)',
    xaxis_title=f'Minutes from Market Open ({TRADING_START_TIME} ET)',
    yaxis_title='Price (USD)',
)

# 4. Show the plot
fig.show()

# --- 5. Functional Principal Component Analysis (FPCA) ---
print("\n--- Performing Functional Principal Component Analysis (FPCA) ---")

mean_curve = fd_object.mean()
fd_centered = fd_object - mean_curve

N_COMPONENTS = 3
fpca = skfda.preprocessing.dim_reduction.FPCA(n_components=N_COMPONENTS)

print(f"Fitting FPCA to find the top {N_COMPONENTS} principal components...")
fpca.fit(fd_centered)


# --- 6. Visualize the FPCA Results (Intuitive Method) ---
print("\n--- Visualizing the Effect of Principal Components ---")

# First, we need to get the "scores" for each day on each component.
# This tells us how much of each component pattern each day contained.
fd_scores = fpca.transform(fd_centered)

# Create a new figure for our FPCA visualization
fig_fpca_effects = go.Figure()

# Add the mean curve as a bold black line
fig_fpca_effects.add_trace(go.Scatter(
    x=grid_points,
    y=mean_curve.data_matrix[0].flatten(),
    mode='lines',
    line=dict(color='black', width=3),
    name='Mean Daily Curve'
))

# Define colors for the components' effects
component_colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # Blue, Orange, Green

# For each component, show its effect by adding/subtracting it from the mean
for i in range(N_COMPONENTS):
    # Get the component curve
    component_curve = fpca.components_[i]
    
    # Calculate the standard deviation of the scores for this component.
    # This gives us a measure of how much this component typically varies.
    score_std_dev = np.std(fd_scores[:, i])
    
    # Calculate the upper and lower bounds of the effect
    upper_bound = mean_curve + score_std_dev * component_curve
    lower_bound = mean_curve - score_std_dev * component_curve
    
    # Add the upper bound trace
    fig_fpca_effects.add_trace(go.Scatter(
        x=grid_points,
        y=upper_bound.data_matrix[0].flatten(),
        mode='lines',
        line=dict(width=1, color=component_colors[i]),
        name=f'+1 Std Dev (Comp {i+1})'
    ))
    
    # Add the lower bound trace, filled to the upper bound
    fig_fpca_effects.add_trace(go.Scatter(
        x=grid_points,
        y=lower_bound.data_matrix[0].flatten(),
        mode='lines',
        line=dict(width=1, color=component_colors[i]),
        fill='tonexty', # This fills the area between this trace and the previous one
        name=f'-1 Std Dev (Comp {i+1})'
    ))


# Customize the layout
fig_fpca_effects.update_layout(
    title='Effect of Principal Components on SPY Intraday Price',
    xaxis_title=f'Minutes from Market Open ({TRADING_START_TIME} ET)',
    # The y-axis is now back to actual price, which is intuitive
    yaxis_title='Price (USD)',
    legend_title='Component Effects'
)

fig_fpca_effects.show()

# --- 7. Analyze FPCA Results ---
# This part remains the same.
print("\nExplained variance ratio by each component:")
for i, ratio in enumerate(fpca.explained_variance_ratio_):
    print(f"  - Component {i+1}: {ratio:.2%}")

# --- 8. Visualize Daily Fingerprints with a Heatmap ---
print("\n--- Generating Heatmap of Daily FPCA Scores ---")

# We already calculated the scores in the previous step.
# They are stored in the `fd_scores` variable.

# For better visualization, it helps to scale the scores.
# We'll use StandardScaler so that each component's scores have a mean of 0 and a standard deviation of 1.
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_scores = scaler.fit_transform(fd_scores)

# Create the heatmap using Plotly Express (which is great for this)
import plotly.express as px

# We need labels for the rows (days)
day_labels = [f"Day {i+1}" for i in range(fd_object.n_samples)]
component_labels = [f"PC {i+1}" for i in range(N_COMPONENTS)]

fig_heatmap = px.imshow(
    scaled_scores,
    labels=dict(x="Principal Component", y="Trading Day", color="Scaled Score"),
    x=component_labels,
    y=day_labels,
    color_continuous_scale='RdBu_r', # Red-White-Blue, a good choice for positive/negative values
    zmin=-2.5, # Setting min/max color range makes the plot consistent
    zmax=2.5
)

fig_heatmap.update_layout(
    title='Daily "Fingerprints": FPCA Score for Each Day'
)

fig_heatmap.show()









# --- IMPORTS FOR PREDICTION PHASE ---
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np
import plotly.express as px

# ===================================================================
# --- PHASE 3: EXPERIMENTAL FRAMEWORK FOR PREDICTION ---
# ===================================================================

def train_and_evaluate_model(X, y, feature_names, ticker, model):
    """
    A reusable function to train and evaluate our model on a given feature set.
    """
     # 1. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, shuffle=False
    )
    
    # 2. Train the PROVIDED model
    if isinstance(model, XGBRegressor):
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    else:
        model.fit(X_train, y_train)
    
    # 3. Evaluate and return results
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Create a results plot
    results_df = pd.DataFrame({'Actual Close': y_test, 'Predicted Close': y_pred})
    fig = px.scatter(
        results_df, x='Actual Close', y='Predicted Close',
        title=f'Performance for {ticker} using features: {", ".join(feature_names)}',
        labels={'Actual Close': 'Actual Closing Price (USD)', 'Predicted Close': 'Predicted Closing Price (USD)'}
    )
    fig.add_shape(
        type='line', x0=y_test.min(), y0=y_test.min(),
        x1=y_test.max(), y1=y_test.max(),
        line=dict(color='Red', width=2, dash='dash'), name='Perfect Prediction'
    )
    
    return mae, fig

def plot_feature_importance(model, feature_names, model_name, ticker):
    """
    Analyzes and plots the feature importances from a tree-based model.
    """
    importances = model.feature_importances_
    # Create a DataFrame for easy sorting and plotting
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    # Create the plot
    fig = px.bar(
        importance_df,
        x='Importance',
        y='Feature',
        orientation='h',
        title=f'Feature Importance for {model_name} on {ticker}'
    )
    fig.update_layout(yaxis={'categoryorder':'total ascending'}) # Show most important at top
    fig.show()
    return importance_df

# --- 9. Generate All Potential Features ---
print("\n--- Phase 3: Generating Feature Sets for Experimentation ---")

# --- 9a. Create Functional Features (FPCA Scores) ---
start_time, end_time = grid_points[0], grid_points[MORNING_PERIOD_MINUTES - 1]

# CORRECTED LINE: Pass the start and end times as a single tuple
fd_object_morning = fd_object.restrict((start_time, end_time))

fpca_morning = skfda.preprocessing.dim_reduction.FPCA(n_components=N_COMPONENTS)
X_fpca = fpca_morning.fit_transform(fd_object_morning)

# --- 9b. Create a Feature Matrix from our Engineered Features ---
# We already calculated everything in the loop, so now we just convert it to a NumPy array.
X_scalar_advanced = np.array(engineered_features_list)

# --- 9d. Define Target Variable (y) ---
y = fd_object.data_matrix[:, -1].flatten()

# --- NEW: Define a Classification Target (y_class) ---
# Goal: Predict if the afternoon session will be UP (1) or DOWN (0)
print("\n--- Creating Classification Target (Afternoon UP/DOWN) ---")

y_class = []
# We use the un-normalized data to get the real prices
# We iterate through the days we ACTUALLY PROCESSED
for i, day_str in enumerate(valid_day_labels):
    full_day_prices = stock_data.loc[day_str]['Close']
    
    # Get the real price at the end of the morning (12:30 PM)
    morning_close_price = full_day_prices.iloc[MORNING_PERIOD_MINUTES - 1]
    
    # Get the final closing price
    final_close_price = full_day_prices.iloc[-1]
    
    # Compare them to define the target
    if final_close_price > morning_close_price:
        y_class.append(1) # Afternoon was UP
    else:
        y_class.append(0) # Afternoon was DOWN or FLAT

y_class = np.array(y_class)
print(f"Classification target created with {np.sum(y_class)} UP days and {len(y_class) - np.sum(y_class)} DOWN days.")

# --- 10. Define Experimental Feature Sets ---
# --- 10. Define Experimental Feature Sets ---
print("\n--- Defining Feature Sets to Test ---")

# Define the names for our new advanced feature set for prettier plots and reports
advanced_scalar_names = [
    'Volatility', 'Return', 'Gap', 'Slope', 'MaxDrawdown', 'SPY_Return', 'SPY_Vol'
]

features_to_test = {
    # Test just the FPCA scores
    "FPCA_Only": {
        "X": X_fpca,
        "names": [f'PC_{i+1}' for i in range(N_COMPONENTS)]
    },
    # Test just our new, powerful scalar features
    "Advanced_Scalar_Only": {
        "X": X_scalar_advanced,
        "names": advanced_scalar_names
    },
    # Test the combination of FPCA and our new scalar features
    "FPCA_and_Advanced_Scalar": {
        "X": np.hstack((X_fpca, X_scalar_advanced)),
        "names": [f'PC_{i+1}' for i in range(N_COMPONENTS)] + advanced_scalar_names
    }
}

# --- 11. Run the Experiments ---
print("\n--- Running Experiments ---")
experiment_results = []

# --- NEW: Define the models we want to test ---
models_to_test = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42, eval_metric='mae', early_stopping_rounds=10)
}

for model_name, model_obj in models_to_test.items():
    print(f"\n--- Testing Model: {model_name} ---")
    for name, features in features_to_test.items():
        print(f"  Testing feature set: {name}...")
        mae, fig = train_and_evaluate_model(
            features["X"], y, features["names"], TICKER_SYMBOL, model=model_obj
        )
        
        # Add the model name to the results for a clear comparison
        experiment_results.append({'Model': model_name, 'Feature Set': name, 'MAE': mae})

# --- 12. Display Final Comparison ---
print("\n\n--- FINAL EXPERIMENT RESULTS ---")
final_results_df = pd.DataFrame(experiment_results).sort_values(by='MAE')
print(final_results_df.to_string(index=False))

# --- NEW: Show the plot for the best performing MODEL and FEATURE SET combination ---
best_result = final_results_df.iloc[0]
best_model_name = best_result['Model']
best_set_name = best_result['Feature Set']

print(f"\nDisplaying plot for the best performing combination: Model='{best_model_name}', Features='{best_set_name}'")

# Get the model object and feature set from our dictionaries
best_model_obj = models_to_test[best_model_name]
best_feature_set = features_to_test[best_set_name]

# Now call the function with all the required arguments
mae, best_fig = train_and_evaluate_model(
    best_feature_set["X"],
    y,
    best_feature_set["names"],
    TICKER_SYMBOL,
    model=best_model_obj  # <-- The missing argument is now provided
)
best_fig.show()

print("\n--- Analyzing Feature Importances from RandomForest on all features ---")
# We choose the RandomForest model because it handles all features well.
# We choose the 'FPCA_and_Advanced_Scalar' set because it contains every feature.
rf_model_for_analysis = models_to_test['RandomForest']
features_for_analysis = features_to_test['FPCA_and_Advanced_Scalar']

# We need to split the data and train the model just like in the main function
X_train, X_test, y_train, y_test = train_test_split(
    features_for_analysis["X"], y, test_size=0.25, shuffle=False
)
rf_model_for_analysis.fit(X_train, y_train)

# Now, call our new function to plot the importances
importance_df = plot_feature_importance(
    rf_model_for_analysis, features_for_analysis["names"], "RandomForest", TICKER_SYMBOL
)
print(importance_df)

# ===================================================================
# --- PHASE 4: NEW EXPERIMENT - PREDICTING AFTERNOON DIRECTION ---
# ===================================================================
print("\n\n\n**************************************************")
print("*** PHASE 4: CLASSIFICATION EXPERIMENTS      ***")
print("**************************************************")

def train_and_evaluate_classifier(X, y, feature_names, ticker, model):
    """
    A reusable function for our new CLASSIFICATION task.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, shuffle=False, stratify=None # No stratify for time series
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    # The classification_report provides precision, recall, f1-score
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    return accuracy, report


# --- Define the classifiers we want to test ---
classifiers_to_test = {
    "LogisticRegression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "RandomForestClassifier": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    "XGBClassifier": XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
}

# --- Run the Classification Experiments ---
class_experiment_results = []
print("\n--- Running Classification Experiments ---")

for model_name, model_obj in classifiers_to_test.items():
    print(f"\n--- Testing Classifier: {model_name} ---")
    # We can reuse the same feature sets from before
    for name, features in features_to_test.items():
        print(f"  Testing feature set: {name}...")
        
        # Use our new classification target y_class
        accuracy, report = train_and_evaluate_classifier(
            features["X"], y_class, features["names"], TICKER_SYMBOL, model=model_obj
        )
        
        class_experiment_results.append({
            'Model': model_name, 
            'Feature Set': name, 
            'Accuracy': accuracy,
            'F1-Score (UP)': report['1']['f1-score'], # How well it predicts UP days
            'F1-Score (DOWN)': report['0']['f1-score'] # How well it predicts DOWN days
        })

# --- Display Final Classification Comparison ---
print("\n\n--- FINAL CLASSIFICATION EXPERIMENT RESULTS ---")
class_results_df = pd.DataFrame(class_experiment_results).sort_values(by='Accuracy', ascending=False)
print(class_results_df.to_string(index=False))