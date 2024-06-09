import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
pd.options.mode.chained_assignment = None
import nolds


# Function to add indicators
def calculate_rsi(df, rsi_period=14):
    # Calculate RSI (Relative Strength Index)
    price_change = df['Close'].diff()
    upward_changes = price_change.clip(lower=0)
    downward_changes = -price_change.clip(upper=0)
    average_upward_changes = upward_changes.rolling(window=rsi_period).mean()
    average_downward_changes = downward_changes.rolling(window=rsi_period).mean()
    relative_strength = average_upward_changes / average_downward_changes
    df['RSI'] = 100 - (100 / (1 + relative_strength))

    return df

def calculate_cci(df, cci_period=20):
    # Calculate CCI (Commodity Channel Index)
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    sma_typical_price = typical_price.rolling(window=cci_period).mean()
    mean_deviation = (typical_price - sma_typical_price).abs().rolling(window=cci_period).mean()
    df['CCI'] = (typical_price - sma_typical_price) / (0.015 * mean_deviation)

    return df

def calculate_emadn(df, ema_short_period=10, ema_long_period=40):
    # Calculate short and long period exponential moving averages
    df['ema_short_column'] = df['Close'].ewm(span=ema_short_period, adjust=False).mean()
    df['ema_long_column'] = df['Close'].ewm(span=ema_long_period, adjust=False).mean()

    # Calculate the difference of exponential moving averages and normalize over the longer one
    df['EMADN'] = (df['ema_short_column'] - df['ema_long_column'])/df['ema_long_column']
    
    df.drop(columns=['ema_short_column', 'ema_long_column'], inplace=True)
    
    return df

def calculate_cpc(df, cpc_window=4):
    # Calculate CPC (Current Price Change)
    rolling_mean_close = df['Close'].rolling(cpc_window).mean()
    df['CPC'] = 1 / (1 + np.exp(-(df['Close'] - rolling_mean_close.shift(1)) / rolling_mean_close.shift(1) * 100))

    return df

def grid_search_hurst(df):
    '''
    Function that does a grid search across different window parameters for the indicators and finds the one that results in highest Hurst exponent (R/S estimate). 
    Given multiple Hurst exponent values within 0.01 of each other, the smaller parameter is kept so that the indicators are as responsive as possible.
    '''
    hurst_results = []

    # Search for best CCI period
    best_cci_period = None
    best_cci_hurst = -1
    for cci_period in range(6, 40, 1):
        df_cci = calculate_cci(df[['Close', 'High', 'Low']].copy(), cci_period)
        df_cci = df_cci.dropna(subset=['CCI'])
        H_rs = nolds.hurst_rs(df_cci['CCI'].dropna())
        if best_cci_hurst == -1 or H_rs > best_cci_hurst or (abs(H_rs - best_cci_hurst) <= 0.01 and cci_period < best_cci_period):
            best_cci_period = cci_period
            best_cci_hurst = H_rs
    hurst_results.append({'Indicator': 'CCI', 'Best Period': int(best_cci_period), 'Hurst Exponent': round(best_cci_hurst, 2)})

    # Search for best EMA periods
    best_ema_short_period = None
    best_ema_long_period = None
    best_emadn_hurst = -1
    for ema_short_period in range(10, 25, 1):
        for ema_long_period in range(26, 60, 2):
            df_emadn = calculate_emadn(df[['Close']].copy(), ema_short_period, ema_long_period)
            df_emadn = df_emadn.dropna(subset=['EMADN'])
            H_rs = nolds.hurst_rs(df_emadn['EMADN'].dropna())
            if best_emadn_hurst == -1 or H_rs > best_emadn_hurst or (abs(H_rs - best_emadn_hurst) <= 0.01 and (ema_short_period < best_ema_short_period or ema_long_period < best_ema_long_period)):
                best_ema_short_period = ema_short_period
                best_ema_long_period = ema_long_period
                best_emadn_hurst = H_rs
    hurst_results.append({'Indicator': 'EMADN', 'Best Short Period': int(best_ema_short_period), 'Best Long Period': int(best_ema_long_period), 'Hurst Exponent': round(best_emadn_hurst,2)})

    # Search for best CPC window
    best_cpc_window = None
    best_cpc_hurst = -1
    for cpc_window in range(5, 40, 1):
        df_cpc = calculate_cpc(df[['Close']].copy(), cpc_window)
        df_cpc = df_cpc.dropna(subset=['CPC'])
        H_rs = nolds.hurst_rs(df_cpc['CPC'].dropna())
        if best_cpc_hurst == -1 or H_rs > best_cpc_hurst or (abs(H_rs - best_cpc_hurst) <= 0.01 and cpc_window < best_cpc_window):
            best_cpc_window = cpc_window
            best_cpc_hurst = H_rs
    hurst_results.append({'Indicator': 'CPC', 'Best Period': int(best_cpc_window), 'Hurst Exponent': round(best_cpc_hurst,2)})

    # Search for best RSI period
    best_rsi_period = None
    best_rsi_hurst = -1
    for rsi_period in range(6, 40, 1):
        df_rsi = calculate_rsi(df[['Close']].copy(), rsi_period)
        df_rsi = df_rsi.dropna(subset=['RSI'])
        H_rs = nolds.hurst_rs(df_rsi['RSI'].dropna())
        if best_rsi_hurst == -1 or H_rs > best_rsi_hurst or (abs(H_rs - best_rsi_hurst) <= 0.01 and rsi_period < best_rsi_period):
            best_rsi_period = rsi_period
            best_rsi_hurst = H_rs
    hurst_results.append({'Indicator': 'RSI', 'Best Period': int(best_rsi_period), 'Hurst Exponent': round(best_rsi_hurst, 2)})

    hurst_results_df = pd.DataFrame(hurst_results)
    hurst_results_df['Best Period'] = hurst_results_df['Best Period'].astype('Int64')
    hurst_results_df['Best Short Period'] = hurst_results_df.get('Best Short Period', pd.Series(dtype='Int64')).astype('Int64')
    hurst_results_df['Best Long Period'] = hurst_results_df.get('Best Long Period', pd.Series(dtype='Int64')).astype('Int64')
    
    return hurst_results_df

def calculate_indicators(df, rsi_period=14, cci_period=20, ema_short_period=10, ema_long_period=40, cpc_window=4):
    calculate_rsi(df, rsi_period)
    calculate_cci(df, cci_period)
    calculate_emadn(df, ema_short_period, ema_long_period)
    calculate_cpc(df, cpc_window)

    # Drop rows with na, depending on parameters chosen for indicators
    df.dropna(inplace=True)
    df.reset_index(inplace=True, drop=True)

    return df

def categorize_change(weekly_return, flat_threshold=0.5):
    '''
    Define a function to categorize the return over a week as Bullish (+1), Flat (0), or Bearish (-1) 
    Set a threshold for what we consider to be a flat week, e.g. 0.5% price change
    '''
    if weekly_return > flat_threshold:
        return 1  # Bullish
    elif weekly_return < -flat_threshold:
        return -1  # Bearish
    else:
        return 0  # Flat

def preprocess_market_behavior(df):
    """
    Preprocesses the DataFrame to calculate the percentage change in closing price from one week to the next,
    and creates a target variable for next week's market behavior.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the 'Close' column.

    Returns:
    pd.DataFrame: The preprocessed DataFrame with new columns for current week percentage change,
                  next week percentage change, and next week market behavior.
    """
    # Calculate the percentage change in closing price from one week to the next
    df['Current_week_pct_change'] = 100 * df['Close'].pct_change(periods=1)
    df['Next_week_pct_change'] = 100 * df['Close'].pct_change(periods=1).shift(-1)

    # Apply the function to the percentage change column to create the target variable
    df['Next_week_market_behavior'] = df['Next_week_pct_change'].apply(categorize_change)

    # Drop the first and last rows as they contain NaN values in the 'Current_week_pct_change' and 'Next_week_pct_change' columns
    df.dropna(subset=['Current_week_pct_change', 'Next_week_pct_change'], inplace=True)
    df.reset_index(inplace=True, drop=True)

    return df

def calculate_returns(df, prediction_column):
    """
    Calculates the portfolio return based on the predictions and prints the total return
    over the dataset and the baseline return (buy and hold over the period).

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the 'Current_week_pct_change' column.
    prediction_column (str): The name of the column containing the predictions.

    Returns:
    pd.DataFrame: The DataFrame with additional columns for portfolio returns and cumulative returns.
    """
    # Calculate the portfolio return based on the prediction
    df['Shifted_Prediction'] = df[prediction_column].shift(1)
    df['Portfolio_Return'] = df.apply(
        lambda row: (row['Current_week_pct_change'] / 100) * row['Shifted_Prediction'], axis=1
    )
    df.drop(columns=['Shifted_Prediction'], inplace=True)
    
    # Fill NaNs, which would be in the first row after shifting
    df['Portfolio_Return'].fillna(0, inplace=True)

    # Calculate cumulative returns
    df['Cumulative_Return'] = (1 + df['Portfolio_Return']).cumprod() - 1

    # Calculate 'high water mark', i.e. highest cumulative return so far, to visualize drawdowns
    df['High_Water_Mark'] = df['Cumulative_Return'].cummax()

    # Calculate total return
    total_return = df['Cumulative_Return'].iloc[-1]
    print(f'Cumulative return over the period: {round(total_return * 100, 1)} %')

    # Calculate baseline return for a buy and hold strategy
    buy_hold_return = (df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1
    print(f'Baseline Return (Buy and Hold over the period): {round(buy_hold_return * 100, 1)} %')

    return df

def calculate_max_drawdown_duration(df):
    uwater = df['Cumulative_Return'] < df['High_Water_Mark']
    runs = (~uwater).cumsum()[uwater]
    drawdown_durations = runs.value_counts(sort=True)
    return drawdown_durations.iloc[0] if not drawdown_durations.empty else 0
'''
def calculate_yearly_returns(df):
    print('─' * 40)
    df['Year'] = df['Date'].dt.year
    years = df['Year'].unique()
    for year in years:
        year_df=calculate_returns(df[df['Year']==year], 'Prediction')
        max_yearly_drawdown=calculate_max_drawdown_duration(year_df)
        yearly_cumulative_return=year_df['Cumulative_Return'].iloc[-1]

        print(f'Return in year {year}: {round(yearly_cumulative_return * 100, 1)} %')
        print(f'Maximum drawdown during year {year}: {max_yearly_drawdown} weeks')
        print('─' * 40)
'''

def calculate_yearly_returns(df):
    """
    Calculates the returns for every year-long slice (52 weeks) of the dataset and prints the start and end date
    for each period, along with the total return, maximum drawdown, and baseline return for each period.
    If the dataset is shorter than a year, it prints a corresponding message.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the 'Date' and 'Close' columns.

    Returns:
    None
    """
    if len(df) < 52:
        print('Dataset shorter than 1 year')
        return

    df = df.sort_values('Date').reset_index(drop=True)
    num_years = len(df) // 52
    print('Yearly Returns and Drawdowns:')
    print('─' * 40)
    for i in range(num_years):
        start_idx = i * 52
        end_idx = start_idx + 52
        year_slice = df.iloc[start_idx:end_idx].copy()

        start_date = year_slice['Date'].iloc[0].strftime('%Y-%m-%d')
        end_date = year_slice['Date'].iloc[-1].strftime('%Y-%m-%d')

        # Calculate returns using the calculate_returns function
        year_slice = calculate_returns(year_slice, 'Prediction')

        # Calculate maximum drawdown duration
        max_yearly_drawdown = calculate_max_drawdown_duration(year_slice)

        yearly_cumulative_return = year_slice['Cumulative_Return'].iloc[-1]

        print(f'Start Date: {start_date}')
        print(f'End Date: {end_date}')
        print(f'Maximum drawdown during the period: {max_yearly_drawdown} weeks')
        print('─' * 40)

def plot_return(df, name, train_test_split=None, save_figure=False):
    plt.style.use('default')

    plt.rcParams['font.sans-serif'] = 'Arial'
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.grid(True, linestyle='--', linewidth=0.5, color='gray')
    if train_test_split!=None:
        # Add line for training/validation split
        split_idx = int(len(df) * train_test_split)
        ax.axvline(x=df['Date'].iloc[split_idx], color='darkgrey', linestyle='--', label='Train/Validation Split')
        
    # Plot cumulative returns and high water mark
    ax.plot(df['Date'], df['Cumulative_Return'], label='Cumulative Return', color='darkgreen', linewidth=1)
    ax.plot(df['Date'], df['High_Water_Mark'], label='Drawdown line', color='darkred', linewidth=1, alpha=0.9)
    ax2=ax.twinx()
    ax2.plot(df['Date'], df['Close'], label='Close price', color='#3498db', alpha=0.5, linewidth=1)
    
    ax.set_title(f'{name} Long-Short Equity Strategy Performance')
    ax.set_xlabel('Date')
    ax.xaxis.set_tick_params(rotation=45)
    ax.set_ylabel('Cumulative Return', color='darkgreen')
    ax.tick_params(axis='y', labelcolor='darkgreen')
    ax2.set_ylabel('Adjusted Close price (USD)', color='#3498db') 
    ax2.tick_params(axis='y', labelcolor='#3498db')
    
    fig.legend(loc='upper left', bbox_to_anchor=(0,1), bbox_transform=ax.transAxes)
    
    if save_figure:
        plt.savefig(f'{name}_gianfis_return.png', transparent=True, dpi=300)

    plt.show()