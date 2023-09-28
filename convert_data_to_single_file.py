from glob import glob
import os
import pandas as pd
from datetime import datetime
import ta
import numpy as np

loadpath = 'raw_data'
savepath = 'modified_data'

TIME_DIFF_1 = 1
TIME_DIFF_24 = 24 # a day
TIME_DIFF_168 = 24 * 7 # a week

class TechnicalIndicators:
    @staticmethod
    def add_SMA(df, window=50):
        df['SMA'] = ta.trend.sma_indicator(df['close'], window)
        return df

    @staticmethod
    def add_EMA(df, window=50):
        df['EMA'] = ta.trend.ema_indicator(df['close'], window)
        return df

    @staticmethod
    def add_RSI(df, window=14):
        df['RSI'] = ta.momentum.rsi(df['close'], window)
        return df

    @staticmethod
    def add_MACD(df):
        df['MACD'] = ta.trend.macd_diff(df['close'])
        return df

    @staticmethod
    def add_Bollinger(df, window=20, std_dev=2):
        bollinger = ta.volatility.BollingerBands(df['close'], window, std_dev)
        df['Bollinger_High'] = bollinger.bollinger_hband()
        df['Bollinger_Low'] = bollinger.bollinger_lband()
        return df

    @staticmethod
    def add_VWAP(df):
        df['VWAP'] = (df['close'] * df['Volume USD']).cumsum() / df['Volume USD'].cumsum()
        return df
    
    @staticmethod
    # Adding Percentage Returns
    def add_percentage_returns(df):
        df['Percentage_Returns'] = df['close'].pct_change()
        return df

    @staticmethod
    # Adding Log Returns
    def add_log_returns(df):
        df['Log_Returns'] = np.log(df['close'] / df['close'].shift(1))
        return df
    
    # @staticmethod
    # def add_boolean_direction(df):
    #     df['next_hour'] = df['close'].shift(-1)
    #     df['Target'] = (df['next_hour'] > df['close']).astype(int)
    #     return df
    
    @staticmethod
    def add_time_diff_columns(df):
        df[f'close_shifted_{TIME_DIFF_1}'] = df['close'].shift(-TIME_DIFF_1)
        df[f'close_shifted_{TIME_DIFF_24}'] = df['close'].shift(-TIME_DIFF_24)
        df[f'close_shifted_{TIME_DIFF_168}'] = df['close'].shift(-TIME_DIFF_168)
        df[f'Target_shifted_{TIME_DIFF_1}'] = (df[f'close_shifted_{TIME_DIFF_1}'] > df['close']).astype(int)
        df[f'Target_shifted_{TIME_DIFF_24}'] = (df[f'close_shifted_{TIME_DIFF_24}'] > df['close']).astype(int)
        df[f'Target_shifted_{TIME_DIFF_168}'] = (df[f'close_shifted_{TIME_DIFF_168}'] > df['close']).astype(int)
        return df
    
    # Z-score Normalization [Commented out]
    # def z_score_normalize(df, column):
    #     df[column] = (df[column] - df[column].mean()) / df[column].std()
    #     return df

    @classmethod
    def add_all_indicators(cls, df):
        df = cls.add_SMA(df)
        df = cls.add_EMA(df)
        df = cls.add_RSI(df)
        df = cls.add_MACD(df)
        df = cls.add_Bollinger(df)
        df = cls.add_VWAP(df)
        df = cls.add_percentage_returns(df)
        df = cls.add_log_returns(df)
        df = cls.add_time_diff_columns(df)
        # df = cls.add_boolean_direction(df)
        return df
    

gemini_files = glob(os.path.join(loadpath, 'Gemini*_1h.csv'))
hitbtc_files = glob(os.path.join(loadpath, 'HitBTC*_1h.csv'))


for gemini_file in gemini_files:
    print(f'opening file: {gemini_file}')
    crypto_name = gemini_file.split('_')[-2][:-3]
    file = pd.read_csv(gemini_file, skiprows=1)
    file = file.drop(columns=['date', 'symbol', f'Volume {crypto_name}'])
    
    # Add indicators
    file = TechnicalIndicators.add_all_indicators(file)
    file.to_csv(os.path.join(savepath, f'gemini_data_{crypto_name}_mod.csv'), index=False)



# for hitbtc_file in hitbtc_files:
#     print(f'opening file: {hitbtc_file}')
#     crypto_name = hitbtc_file.split('_')[-2][:-3]
#     file = pd.read_csv(hitbtc_file, skiprows=1)
#     file = file.drop(columns=['Date', 'Symbol', f'Volume {crypto_name}'])

#     file = TechnicalIndicators.add_all_indicators(file)
#     file.to_csv(os.path.join(savepath, f'hitbtc_data_{crypto_name}-{datetime.now()}.csv'), index=False)