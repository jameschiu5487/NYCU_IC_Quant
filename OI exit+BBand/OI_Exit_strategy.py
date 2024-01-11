
import vectorbt as vbt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import numba as nb
import os
from datetime import datetime
import json
from tabulate import tabulate
import sys
sys.path.append('../..')
# from src.strategy.PositionSizer import PositionSizer


def timestamp_to_int(dt):  # takes a datetime object return timestamp in int
    return int(datetime.timestamp(dt) * 1000)

# takes a timestamp in int return datetime object


def timestampt_to_datetime(ts):
    return datetime.fromtimestamp(ts / 1000)


def get_data():
    OHLC_data = pd.read_csv(
        "/Users/frank/Downloads/BTCUSDT永續15m_1_1_2023_to_11_6_2023.csv")
    OHLC_data = OHLC_data.iloc[:, 1:-2]
    OHLC_data['time'] = pd.to_datetime(OHLC_data['time'])
    OHLC_data.set_index('time', inplace=True)
    return OHLC_data


def crossover(over, down):
    a1 = over
    b1 = down
    a2 = a1.shift(1)
    b2 = b1.shift(1)
    crossover = (a1 > a2) & (a1 > b1) & (b2 > a2)
    return crossover
# 死亡交叉


def crossunder(down, over):
    a1 = down
    b1 = over
    a2 = a1.shift(1)
    b2 = b1.shift(1)
    crossdown = (a1 < a2) & (a1 < b1) & (b2 < a2)
    return crossdown


def crossover2(over, down):
    a1 = over
    b1 = down
    a2 = a1.shift(1)
    crossover = (a1 > a2) & (a1 > b1) & (b1 > a2)
    return crossover
# 死亡交叉


def crossunder2(down, over):
    a1 = down
    b1 = over
    a2 = a1.shift(1)
    crossdown = (a1 < a2) & (a1 < b1) & (b1 < a2)
    return crossdown


class Strategy():

    def __init__(self, df, configs, **kwargs):
        super().__init__(**kwargs)
        self.configs = configs
        self.freq = self.configs['freq']
        self.fee = self.configs['fee']
        self.start = self.configs['start']
        self.end = self.configs['end']
        self.df = self.resample_df(df=df, freq=self.freq)

    def resample_df(self, df, freq='1h'):
        cols = ['open', 'high', 'low', 'close', 'volume']
        agg = ['first', 'max',  'min', 'last', 'sum']
        df = df[cols]
        df = df.resample(freq).agg(dict(zip(cols, agg)))
        return df.dropna()

    def strategy(self, side='both', **params):
        params = params['params']
        df = self.df.copy()
        # params
        OI_window_MA_f = int(params['OI_window_MA_f'])
        OI_window_MA_s = int(params['OI_window_MA_s']) + OI_window_MA_f
        # 因為你慢線一定要大於快線，所以直接用快線加上慢線的參數讓他等於慢線這樣就不會有慢線小於快線的參數浪費
        RV_window = int(params['RV_window'])
        RV_period = int(params['RV_period'])
        BBand_mult = int(params['BBand_mult'])
        BBand_window = int(params['BBand_window'])

        def get_OI_data(freq):
            OI_data = pd.read_hdf(
                "/Users/frank/Downloads/BTCUSDT_PERPETUAL.h5 2")
            # Apply the conversion function to each element of the index
            OI_data.index = OI_data.index.map(timestampt_to_datetime)

            # Convert the 'sumOpenInterest' and 'sumOpenInterestValue' columns to float
            OI_data['sumOpenInterest'] = OI_data['sumOpenInterest'].astype(
                float)
            OI_data['sumOpenInterestValue'] = OI_data['sumOpenInterestValue'].astype(
                float)
            OI_data = OI_data.iloc[:, 1:]
            OI_data = OI_data
            OI_data = OI_data.resample(freq).sum()
            OI_data = OI_data.iloc[:-1, :]
            OI_data.replace(
                to_replace=OI_data.iloc[3501, 0], value=OI_data.iloc[3500, 0], inplace=True)
            OI_data.replace(
                to_replace=OI_data.iloc[3502, 0], value=OI_data.iloc[3500, 0], inplace=True)
            OI_data.replace(to_replace=249314.980,
                                value=373682.328, inplace=True)
            OI_data.replace(
                to_replace=OI_data.iloc[12898, 0], value=OI_data.iloc[12897, 0], inplace=True)
            OI_data.replace(
                to_replace=OI_data.iloc[12899, 0], value=OI_data.iloc[12897, 0], inplace=True)
            return OI_data

        def adjust_date(OI_data, df):
            start = self.start
            end = self.end
            df = df.loc[start:end, :]
            OI_data = OI_data.loc[start:end, :]
            return OI_data, df

        OI_data = get_OI_data(self.freq)
        OI_data, df = adjust_date(OI_data,  df)
        OI_data['MA_fast'] = OI_data['sumOpenInterest'].rolling(
            OI_window_MA_f).mean()
        OI_data['MA_slow'] = OI_data['sumOpenInterest'].rolling(
            OI_window_MA_s).mean()
        df.loc[:, 'log_rtn_sq'] = np.square(
            np.log(df['close']/df['close'].shift(1)))
        df.loc[:, 'RV'] = np.sqrt(df['log_rtn_sq'].rolling(RV_window).sum())
        df.loc[:, 'RV_pctrank'] = df['RV'].rolling(
            RV_period).rank(pct=True)*100
        df.loc[:, 'RV_filter'] = np.where(
            (df['RV_pctrank'] > 30) & (df['RV_pctrank'] < 70), 1, 0)
        df.loc[:, 'SMA'] = df['close'].rolling(BBand_window).mean()
        df.loc[:, 'upper'] = df['SMA'] + BBand_mult * \
            df['close'].rolling(BBand_window).std()
        df.loc[:, 'lower'] = df['SMA'] - BBand_mult * \
            df['close'].rolling(BBand_window).std()

        long_entry = np.where((df['close'] > df['upper']) & (
            OI_data['MA_fast'] > OI_data['MA_slow']) & (df['RV_filter']), True, False)
        long_exit = np.where(crossunder(
            OI_data['MA_fast'], OI_data['MA_slow']), True, False)
        short_entry = np.where((df['close'] < df['lower']) & (
            OI_data['MA_fast'] > OI_data['MA_slow']) & (df['RV_filter']), True, False)
        short_exit = np.where(crossunder(
            OI_data['MA_fast'], OI_data['MA_slow']), True, False)

        if side == 'long':
            short_entry = False
            short_exit = False

        elif side == 'short':
            long_entry = False
            long_exit = False
        price = df['open'].shift(-1)
        pf = vbt.Portfolio.from_signals(price,  # type: ignore
                                        open=df['open'],
                                        high=df['high'],
                                        low=df['low'],
                                        entries=long_entry,
                                        exits=long_exit,
                                        short_entries=short_entry,
                                        short_exits=short_exit,
                                        fees=self.fee,
                                        sl_stop=np.nan/100,
                                        upon_opposite_entry='reverse'
                                        )
        return pf, params