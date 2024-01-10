import talib as ta
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import vectorbt as vbt
from datetime import datetime
import plotly.offline as pyo


#def
def crossover(over, down):
    a1 = over
    b1 = down
    a2 = a1.shift(1)
    b2 = b1.shift(1)
    crossover = (a1 > a2) & (a1 > b1) & (b2 > a2)
    return crossover
def crossdown(down, over):
    a1 = down
    b1 = over
    a2 = a1.shift(1)
    b2 = b1.shift(1)
    crossdown = (a1 < a2) & (a1 < b1) & (b2 < a2)
    return crossdown

#read data
df = pd.read_csv('/Users/mona/Desktop/VS code/Quant/BTCUSDT-1h-data.csv')
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df = df.set_index('Timestamp')
df = df.iloc[20710:,:]
df = df.resample('2H').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'})

#rvi&kc
rvi_length = 21   #平滑指標
rvi_ma_length = 20
df['Close'] = df['Close'].astype(float)
price_diff = np.diff(df['Close'])
df = df.iloc[1:, :]
df['price_diff'] = price_diff
df['stddev'] = df['Close'].rolling(window=rvi_length).std()
df['upper'] = np.where(df['price_diff'] <= 0, 0, df['stddev'])
df['upper'] = df['upper'].ewm(span=rvi_length, adjust=False).mean()
df['lower'] = np.where(df['price_diff'] > 0, 0, df['stddev'])
df['lower'] = df['lower'].ewm(span=rvi_length, adjust=False).mean()
df['rvi'] = df['upper'] / (df['upper'] + df['lower']) *100
df['rvi_ma'] = df['rvi'].rolling(rvi_ma_length).mean()
df.dropna(inplace=True)

kclength = 47
df['sma'] = df['Close'].rolling(kclength).mean()
df['basis'] = df['Close'].ewm(span=kclength, adjust=False).mean()
df['atr'] = ta.ATR(df['High'], df['Low'], df['Close'], timeperiod=kclength)
kc_mult = 2.9456
df['kc_upper'] = df['basis'] + kc_mult * df['atr']
df['kc_lower'] = df['basis'] - kc_mult * df['atr']
df.dropna(inplace=True)

#plot
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(df['Close'], label='close Prices')
plt.plot(df['basis'], label='basis')
plt.plot(df['kc_upper'], label='kc_upper')
plt.plot(df['kc_lower'], label='kc_lower')
plt.title('close Prices & KC')
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(df['rvi'], label='RVI')
plt.plot(df['rvi_ma'], label='RVI MA')
plt.title('RVI')
plt.legend()
plt.show()

#signal
signal = pd.DataFrame(columns=['long_entry', 'short_entry', 'long_exit', 'short_exit'])
signal['long_entry'] = np.where((crossover(df['Close'], df['kc_upper']) & (df['rvi'] > df['rvi_ma'])), 1, 0)
signal['long_exit'] = np.where((crossdown(df['Close'], df['basis']) & (df['rvi'] < df['rvi_ma'])), 1, 0)  # basis or sma?
signal['short_entry'] = np.where((crossdown(df['Close'], df['kc_lower']) & (df['rvi'] < df['rvi_ma'])), 1, 0)
signal['short_exit'] = np.where((crossover(df['Close'], df['basis']) & (df['rvi'] > df['rvi_ma'])), 1, 0)

#pf
price = df['Open'].shift(-1).dropna()
pf = vbt.Portfolio.from_signals(price,
                               entries = signal['long_entry'].iloc[:-1],
                               exits = signal['long_exit'].iloc[:-1],
                               short_entries = signal['short_entry'].iloc[:-1],
                               short_exits = signal['short_exit'].iloc[:-1],
                               freq='2H',
                               fees = 0.0005)

print(pf.stats())
fig = pf.plot(subplots=['cum_returns', 'orders' , 'trade_pnl'])
pyo.plot(fig)
