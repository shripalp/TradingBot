from ib_insync import IB, Stock, util, LimitOrder
import pandas as pd
import ta  # Technical analysis library

# Connect to IBKR
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)

# Define stock (AAPL - Apple Inc.)
contract = Stock('NIO', 'SMART', 'USD')
ib.qualifyContracts(contract)

# Fetch historical data (past 30 days)
bars = ib.reqHistoricalData(
    contract,
    endDateTime='',
    durationStr='6 M',
    barSizeSetting='1 day',
    whatToShow='ADJUSTED_LAST',
    useRTH=True
)

# Convert data to Pandas DataFrame
df = util.df(bars)


# Calculate RSI, MACD, Bollinger Bands
df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
df['MACD'] = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9).macd()
bollinger = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
df['Bollinger_Upper'] = bollinger.bollinger_hband()
df['Bollinger_Lower'] = bollinger.bollinger_lband()

# Drop NaN values to avoid errors
df = df.dropna()

df['Buy_Signal'] = (df['RSI'] < 45) & (df['close'] < df['Bollinger_Lower'])
df['Sell_Signal'] = (df['RSI'] > 55) & (df['close'] > df['Bollinger_Upper'])

#manually add buy/sell signal
#df.loc[df.index[-1], 'Buy_Signal'] = True  # Force Buy Signal
#df.loc[df.index[-1], 'Sell_Signal'] = False  # Disable Sell Signal


 #print(df[['date', 'close', 'RSI', 'MACD', 'Bollinger_Upper', 'Bollinger_Lower', 'Buy_Signal', 'Sell_Signal']].tail(20))

print(df[['date', 'close', 'RSI', 'Buy_Signal', 'Sell_Signal']].tail(20))


#print(df[['RSI', 'MACD', 'Bollinger_Upper', 'Bollinger_Lower']].isna().sum())


# Check latest Buy/Sell signal
latest_data = df.iloc[-1]


    
entry_price = latest_data['close']
stop_loss = entry_price * 0.95  # 5% below entry
take_profit = entry_price * 1.10  # 10% above entry

if latest_data['Buy_Signal']:
    print(f"🔹 Buy Order at ${entry_price:.2f} with Stop-Loss at ${stop_loss:.2f} and Take-Profit at ${take_profit:.2f}")
    order = LimitOrder('BUY', 10, entry_price)  # Buy at entry price
    trade = ib.placeOrder(contract, order)

elif latest_data['Sell_Signal']:
    print(f"🔸 Sell Order at ${entry_price:.2f} with Stop-Loss at ${take_profit:.2f} and Take-Profit at ${stop_loss:.2f}")
    order = LimitOrder('SELL', 10, entry_price)
    trade = ib.placeOrder(contract, order)
else:
    print("📉 No trade signals detected.")

