import asyncio
import threading
import time
import schedule
from ib_insync import IB, Stock, MarketOrder, util
import pandas as pd
import ta

# Create a global IB instance
ib = IB()

# Create a global asyncio event loop
loop = asyncio.new_event_loop()

def start_loop():
    """ Runs the asyncio event loop in a separate thread. """
    asyncio.set_event_loop(loop)
    loop.run_forever()

# Start the background event loop thread
threading.Thread(target=start_loop, daemon=True).start()

async def connect_ibkr():
    """ Asynchronously connects to IBKR TWS, ensuring a clean reconnect if needed. """
    if ib.isConnected():
        print("🔌 Closing existing IBKR connection...")
        ib.disconnect()
        await asyncio.sleep(2)  # Ensures full disconnect

    print("🔄 Connecting to IBKR...")
    for _ in range(5):  # Retry up to 5 times
        try:
            await ib.connectAsync('127.0.0.1', 7497, clientId=1)
            if ib.isConnected():
                print("✅ Successfully connected to IBKR API!")
                return  # Exit function as soon as connected
        except Exception as ex:
            print(f"❌ IBKR Connection Error: {ex}")

        print("🔄 Retrying connection in 5 seconds...")
        await asyncio.sleep(5)

    print("❌ Failed to connect to IBKR after multiple attempts.")


async def fetch_market_data():
    """ Fetch historical data and calculate indicators asynchronously. """
    if not ib.isConnected():
        print("❌ Cannot fetch market data: IBKR is not connected.")
        return None, None

    contract = Stock('TSLA', 'SMART', 'USD')

    try:
        await ib.qualifyContractsAsync(contract)

        bars = await ib.reqHistoricalDataAsync(
            contract,
            endDateTime='',
            durationStr='6 M',
            barSizeSetting='1 day',
            whatToShow='ADJUSTED_LAST',
            useRTH=True
        )

        df = util.df(bars)

        # Indicators
        df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        bollinger = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['Bollinger_Upper'] = bollinger.bollinger_hband()
        df['Bollinger_Lower'] = bollinger.bollinger_lband()

        df.dropna(inplace=True)

        # Buy/Sell logic
        df['Buy_Signal'] = (df['RSI'] < 40) & (df['close'] < df['Bollinger_Lower'])
        df['Sell_Signal'] = (df['RSI'] > 60) & (df['close'] > df['Bollinger_Upper'])

        return df, contract
    except Exception as ex:
        print(f"❌ Error fetching market data: {ex}")
        return None, None


async def execute_trades():
    """ Checks for Buy/Sell signals and executes trades via IBKR. """
    df, contract = await fetch_market_data()
    if df is None or contract is None:
        print("⚠️ Skipping trade execution due to missing market data.")
        return

    latest = df.iloc[-1]
    print(latest[['date', 'close', 'RSI', 'Buy_Signal', 'Sell_Signal']])

    if latest['Buy_Signal']:
        print(f"🔹 Buy Signal at {latest['close']:.2f}, placing BUY order.")
        order = MarketOrder('BUY', 10)
        ib.placeOrder(contract, order)
    elif latest['Sell_Signal']:
        print(f"🔸 Sell Signal at {latest['close']:.2f}, placing SELL order.")
        order = MarketOrder('SELL', 10)
        ib.placeOrder(contract, order)
    else:
        print("📉 No trade signals detected this run.")

def run_trading_bot():
    """ Called by schedule every X minutes. Push async tasks to the background loop. """
    print("🔄 Checking for new trade signals...")

    # Step 1: Ensure IBKR is connected before anything else
    connect_future = asyncio.run_coroutine_threadsafe(connect_ibkr(), loop)
    connect_future.result(timeout=60)  # Wait until IBKR is connected

    if not ib.isConnected():
        print("❌ Skipping trade execution because IBKR is not connected.")
        return  # Don't proceed with fetching data

    # Step 2: Execute trades only if IBKR is connected
    asyncio.run_coroutine_threadsafe(execute_trades(), loop)

# 🔥 Set to Run Every 1 Minute for Testing
schedule.every(1).minutes.do(run_trading_bot)

print("🚀 Trading Bot Started. Checking for signals every 1 minute.")

while True:
    schedule.run_pending()
    time.sleep(1)




