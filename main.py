import asyncio
import pandas as pd
import numpy as np
from ib_insync import *
import datetime
import csv

# Connect to IBKR
ib = IB()

PREFERRED_ACCOUNT = 'DUA296813'  # IBKR account number

async def connect_ibkr():
    """Asynchronously connects to IBKR TWS or IB Gateway"""
    await ib.connectAsync('127.0.0.1', 7497, clientId=1)
    print("✅ Connected to IBKR")
    
def safe_async_call(coro):
    """Safely run an async function from a synchronous environment."""
    loop = asyncio.get_event_loop()
    return asyncio.run_coroutine_threadsafe(coro, loop).result()

# Fetch historical stock data
async def get_stock_data(symbol, duration='1 Y', bar_size='1 day'):
    """Fetch historical stock data, including volume."""
    print(f"📡 Fetching market data for {symbol}...")

    contract = Stock(symbol, 'SMART', 'USD')
    try:
        bars = await ib.reqHistoricalDataAsync(
            contract,
            endDateTime='',
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow='TRADES',  # ✅ Ensure volume is included
            useRTH=True
        )

        if not bars:
            print(f"⚠ No historical data found for {symbol}. Skipping.")
            return None

        df = pd.DataFrame(bars)
        df['close'] = df['close'].astype(float)

        # ✅ Ensure 'volume' column exists
        if 'volume' not in df.columns:
            df['volume'] = np.nan  # Add empty column to prevent KeyError

        print(f"✅ Received {len(df)} data points for {symbol}. Latest close: ${df['close'].iloc[-1]:.2f}")
        return df

    except Exception as e:
        print(f"⚠ Error fetching market data for {symbol}: {e}")
        return None

    
# Calculate RSI
def calculate_rsi(df, period=14):
    """Calculate RSI (Relative Strength Index)."""
    delta = df['close'].diff()
    gain = np.where(delta > 0, delta, 0).astype(float)
    loss = np.where(delta < 0, -delta, 0).astype(float)

    avg_gain = pd.Series(gain).rolling(window=period, min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

# Calculate Moving Averages
def calculate_moving_averages(df):
    """Calculate 50-day and 200-day moving averages."""
    df['SMA_50'] = df['close'].rolling(50).mean()
    df['SMA_200'] = df['close'].rolling(200).mean()
    return df

# Detect Breakouts
def check_breakout(df, period=50):
    """Check if the stock breaks out above 20-day high."""
    df['20_day_high'] = df['close'].rolling(period).max()
    df['Breakout'] = (df['close'] > df['20_day_high']) & (df['volume'] > df['volume'].rolling(20).mean())
    return df

def log_trade_signals(symbol, df):
    """Log trade signals for debugging."""
    latest_rsi = df['RSI'].iloc[-1]
    latest_sma_50 = df['SMA_50'].iloc[-1]
    latest_sma_200 = df['SMA_200'].iloc[-1]
    breakout_signal = df['Breakout'].iloc[-1]

    print(f"📊 Checking trade signals for {symbol}:")
    print(f"   - RSI: {latest_rsi:.2f} (Oversold < 30)")
    print(f"   - 50 SMA: {latest_sma_50:.2f}")
    print(f"   - 200 SMA: {latest_sma_200:.2f}")
    print(f"   - Breakout Signal: {'YES' if breakout_signal else 'NO'}")
    
async def get_account_balance():
    """Fetch and display IBKR account balance details asynchronously."""
    print("\n🔄 Fetching IBKR account balance...")

    # Fetch account summary asynchronously
    account_summary = await ib.accountSummaryAsync()

    if not account_summary:
        print("⚠ Unable to fetch account balance.")
        return

    # Extract values from the list
    net_liq = cash_balance = buying_power = None

    for item in account_summary:
        if item.tag == 'NetLiquidation':
            net_liq = float(item.value)
        elif item.tag == 'CashBalance':
            cash_balance = float(item.value)
        elif item.tag == 'BuyingPower':
            buying_power = float(item.value)

    # Check if values were found
    if net_liq is None or cash_balance is None or buying_power is None:
        print("⚠ Could not retrieve all account values.")
        return

    print("\n💰 Account Balance:")
    print("=" * 50)
    print(f"💵 Cash Balance: ${cash_balance:,.2f}")
    print(f"📊 Net Liquidation Value: ${net_liq:,.2f}")
    print(f"⚡ Buying Power: ${buying_power:,.2f}")
    print("=" * 50)

async def get_positions():
    """Fetch open positions and ensure market data is available."""
    positions = ib.positions()

    if not positions:
        print("📭 No open positions found.")
        return

    print("\n📊 Open Positions:")
    print("=" * 50)

    for position in positions:
        symbol = position.contract.symbol
        quantity = position.position
        avg_cost = position.avgCost
        stop_loss = trade_log.get(symbol, {}).get("stop_loss", "N/A")
        take_profit = trade_log.get(symbol, {}).get("take_profit", "N/A")
        contract = Stock(symbol, 'SMART', 'USD')

        try:
            tick = ib.reqMktData(contract, genericTickList="", snapshot=True)
            await asyncio.sleep(2)  # ✅ Allow IBKR to return data

            current_price = tick.last if tick.last is not None else avg_cost  # Use last known price if missing
            unrealized_pnl = (current_price - avg_cost) * quantity

        except Exception as e:
            print(f"⚠ Error fetching market data for {symbol}: {e}")
            current_price = avg_cost
            unrealized_pnl = 0.0

        print(f"📈 {symbol}: {quantity} shares")
        print(f"   - Average Cost: ${avg_cost:.2f}")
        print(f"   - Current Price: ${current_price:.2f}")
        print(f"   - Unrealized P&L: ${unrealized_pnl:.2f}")
        print(f"   - 🛑 Stop-Loss: ${stop_loss}")
        print(f"   - 🎯 Take-Profit: ${take_profit}")
        print("-" * 50)



async def cancel_order(symbol):
    """Cancel an open order for a given stock if it is still active."""
    print(f"🚫 Attempting to cancel {symbol} order...")

    # Get all open trades
    open_trades = ib.openTrades()

    found = False
    for trade in open_trades:
        if trade.contract.symbol == symbol:
            found = True
            if trade.orderStatus.status in ["PreSubmitted", "Submitted"]:  # ✅ Only cancel active orders
                print(f"🚫 Canceling {symbol} order (Order ID: {trade.order.orderId})...")
                ib.cancelOrder(trade.order)
                await asyncio.sleep(2)  # ✅ Allow IBKR time to process the cancellation
                print(f"✅ Order {trade.order.orderId} for {symbol} successfully canceled.")
            else:
                print(f"⚠️ Order {trade.order.orderId} for {symbol} is already filled or cancelled.")
    
    if not found:
        print(f"⚠️ No active order found for {symbol}. Skipping cancellation.")



def log_trade(symbol, action, price, quantity, status):
    """Logs trades to a CSV file for debugging."""
    with open("trade_log.csv", "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([datetime.datetime.now(), symbol, action, price, quantity, status])





# Trade tracking
trade_log = {}

async def place_order(symbol, quantity, action, limit_price, breakout_price=None, stop_loss_pct=3, take_profit_pct=5):
    """Places a Limit Order asynchronously and ensures order_time is set for pending trades."""

    # ✅ Prevent duplicate trades at the same breakout level or if a trade is pending
    if symbol in trade_log and (trade_log[symbol].get("status") == "pending" or trade_log[symbol].get("last_breakout") == breakout_price):
        print(f"⏳ Skipping {symbol} - Order already placed or pending at breakout level ${breakout_price:.2f}.")
        return  

    contract = Stock(symbol, 'SMART', 'USD')

  # ✅ Define Market Order
    order = MarketOrder(action, quantity, account=PREFERRED_ACCOUNT)
    trade = ib.placeOrder(contract, order)
    
    await asyncio.sleep(2)  # ✅ Allow order execution time
    
    

    # ✅ Calculate Stop-Loss & Take-Profit Before Order Execution
    stop_loss_price = limit_price * (1 - stop_loss_pct / 100)
    take_profit_price = limit_price * (1 + take_profit_pct / 100)

    # ✅ Save trade details in log immediately (even if order is pending)
    trade_log[symbol] = {
        "status": "pending",
        "entry_price": limit_price,  # Save entry price before execution
        "stop_loss": stop_loss_price,
        "take_profit": take_profit_price,
        "last_breakout": breakout_price,
        "limit_price": limit_price
    }

    print(f"📌 Placed {action} Limit Order for {symbol}: {quantity} shares at ${limit_price:.2f}")
    print(f"   - 🛑 Stop-Loss: ${stop_loss_price:.2f}")
    print(f"   - 🎯 Take-Profit: ${take_profit_price:.2f}")

    # ✅ Check if the order is filled and update trade log
    if trade.orderStatus.status == 'Filled':
        entry_price = trade.orderStatus.avgFillPrice
        trade_log[symbol]["entry_price"] = entry_price
        print(f"✅ Trade Executed: {action} {quantity} shares of {symbol} at ${entry_price:.2f}")

    else:
        print(f"⚠ Limit Order for {symbol} not filled yet. Status: {trade.orderStatus.status}. Retrying in next cycle.")


# Monitor Active Trades
async def monitor_trades(market_data_cache):
    """Monitors active trades and ensures stop-loss & take-profit are tracked."""

    open_positions = ib.positions()

    for position in open_positions:
        symbol = position.contract.symbol
        quantity = position.position
        entry_price = trade_log.get(symbol, {}).get("entry_price")
        stop_loss = trade_log.get(symbol, {}).get("stop_loss")
        take_profit = trade_log.get(symbol, {}).get("take_profit")

        # ✅ Ensure we extract a single float value for current price
        current_price = market_data_cache.get(symbol, {}).get("close")
        if isinstance(current_price, pd.Series):  # ✅ Convert Series to float
            current_price = current_price.iloc[-1]

        # ✅ Proper condition check
        if entry_price is None or stop_loss is None or take_profit is None or current_price is None:
            print(f"⚠ Missing trade data for {symbol}. Skipping monitoring.")
            continue

        print(f"📊 Monitoring {symbol} | Entry: ${entry_price:.2f} | Stop-Loss: ${stop_loss:.2f} | Take-Profit: ${take_profit:.2f} | Current: ${current_price:.2f}")

        # ✅ Execute Stop-Loss
        if current_price <= stop_loss:
            print(f"🚨 STOP-LOSS HIT: Selling {symbol} at ${current_price:.2f} (Stop-Loss: ${stop_loss:.2f})")
            await place_order(symbol, quantity, "SELL", current_price * 0.995)
            trade_log.pop(symbol, None)

        # ✅ Execute Take-Profit
        if current_price >= take_profit:
            print(f"🎯 TAKE-PROFIT HIT: Selling {symbol} at ${current_price:.2f} (Target: ${take_profit:.2f})")
            await place_order(symbol, quantity, "SELL", current_price * 1.005)
            trade_log.pop(symbol, None)


# Run Trading Strategy
async def run_trading_bot():
    """Runs the trading strategy and displays account info."""
    print("\n🔄 Running trading bot cycle...")

    # Display account balance and open positions before trading
    await get_account_balance()
    await get_positions()

    stocks = ["NIO", "COIN", "TQQQ"]
    market_data_cache = {}
    
  
    # Fetch market data once per stock
    tasks = [get_stock_data(stock) for stock in stocks]
    stock_data_list = await asyncio.gather(*tasks)

    # Store data in cache
    for i, stock in enumerate(stocks):
        if stock_data_list[i] is not None:
            market_data_cache[stock] = stock_data_list[i]

    # Execute trade signals
    for stock, df in market_data_cache.items():
        df = calculate_rsi(df)
        df = calculate_moving_averages(df)
        df = check_breakout(df)

        log_trade_signals(stock, df)

        latest_rsi = df['RSI'].iloc[-1]
        latest_sma_50 = df['SMA_50'].iloc[-1]
        latest_sma_200 = df['SMA_200'].iloc[-1]
        breakout_signal = df['Breakout'].iloc[-1]
        current_price = df['close'].iloc[-1]
        breakout_price = df['20_day_high'].iloc[-1]

        limit_price = current_price * 1.0  # Buy slightly below market price

         # ✅ Buy if RSI is below 30 (Oversold)
        if latest_rsi < 30 and stock not in trade_log:
            print(f"📢 BUY {stock} - RSI Oversold")
            await place_order(stock, 1, "BUY", limit_price)
            
         # ✅ Buy if 50 SMA crosses above 200 SMA (Golden Cross)
        elif latest_sma_50 > latest_sma_200 and stock not in trade_log:
            print(f"📢 BUY {stock} - Golden Cross Detected (SMA_50: {latest_sma_50:.2f}, SMA_200: {latest_sma_200:.2f})")
            await place_order(stock, 1, "BUY", limit_price)
            
         # ✅ Buy if breakout occurs
        elif breakout_signal and (stock not in trade_log or trade_log[stock]["last_breakout"] != breakout_price):
            print(f"📢 BUY {stock} - Breakout Detected at ${breakout_price:.2f}")
            await place_order(stock, 1, "BUY", limit_price, breakout_price=breakout_price)

    await monitor_trades(market_data_cache)

    
    
# Main loop to run the bot continuously
async def main():
    await connect_ibkr()
    while True:
        await run_trading_bot()
        print("⏳ Waiting for next cycle...")
        await asyncio.sleep(300)  # Runs every 5 minutes

# Async loop fix for Windows
if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(main())


