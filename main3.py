import asyncio
import pandas as pd
import numpy as np
from ib_insync import *
import datetime

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
async def get_stock_data(symbol, duration='30 D', bar_size='1 hour'):
    """Fetch historical stock data asynchronously from IBKR."""
    print(f"📡 Fetching market data for {symbol}(delayed)...")

    contract = Stock(symbol, 'SMART', 'USD')
    bars = await ib.reqHistoricalDataAsync(
        contract,
        endDateTime='',
        durationStr=duration,
        barSizeSetting=bar_size,
        whatToShow='MIDPOINT',
        useRTH=True,
        
    )

    if not bars:
        print(f"⚠ No historical data found for {symbol}. Skipping.")
        return None

    df = pd.DataFrame(bars)
    df['close'] = df['close'].astype(float)

    print(f"✅ Received {len(df)} data points for {symbol}. Latest close: ${df['close'].iloc[-1]:.2f}")
    return df

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
    """Fetch open positions for the IBKR account and display them."""
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
        market_price = ib.reqMktData(position.contract, snapshot=True).last
        unrealized_pnl = (market_price - avg_cost) * quantity if market_price else "N/A"

        print(f"📈 {symbol}: {quantity} shares")
        print(f"   - Average Cost: ${avg_cost:.2f}")
        print(f"   - Current Price: ${market_price:.2f}" if market_price else "   - Current Price: Not available")
        print(f"   - Unrealized P&L: ${unrealized_pnl:.2f}" if isinstance(unrealized_pnl, float) else "   - Unrealized P&L: Not available")
        print("-" * 50)


async def cancel_order(symbol):
    """Cancel an open order for a given stock."""
    print(f"🚫 Attempting to cancel {symbol} order...")

    # Get all open trades
    open_trades = ib.openTrades()
    
    # Find the order for this stock
    for trade in open_trades:
        if trade.contract.symbol == symbol:
            print(f"🚫 Canceling {symbol} order (Order ID: {trade.order.orderId})...")
            ib.cancelOrder(trade.order)  # Cancel the order
            await asyncio.sleep(2)  # Allow IBKR time to process the cancellation
            
            # Check if the order is successfully canceled
            if trade.orderStatus.status == "Cancelled":
                print(f"✅ Order for {symbol} successfully canceled.")
                return
            else:
                print(f"⚠ Failed to confirm cancellation of {symbol} order.")

    print(f"⚠ No open order found for {symbol}.")





# Trade tracking
trade_log = {}

async def place_order(symbol, quantity, action, limit_price, breakout_price=None, stop_loss_pct=3, take_profit_pct=10):
    """Places a Limit Order asynchronously and ensures order_time is set for pending trades."""
    
    
    # Prevent duplicate trades at the same breakout level or if a trade is pending
    if symbol in trade_log and (trade_log[symbol].get("status") == "pending" or trade_log[symbol].get("last_breakout") == breakout_price):
        print(f"⏳ Skipping {symbol} - Order already placed or pending at breakout level ${breakout_price:.2f}.")
        return  

    contract = Stock(symbol, 'SMART', 'USD')

    # Define limit order
    order = LimitOrder(action, quantity, limit_price)
    
    trade = ib.placeOrder(contract, order)
    await asyncio.sleep(2)  # Allow order execution time

    # Mark trade as pending and store order time
    trade_log[symbol] = {
        "status": "pending",
        "last_breakout": breakout_price,
        "order_time": datetime.datetime.now(),
        "limit_price": limit_price  # Store the limit price
    }

    print(f"📌 Placed {action} Limit Order for {symbol}: {quantity} shares at ${limit_price:.2f}")

    if trade.orderStatus.status == 'Filled':
        entry_price = trade.orderStatus.avgFillPrice
        stop_loss_price = entry_price * (1 - stop_loss_pct / 100)
        take_profit_price = entry_price * (1 + take_profit_pct / 100)

        # Update trade log with actual trade details
        trade_log[symbol] = {
            "entry_price": entry_price,
            "stop_loss": stop_loss_price,
            "take_profit": take_profit_price,
            "status": "open",
            "last_breakout": breakout_price,
            "limit_price": limit_price
        }

        print(f"✅ Trade Executed: {action} {quantity} shares of {symbol} at ${entry_price:.2f}")
        print(f"   - 🛑 Stop-Loss: ${stop_loss_price:.2f}")
        print(f"   - 🎯 Take-Profit: ${take_profit_price:.2f}")

    else:
        print(f"⚠ Limit Order for {symbol} not filled. Status: {trade.orderStatus.status}. Retrying in next cycle.")



# Monitor Active Trades
async def monitor_trades(market_data_cache):
    """Checks stop-loss, take-profit, and cancels pending orders using stored data."""
    print("🔎 Monitoring active trades...")

    for symbol, trade_info in list(trade_log.items()):
        df = market_data_cache.get(symbol)

        if df is None:
            print(f"⚠ No cached data for {symbol}. Skipping monitoring.")
            continue

        current_price = df['close'].iloc[-1]
        print(f"📊 {symbol} - Current Price: ${current_price:.2f}")

        if trade_info["status"] == "pending":
            # Check if order_time exists before using it
            if "order_time" in trade_info:
                time_elapsed = (datetime.datetime.now() - trade_info["order_time"]).total_seconds() / 60  # Minutes
                if time_elapsed > 10:
                    print(f"⚠ Canceling {symbol} order. It has been pending for {time_elapsed:.1f} minutes.")
                    await cancel_order(symbol)  # Now this function is properly defined
                    trade_log.pop(symbol)  # Remove from pending trades
            else:
                print(f"⚠ Missing order_time for {symbol}. Skipping cancellation check.")
            continue

        if current_price <= trade_info["stop_loss"]:
            print(f"❌ STOP-LOSS HIT! Selling {symbol} at ${current_price:.2f}")
            await place_order(symbol, 1, "SELL")
            trade_log.pop(symbol)

        elif current_price >= trade_info["take_profit"]:
            print(f"✅ TAKE-PROFIT HIT! Selling {symbol} at ${current_price:.2f}")
            await place_order(symbol, 1, "SELL")
            trade_log.pop(symbol)


# Run Trading Strategy
async def run_trading_bot():
    """Runs the trading strategy and displays account info."""
    print("\n🔄 Running trading bot cycle...")

    # Display account balance and open positions before trading
    await get_account_balance()
    await get_positions()

    stocks = ["intc"]
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
        breakout_signal = df['Breakout'].iloc[-1]
        current_price = df['close'].iloc[-1]
        breakout_price = df['20_day_high'].iloc[-1]

        limit_price = current_price * 0.995  # Buy slightly below market price

        if latest_rsi < 30 and stock not in trade_log:
            print(f"📢 BUY {stock} - RSI Oversold")
            await place_order(stock, 1, "BUY", limit_price)

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


