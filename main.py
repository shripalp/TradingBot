from ib_insync import IB

# Connect to IBKR TWS
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)

if ib.isConnected():
    print("✅ Successfully connected to IBKR API!")
else:
    print("❌ Connection failed.")
