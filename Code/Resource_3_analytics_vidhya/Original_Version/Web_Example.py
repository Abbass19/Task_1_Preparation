import yfinance as yf

ticker = 'HDFCBANK.NS'  # Correct NSE ticker for HDFC Bank

# Download 5 years of daily data
data = yf.download(ticker, period='5y')

# Check if data is loaded correctly
if data.empty:
    print("No data found. Check the ticker symbol or your internet connection.")
else:
    data.to_csv('hdfc_bank_5years.csv')
    print("Data downloaded and saved to 'hdfc_bank_5years.csv'")
