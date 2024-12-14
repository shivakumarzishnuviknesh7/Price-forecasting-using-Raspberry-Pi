from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
import streamlit as st

# Constants
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Streamlit app title
st.title("Stock Price Prediction App")

# Dropdown for stock selection
#stocks = ['TATAMOTORS.NS']
stocks = ('BAJAJFINSV.NS', 'TATAMOTORS.NS', 'BAJFINANCE.NS', 'JSWSTEEL.NS', 'COALINDIA.NS', 'ICICIBANK.NS', 'RELIANCE.NS', 'APOLLOHOSP.NS', 'ITC.NS', 'HDFCBANK.NS', 'TATASTEEL.NS', 'GRASIM.NS', 'AXISBANK.NS', 'KOTAKBANK.NS', 'SBIN.NS', 'HDFCBANK.NS', 'INDUSINDBK.NS', 'HCLTECH.NS', 'HINDALCO.NS', 'INFY.NS', 'M&M.NS', 'BAJAJ-AUTO.NS', 'TATAMOTORS.NS', 'TECHM.NS', 'NESTLEIND.NS', 'ONGC.NS', 'BHARTIARTL.NS', 'TCS.NS', 'HINDUNILVR.NS', 'BRITANNIA.NS', 'NTPC.NS', 'EICHERMOT.NS', 'TATACONSUM.NS', 'MARUTI.NS', 'ADANIPORTS.NS', 'HEROMOTOCO.NS', 'SBILIFE.NS', 'POWERGRID.NS', 'DIVISLAB.NS', 'ULTRACEMCO.NS', 'LT.NS', 'SUNPHARMA.NS', 'UPL.NS', 'CIPLA.NS', 'ASIANPAINT.NS', 'DRREDDY.NS', 'BPCL.NS', 'BPCL.NS', 'HDFCLIFE.NS', 'SHREECEM.NS', 'WIPRO.NS')
#selected_stock = st.selectbox("Select a stock ticker", stocks)

selected_stock = st.selectbox('Select company for prediction', stocks)

# Slider for prediction period
n_years = st.slider('Years of prediction:', 1, 10)
period = n_years * 365

# Function to load stock data
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

# Load the data
data = load_data(selected_stock)
st.write("### Raw Data", data.tail())

# Preprocessing the data for Prophet
data = data[["Date", "Close"]]
data.columns = ["ds", "y"]

# Show data
st.write("### Data for Prophet Model", data)

# Train the model
prophet = Prophet(daily_seasonality=True)
prophet.fit(data)

# Predictions
future_dates = prophet.make_future_dataframe(periods=period)
predictions = prophet.predict(future_dates)

# Visualize Predictions
st.write("### Predictions")
fig = plot_plotly(prophet, predictions)
st.plotly_chart(fig)

# Show forecast components
st.write("### Forecast Components")
components_fig = prophet.plot_components(predictions)
st.write(components_fig)


#usb stick