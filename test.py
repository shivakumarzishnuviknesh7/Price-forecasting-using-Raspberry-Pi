import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

# Constants
START = "2022-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Streamlit App Title
st.title('Stock Price Prediction Application')

# Dropdown for stock selection
stocks = (
     'TATAMOTORS.NS'
)
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

# Load and display data
data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Last Five Days of Data')
st.write(data.tail())

# Prepare data for forecasting
#df_train = data[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})
data.columns = ["ds","y"]

#training
prophet = Prophet(daily_seasonality=True)
prophet.fit(data)

#predictions
future_dates=prophet.make_future_dataframe(period)
predictions =prophet.predict(future_dates)

#viscualize
plot_plotly(prophet,predictions)




