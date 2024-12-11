
import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly

from plotly import graph_objs as go

# Constants
START = "2022-01-01"
TODAY = date.today().strftime("%Y-%m-%d")
#from datetime import timedelta
#TODAY = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")


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

# Plot raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="Stock Open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Stock Close"))
    fig.layout.update(title_text='Time Series Data with Range Slider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# Prepare data for forecasting
df_train = data[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})

# Forecasting using Prophet
m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Display forecast data
st.subheader('Forecast Data')
st.write(forecast.tail())

# Plot forecast
st.write('Stock Predicted Price')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

# Plot forecast components
st.write('Forecast Price Trends')
fig2 = m.plot_components(forecast)
st.write(fig2)
