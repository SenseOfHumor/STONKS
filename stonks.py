import streamlit as st
from datetime import date

#import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go


START = "2014-01-01"
TODAY = date.today().strftime("%Y-%m-%d") #format of the date
st.title("STOONKS 📈")

stocks = ("AAPL", "GOOG", "MSFT", "GME")

selected = st.selectbox("Pick Your Stonk 🎺", stocks)
years = st.slider("Years under consideration",1,10)
period = years * 365

#cache the data so it doesn't load everytime
@st.cache_data

def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

#for the looks - loading nat type thing
dataLoadState = st.text("Loading Data...")
data = load_data(selected)
dataLoadState.text("Done")

st.markdown("---")
train = data[['Date', 'Close']]
train = train.rename(columns={"Date": "ds", "Close": "y"})

pro = Prophet()
pro.fit(train)

future = pro.make_future_dataframe(periods=period)
forecast = pro.predict(future)

st.subheader('Future Data 📊')
st.write(forecast.tail())

st.subheader('\nBad Statistical Model 📈')
forfig = plot_plotly(pro, forecast)
forfig.update_layout(autosize=False, width=500, height=500)  # Adjust the size of the chart
st.plotly_chart(forfig)








