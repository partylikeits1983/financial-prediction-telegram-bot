import pandas as pd
from pandas_datareader import data, wb  # Package and modules for importing data; this code may change depending on pandas version
import datetime
 
# We will look at stock prices starting on January 1, 2016
start = datetime.datetime(2018,1,1)
end = datetime.date.today()
 
# Let's get Apple stock data; Apple's ticker symbol is AAPL
# First argument is the series we want, second is the source ("yahoo" for Yahoo! Finance), third is the start date, fourth is the end date
graph = data.DataReader("CL=F", "yahoo", start, end)
 
type(graph)

import matplotlib.pyplot as plt   # Import matplotlib
# This line is necessary for the plot to appear in a Jupyter notebook

# Control the default size of figures in this Jupyter notebook



graph.plot(grid = True)


stock_return = graph.apply(lambda x: x / x[0])
stock_return.head()

stock_return.plot(grid = True).axhline(y = 1, color = "black", lw = 2)


import numpy as np
 
stock_change = graph.apply(lambda x: np.log(x) - np.log(x.shift(1))) # shift moves dates back by 1.
stock_change.head()




stock_change.plot(grid = True).axhline(y = 0, color = "black", lw = 2)


graph["20d"] = np.round(graph["Close"].rolling(window = 20, center = False).mean(), 2)
#pandas_candlestick_ohlc(apple.loc['2016-01-04':'2016-08-07',:], otherseries = "20d")


from fbprophet import Prophet


df = pd.DataFrame()
df['ds'] = stock_return.index
#df['y_orig']=daily_df.Pageviews.values
df['y']=graph['Close'].apply(lambda x: np.log(x)).values
df.tail()


m0 = Prophet(yearly_seasonality=True)
m0.fit(df)
#n_add = 365 - len()
n_add = 100
print("adding {n} days to reach the end of 2017.".format(n=n_add))
future = m0.make_future_dataframe(periods=n_add) # generate frame going to end of 2017; 112 added on 9/11/2017
future.tail()


forecast = m0.predict(future)
forecast[['ds','yhat','yhat_lower','yhat_upper']].tail()


forcast = m0.plot(forecast, ylabel='$\ln($stock_return$)$');


forcast.savefig('/home/ubuntu/Desktop/TelegramBot/charts/OILforcast.jpeg', dpi=400, bbox_inches='tight')


trend = m0.plot_components(forecast);


trend.savefig('/home/ubuntu/Desktop/TelegramBot/charts/OILtrend.jpeg', dpi=400, bbox_inches='tight')



