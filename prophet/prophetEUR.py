import pandas as pd
from pandas_datareader import data, wb  # Package and modules for importing data; this code may change depending on pandas version
import matplotlib.pyplot as plt   
    
import datetime
from datetime import date 
import numpy as np
from fbprophet import Prophet
    
end = date.today()
d = datetime.timedelta(days=365)
start = end - d
    
graph = data.DataReader("EURUSD=X", "yahoo", start, end)
 
type(graph)

graph.plot(grid = True)
stock_return = graph.apply(lambda x: x / x[0])
stock_return.head()
stock_return.plot(grid = True).axhline(y = 1, color = "black", lw = 2)
stock_change = graph.apply(lambda x: np.log(x) - np.log(x.shift(1))) # shift moves dates back by 1.
stock_change.head()
stock_change.plot(grid = True).axhline(y = 0, color = "black", lw = 2)
graph["20d"] = np.round(graph["Close"].rolling(window = 20, center = False).mean(), 2)


df = pd.DataFrame()
df['ds'] = stock_return.index
df['y']=graph['Close'].apply(lambda x: np.log(x)).values
df.tail()

m0 = Prophet(yearly_seasonality=True)
m0.fit(df)
#how many days in the future to show predictions for 
n_add = 100
print("adding {n} days.".format(n=n_add))
future = m0.make_future_dataframe(periods=n_add) 
future.tail()

forecast = m0.predict(future)
forecast[['ds','yhat','yhat_lower','yhat_upper']].tail()
forcast = m0.plot(forecast, ylabel='$\ln($stock_return$)$');
forcast.savefig('/home/ubuntu/Desktop/TelegramBot/charts/EURforcast.jpeg', dpi=400, bbox_inches='tight')


trend = m0.plot_components(forecast);
trend.savefig('/home/ubuntu/Desktop/TelegramBot/charts/EURtrend.jpeg', dpi=400, bbox_inches='tight')

