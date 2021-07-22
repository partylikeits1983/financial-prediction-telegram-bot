from yahoo_fin import stock_info as si
import pandas as pd
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot

end = date.today()
d = datetime.timedelta(days=730)
start = end - d

ticker = "ES=F"
s = si.get_data(ticker, start, end)
s['Date'] = s.index
s.rename({'close': 'Close'}, axis=1, inplace=True)
s.head()

df = pd.DataFrame()
df['ds'] = (s['Date'])
df['y'] = s['Close']
df.head()

m = Prophet()
m.fit(df)

future = m.make_future_dataframe(periods=12 * 8, freq='D')
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend', 'trend_lower', 'trend_upper']].tail()

fig1 = m.plot(forecast)
fig1.savefig('/home/ubuntu/Desktop/TelegramBot/charts/SP500forcast.jpeg', dpi=400, bbox_inches='tight')

fig2 = m.plot_components(forecast)
fig2.savefig('/home/ubuntu/Desktop/TelegramBot/charts/SP500trend.jpeg', dpi=400, bbox_inches='tight')

fig = m.plot(forecast)
a = add_changepoints_to_plot(fig.gca(), m, forecast)
fig.savefig('/home/ubuntu/Desktop/TelegramBot/charts/SP500forcastwithlines.jpeg', dpi=400, bbox_inches='tight')
