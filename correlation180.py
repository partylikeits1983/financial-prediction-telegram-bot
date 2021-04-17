import pandas as pd
import yfinance as yf
import yahoofinancials

import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from datetime import date

#####how many days the correlation is looking at
today = date.today()
d = datetime.timedelta(days=180)
start = today - d


def plot_tickers(tickers, start, end, interval, track, log_plot, normalize):
    data_df = yf.download(tickers,
                          start=start,
                          end=end,
                          interval=interval,
                          progress=False)

    # Drop any NaNs (e.g. when comparing SPY to 'BTC-USD')
    data_df = data_df.dropna()

    # normalize df
    if normalize == True:
        data_df = (data_df - data_df.mean()) / data_df.std()
    else:
        pass

    # Plot tickers
    ticker_list = tickers.split(' ')

    for ticker in ticker_list:
        if ticker == 'BTC-USD':  # put BTC on right axis
            ax = data_df[track, ticker].plot(secondary_y=True, figsize=(14, 10), legend=True, logy=log_plot, grid=True)
        else:
            ax = data_df[track, ticker].plot(figsize=(14, 10), legend=True, logy=log_plot, grid=True)

    ax.get_legend().set_bbox_to_anchor((1.3, 1))

    return data_df


def calc_correlation(data_df, track):
    # Get correlation and sort by sum
    sum_corr = data_df[track].corr().sum().sort_values(ascending=True).index.values

    data_df[track][sum_corr].corr()

    # Call the df with the list from summed correlation, sorted ascending.
    plt.figure(figsize=(13, 8))
    ax = sns.heatmap(data_df[track][sum_corr].corr(),
                     annot=True,
                     cmap="Blues")

    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)


'''Change inputs here:
'''

# Enter tickers to plot/compare
tickers = 'BTC-USD ETH-USD EURUSD=X RUB=X CL=F GLD SPY TSLA PYPL XMR-USD ^RUT'

# Timeframe
start = '{}'.format(start)
end = '{}'.format(today)

# Time interval: can be 1m, 1h, 1d
interval = '1d'

# key to track: 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'
track = 'Close'

# plot options
log_plot = False
normalize = True

# plot trends
data_df = plot_tickers(tickers, start, end, interval, track, log_plot, normalize)
plt.savefig('/home/ubuntu/Desktop/TelegramBot/charts/visualcorrelation180.jpeg', dpi=400, bbox_inches='tight')

# calculate and plot correlations
calc_correlation(data_df, track)
plt.savefig('/home/ubuntu/Desktop/TelegramBot/charts/correlationmatrix180.jpeg', dpi=400, bbox_inches='tight')


