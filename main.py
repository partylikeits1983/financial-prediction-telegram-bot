from datetime import date, time, tzinfo, timezone, datetime
import datetime
import pytz
import schedule
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as web
import datetime
from datetime import date
from yahoo_fin import stock_info as si
import csv

import telegram
from telegram import Update
from telegram.ext import Updater, CommandHandler, CallbackContext

import logging

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

TOKEN = 'token'

today = date.today()

bot = telegram.Bot(TOKEN)


def start(update: Update, context: CallbackContext) -> None:
    update.message.reply_text('Hello! Welcome to the Financial Forecast and Price Prediction Telegram bot!'
                              ' This bot gives daily price correlations and future price predictions of'
                              ' Bitcoin, Ether, Monero, USD/EUR, USD/RUB, PYPL, TSLA, SP500, and the Russel 2000!')

    context.bot.sendChatAction(chat_id=update.message.chat_id, action=telegram.ChatAction.TYPING)

    update.message.reply_text('Type /update to see the latest correlation matricies and price predictions.'
                              ' Type /list to see the list of analyzed assets.'
                              ' Type /info for more info. All charts are updated daily.')
   
    from datetime import datetime
    user = update.message.from_user
    chat_id = update.message.chat_id
    first_name = update.message.chat.first_name
    last_name = update.message.chat.last_name
    username = update.message.chat.username
    fullname = "{} {}".format(first_name, last_name)
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d %H:%M:%S")
    logfile = [dt_string, chat_id, fullname, username, 'update']
    
    with open('/home/ubuntu/Desktop/telegrambotlog.csv', 'a', newline='') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(logfile)
    
    print("{} Name: {} {} Username: {} Chat ID: {} Function: Start". format(dt_string, first_name, last_name , username, chat_id))
    
    

def list(update: Update, context: CallbackContext) -> None:
    update.message.reply_text(
        'Type /BTC /ETH /XMR /EUR /RUB\n /PYPL /TSLA /RUS2000 /SP500 to get specific information about each asset.'
        )


def update(update, context):
    context.bot.sendChatAction(chat_id=update.message.chat_id, action=telegram.ChatAction.TYPING)

    photo = open('/home/ubuntu/Desktop/TelegramBot/charts/correlationmatrix180.jpeg', 'rb')
    caption = "180 day correlation matrix {}".format(today)
    chat_id = update.message.chat_id
    context.bot.send_photo(chat_id, photo, caption)   
    
    photo = open('/home/ubuntu/Desktop/TelegramBot/charts/correlationmatrix30.jpeg', 'rb')
    caption = "30 day correlation matrix {}".format(today)
    chat_id = update.message.chat_id
    context.bot.send_photo(chat_id, photo, caption)

    photo = open('/home/ubuntu/Desktop/TelegramBot/charts/yield.jpeg', 'rb')
    caption = "Yield Curve {}".format(today)
    chat_id = update.message.chat_id
    context.bot.send_photo(chat_id, photo, caption)
    
    BTC1 = open('/home/ubuntu/Desktop/TelegramBot/predictions/futurepriceBTC.txt', 'r').read()
    BTC2 = open('/home/ubuntu/Desktop/TelegramBot/predictions/meanabsoluteerrorBTC.txt', 'r').read()
    
    if type(BTC1) or type(BTC2) == int or float:
        #calculating Δ%
        nbtc = float(BTC1)
        s = pd.Series([si.get_live_price("BTC-USD"), nbtc])
        s.pct_change()
        normal_sum = s.pct_change()
        normal_sum.at[1]
        dvBTC = str(round((normal_sum.at[1] * 100), 2))
        #calculating % error
        nbtc2 = float(BTC2)
        e = (nbtc2 / nbtc) * 100
        eBTC = '%.2f' % e
        BTC = 'Predicted price of Bitcoin in 1 day $%s   (Δ%s%%)   %s%% error' % (BTC1, dvBTC, eBTC)
        
    else:
        text = 'Yahoo Finance is currently missing data for BTC-USD. Could not run prediction model at this time.'
        chat_id = update.message.chat_id
        bot.send_message(chat_id, text)

    ETH1 = open('/home/ubuntu/Desktop/TelegramBot/predictions/futurepriceETH.txt', 'r').read()
    ETH2 = open('/home/ubuntu/Desktop/TelegramBot/predictions/meanabsoluteerrorETH.txt', 'r').read()

    if type(ETH1) or type(ETH2) == int or float:
        #calculating Δ%
        neth = float(ETH1)
        s = pd.Series([si.get_live_price("ETH-USD"), neth])
        s.pct_change()
        normal_sum = s.pct_change()
        normal_sum.at[1]
        dvETH = str(round((normal_sum.at[1] * 100), 2))
        #calculating % error
        neth2 = float(ETH2)
        e = (neth2 / neth) * 100
        eETH = '%.2f' % e
        ETH = 'Predicted price of Ether in 1 day $%s   (Δ%s%%)   %s%% error' % (ETH1, dvETH, eETH)

    else:
        text = 'Yahoo Finance is currently missing data for ETH-USD. Could not run prediction model at this time.'
        chat_id = update.message.chat_id
        bot.send_message(chat_id, text)

    XMR1 = open('/home/ubuntu/Desktop/TelegramBot/predictions/futurepriceXMR.txt', 'r').read()
    XMR2 = open('/home/ubuntu/Desktop/TelegramBot/predictions/meanabsoluteerrorXMR.txt', 'r').read()

    if type(XMR1) or type(XMR2) == int or float:
        #calculating Δ%
        nxmr = float(XMR1)
        s = pd.Series([si.get_live_price("XMR-USD"), nxmr])
        s.pct_change()
        normal_sum = s.pct_change()
        normal_sum.at[1]
        dvXMR = str(round((normal_sum.at[1] * 100), 2))
        #calculating % error
        nxmr2 = float(XMR2)
        e = (nxmr2 / nxmr) * 100
        eXMR = '%.2f' % e
        XMR = 'Predicted price of Monero in 1 day $%s   (Δ%s%%)   %s%% error' % (XMR1, dvXMR, eXMR)
        

    else:
        text = 'Yahoo Finance is currently missing data for XMR-USD. Could not run prediction model at this time.'
        chat_id = update.message.chat_id
        bot.send_message(chat_id, text)

    PYPL1 = open('/home/ubuntu/Desktop/TelegramBot/predictions/futurepricePYPL.txt', 'r').read()
    PYPL2 = open('/home/ubuntu/Desktop/TelegramBot/predictions/meanabsoluteerrorPYPL.txt', 'r').read()

    if type(PYPL1) or type(PYPL2) == int or float:
        #calculating Δ%
        npypl = float(PYPL1)
        s = pd.Series([si.get_live_price("PYPL"), npypl])
        s.pct_change()
        normal_sum = s.pct_change()
        normal_sum.at[1]
        dvPYPL = str(round((normal_sum.at[1] * 100), 2))
        #calculating % error
        npypl2 = float(PYPL2)
        e = (npypl2 / npypl) * 100
        ePYPL = '%.2f' % e
        PYPL = 'Predicted price of PayPal stock in 7 days $%s (Δ%s%%) %s%% error' % (PYPL1, dvPYPL, ePYPL)
        
    else:
        text = 'Yahoo Finance is currently missing data for PYPL. Could not run prediction model at this time.'
        chat_id = update.message.chat_id
        bot.send_message(chat_id, text)

    TSLA1 = open('/home/ubuntu/Desktop/TelegramBot/predictions/futurepriceTSLA.txt', 'r').read()
    TSLA2 = open('/home/ubuntu/Desktop/TelegramBot/predictions/meanabsoluteerrorTSLA.txt', 'r').read()

    if type(TSLA1) or type(TSLA2) == int or float:
        #calculating Δ%
        ntsla = float(TSLA1)
        s = pd.Series([si.get_live_price("TSLA"), ntsla])
        s.pct_change()
        normal_sum = s.pct_change()
        normal_sum.at[1]
        dvTSLA = str(round((normal_sum.at[1] * 100), 2))
        #calculating % error
        ntsla2 = float(TSLA2)
        e = (ntsla2 / ntsla) * 100
        eTSLA = '%.2f' % e
        TSLA = 'Predicted price of Tesla stock in 7 days $%s (Δ%s%%) %s%% error' % (TSLA1, dvTSLA, eTSLA)

    else:
        text = 'Yahoo Finance is currently missing data for TSLA. Could not run prediction model at this time.'
        chat_id = update.message.chat_id
        bot.send_message(chat_id, text)

    EUR1 = open('/home/ubuntu/Desktop/TelegramBot/predictions/futurepriceEUR.txt', 'r').read()
    EUR2 = open('/home/ubuntu/Desktop/TelegramBot/predictions/meanabsoluteerrorEUR.txt', 'r').read()

    if type(EUR1) or type(EUR2) == int or float:
        #calculating Δ%
        neur = float(EUR1)
        s = pd.Series([si.get_live_price("EURUSD=X"), neur])
        s.pct_change()
        normal_sum = s.pct_change()
        normal_sum.at[1]
        dvEUR = str(round((normal_sum.at[1] * 100), 2))
        #calculating % error
        neur2 = float(EUR2)
        e = (neur2 / neur) * 100
        eEUR = '%.2f' % e
        EUR = 'Predicted exchange rate of EUR/USD in 7 days $%s (Δ%s%%) %s%% error' % (EUR1, dvEUR, eEUR)
        

    else:
        text = 'Yahoo Finance is currently missing data for EUR/USD. Could not run prediction model at this time.'
        chat_id = update.message.chat_id
        bot.send_message(chat_id, text)

    RUB1 = open('/home/ubuntu/Desktop/TelegramBot/predictions/futurepriceRUB.txt', 'r').read()
    RUB2 = open('/home/ubuntu/Desktop/TelegramBot/predictions/meanabsoluteerrorRUB.txt', 'r').read()

    if type(RUB1) or type(RUB2) == int or float:
        #calculating Δ%
        nrub = float(RUB1)
        s = pd.Series([si.get_live_price("RUB=X"), nrub])
        s.pct_change()
        normal_sum = s.pct_change()
        normal_sum.at[1]
        dvRUB = str(round((normal_sum.at[1] * 100), 2))
        #calculating % error
        nrub2 = float(RUB2)
        e = (nrub2 / nrub) * 100
        eRUB = '%.2f' % e
        RUB = 'Predicted exchange rate of USD/RUB in 7 days $%s (Δ%s%%) %s%% error' % (RUB1, dvRUB, eRUB)

    else:
        text = 'Yahoo Finance is currently missing data for USD/RUB. Could not run prediction model at this time.'
        chat_id = update.message.chat_id
        bot.send_message(chat_id, text)

    SP1 = open('/home/ubuntu/Desktop/TelegramBot/predictions/futurepriceSP500.txt', 'r').read()
    SP2 = open('/home/ubuntu/Desktop/TelegramBot/predictions/meanabsoluteerrorSP500.txt', 'r').read()

    if type(SP1) or type(SP2) == int or float:
        #calculating Δ%
        nsp = float(SP1)
        s = pd.Series([si.get_live_price("^GSPC"), nsp])
        s.pct_change()
        normal_sum = s.pct_change()
        normal_sum.at[1]
        dvSP = str(round((normal_sum.at[1] * 100), 2))
        #calculating % error
        nsp2 = float(SP2)
        e = (nsp2 / nsp) * 100
        eSP = '%.2f' % e
        SP = 'Predicted value of the S&P500 in 7 days %s points (Δ%s%%) %s%% error' % (SP1, dvSP, eSP)

    else:
        text = 'Yahoo Finance is currently missing data for SP500. Could not run prediction model at this time.'
        chat_id = update.message.chat_id
        bot.send_message(chat_id, text)

    RUS1 = open('/home/ubuntu/Desktop/TelegramBot/predictions/futurepriceRUS.txt', 'r').read()
    RUS2 = open('/home/ubuntu/Desktop/TelegramBot/predictions/meanabsoluteerrorRUS.txt', 'r').read()

    if type(RUS1) or type(RUS2) == int or float:
        #calculating Δ%
        nrus = float(RUS1)
        s = pd.Series([si.get_live_price("^RUT"), nrus])
        s.pct_change()
        normal_sum = s.pct_change()
        normal_sum.at[1]
        dvRUS = str(round((normal_sum.at[1] * 100), 2))
        #calculating % error
        nrus2 = float(RUS2)
        e = (nrus2 / nrus) * 100
        eRUS = '%.2f' % e
        RUS = 'Predicted value of the Russel 2000 in 7 days %s points (Δ%s%%) %s%% error' % (RUS1, dvRUS, eRUS)

    else:
        text = 'Yahoo Finance is currently missing data for Russel 2000. Could not run prediction model at this time.'
        chat_id = update.message.chat_id
        bot.send_message(chat_id, text)

   
    dailyupdate = ("▪️{}\n\n▪️{}\n\n▪️{}\n\n▪️{}\n\n▪️{}\n\n▪️{}\n\n▪️{}\n\n🇪🇺{}\n\n🇷🇺{}").format(BTC,ETH,XMR,PYPL,TSLA,SP,RUS,EUR,RUB)
    bot.send_message(chat_id, dailyupdate)

    update.message.reply_text(
        'Type /BTC /ETH /XMR /EUR /RUB\n /PYPL /TSLA /RUS2000 /SP500 to get specific information about each asset.'
        )
    

    from datetime import datetime
    user = update.message.from_user
    chat_id = update.message.chat_id
    first_name = update.message.chat.first_name
    last_name = update.message.chat.last_name
    username = update.message.chat.username
    fullname = "{} {}".format(first_name, last_name)
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d %H:%M:%S")
    logfile = [dt_string, chat_id, fullname, username, 'update']
    
    with open('/home/ubuntu/Desktop/telegrambotlog.csv', 'a', newline='') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(logfile)
    
    print("{} Name: {} {} Username: {} Chat ID: {} Function: Update". format(dt_string, first_name, last_name , username, chat_id))
   
    '''
    GOLD1 = open('/home/ubuntu/Desktop/TelegramBot/predictions/futurepriceGOLD.txt', 'r').read()
    GOLD2 = open('/home/ubuntu/Desktop/TelegramBot/predictions/meanabsoluteerrorGOLD.txt', 'r').read()
    text = 'Predicted price of Gold in 7 days $%s mean absolute error %s' % (GOLD1, GOLD2)
    chat_id = update.message.chat_id
    bot.send_message(chat_id, text)

    OIL1 = open('/home/ubuntu/Desktop/TelegramBot/predictions/futurepriceOIL.txt', 'r').read()
    OIL2 = open('/home/ubuntu/Desktop/TelegramBot/predictions/meanabsoluteerrorOIL.txt', 'r').read()
    text = 'Predicted price of Crude Oil in 7 days $%s mean absolute error %s' % (OIL1, OIL2)
    chat_id = update.message.chat_id
    bot.send_message(chat_id, text)

    '''


def BTC(update, context):
    context.bot.sendChatAction(chat_id=update.message.chat_id, action=telegram.ChatAction.TYPING)
    photo = open('/home/ubuntu/Desktop/TelegramBot/charts/BTCforcast.jpeg', 'rb')
    caption = "Bitcoin forcast chart for {}".format(today)
    chat_id = update.message.chat_id
    context.bot.send_photo(chat_id, photo, caption)

    photo = open('/home/ubuntu/Desktop/TelegramBot/charts/BTCtrend.jpeg', 'rb')
    caption = "Bitcoin performance {}".format(today)
    chat_id = update.message.chat_id
    context.bot.send_photo(chat_id, photo, caption)

    BTC1 = open('/home/ubuntu/Desktop/TelegramBot/predictions/futurepriceBTC.txt', 'r').read()
    BTC2 = open('/home/ubuntu/Desktop/TelegramBot/predictions/meanabsoluteerrorBTC.txt', 'r').read()

    if type(BTC1) == int or float:
        #calculating Δ%
        nbtc = float(BTC1)
        s = pd.Series([si.get_live_price("BTC-USD"), nbtc])
        s.pct_change()
        normal_sum = s.pct_change()
        normal_sum.at[1]
        dvBTC = str(round((normal_sum.at[1] * 100), 2))
        #calculating % error
        nbtc2 = float(BTC2)
        e = (nbtc2 / nbtc) * 100
        eBTC = '%.2f' % e
        BTC = 'Predicted price of Bitcoin in 1 day $%s   (Δ%s%%)   %s%% error' % (BTC1, dvBTC, eBTC)
        chat_id = update.message.chat_id
        bot.send_message(chat_id, BTC)
    else:
        text = 'Yahoo Finance is missing data for this asset. Could not run prediction model at this time.'
        chat_id = update.message.chat_id
        bot.send_message(chat_id, text)


def ETH(update, context):
    context.bot.sendChatAction(chat_id=update.message.chat_id, action=telegram.ChatAction.TYPING)
    photo = open('/home/ubuntu/Desktop/TelegramBot/charts/ETHforcast.jpeg', 'rb')
    caption = "Ether future price chart for {}".format(today)
    chat_id = update.message.chat_id
    context.bot.send_photo(chat_id, photo, caption)

    photo = open('/home/ubuntu/Desktop/TelegramBot/charts/ETHtrend.jpeg', 'rb')
    caption = "Ether performance {}".format(today)
    chat_id = update.message.chat_id
    context.bot.send_photo(chat_id, photo, caption)

    ETH1 = open('/home/ubuntu/Desktop/TelegramBot/predictions/futurepriceETH.txt', 'r').read()
    ETH2 = open('/home/ubuntu/Desktop/TelegramBot/predictions/meanabsoluteerrorETH.txt', 'r').read()

    if type(ETH1) or type(ETH2) == int or float:
        #calculating Δ%
        neth = float(ETH1)
        s = pd.Series([si.get_live_price("ETH-USD"), neth])
        s.pct_change()
        normal_sum = s.pct_change()
        normal_sum.at[1]
        dvETH = str(round((normal_sum.at[1] * 100), 2))
        #calculating % error
        neth2 = float(ETH2)
        e = (neth2 / neth) * 100
        eETH = '%.2f' % e
        ETH = 'Predicted price of Ether in 1 day $%s   (Δ%s%%)   %s%% error' % (ETH1, dvETH, eETH)

    else:
        text = 'Yahoo Finance is currently missing data for ETH-USD. Could not run prediction model at this time.'
        chat_id = update.message.chat_id
        bot.send_message(chat_id, ETH)


def EUR(update, context):
    context.bot.sendChatAction(chat_id=update.message.chat_id, action=telegram.ChatAction.TYPING)
    photo = open('/home/ubuntu/Desktop/TelegramBot/charts/EURforcast.jpeg', 'rb')
    caption = "USD/EUR future price chart for {}".format(today)
    chat_id = update.message.chat_id
    context.bot.send_photo(chat_id, photo, caption)

    photo = open('/home/ubuntu/Desktop/TelegramBot/charts/EURtrend.jpeg', 'rb')
    caption = "EURO performance {}".format(today)
    chat_id = update.message.chat_id
    context.bot.send_photo(chat_id, photo, caption)

    EUR1 = open('/home/ubuntu/Desktop/TelegramBot/predictions/futurepriceEUR.txt', 'r').read()
    EUR2 = open('/home/ubuntu/Desktop/TelegramBot/predictions/meanabsoluteerrorEUR.txt', 'r').read()

    if type(EUR1) or type(EUR2) == int or float:
        #calculating Δ%
        neur = float(EUR1)
        s = pd.Series([si.get_live_price("EURUSD=X"), neur])
        s.pct_change()
        normal_sum = s.pct_change()
        normal_sum.at[1]
        dvEUR = str(round((normal_sum.at[1] * 100), 2))
        #calculating % error
        neur2 = float(EUR2)
        e = (neur2 / neur) * 100
        eEUR = '%.2f' % e
        EUR = 'Predicted exchange rate of EUR/USD in 7 days $%s (Δ%s%%) %s%% error' % (EUR1, dvEUR, eEUR)
        chat_id = update.message.chat_id
        bot.send_message(chat_id, EUR)

    else:
        text = 'Yahoo Finance is currently missing data for EUR/USD. Could not run prediction model at this time.'
        chat_id = update.message.chat_id
        bot.send_message(chat_id, text)


def RUB(update, context):
    context.bot.sendChatAction(chat_id=update.message.chat_id, action=telegram.ChatAction.TYPING)
    photo = open('/home/ubuntu/Desktop/TelegramBot/charts/RUBforcast.jpeg', 'rb')
    caption = "USD/RUB exchange rate prediction chart for {}".format(today)
    chat_id = update.message.chat_id
    context.bot.send_photo(chat_id, photo, caption)

    photo = open('/home/ubuntu/Desktop/TelegramBot/charts/RUBtrend.jpeg', 'rb')
    caption = "Ruble performance {}".format(today)
    chat_id = update.message.chat_id
    context.bot.send_photo(chat_id, photo, caption)

    RUB1 = open('/home/ubuntu/Desktop/TelegramBot/predictions/futurepriceRUB.txt', 'r').read()
    RUB2 = open('/home/ubuntu/Desktop/TelegramBot/predictions/meanabsoluteerrorRUB.txt', 'r').read()

    if type(RUB1) or type(RUB2) == int or float:
        #calculating Δ%
        nrub = float(RUB1)
        s = pd.Series([si.get_live_price("RUB=X"), nrub])
        s.pct_change()
        normal_sum = s.pct_change()
        normal_sum.at[1]
        dvRUB = str(round((normal_sum.at[1] * 100), 2))
        #calculating % error
        nrub2 = float(RUB2)
        e = (nrub2 / nrub) * 100
        eRUB = '%.2f' % e
        RUB = 'Predicted exchange rate of USD/RUB in 7 days $%s (Δ%s%%) %s%% error' % (RUB1, dvRUB, eRUB)
        chat_id = update.message.chat_id
        bot.send_message(chat_id, RUB)

    else:
        text = 'Yahoo Finance is currently missing data for USD/RUB. Could not run prediction model at this time.'
        chat_id = update.message.chat_id
        bot.send_message(chat_id, text)


def XMR(update, context):
    context.bot.sendChatAction(chat_id=update.message.chat_id, action=telegram.ChatAction.TYPING)
    photo = open('/home/ubuntu/Desktop/TelegramBot/charts/XMRforcast.jpeg', 'rb')
    caption = "Monero price prediction chart for {}".format(today)
    chat_id = update.message.chat_id
    context.bot.send_photo(chat_id, photo, caption)

    photo = open('/home/ubuntu/Desktop/TelegramBot/charts/XMRtrend.jpeg', 'rb')
    caption = "Monero performance {}".format(today)
    chat_id = update.message.chat_id
    context.bot.send_photo(chat_id, photo, caption)

    XMR1 = open('/home/ubuntu/Desktop/TelegramBot/predictions/futurepriceXMR.txt', 'r').read()
    XMR2 = open('/home/ubuntu/Desktop/TelegramBot/predictions/meanabsoluteerrorXMR.txt', 'r').read()

    if type(XMR1) or type(XMR2) == int or float:
        #calculating Δ%
        nxmr = float(XMR1)
        s = pd.Series([si.get_live_price("XMR-USD"), nxmr])
        s.pct_change()
        normal_sum = s.pct_change()
        normal_sum.at[1]
        dvXMR = str(round((normal_sum.at[1] * 100), 2))
        #calculating % error
        nxmr2 = float(XMR2)
        e = (nxmr2 / nxmr) * 100
        eXMR = '%.2f' % e
        XMR = 'Predicted price of Monero in 1 day $%s   (Δ%s%%)   %s%% error' % (XMR1, dvXMR, eXMR)
        chat_id = update.message.chat_id
        bot.send_message(chat_id, XMR)
    else:
        text = 'Yahoo Finance is currently missing data for XMR-USD. Could not run prediction model at this time.'
        chat_id = update.message.chat_id
        bot.send_message(chat_id, text)


def PYPL(update, context):
    context.bot.sendChatAction(chat_id=update.message.chat_id, action=telegram.ChatAction.TYPING)
    photo = open('/home/ubuntu/Desktop/TelegramBot/charts/PYPLforcast.jpeg', 'rb')
    caption = "PayPal price prediction chart for {}".format(today)
    chat_id = update.message.chat_id
    context.bot.send_photo(chat_id, photo, caption)

    photo = open('/home/ubuntu/Desktop/TelegramBot/charts/PYPLtrend.jpeg', 'rb')
    caption = "PYPL performance {}".format(today)
    chat_id = update.message.chat_id
    context.bot.send_photo(chat_id, photo, caption)

    PYPL1 = open('/home/ubuntu/Desktop/TelegramBot/predictions/futurepricePYPL.txt', 'r').read()
    PYPL2 = open('/home/ubuntu/Desktop/TelegramBot/predictions/meanabsoluteerrorPYPL.txt', 'r').read()

    if type(PYPL1) or type(PYPL2) == int or float:
        #calculating Δ%
        npypl = float(PYPL1)
        s = pd.Series([si.get_live_price("PYPL"), npypl])
        s.pct_change()
        normal_sum = s.pct_change()
        normal_sum.at[1]
        dvPYPL = str(round((normal_sum.at[1] * 100), 2))
        #calculating % error
        npypl2 = float(PYPL2)
        e = (npypl2 / npypl) * 100
        ePYPL = '%.2f' % e
        PYPL = 'Predicted price of PayPal stock in 7 days $%s (Δ%s%%) %s%% error' % (PYPL1, dvPYPL, ePYPL)
        chat_id = update.message.chat_id
        bot.send_message(chat_id, PYPL)

    else:
        text = 'Yahoo Finance is currently missing data for PYPL. Could not run prediction model at this time.'
        chat_id = update.message.chat_id
        bot.send_message(chat_id, text)


def TSLA(update, context):
    context.bot.sendChatAction(chat_id=update.message.chat_id, action=telegram.ChatAction.TYPING)
    photo = open('/home/ubuntu/Desktop/TelegramBot/charts/TSLAforcast.jpeg', 'rb')
    caption = "TSLA price prediction chart for {}".format(today)
    chat_id = update.message.chat_id
    context.bot.send_photo(chat_id, photo, caption)

    photo = open('/home/ubuntu/Desktop/TelegramBot/charts/TSLAtrend.jpeg', 'rb')
    caption = "TSLA performance {}".format(today)
    chat_id = update.message.chat_id
    context.bot.send_photo(chat_id, photo, caption)

    TSLA1 = open('/home/ubuntu/Desktop/TelegramBot/predictions/futurepriceTSLA.txt', 'r').read()
    TSLA2 = open('/home/ubuntu/Desktop/TelegramBot/predictions/meanabsoluteerrorTSLA.txt', 'r').read()

    if type(TSLA1) or type(TSLA2) == int or float:
        #calculating Δ%
        ntsla = float(TSLA1)
        s = pd.Series([si.get_live_price("TSLA"), ntsla])
        s.pct_change()
        normal_sum = s.pct_change()
        normal_sum.at[1]
        dvTSLA = str(round((normal_sum.at[1] * 100), 2))
        #calculating % error
        ntsla2 = float(TSLA2)
        e = (ntsla2 / ntsla) * 100
        eTSLA = '%.2f' % e
        TSLA = 'Predicted price of Tesla stock in 7 days $%s (Δ%s%%) %s%% error' % (TSLA1, dvTSLA, eTSLA)
        chat_id = update.message.chat_id
        bot.send_message(chat_id, TSLA)

    else:
        text = 'Yahoo Finance is currently missing data for TSLA. Could not run prediction model at this time.'
        chat_id = update.message.chat_id
        bot.send_message(chat_id, text)


def RUS2000(update, context):
    context.bot.sendChatAction(chat_id=update.message.chat_id, action=telegram.ChatAction.TYPING)
    photo = open('/home/ubuntu/Desktop/TelegramBot/charts/RUSforcast.jpeg', 'rb')
    caption = "USD/RUB exchange rate prediction chart for {}".format(today)
    chat_id = update.message.chat_id
    context.bot.send_photo(chat_id, photo, caption)

    photo = open('/home/ubuntu/Desktop/TelegramBot/charts/RUStrend.jpeg', 'rb')
    caption = "Russel 2000 performance {}".format(today)
    chat_id = update.message.chat_id
    context.bot.send_photo(chat_id, photo, caption)

    RUS1 = open('/home/ubuntu/Desktop/TelegramBot/predictions/futurepriceRUS.txt', 'r').read()
    RUS2 = open('/home/ubuntu/Desktop/TelegramBot/predictions/meanabsoluteerrorRUS.txt', 'r').read()

    if type(RUS1) or type(RUS2) == int or float:
        #calculating Δ%
        nrus = float(RUS1)
        s = pd.Series([si.get_live_price("^RUT"), nrus])
        s.pct_change()
        normal_sum = s.pct_change()
        normal_sum.at[1]
        dvRUS = str(round((normal_sum.at[1] * 100), 2))
        #calculating % error
        nrus2 = float(RUS2)
        e = (nrus2 / nrus) * 100
        eRUS = '%.2f' % e
        RUS = 'Predicted value of the Russel 2000 in 7 days %s points (Δ%s%%) %s%% error' % (RUS1, dvRUS, eRUS)
        chat_id = update.message.chat_id
        bot.send_message(chat_id, RUS)

    else:
        text = 'Yahoo Finance is currently missing data for Russel 2000. Could not run prediction model at this time.'
        chat_id = update.message.chat_id
        bot.send_message(chat_id, text)


def SP500(update, context):
    context.bot.sendChatAction(chat_id=update.message.chat_id, action=telegram.ChatAction.TYPING)
    photo = open('/home/ubuntu/Desktop/TelegramBot/charts/SP500forcast.jpeg', 'rb')
    caption = "SP500 prediction chart for {}".format(today)
    chat_id = update.message.chat_id
    context.bot.send_photo(chat_id, photo, caption)

    photo = open('/home/ubuntu/Desktop/TelegramBot/charts/SP500trend.jpeg', 'rb')
    caption = "SP500 performance {}".format(today)
    chat_id = update.message.chat_id
    context.bot.send_photo(chat_id, photo, caption)

    SP1 = open('/home/ubuntu/Desktop/TelegramBot/predictions/futurepriceSP500.txt', 'r').read()
    SP2 = open('/home/ubuntu/Desktop/TelegramBot/predictions/meanabsoluteerrorSP500.txt', 'r').read()

    if type(SP1) or type(SP2) == int or float:
                #calculating Δ%
        nsp = float(SP1)
        s = pd.Series([si.get_live_price("^GSPC"), nsp])
        s.pct_change()
        normal_sum = s.pct_change()
        normal_sum.at[1]
        dvSP = str(round((normal_sum.at[1] * 100), 2))
        #calculating % error
        nsp2 = float(SP2)
        e = (nsp2 / nsp) * 100
        eSP = '%.2f' % e
        SP = 'Predicted value of the S&P500 in 7 days %s points (Δ%s%%) %s%% error' % (SP1, dvSP, eSP)
        chat_id = update.message.chat_id
        bot.send_message(chat_id, SP)

    else:
        text = 'Yahoo Finance is currently missing data for SP500. Could not run prediction model at this time.'
        chat_id = update.message.chat_id
        bot.send_message(chat_id, text)


'''

def GOLD(update, context):
    photo = open('/home/ubuntu/Desktop/TelegramBot/charts/GOLDforcast.jpeg', 'rb')
    caption = "Gold price prediction chart for {}".format(today)
    chat_id = update.message.chat_id
    context.bot.send_photo(chat_id, photo, caption)

    photo = open('/home/ubuntu/Desktop/TelegramBot/charts/GOLDtrend.jpeg', 'rb')
    caption = "Gold performance {}".format(today)
    chat_id = update.message.chat_id
    context.bot.send_photo(chat_id, photo, caption)

    GOLD1 = open('/home/ubuntu/Desktop/TelegramBot/predictions/futurepriceGOLD.txt', 'r').read()
    GOLD2 = open('/home/ubuntu/Desktop/TelegramBot/predictions/meanabsoluteerrorGOLD.txt', 'r').read()
    text = 'Predicted price of Gold in 7 days $%s mean absolute error %s' % (GOLD1, GOLD2)
    chat_id = update.message.chat_id
    bot.send_message(chat_id, text)

def OIL(update, context):
    photo = open('/home/ubuntu/Desktop/TelegramBot/charts/OILforcast.jpeg', 'rb')
    caption = "Crude oil price prediction chart for {}".format(today)
    chat_id = update.message.chat_id
    context.bot.send_photo(chat_id, photo, caption)

    photo = open('/home/ubuntu/Desktop/TelegramBot/charts/OILtrend.jpeg', 'rb')
    caption = "Crude oil performance {}".format(today)
    chat_id = update.message.chat_id
    context.bot.send_photo(chat_id, photo, caption)

    OIL1 = open('/home/ubuntu/Desktop/TelegramBot/predictions/futurepriceOIL.txt', 'r').read()
    OIL2 = open('/home/ubuntu/Desktop/TelegramBot/predictions/meanabsoluteerrorOIL.txt', 'r').read()
    text = 'Predicted price of crude oil in 7 days $%s mean absolute error %s' % (OIL1, OIL2)
    chat_id = update.message.chat_id
    bot.send_message(chat_id, text)

'''


def info(update: Update, context: CallbackContext) -> None:
    context.bot.sendChatAction(chat_id=update.message.chat_id, action=telegram.ChatAction.TYPING)
    update.message.reply_text(
        'This bot uses deep learning LSTM models to analyze time series data of asset prices. All charts and predictions are updated daily.'
        ' Type /update to get stock price correlations.')

    update.message.reply_text(
        'This bot was created by Alexander Lee and uses Tensorflow to analyze time series data. (Go to https://www.tensorflow.org to read more)'
        )

    update.message.reply_text(
        'What does it mean when the bot sends "percent error" alongside the price prediction? This is the machine learning model\'s average prediction error when it was training on the historical price data. In machine learning this is called "mean absolute error". Read more about it here: https://en.wikipedia.org/wiki/Mean_absolute_error'
        )

    update.message.reply_text(
        'Type /moreinfo for more information'
    )
    
    from datetime import datetime
    user = update.message.from_user
    chat_id = update.message.chat_id
    first_name = update.message.chat.first_name
    last_name = update.message.chat.last_name
    username = update.message.chat.username
    fullname = "{} {}".format(first_name, last_name)
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d %H:%M:%S")
    logfile = [dt_string, chat_id, fullname, username, 'update']
    
    with open('/home/ubuntu/Desktop/telegrambotlog.csv', 'a', newline='') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(logfile)
    
    print("{} Name: {} {} Username: {} Chat ID: {} Function: More Info". format(dt_string, first_name, last_name , username, chat_id))


def moreinfo(update: Update, context: CallbackContext) -> None:
    context.bot.sendChatAction(chat_id=update.message.chat_id, action=telegram.ChatAction.TYPING)
    update.message.reply_text(
        'If you like this bot please consider helping keeping it up and running! My ETH address: 0xC2e647AD0a1dF0EC67dC26EB39f3fD57171e13Fe\n This bot consumes on average 30w per hour @ 0.087 cents per kWh ~ $3 a month.')

    update.message.reply_text('Disclaimer: This bot is provided for informational '
                              ' and entertainment purposes only. Any price prediction data generated by this bot does not constitute investment advice.'
                              ' If you like this bot, please consider sharing it!')


### Send updates every x amount of time

def unknown(bot, update):
    bot.sendMessage(chat_id=update.message.chat_id, text="Sorry, I didn't understand that command.")




def main():
    """Run bot."""
    # Create the Updater and pass it your bot's token.
    # Make sure to set use_context=True to use the new context based callbacks
    # Post version 12 this will no longer be necessary
    updater = Updater("token", use_context=True)

    # Get the dispatcher to register handlers
    dispatcher = updater.dispatcher

    # on different commands - answer in Telegram
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("help", start))
    dispatcher.add_handler(CommandHandler("list", list))
    dispatcher.add_handler(CommandHandler("update", update))
    dispatcher.add_handler(CommandHandler("info", info))
    dispatcher.add_handler(CommandHandler("moreinfo", moreinfo))
    dispatcher.add_handler(CommandHandler("BTC", BTC))
    dispatcher.add_handler(CommandHandler("ETH", ETH))
    dispatcher.add_handler(CommandHandler("XMR", XMR))
    dispatcher.add_handler(CommandHandler("PYPL", PYPL))
    dispatcher.add_handler(CommandHandler("TSLA", TSLA))
    dispatcher.add_handler(CommandHandler("EUR", EUR))
    dispatcher.add_handler(CommandHandler("RUB", RUB))
    dispatcher.add_handler(CommandHandler("SP500", SP500))
    dispatcher.add_handler(CommandHandler("RUS2000", RUS2000))
      

    # dispatcher.add_handler(CommandHandler("GOLD", GOLD))
    # dispatcher.add_handler(CommandHandler("OIL", OIL))

    # Start the Bot
    updater.start_polling()

    # Block until you press Ctrl-C or the process receives SIGINT, SIGTERM or
    # SIGABRT. This should be used most of the time, since start_polling() is
    # non-blocking and will stop the bot gracefully.
    updater.idle()


if __name__ == '__main__':
    main()

