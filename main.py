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
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update, ReplyKeyboardMarkup
from telegram.ext import Updater, CommandHandler, CallbackQueryHandler, CallbackContext

import logging

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

TOKEN = 'add your telegram token here'

today = date.today()

bot = telegram.Bot(TOKEN)


def start(update: Update, context: CallbackContext) -> None:

    context.bot.sendChatAction(chat_id=update.message.chat_id, action=telegram.ChatAction.TYPING)

    keyboard = [['/update', '/matricies'],
               ['/yieldcurve', '/info'],
               ['/list'],]

    reply_markup = telegram.ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    bot.sendMessage(update.message.chat_id, text='Hello! Welcome to the Financial Forecast and Price Prediction Telegram bot!'
                              ' This bot gives daily price correlations and future price predictions of'
                              ' Bitcoin, Ether, Monero, USD/EUR, USD/RUB, PYPL, TSLA, SP500, and the Russel 2000!', reply_markup=reply_markup)

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
        'Commands: \n/BTC \n/ETH \n/UNI \n/GOLD \n/OIL  \n/SP500 \n/EUR \n/RUB'
        )


def update(update, context):
    context.bot.sendChatAction(chat_id=update.message.chat_id, action=telegram.ChatAction.TYPING)


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


    UNI1 = open('/home/ubuntu/Desktop/TelegramBot/predictions/futurepriceUNI.txt', 'r').read()
    UNI2 = open('/home/ubuntu/Desktop/TelegramBot/predictions/meanabsoluteerrorUNI.txt', 'r').read()

    if type(UNI1) or type(UNI2) == int or float:
        #calculating Δ%
        nuni = float(UNI1)
        s = pd.Series([si.get_live_price("UNI3-USD"), nuni])
        s.pct_change()
        normal_sum = s.pct_change()
        normal_sum.at[1]
        dvUNI = str(round((normal_sum.at[1] * 100), 2))
        #calculating % error
        nuni2 = float(UNI2)
        e = (nuni2 / nuni) * 100
        eUNI = '%.2f' % e
        UNI = 'Predicted price of Uniswap in 1 day $%s   (Δ%s%%)   %s%% error' % (UNI1, dvUNI, eUNI)

    else:
        text = 'Yahoo Finance is currently missing data for Uniswap. Could not run prediction model at this time.'
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
        s = pd.Series([si.get_live_price("ES=F"), nsp])
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

    GOLD1 = open('/home/ubuntu/Desktop/TelegramBot/predictions/futurepriceGOLD.txt', 'r').read()
    GOLD2 = open('/home/ubuntu/Desktop/TelegramBot/predictions/meanabsoluteerrorGOLD.txt', 'r').read()

    if type(GOLD1) or type(GOLD2) == int or float:
        #calculating Δ%
        ngold = float(GOLD1)
        s = pd.Series([si.get_live_price("GC=F"), ngold])
        s.pct_change()
        normal_sum = s.pct_change()
        normal_sum.at[1]
        dvGOLD = str(round((normal_sum.at[1] * 100), 2))
        #calculating % error
        ngold2 = float(GOLD2)
        e = (ngold2 / ngold) * 100
        eGOLD = '%.2f' % e
        GOLD = 'Predicted value of Gold in 7 days %s points (Δ%s%%) %s%% error' % (GOLD1, dvGOLD, eGOLD)

    else:
        text = 'Yahoo Finance is currently missing data for Gold. Could not run prediction model at this time.'
        chat_id = update.message.chat_id
        bot.send_message(chat_id, text)

    OIL1 = open('/home/ubuntu/Desktop/TelegramBot/predictions/futurepriceOIL.txt', 'r').read()
    OIL2 = open('/home/ubuntu/Desktop/TelegramBot/predictions/meanabsoluteerrorOIL.txt', 'r').read()

    if type(OIL1) or type(OIL2) == int or float:
        #calculating Δ%
        noil = float(OIL1)
        s = pd.Series([si.get_live_price("CL=F"), nsp])
        s.pct_change()
        normal_sum = s.pct_change()
        normal_sum.at[1]
        dvOIL = str(round((normal_sum.at[1] * 100), 2))
        #calculating % error
        noil2 = float(OIL2)
        e = (noil2 / noil) * 100
        eOIL = '%.2f' % e
        OIL = 'Predicted value of Crude Oil in 7 days %s points (Δ%s%%) %s%% error' % (OIL1, dvOIL, eOIL)

    else:
        text = 'Yahoo Finance is currently missing data for Crude Oil. Could not run prediction model at this time.'
        chat_id = update.message.chat_id
        bot.send_message(chat_id, text)


    chat_id = update.message.chat_id
    dailyupdate = ("▪️{}\n\n▪️{}\n\n▪️{}\n\n▪️{}\n\n▪️{}\n\n▪️{}\n\n🇪🇺{}\n\n🇷🇺{}").format(BTC,ETH,UNI,SP,GOLD,OIL,EUR,RUB)
    bot.send_message(chat_id, dailyupdate)

    update.message.reply_text(
        'Type /list to view specific assets'
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



def matricies(update, context):
    context.bot.sendChatAction(chat_id=update.message.chat_id, action=telegram.ChatAction.TYPING)
    photo = open('/home/ubuntu/Desktop/TelegramBot/charts/correlationmatrix180.jpeg', 'rb')
    caption = "180 day correlation matrix {}".format(today)
    chat_id = update.message.chat_id
    context.bot.send_photo(chat_id, photo, caption)

    photo = open('/home/ubuntu/Desktop/TelegramBot/charts/correlationmatrix30.jpeg', 'rb')
    caption = "30 day correlation matrix {}".format(today)
    chat_id = update.message.chat_id
    context.bot.send_photo(chat_id, photo, caption)


def yieldcurve(update, context):
    context.bot.sendChatAction(chat_id=update.message.chat_id, action=telegram.ChatAction.TYPING)
    photo = open('/home/ubuntu/Desktop/TelegramBot/charts/yield.jpeg', 'rb')
    caption = "Yield Curve {}".format(today)
    chat_id = update.message.chat_id
    context.bot.send_photo(chat_id, photo, caption)


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

    photo = open('/home/ubuntu/Desktop/TelegramBot/charts/BTCforcastwithlines.jpeg', 'rb')
    caption = "Bitcoin performance {}".format(today)
    chat_id = update.message.chat_id
    context.bot.send_photo(chat_id, photo, caption)

    photo = open('/home/ubuntu/Desktop/TelegramBot/charts/BTCml.jpeg', 'rb')
    caption = "Actual vs Predicted price of ML model {}".format(today)
    chat_id = update.message.chat_id
    context.bot.send_photo(chat_id, photo, caption)

    BTC1 = open('/home/ubuntu/Desktop/TelegramBot/predictions/futurepriceBTC.txt', 'r').read()
    BTC2 = open('/home/ubuntu/Desktop/TelegramBot/predictions/meanabsoluteerrorBTC.txt', 'r').read()
    BTC3 = open('/home/ubuntu/Desktop/TelegramBot/predictions/buyprofitBTC.txt', 'r').read()
    BTC4 = open('/home/ubuntu/Desktop/TelegramBot/predictions/sellprofitBTC.txt', 'r').read()
    BTC5 = open('/home/ubuntu/Desktop/TelegramBot/predictions/totalprofitBTC.txt', 'r').read()
    BTC6 = open('/home/ubuntu/Desktop/TelegramBot/predictions/profitpertradeBTC.txt', 'r').read()


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
        BTC = 'Predicted price of Bitcoin in 1 day $%s   (Δ%s%%)\n Model Error: %s%% \n Total buy profit: %s\n Total sell profit: %s \n Total profit: %s \n Profit per trade: %s \n' % (BTC1, dvBTC, eBTC, BTC3, BTC4, BTC5, BTC6)
        chat_id = update.message.chat_id
        bot.send_message(chat_id, BTC)
    else:
        text = 'Yahoo Finance is missing data for this asset. Could not run prediction model at this time.'
        chat_id = update.message.chat_id
        bot.send_message(chat_id, text)


def ETH(update, context):
    context.bot.sendChatAction(chat_id=update.message.chat_id, action=telegram.ChatAction.TYPING)
    photo = open('/home/ubuntu/Desktop/TelegramBot/charts/ETHforcast.jpeg', 'rb')
    caption = "Ether forcast chart for {}".format(today)
    chat_id = update.message.chat_id
    context.bot.send_photo(chat_id, photo, caption)

    photo = open('/home/ubuntu/Desktop/TelegramBot/charts/ETHtrend.jpeg', 'rb')
    caption = "Ether performance {}".format(today)
    chat_id = update.message.chat_id
    context.bot.send_photo(chat_id, photo, caption)

    photo = open('/home/ubuntu/Desktop/TelegramBot/charts/ETHforcastwithlines.jpeg', 'rb')
    caption = "Ether performance {}".format(today)
    chat_id = update.message.chat_id
    context.bot.send_photo(chat_id, photo, caption)

    photo = open('/home/ubuntu/Desktop/TelegramBot/charts/ETHml.jpeg', 'rb')
    caption = "Actual vs Predicted price of ML model {}".format(today)
    chat_id = update.message.chat_id
    context.bot.send_photo(chat_id, photo, caption)

    ETH1 = open('/home/ubuntu/Desktop/TelegramBot/predictions/futurepriceETH.txt', 'r').read()
    ETH2 = open('/home/ubuntu/Desktop/TelegramBot/predictions/meanabsoluteerrorETH.txt', 'r').read()
    ETH3 = open('/home/ubuntu/Desktop/TelegramBot/predictions/buyprofitETH.txt', 'r').read()
    ETH4 = open('/home/ubuntu/Desktop/TelegramBot/predictions/sellprofitETH.txt', 'r').read()
    ETH5 = open('/home/ubuntu/Desktop/TelegramBot/predictions/totalprofitETH.txt', 'r').read()
    ETH6 = open('/home/ubuntu/Desktop/TelegramBot/predictions/profitpertradeETH.txt', 'r').read()


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
        ETH = 'Predicted price of Ether in 1 day $%s   (Δ%s%%)\n Model Error: %s%% \n Total buy profit: %s\n Total sell profit: %s \n Total profit: %s \n Profit per trade: %s \n' % (ETH1, dvETH, eETH, ETH3, ETH4, ETH5, ETH6)
        chat_id = update.message.chat_id
        bot.send_message(chat_id, ETH)
    else:
        text = 'Yahoo Finance is missing data for Ether. Could not run prediction model at this time.'
        chat_id = update.message.chat_id
        bot.send_message(chat_id, text)

def UNI(update, context):
    context.bot.sendChatAction(chat_id=update.message.chat_id, action=telegram.ChatAction.TYPING)
    photo = open('/home/ubuntu/Desktop/TelegramBot/charts/UNIforcast.jpeg', 'rb')
    caption = "Uniswap forcast chart for {}".format(today)
    chat_id = update.message.chat_id
    context.bot.send_photo(chat_id, photo, caption)

    photo = open('/home/ubuntu/Desktop/TelegramBot/charts/UNItrend.jpeg', 'rb')
    caption = "Uniswap performance {}".format(today)
    chat_id = update.message.chat_id
    context.bot.send_photo(chat_id, photo, caption)

    photo = open('/home/ubuntu/Desktop/TelegramBot/charts/UNIforcastwithlines.jpeg', 'rb')
    caption = "Uniswap performance {}".format(today)
    chat_id = update.message.chat_id
    context.bot.send_photo(chat_id, photo, caption)

    photo = open('/home/ubuntu/Desktop/TelegramBot/charts/UNIml.jpeg', 'rb')
    caption = "Actual vs Predicted price of ML model {}".format(today)
    chat_id = update.message.chat_id
    context.bot.send_photo(chat_id, photo, caption)

    UNI1 = open('/home/ubuntu/Desktop/TelegramBot/predictions/futurepriceUNI.txt', 'r').read()
    UNI2 = open('/home/ubuntu/Desktop/TelegramBot/predictions/meanabsoluteerrorUNI.txt', 'r').read()
    UNI3 = open('/home/ubuntu/Desktop/TelegramBot/predictions/buyprofitUNI.txt', 'r').read()
    UNI4 = open('/home/ubuntu/Desktop/TelegramBot/predictions/sellprofitUNI.txt', 'r').read()
    UNI5 = open('/home/ubuntu/Desktop/TelegramBot/predictions/totalprofitUNI.txt', 'r').read()
    UNI6 = open('/home/ubuntu/Desktop/TelegramBot/predictions/profitpertradeUNI.txt', 'r').read()


    if type(UNI1) or type(UNI2) == int or float:
        #calculating Δ%
        nuni = float(UNI1)
        s = pd.Series([si.get_live_price("UNI3-USD"), nuni])
        s.pct_change()
        normal_sum = s.pct_change()
        normal_sum.at[1]
        dvUNI = str(round((normal_sum.at[1] * 100), 2))
        #calculating % error
        nuni2 = float(UNI2)
        e = (nuni2 / nuni) * 100
        eUNI = '%.2f' % e
        UNI = 'Predicted price of Uniswap in 1 day $%s   (Δ%s%%)\n Model Error: %s%% \n Total buy profit: %s\n Total sell profit: %s \n Total profit: %s \n Profit per trade: %s \n' % (UNI1, dvUNI, eUNI, UNI3, UNI4, UNI5, UNI6)
        chat_id = update.message.chat_id
        bot.send_message(chat_id, UNI)
    else:
        text = 'Yahoo Finance is missing data for Uniswap. Could not run prediction model at this time.'
        chat_id = update.message.chat_id
        bot.send_message(chat_id, text)


def EUR(update, context):
    context.bot.sendChatAction(chat_id=update.message.chat_id, action=telegram.ChatAction.TYPING)
    photo = open('/home/ubuntu/Desktop/TelegramBot/charts/EURforcast.jpeg', 'rb')
    caption = "Euro forcast chart for {}".format(today)
    chat_id = update.message.chat_id
    context.bot.send_photo(chat_id, photo, caption)

    photo = open('/home/ubuntu/Desktop/TelegramBot/charts/EURtrend.jpeg', 'rb')
    caption = "Euro performance {}".format(today)
    chat_id = update.message.chat_id
    context.bot.send_photo(chat_id, photo, caption)

    photo = open('/home/ubuntu/Desktop/TelegramBot/charts/EURforcastwithlines.jpeg', 'rb')
    caption = "Euro performance {}".format(today)
    chat_id = update.message.chat_id
    context.bot.send_photo(chat_id, photo, caption)

    photo = open('/home/ubuntu/Desktop/TelegramBot/charts/EURml.jpeg', 'rb')
    caption = "Actual vs Predicted price of ML model {}".format(today)
    chat_id = update.message.chat_id
    context.bot.send_photo(chat_id, photo, caption)

    EUR1 = open('/home/ubuntu/Desktop/TelegramBot/predictions/futurepriceEUR.txt', 'r').read()
    EUR2 = open('/home/ubuntu/Desktop/TelegramBot/predictions/meanabsoluteerrorEUR.txt', 'r').read()
    EUR3 = open('/home/ubuntu/Desktop/TelegramBot/predictions/buyprofitEUR.txt', 'r').read()
    EUR4 = open('/home/ubuntu/Desktop/TelegramBot/predictions/sellprofitEUR.txt', 'r').read()
    EUR5 = open('/home/ubuntu/Desktop/TelegramBot/predictions/totalprofitEUR.txt', 'r').read()
    EUR6 = open('/home/ubuntu/Desktop/TelegramBot/predictions/profitpertradeEUR.txt', 'r').read()


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
        EUR = 'Predicted price of Euro in 7 days $%s   (Δ%s%%)\n Model Error: %s%% \n Total buy profit: %s\n Total sell profit: %s \n Total profit: %s \n Profit per trade: %s \n' % (EUR1, dvEUR, eEUR, EUR3, EUR4, EUR5, EUR6)
        chat_id = update.message.chat_id
        bot.send_message(chat_id, EUR)
    else:
        text = 'Yahoo Finance is missing data for this asset. Could not run prediction model at this time.'
        chat_id = update.message.chat_id
        bot.send_message(chat_id, text)


def RUB(update, context):
    context.bot.sendChatAction(chat_id=update.message.chat_id, action=telegram.ChatAction.TYPING)
    photo = open('/home/ubuntu/Desktop/TelegramBot/charts/RUBforcast.jpeg', 'rb')
    caption = "Ruble forcast chart for {}".format(today)
    chat_id = update.message.chat_id
    context.bot.send_photo(chat_id, photo, caption)

    photo = open('/home/ubuntu/Desktop/TelegramBot/charts/RUBtrend.jpeg', 'rb')
    caption = "Ruble performance {}".format(today)
    chat_id = update.message.chat_id
    context.bot.send_photo(chat_id, photo, caption)

    photo = open('/home/ubuntu/Desktop/TelegramBot/charts/RUBforcastwithlines.jpeg', 'rb')
    caption = "Ruble performance {}".format(today)
    chat_id = update.message.chat_id
    context.bot.send_photo(chat_id, photo, caption)

    photo = open('/home/ubuntu/Desktop/TelegramBot/charts/RUBml.jpeg', 'rb')
    caption = "Actual vs Predicted price of ML model {}".format(today)
    chat_id = update.message.chat_id
    context.bot.send_photo(chat_id, photo, caption)

    RUB1 = open('/home/ubuntu/Desktop/TelegramBot/predictions/futurepriceRUB.txt', 'r').read()
    RUB2 = open('/home/ubuntu/Desktop/TelegramBot/predictions/meanabsoluteerrorRUB.txt', 'r').read()
    RUB3 = open('/home/ubuntu/Desktop/TelegramBot/predictions/buyprofitRUB.txt', 'r').read()
    RUB4 = open('/home/ubuntu/Desktop/TelegramBot/predictions/sellprofitRUB.txt', 'r').read()
    RUB5 = open('/home/ubuntu/Desktop/TelegramBot/predictions/totalprofitRUB.txt', 'r').read()
    RUB6 = open('/home/ubuntu/Desktop/TelegramBot/predictions/profitpertradeRUB.txt', 'r').read()


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
        RUB = 'Predicted price of Ruble in 7 days $%s   (Δ%s%%)\n Model Error: %s%% \n Total buy profit: %s\n Total sell profit: %s \n Total profit: %s \n Profit per trade: %s \n' % (RUB1, dvRUB, eRUB, RUB3, RUB4, RUB5, RUB6)
        chat_id = update.message.chat_id
        bot.send_message(chat_id, RUB)
    else:
        text = 'Yahoo Finance is missing data for this currency pair. Could not run prediction model at this time.'
        chat_id = update.message.chat_id
        bot.send_message(chat_id, text)



def SP500(update, context):
    context.bot.sendChatAction(chat_id=update.message.chat_id, action=telegram.ChatAction.TYPING)
    photo = open('/home/ubuntu/Desktop/TelegramBot/charts/SP500forcast.jpeg', 'rb')
    caption = "SP500 forcast chart for {}".format(today)
    chat_id = update.message.chat_id
    context.bot.send_photo(chat_id, photo, caption)

    photo = open('/home/ubuntu/Desktop/TelegramBot/charts/SP500trend.jpeg', 'rb')
    caption = "SP500 performance {}".format(today)
    chat_id = update.message.chat_id
    context.bot.send_photo(chat_id, photo, caption)

    photo = open('/home/ubuntu/Desktop/TelegramBot/charts/SP500forcastwithlines.jpeg', 'rb')
    caption = "SP500 performance {}".format(today)
    chat_id = update.message.chat_id
    context.bot.send_photo(chat_id, photo, caption)

    photo = open('/home/ubuntu/Desktop/TelegramBot/charts/SP500ml.jpeg', 'rb')
    caption = "Actual vs Predicted price of ML model {}".format(today)
    chat_id = update.message.chat_id
    context.bot.send_photo(chat_id, photo, caption)

    SP5001 = open('/home/ubuntu/Desktop/TelegramBot/predictions/futurepriceSP500.txt', 'r').read()
    SP5002 = open('/home/ubuntu/Desktop/TelegramBot/predictions/meanabsoluteerrorSP500.txt', 'r').read()
    SP5003 = open('/home/ubuntu/Desktop/TelegramBot/predictions/buyprofitSP500.txt', 'r').read()
    SP5004 = open('/home/ubuntu/Desktop/TelegramBot/predictions/sellprofitSP500.txt', 'r').read()
    SP5005 = open('/home/ubuntu/Desktop/TelegramBot/predictions/totalprofitSP500.txt', 'r').read()
    SP5006 = open('/home/ubuntu/Desktop/TelegramBot/predictions/profitpertradeSP500.txt', 'r').read()


    if type(SP5001) or type(SP5002) == int or float:
        #calculating Δ%
        nsp500 = float(SP5001)
        s = pd.Series([si.get_live_price("ES=F"), nsp500])
        s.pct_change()
        normal_sum = s.pct_change()
        normal_sum.at[1]
        dvSP500 = str(round((normal_sum.at[1] * 100), 2))
        #calculating % error
        nsp5002 = float(SP5002)
        e = (nsp5002 / nsp500) * 100
        eSP500 = '%.2f' % e
        SP500 = 'Predicted price of SP500 in 7 days $%s   (Δ%s%%)\n Model Error: %s%% \n Total buy profit: %s\n Total sell profit: %s \n Total profit: %s \n Profit per trade: %s \n' % (SP5001, dvSP500, eSP500, SP5003, SP5004, SP5005, SP5006)
        chat_id = update.message.chat_id
        bot.send_message(chat_id, SP500)
    else:
        text = 'Yahoo Finance is missing data for this asset. Could not run prediction model at this time.'
        chat_id = update.message.chat_id
        bot.send_message(chat_id, text)


def GOLD(update, context):
    context.bot.sendChatAction(chat_id=update.message.chat_id, action=telegram.ChatAction.TYPING)
    photo = open('/home/ubuntu/Desktop/TelegramBot/charts/GOLDforcast.jpeg', 'rb')
    caption = "Gold forcast chart for {}".format(today)
    chat_id = update.message.chat_id
    context.bot.send_photo(chat_id, photo, caption)

    photo = open('/home/ubuntu/Desktop/TelegramBot/charts/GOLDtrend.jpeg', 'rb')
    caption = "Gold performance {}".format(today)
    chat_id = update.message.chat_id
    context.bot.send_photo(chat_id, photo, caption)

    photo = open('/home/ubuntu/Desktop/TelegramBot/charts/GOLDforcastwithlines.jpeg', 'rb')
    caption = "Gold performance {}".format(today)
    chat_id = update.message.chat_id
    context.bot.send_photo(chat_id, photo, caption)

    photo = open('/home/ubuntu/Desktop/TelegramBot/charts/GOLDml.jpeg', 'rb')
    caption = "Actual vs Predicted price of ML model {}".format(today)
    chat_id = update.message.chat_id
    context.bot.send_photo(chat_id, photo, caption)

    GOLD1 = open('/home/ubuntu/Desktop/TelegramBot/predictions/futurepriceGOLD.txt', 'r').read()
    GOLD2 = open('/home/ubuntu/Desktop/TelegramBot/predictions/meanabsoluteerrorGOLD.txt', 'r').read()
    GOLD3 = open('/home/ubuntu/Desktop/TelegramBot/predictions/buyprofitGOLD.txt', 'r').read()
    GOLD4 = open('/home/ubuntu/Desktop/TelegramBot/predictions/sellprofitGOLD.txt', 'r').read()
    GOLD5 = open('/home/ubuntu/Desktop/TelegramBot/predictions/totalprofitGOLD.txt', 'r').read()
    GOLD6 = open('/home/ubuntu/Desktop/TelegramBot/predictions/profitpertradeGOLD.txt', 'r').read()


    if type(GOLD1) or type(GOLD2) == int or float:
        #calculating Δ%
        ngold = float(GOLD1)
        s = pd.Series([si.get_live_price("GC=F"), ngold])
        s.pct_change()
        normal_sum = s.pct_change()
        normal_sum.at[1]
        dvGOLD = str(round((normal_sum.at[1] * 100), 2))
        #calculating % error
        ngold2 = float(GOLD2)
        e = (ngold2 / ngold) * 100
        eGOLD = '%.2f' % e
        GOLD = 'Predicted price of Gold in 7 days $%s   (Δ%s%%)\n Model Error: %s%% \n Total buy profit: %s\n Total sell profit: %s \n Total profit: %s \n Profit per trade: %s \n' % (GOLD1, dvGOLD, eGOLD, GOLD3, GOLD4, GOLD5, GOLD6)
        chat_id = update.message.chat_id
        bot.send_message(chat_id, GOLD)
    else:
        text = 'Yahoo Finance is missing data for this asset. Could not run prediction model at this time.'
        chat_id = update.message.chat_id
        bot.send_message(chat_id, text)



def OIL(update, context):
    context.bot.sendChatAction(chat_id=update.message.chat_id, action=telegram.ChatAction.TYPING)
    photo = open('/home/ubuntu/Desktop/TelegramBot/charts/OILforcast.jpeg', 'rb')
    caption = "Oil forcast chart for {}".format(today)
    chat_id = update.message.chat_id
    context.bot.send_photo(chat_id, photo, caption)

    photo = open('/home/ubuntu/Desktop/TelegramBot/charts/OILtrend.jpeg', 'rb')
    caption = "Oil performance {}".format(today)
    chat_id = update.message.chat_id
    context.bot.send_photo(chat_id, photo, caption)

    photo = open('/home/ubuntu/Desktop/TelegramBot/charts/OILforcastwithlines.jpeg', 'rb')
    caption = "Oil performance {}".format(today)
    chat_id = update.message.chat_id
    context.bot.send_photo(chat_id, photo, caption)

    photo = open('/home/ubuntu/Desktop/TelegramBot/charts/OILml.jpeg', 'rb')
    caption = "Actual vs Predicted price of ML model {}".format(today)
    chat_id = update.message.chat_id
    context.bot.send_photo(chat_id, photo, caption)

    OIL1 = open('/home/ubuntu/Desktop/TelegramBot/predictions/futurepriceOIL.txt', 'r').read()
    OIL2 = open('/home/ubuntu/Desktop/TelegramBot/predictions/meanabsoluteerrorOIL.txt', 'r').read()
    OIL3 = open('/home/ubuntu/Desktop/TelegramBot/predictions/buyprofitOIL.txt', 'r').read()
    OIL4 = open('/home/ubuntu/Desktop/TelegramBot/predictions/sellprofitOIL.txt', 'r').read()
    OIL5 = open('/home/ubuntu/Desktop/TelegramBot/predictions/totalprofitOIL.txt', 'r').read()
    OIL6 = open('/home/ubuntu/Desktop/TelegramBot/predictions/profitpertradeOIL.txt', 'r').read()


    if type(OIL1) or type(OIL2) == int or float:
        #calculating Δ%
        noil = float(OIL1)
        s = pd.Series([si.get_live_price("CL=F"), noil])
        s.pct_change()
        normal_sum = s.pct_change()
        normal_sum.at[1]
        dvOIL = str(round((normal_sum.at[1] * 100), 2))
        #calculating % error
        noil2 = float(OIL2)
        e = (noil2 / noil) * 100
        eOIL = '%.2f' % e
        OIL = 'Predicted price of Oil in 7 days $%s   (Δ%s%%)\n Model Error: %s%% \n Total buy profit: %s\n Total sell profit: %s \n Total profit: %s \n Profit per trade: %s \n' % (OIL1, dvOIL, eOIL, OIL3, OIL4, OIL5, OIL6)
        chat_id = update.message.chat_id
        bot.send_message(chat_id, OIL)
    else:
        text = 'Yahoo Finance is missing data for this asset. Could not run prediction model at this time.'
        chat_id = update.message.chat_id
        bot.send_message(chat_id, text)



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


def button(update: Update, _: CallbackContext) -> None:
    query = update.callback_query

    # CallbackQueries need to be answered, even if no notification to the user is needed
    # Some clients may have trouble otherwise. See https://core.telegram.org/bots/api#callbackquery
    query.answer()

    query.edit_message_text(text=f"Selected option: {query.data}")


def main():
    """Run bot."""
    # Create the Updater and pass it your bot's token.
    # Make sure to set use_context=True to use the new context based callbacks
    # Post version 12 this will no longer be necessary
    updater = Updater(TOKEN, use_context=True)

    # Get the dispatcher to register handlers
    dispatcher = updater.dispatcher

    # on different commands - answer in Telegram
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("help", start))
    dispatcher.add_handler(CommandHandler("list", list))
    dispatcher.add_handler(CommandHandler("update", update))
    dispatcher.add_handler(CommandHandler("matricies", matricies))
    dispatcher.add_handler(CommandHandler("yieldcurve", yieldcurve))
    dispatcher.add_handler(CommandHandler("info", info))
    dispatcher.add_handler(CommandHandler("moreinfo", moreinfo))
    dispatcher.add_handler(CommandHandler("BTC", BTC))
    dispatcher.add_handler(CommandHandler("ETH", ETH))
    dispatcher.add_handler(CommandHandler("UNI", UNI))
    dispatcher.add_handler(CommandHandler("EUR", EUR))
    dispatcher.add_handler(CommandHandler("RUB", RUB))
    dispatcher.add_handler(CommandHandler("GOLD", GOLD))
    dispatcher.add_handler(CommandHandler("OIL", OIL))
    dispatcher.add_handler(CommandHandler("SP500", SP500))

    # Start the Bot
    updater.start_polling()

    # Block until you press Ctrl-C or the process receives SIGINT, SIGTERM or
    # SIGABRT. This should be used most of the time, since start_polling() is
    # non-blocking and will stop the bot gracefully.
    updater.idle()


if __name__ == '__main__':
    main()
