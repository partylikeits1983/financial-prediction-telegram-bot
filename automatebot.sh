#!/bin/bash

cd 
cd /home/ubuntu/Desktop/TelegramBot

python3 correlation30.py
python3 correlation180.py

cd
cd /home/ubuntu/Desktop/TelegramBot/prophet

python3 prophetBTC.py
python3 prophetETH.py
python3 prophetXMR.py
python3 prophetPYPL.py
python3 prophetTSLA.py
python3 prophetRUS.py
python3 prophetEUR.py
python3 prophetRUB.py
python3 prophetSP500.py
#python3 prophetGOLD.py
#python3 prophetOIL.py


cd
rm -r -v /home/ubuntu/Desktop/TelegramBot/tensorflow/tensorflowdata/*
cd /home/ubuntu/Desktop/TelegramBot/tensorflow

python3 pricepredictionBTC.py
python3 pricepredictionETH.py
python3 pricepredictionEUR.py
python3 pricepredictionRUB.py
python3 pricepredictionXMR.py
python3 pricepredictionPYPL.py
python3 pricepredictionTSLA.py
python3 pricepredictionSP500.py
python3 pricepredictionRUS.py



#python3 pricepredictionGOLD.py
#python3 pricepredictionOIL.py
