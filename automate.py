import os
import time as t
from datetime import datetime
import time as t
import schedule

time1= '02:30'
time2= '10:30'

def automate():
   os.system("gnome-terminal -e 'bash -c \"./automatebot.sh; exec bash\"'")
    #os.system("./automatebot.sh")
    
schedule.every().day.at(time1).do(automate)
schedule.every().day.at(time2).do(automate)

while True:
    schedule.run_pending()
    t.sleep(1)
    print(f'Waiting until {time1} or {time2} to update models.')
