import time
import requests
import random
while(1):
    #sitedata=requests.get("https://api.coindesk.com/v1/bpi/currentprice.json")
    sitedata=requests.get("https://www.google.com/")
    #print(sitedata.content)
    numb = random.randrange(90)
    print("Adormece " + str(numb) + " segundos")
    time.sleep(numb)
