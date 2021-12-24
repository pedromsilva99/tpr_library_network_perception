import time
import requests
while(1):
    #sitedata=requests.get("https://api.coindesk.com/v1/bpi/currentprice.json")
    sitedata=requests.get("https://www.google.com/")
    #print(sitedata.content)
    print("Adormece 10 segundos")
    time.sleep(10)
    