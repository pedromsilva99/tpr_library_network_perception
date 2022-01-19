import time
import requests
import random
while(1):
    ls=["https://www.google.com/", "https://en.wikipedia.org/wiki/Main_Page", "https://www.youtube.com/", "https://twitter.com/", "https://www.sporcle.com/",
    "https://vimeo.com/", "https://www.worldometers.info/", "https://stackoverflow.com/", "https://github.com/", "https://www.nytimes.com/"]
    website = random.randrange(10)
    sitedata=requests.get(ls[website])
    print(ls[website])
    numb = random.randrange(120)
    print("Adormece " + str(numb) + " segundos")
    time.sleep(numb)
