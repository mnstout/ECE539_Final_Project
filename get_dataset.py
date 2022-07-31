import pandas as pd
import random
import numpy as np
from pandas_datareader.data import DataReader
import yfinance as yf
import datetime

# Open a list of all symbols
with open("all_symbols.txt") as file:
    names = file.read().split('\n')

# Reduce list of all symbols to 40% the original size
random.seed(69)
reduced = random.sample(names, round(len(names)*.4))
reduced.sort()

# Get First Company
randDate = random.randint(120,360)
start = (datetime.date.today() - datetime.timedelta(randDate))
end = (datetime.date.today() - datetime.timedelta(randDate-90))
df = yf.download(reduced[0], start=start,end=end)
df['Symbol'] = reduced[0]
df['date'] = df.index.tolist()


for Sym in reduced:
    if Sym == reduced[0]:
        continue
    randDate = random.randint(120,360)
    start = (datetime.date.today() - datetime.timedelta(randDate))
    end = (datetime.date.today() - datetime.timedelta(randDate-90))
    down = yf.download(Sym, start=start,end=end)
    if not down.empty:
        down['Symbol'] = Sym
        down['date'] = down.index.tolist()
        df = df.append(down, ignore_index=True)
        
    else:
        reduced.pop(reduced.index(Sym))
    

# Save Dataset
df.to_csv('dataset.csv',index=False)

# Save Reduced List
with open('used_symbols.txt', 'w') as file:
    for i in reduced:
        file.write(i)
        file.write("\n")