import pandas as pd
import random
import numpy as np
from pandas_datareader.data import DataReader
import yfinance as yf
import datetime
import os.path

# Open a list of all symbols
with open("all_symbols.txt") as file:
    names = file.read().split('\n')

# Reduce list of all symbols to 40% the original size
random.seed(69)
reduced = random.sample(names, round(len(names)*.25))
reduced.sort()

# Get First Company
randDate = random.randint(360,480)
start = (datetime.date.today() - datetime.timedelta(randDate))
end = (datetime.date.today() - datetime.timedelta(randDate-360))
df = yf.download(reduced[0], start=start,end=end)
df['Symbol'] = reduced[0]
df['date'] = df.index.tolist()

if os.path.exists('PREdataset.csv'):
    df = pd.read_csv (r'C:\Users\Nate\Desktop\CS539\PREdataset.csv', parse_dates=True)
else:
    for Sym in reduced:
        if Sym == reduced[0]:
            continue
        randDate = random.randint(360,480)
        start = (datetime.date.today() - datetime.timedelta(randDate))
        end = (datetime.date.today() - datetime.timedelta(randDate-360))
        down = yf.download(Sym, start=start,end=end)
        if not down.empty:
            down['Symbol'] = Sym
            down['date'] = down.index.tolist()
            df = df.append(down, ignore_index=True)
        
        else:
            reduced.pop(reduced.index(Sym))

# Save Dataset (After Download, before Fix)
df.to_csv('PREdataset.csv',index=False)

# Remove extra names 240
rElement = []
for name in reduced:
    if name not in df['Symbol'].values:
        rElement.append(name)
    tmp = df[df["Symbol"] == name]
    if (tmp.shape[0] < 240):
        rElement.append(name)
for r in rElement:
    df = df.drop(df[df['Symbol'] == r].index)
    if r in reduced:
        reduced.remove(r)
    
print(len(pd.unique(df['Symbol'])))
print(len(reduced))

# Save Dataset
df.to_csv('dataset.csv',index=False)

# Save Reduced List
with open('used_symbols.txt', 'w') as file:
    for i in reduced:
        file.write(i)
        file.write("\n")
