import numpy as np
import pandas as pd
import random

# Open the full dataset (Contains all companies)
data = pd.read_csv (r'C:\Users\Nate\Desktop\CS539\fh_5yrs.csv')
df = pd.DataFrame(data, columns= ['date','volume','open','high','low','close','adjclose','symbol'])

# Open a list of all symbols
with open("all_symbols.txt") as file:
    names = file.read().split('\n')

# Reduce list of all symbols to 5% the original size
random.seed(69)
reduced = random.sample(names, round(len(names)*.05))
reduced.sort()

with open('reduced_symbols.txt', 'w') as file:
    for i in reduced:
        file.write(i)
        file.write("\n")

reduced_data = df[df["symbol"] == reduced[0]]
for i in reduced:
    if (i == reduced[0]):
        continue
    reduced_data = reduced_data.append(df[df["symbol"] == i], ignore_index=True)

reduced_data.to_csv('reduced_data.csv',index=False)