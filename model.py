from ctypes.wintypes import RGB
from turtle import width
from matplotlib.colors import rgb2hex
from matplotlib.markers import MarkerStyle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d 

# Plot function for value in table
# name is the code for any company in the table
# info is one of the column names to plot
def plot(df, name, info):
    company = df[df["symbol"] == name]
    company.set_index(['date'])

    #plt.plot(company["date"], company[info])
    fig,ax = plt.subplots()
    company.plot(x='date',y=info, ax=ax)
    ax.set_axisbelow(True)
    ax.grid(linestyle='-', linewidth='0.1', color='black')
    plt.ylabel(info)
    plt.title(f"{info} for {name}")
    plt.show()

# Import Data
data = pd.read_csv (r'C:\Users\Nate\Desktop\CS539\reduced_data.csv', parse_dates=True)
df = pd.DataFrame(data, columns= ['date','volume','open','high','low','close','adjclose','symbol'])
df.date = pd.to_datetime(df['date'], format='%Y-%m-%d')

# Open a list of symbols
with open("reduced_symbols.txt") as file:
    names = file.read().split('\n')

# Plot
plot(df, names[69], "close")

# Closing plots for all listed companies
fig2,ax2 = plt.subplots()
for name in names:
    company = df[df["symbol"] == name]
    company = company[['date','close']]
    company['close'] = company['close']
    company.plot(x='date',y='close', ax=ax2, legend = False, ylim=(0,1000), color=np.random.rand(3,))
ax2.set_axisbelow(True)
ax2.grid(linestyle='-', linewidth='0.1', color='black')
plt.ylabel('Closing Amounts')
plt.title('Closing amounts for all companies in the dataset')
plt.show()

