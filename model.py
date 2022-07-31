from ctypes.wintypes import RGB
from turtle import width
from matplotlib.colors import rgb2hex
from matplotlib.markers import MarkerStyle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d 
import random

# Plot function for value in table
# name is the code for any company in the table
# info is one of the column names to plot
def plot(df, name):
    company = df[df["Symbol"] == name]
    company.set_index(['date'])

    fig,ax = plt.subplots(nrows=3, ncols=2)
    
    # Volume
    company.plot(x='date',y='Volume', ax=ax[0,0])
    ax[0,0].ticklabel_format(style='plain',axis='y')
    ax[0,0].set_axisbelow(True)
    ax[0,0].grid(linestyle='-', linewidth='0.1', color='black')
    ax[0,0].set_ylabel("Volume, $USD")
    ax[0,0].set_title(f"Volume of {name}")

    # Open
    company.plot(x='date',y='Open', ax=ax[0,1])
    ax[0,1].ticklabel_format(style='plain',axis='y')
    ax[0,1].set_axisbelow(True)
    ax[0,1].grid(linestyle='-', linewidth='0.1', color='black')
    ax[0,1].set_ylabel("Open Value, $USD")
    ax[0,1].set_title(f"Opening Value of {name}")

    # High
    company.plot(x='date',y='High', ax=ax[1,0])
    ax[1,0].ticklabel_format(style='plain',axis='y')
    ax[1,0].set_axisbelow(True)
    ax[1,0].grid(linestyle='-', linewidth='0.1', color='black')
    ax[1,0].set_ylabel("High, $USD")
    ax[1,0].set_title(f"Highest Value of {name}")

    # Low
    company.plot(x='date',y='Low', ax=ax[1,1])
    ax[1,1].ticklabel_format(style='plain',axis='y')
    ax[1,1].set_axisbelow(True)
    ax[1,1].grid(linestyle='-', linewidth='0.1', color='black')
    ax[1,1].set_ylabel("Low, $USD")
    ax[1,1].set_title(f"Lowest Value of {name}")

    # Close
    company.plot(x='date',y='Close', ax=ax[2,0])
    ax[2,0].ticklabel_format(style='plain',axis='y')
    ax[2,0].set_axisbelow(True)
    ax[2,0].grid(linestyle='-', linewidth='0.1', color='black')
    ax[2,0].set_ylabel("Close, $USD")
    ax[2,0].set_title(f"Closing Value of {name}")

    # adjclose
    company.plot(x='date',y='Adj Close', ax=ax[2,1])
    ax[2,1].ticklabel_format(style='plain',axis='y')
    ax[2,1].set_axisbelow(True)
    ax[2,1].grid(linestyle='-', linewidth='0.1', color='black')
    ax[2,1].set_ylabel("Adjusted Close, $USD")
    ax[2,1].set_title(f"Adjusted Closing Value of {name}")

    plt.show()

# Import Data
data = pd.read_csv (r'C:\Users\Nate\Desktop\CS539\dataset.csv', parse_dates=True)
df = pd.DataFrame(data, columns= ['Open','High','Low','Close','Adj Close','Volume','Symbol','date'])
df.date = pd.to_datetime(df['date'], format='%Y-%m-%d')

# Open a list of symbols
with open("used_symbols.txt") as file:
    names = file.read().split('\n')

# Plot
plot(df, names[random.randint(0,len(names))])
