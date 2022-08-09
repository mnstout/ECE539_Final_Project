from ctypes.wintypes import RGB
from sqlite3 import Row
from turtle import width
from matplotlib.colors import rgb2hex
from matplotlib.markers import MarkerStyle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d 
import random
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import joblib
import os.path
import tensorflow_text as text

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
    

# Return model and scalers
def getModel(x,y):
    
    if os.path.exists('RENAME TO MODEL TO JUST REUSE'):
        model = keras.models.load_model('model')
        
    else:
        # Model Design
        model = Sequential()
        model.add(LSTM(120, return_sequences=True, input_shape= (x.shape[1], x.shape[2])))
        model.add(LSTM(60, return_sequences=False))
        model.add(Dense(30))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
    
        history = model.fit(x,y, epochs = 50, batch_size = 20)
        plt.plot(history.history['loss'], label='train')
        plt.legend()
        plt.show()
        
        # save the model
        model.save('model')
    
    return model

def ModelProcess(data, pred_days):
    x = []
    y = []
    
    print(data.shape[0])
    
    for i in range(pred_days, data.shape[0]):
        x.append(data[i-pred_days:i, :])
        y.append(data[i,:])
        
    x, y = np.array(x), np.array(y)
    x = np.reshape(x, (x.shape[0], x.shape[1], 6))
    
    return x,y

def testProcess(data, dataS, pred_days):
    testShift = data.shape[0] - (pred_days)
    x = []
    y = data.to_numpy()[data.shape[0]-pred_days:, :]
    for i in range(0, pred_days):
        x.append(dataS[testShift+i-pred_days:i+testShift, :])
    x = np.array(x)
    x = np.reshape(x, (x.shape[0], x.shape[1], 6))
    
    return x,y

# Import Data
data = pd.read_csv (r'C:\Users\Nate\Desktop\CS539\dataset.csv', parse_dates=True)
df = pd.DataFrame(data, columns= ['Open','High','Low','Close','Adj Close','Volume','Symbol','date'])
df.date = pd.to_datetime(df['date'], format='%Y-%m-%d')

# Open a list of symbols
with open("used_symbols.txt") as file:
    names = file.read().split('\n')

## Plot Random Data
#plot(df, names[random.randint(0,len(names))])

## Data prep for model
# Open,High,Low,Close,Adj Close,Volume,Symbol,date

###################- MODEL -########################

# Choose a dataset to use for this run
# ***** EDIT THE names[x] BELOW TO CHANGE COMPANY ***** # 
Company = names[10]
dC = df[df["Symbol"] == Company]
dCD = dC.copy().drop(["Symbol","date"], axis =1)

# Scale Dataset
scaler = MinMaxScaler(feature_range=(0,1))
dCS = scaler.fit_transform(dCD)

# Number of days to predict, train (All entries have 240-260 entries)
pred_days = 30
training_len = dCS.shape[0] - 30
testing_len = 30
dTr = dCS[0:training_len]

# Prep Data for model
x,y = ModelProcess(dTr, pred_days)

# Get Trained Model
model = getModel(x,y)

# Prep Data for Testing
x2,y2 = testProcess(dCD, dCS, pred_days)

# Predict
pred = model.predict(x2)

# Scale Properly
zero = np.zeros(shape=(30,6))
zero[:,3] = pred[:,0]
pred = scaler.inverse_transform(zero)


# Prepare to Plot the data
train = dC[:training_len]
train.set_index(['date'])
valid = dC[training_len:]
valid['Predictions'] = pred[:,3]

#fig,ax = plt.plot()
#ax.set_xlabel('Date', fontsize=18)
#ax.set_ylabel('Close Price USD ($)', fontsize=18)
#ax.set_title('Model')
#train.plot(x='date',y='Close', ax=ax)
#valid.plot(x='date',y=['Close','Predictions'])
#ax.legend(['Train', 'Val', 'Predictions'], loc='lower right')
#plt.show()

# Visualize the data
plt.figure(figsize=(16,6))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['date'],train['Close'])
plt.plot(valid['date'],valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()
