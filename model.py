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

def prepInput(Xs):
    xf = np.zeros(shape=(Xs.shape[0],60,120,Xs.shape[2]))
    row = -1
    for i in xf:
        row += 1
        set = -1
        for j in i:
            set += 1
            j[:,:] = Xs[row,set:set+120,:]
                    
    xf = xf.reshape(xf.shape[0],xf.shape[1],-1)
    xf = np.asarray(xf).astype('float32')
    return xf
    

# Return model and scalers
def getModel(x,y):
    
    if os.path.exists('model'):
        model = keras.models.load_model('model')
        
    else:
        Xs = x[:,:,0:6]
        Ys = y[:,:,3]
        
        xf = prepInput(Xs)
        
        print(xf.shape)
        print(Ys.shape)      
        
        Ys = np.asarray(Ys).astype('float32')
    
        model = Sequential()
        model.add(LSTM(160, return_sequences=True, input_shape=(xf.shape[1],xf.shape[2])))
        model.add(LSTM(80, return_sequences=False))
        model.add(Dense(40))
        model.add(Dense(1))
        model.compile(loss='mae', optimizer='adam')
    
        history = model.fit(xf,Ys, epochs = 100, batch_size = 15)
        plt.plot(history.history['loss'], label='train')
        plt.legend()
        plt.show()
        
        # save the model
        model.save('model')
    
    return model

def prepPred(x):
    xf = np.zeros(shape=(60,60,120,6))
    
    row = -1
    for a in xf:
        row += 1
        for b in list(range(60)):
            a[b,:,:] = x[row:row+120,:]
    
    xf = xf.reshape(60,60,720)
    xf = np.asarray(xf).astype('float32')
    return xf

# Import Data
data = pd.read_csv (r'C:\Users\Nate\Desktop\CS539\dataset.csv', parse_dates=True)
df = pd.DataFrame(data, columns= ['Open','High','Low','Close','Adj Close','Volume','Symbol','date'])
df.date = pd.to_datetime(df['date'], format='%Y-%m-%d')

# Open a list of symbols
with open("used_symbols.txt") as file:
    names = file.read().split('\n')

## Plot Random Data
# plot(df, names[random.randint(0,len(names))])

## Data prep for model
# Open,High,Low,Close,Adj Close,Volume,Symbol,date
x = []
y = []

# Scale
scaler = MinMaxScaler(feature_range=(0,1))
df_orig = df.copy()
df[["Open","High","Low","Close","Adj Close","Volume"]] = scaler.fit_transform(df[["Open","High","Low","Close","Adj Close","Volume"]])

for i in names:
    tmp = df[df["Symbol"] == i]
    if (tmp.shape[0] >= 240):
        x.append(tmp[0:180])
        y.append(tmp[180:240])

x = np.reshape(np.array(x),(1301,180,8))
y = np.reshape(np.array(y),(1301,60,8))

# Split train/test set, not randomly selected
xTrain = x[0:round((x.shape[0] * .75))]
xTest = x[xTrain.shape[0]::]
yTrain = y[0:round((y.shape[0] * .75))]
yTest = y[yTrain.shape[0]::]

# Get Trained Model
model = getModel(xTrain,yTrain)


xPred = prepPred(xTest[4,:,0:6])
yPred = model.predict(xPred)

print(xTest.shape)
print(xPred.shape)
print(yPred.shape)

yTest = yTest[:,:,3]
plt.figure(figsize=(16,6))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(yTest[4])
plt.plot(yPred)
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()
