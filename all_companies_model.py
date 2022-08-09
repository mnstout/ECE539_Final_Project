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
from keras.layers import Dense, Dropout
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
    
    # Model Design
    model = Sequential()
    model.add(LSTM(120, return_sequences=True, input_shape= (x.shape[1], x.shape[2])))
    model.add(LSTM(60, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(15))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mae')
    
    history = model.fit(x,y, epochs = 5, batch_size = 2000)
    plt.plot(history.history['loss'], label='train')
    plt.legend()
    plt.show()
        
    # save the model
    model.save('model')
    
    return model

def ModelProcess(training_list, dTrain, future_days, scan_range, sample_shape):
    x = []
    y = []
    
    for name in training_list:
        tmp = dTrain[dTrain["Symbol"] == name].to_numpy()
        for i in range(scan_range, sample_shape-future_days):
            x.append(tmp[i-scan_range:i, 0:5])
            y.append(tmp[i+future_days,3])
        
    x, y = np.array(x), np.array(y)
    x = np.reshape(x, (x.shape[0], x.shape[1], 5))
    
    x = np.asarray(x[:,:,0:5]).astype('float32')
    y = np.asarray(y[:]).astype('float32')
    
    return x,y

def testProcess(data, future, scan_range, companies, sample_shape):
    x = []
    y = []
    
    for name in companies:
        tmp = data[data["Symbol"] == name].to_numpy()
        for i in range(sample_shape-(future*2), sample_shape-future):
            x.append(tmp[i-scan_range:i, 0:5])
            y.append(tmp[i+future,3])
    
    x, y = np.array(x), np.array(y)
    x = np.reshape(x, (x.shape[0], x.shape[1], 5))
    
    x = np.asarray(x[:,:,0:5]).astype('float32')
    y = np.asarray(y[:]).astype('float32')
    
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

# Scale Dataset
scaler = MinMaxScaler(feature_range=(0,1))
dCS = df.copy()
dCS[["Open","High","Low","Close","Adj Close","Volume"]] = scaler.fit_transform(dCS[["Open","High","Low","Close","Adj Close","Volume"]])

# CONTROLLING COMMANDS
future_days = 30            # Number of days in the future to predict <Also distance between last sample and predicted point
scan_range = 90             # Range of samples per predicted point
percent_training = .96      # Percent of companies used in training set <leaves ~50 Companies for testing at .96>
sample_shape = 240          # Size of the datasets we are using

# Split datasets
training_list = names[:round(len(names) * percent_training)]
testing_list = names[round(len(names) * percent_training):]

dTrain = dCS.loc[dCS["Symbol"].isin(training_list)]
dTest = dCS.loc[dCS["Symbol"].isin(testing_list)]

# Check if a model has already been designed, and if so skip these steps
if os.path.exists('model'):
        model = keras.models.load_model('model')
else:
    # Prep Data for model
    x,y = ModelProcess(training_list, dTrain, future_days, scan_range, sample_shape)

    # Get Trained Model
    model = getModel(x,y)

# Choose Company for predicting (This will select 3 companies)
num_companies = 3
companies = random.sample(testing_list,num_companies)
dTest = dTest.loc[dTest["Symbol"].isin(companies)]

# Prep Data for Testing
x2,y2 = testProcess(dTest, future_days, scan_range, companies, sample_shape)

# Predict
pred = model.predict(x2)

# Scale Properly
zero = np.zeros(shape=(30*num_companies,6))
zero[:,3] = pred[:,0]
pred = scaler.inverse_transform(zero)
pred = pred[:,3]

# Plot the 3
fig,ax = plt.subplots(nrows=num_companies, ncols=1)
    
for i in range(3):
    # Data not used by anything
    data_ND = dTest[dTest['Symbol'] == companies[i]].head(sample_shape-future_days-scan_range)
    data_ND[["Open","High","Low","Close","Adj Close","Volume"]] = scaler.inverse_transform(data_ND[["Open","High","Low","Close","Adj Close","Volume"]])
    data_ND.set_index(['date'])
    
    # Data used to scan future days
    data_SD = dTest[dTest['Symbol'] == companies[i]].head(sample_shape-future_days).tail(scan_range)
    data_SD[["Open","High","Low","Close","Adj Close","Volume"]] = scaler.inverse_transform(data_SD[["Open","High","Low","Close","Adj Close","Volume"]])
    data_SD.set_index(['date'])
    
    # Future Data
    data_FD = dTest[dTest['Symbol'] == companies[i]].head(sample_shape).tail(future_days)
    data_FD['Close'] = y2[i*30:(i+1)*30]
    data_FD[["Open","High","Low","Close","Adj Close","Volume"]] = scaler.inverse_transform(data_FD[["Open","High","Low","Close","Adj Close","Volume"]])
    data_FD['Predictions'] = pred[i*30:(i+1)*30]
    data_FD.set_index(['date'])
    
    data_ND.plot(x='date', y='Close', ax=ax[i])
    data_SD.plot(x='date', y='Close', ax=ax[i])
    data_FD.plot(x='date', y='Close', ax=ax[i])
    data_FD.plot(x='date', y='Predictions', ax=ax[i])
    ax[i].set_ylabel('Close Price USD ($)')
    ax[i].set_xlabel('Date')
    ax[i].set_title(f"Model for {companies[i]}")
    ax[i].legend(['History', 'Scanned History', 'Real', 'Predicted'])

plt.show()