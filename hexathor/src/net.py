from scipy import signal
from pylab import rcParams
import pylab
from pdb import set_trace as bp
from matplotlib.pyplot import figure
import argparse
import datetime
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters
import random
import statsmodels as sm
from tools import Tools as tl
register_matplotlib_converters()


def args():
	parser = argparse.ArgumentParser(description='LSTM RNN for Time Series Forecasting')
	parser.add_argument('ticker', type=str, help='Provide a ticker name from Yahoo!Finance')
	parser.add_argument('date', type=lambda s: datetime.datetime.strptime(s, '%Y-%m-%d').date(), help='Fetch data from: yyyy,m,d')
	parser.add_argument('smooth', type=int, help='Number of times for centered moving average')
	args = parser.parse_args()
	return args




import numpy
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
# train_size - range (0.0 - 1.0)

def create_dataset(dataset, look_back=5):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i+look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)


def lstm(train_size, past_values, epochs, units, data, batchsize, name):
	dataset = data.values
	dataset = dataset.astype('float32')
	scaler = MinMaxScaler(feature_range=(0, 1))
	dataset = scaler.fit_transform(dataset)


	train_size = int(len(dataset) * train_size)
	test_size = len(dataset) - train_size

	train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
	print("Using",len(train),"samples for training and" , len(test) , "for testing.")
	
	#trainX, trainY = tl.split_window(train, past_values)
	#testX, testY = tl.split_window(test, past_values)
	trainX, trainY = create_dataset(train, past_values)
	testX, testY = create_dataset(test, past_values)




	trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
	testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

	# defining model
	model = Sequential()
	model.add(LSTM(units, input_shape=(1, past_values)))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	model.fit(trainX, trainY, epochs, batchsize, verbose=3)

	trainPredict = model.predict(trainX)
	testPredict = model.predict(testX)

	model.summary()

	trainPredict = scaler.inverse_transform(trainPredict)
	trainY = scaler.inverse_transform([trainY])
	testPredict = scaler.inverse_transform(testPredict)
	testY = scaler.inverse_transform([testY])


	rmse = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
	nmse = tl.NMSE(testY[0], testPredict[:, 0])
	print("RMSE = ", rmse)
	print("NMSE = ", nmse)
	# shift train predictions for plotting
	trainPredictPlot = np.empty_like(dataset)
	trainPredictPlot[:, :] = np.nan
	trainPredictPlot[past_values:len(trainPredict)+past_values, :] = trainPredict

	# shift test predictions for plotting
	testPredictPlot = np.empty_like(dataset)
	testPredictPlot[:, :] = np.nan
	testPredictPlot[len(trainPredict)+(past_values*2)+1  :len(dataset)-1:, :] = testPredict
	figure(num=None, figsize=(16, 9), dpi=60, facecolor='w', edgecolor='k')

	# plot baseline and predictions
	plt.title("LSTM: " + name)
	plt.plot(scaler.inverse_transform(dataset), label='true', lw=0.8, color='green', ls='dashdot')
	plt.plot(trainPredictPlot, label='training', lw=0.3, color='blue')
	plt.plot(testPredictPlot, label='testing', lw=0.3, color='red')
	plt.legend()
	plt.show()

def main(args):
	numpy.random.seed(1)
	ticker = args.ticker
	date = args.date
	smooth = args.smooth
	prices, name = tl.get_prices(date, ticker)
	# prices to dataframe;
	pnorm = tl.normalize(prices)
	ptrend = signal.detrend(pnorm)
	for _ in range(smooth):
		ptrend = tl.moving_average(ptrend)
	prices = pd.DataFrame(ptrend)


	#plt.title(name + " from: " + str(date))
	#plt.plot(prices)
	#plt.show()

	#lstm(train_size, past_values, epochs, units, data, batchsize, name):
	lstm(0.67, 50, 50, 20, prices, 20, name)
if __name__ == '__main__':
	main()