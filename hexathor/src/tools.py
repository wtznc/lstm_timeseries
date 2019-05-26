'''
Contains methods used for dealing and preprocessing time series data
'''
import fix_yahoo_finance as yf
import matplotlib.pyplot as plt
import datetime
import numpy as np
import random
import statsmodels.api as sm
from scipy import signal
class Tools:
	def get_prices(from_date, ticker):
		y, m, d = from_date
		start = datetime.datetime(y, m, d)
		end = datetime.datetime.now()
		tic = yf.Ticker(ticker)
		index = yf.download(ticker, start=start, end=end)
		name = tic.info['shortName']
		prices = index['Close']
		prices = prices.dropna()
		return prices, name

	def split_window(data, window):
		X = []
		y = []
		for i in range(len(data) - window):
			X.append(data[i:i+window])
			y.append(data[window+i])
		return np.asarray(X), np.asarray(y)

	def normalize(data):
		ymin = data.min()
		ymax = data.max()
		normalized = 2.0 * ((data - ymin) / (ymax - ymin) - 0.5)
		return normalized

	def moving_average(data, periods=3):
		weights = np.ones(periods) / periods
		return np.convolve(data, weights, mode='valid')

	def generate_mackey_glass(sample_length=1500, b=0.1, c=0.2, tau=17, init=None):
		if init != None:
			chaotic = init
		else:
			chaotic = []
			for i in range(0, 20):
				chaotic.append(random.uniform(0, 1))

		for t in range(len(chaotic) - 1, sample_length+49):
			chaotic.append(chaotic[t] + c*chaotic[t-tau] / (1+chaotic[t-tau] ** 10) - b*chaotic[t])
		chaotic = chaotic[50: sample_length+50]
		return np.asarray(chaotic)

	def get_sunspots():
		sunspots = sm.datasets.get_sunspots
		sunspots_pandas = sunspots.load_pandas()
		sunactivity = sunspots_pandas['SUNACTIVITY']
		return np.asarray(sunactivity)

	# based on "the future of time series" paper, about santa fe competition that took place in 1992
	def NMSE(true, predicted):
		nominator = []
		for x in range(0, len(real)):
			error = (real[x] - predicted[x]) ** 2
			nominator.append(error)
		denominator = []
		for x in range(1, len(real)):
			ertrue = (real[x] - real[x-1]) ** 2
			denominator.append(ertrue)
		nmse = np.sum(nominator) / np.sum(denominator)
		return nmse

	