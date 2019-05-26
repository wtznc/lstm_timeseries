# LSTM RNN for one-step ahead predictions
### 1. How to use?
Run `python hexathor/src/ 'ticker' 'yyyy-m-d' 'smooth'` or: \
`python hexathor/src/ 'AAPL' 2010-1-1 6` where `smooth = #`of times we apply centered moving average to smooth the series
### 2. Scores
RMSE =  0.005215463712805344\
NMSE =  0.31861399079963115