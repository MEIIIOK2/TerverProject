import scipy.stats
import numpy as np
import pandas as pd


import yfinance as yf
import pandas as pd

revpr = 0.021
noriskrev=0.000172

import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
# load dataset




tickers = ['CCL','AAL','PYPL','BABA','KO','SBRCY','YNDX',
'MPC','JWN','MRNA','LIN','TMUS','GOOG','MSFT','AAPL','BP','T','ZM','FB','TWTR']
forecasts=[]
globTicker = pd.DataFrame()
risk =[]
intervals = []
rdd=0
for tick in tickers:
    tickerData = yf.Ticker(tick)
    tickerDF = tickerData.history(interval='1wk',start='2020-04-01', end='2021-03-19')
    tickerDF = tickerDF.dropna()
    #print(tickerDF['Close'])
    globTicker[tick] = tickerDF['Close'].values
    #series = data[companies[i]]
    rev = []

    X = tickerDF['Close'].values

    #tickerDF['Close'].plot()
    #plt.show()
    X = X.astype('float32')


    for j in range(len(X)-1):
        rev.append(((X[j+1]-X[j])/X[j]))
    am = pd.DataFrame(rev)
    X=am.values
    X=X.astype('float32')
    print(len(X))
    ######################################
    # расчет риска активов
    risk.append(np.std(X))
    ##################

    size = len(X) - 1
    train, test = X[0:size], X[size:]
    # fit an ARIMA model
    model =  ARIMA(train, order=(3, 1, 2))
    model_fit = model.fit()
    # forecast
    result = model_fit.get_forecast()
    # summarize forecast and confidence intervals
    print(tick)
    print('Expected: %.3f' % result.predicted_mean)
    print('Forecast: %.3f' % test[0])
    print('Standard Error: %.3f' % result.se_mean)
    ci = result.conf_int(alpha=0.99)
    print('95%% Interval: %.10f to %.10f' % (ci[0, 0], ci[0, 1]))
    forecasts.append(result.predicted_mean)
    intervals.append([result.predicted_mean,ci[0,0],ci[0,1],risk[rdd]])
    rdd+=1
print(intervals)
araar = pd.DataFrame(intervals)
araar.to_csv('toexcel.csv',index=None,decimal=',',header=None)
print(len(risk))
expected=[]
print(len(tickers))
print(forecasts)
nottopop=[]
newrisk=[]
for i in range(len(forecasts)):
   if forecasts[i][0]>0:
        newrisk.append(risk[i])
        expected.append(forecasts[i][0])
        nottopop.append(tickers[i])
        print(tickers[i])

for tic in tickers:
    if tic not in nottopop:
        globTicker.pop(tic)
risk = newrisk
print(len(risk))
risks = pd.DataFrame(risk)
risks.to_csv('риск.csv',index=False,decimal=',',header=None)
forecasts = pd.DataFrame(expected)
forecasts.to_csv('revenues.csv',index=False,decimal=',',header=None)
print(globTicker)
globTicker.to_csv('Newhistory.csv',index=False,decimal=',')
# split into train and test sets



data = pd.read_csv("Newhistory.csv",decimal=',')

companies = data.columns

for i in range(len(companies)-1):
    prices = data[companies[i]].array

    change = []
    for j in range(len(prices)-1):
        change.append((prices[j+1]-prices[j])/prices[j+1])
    change.append(0)
    data[f'{companies[i]} change'] = change
#######################################################################
#расчет кореляции активов

correlation=[]
for i in range(len(companies)):
    temp=[]
    for j in range(len(companies)):
        temp.append(scipy.stats.pearsonr(data[companies[i]].array,data[companies[j]].array)[0])
    correlation.append(temp)
print(correlation)
dat = pd.DataFrame(correlation)
print(dat)
dat.to_csv('кореляция.csv',index=False,decimal=',',header=False)
#####################################################################
#расчет матрицы ковариации
correlation = pd.read_csv('кореляция.csv',decimal=',',header=None)


correlation=correlation.values
print(risk)
print(correlation)

print(correlation.shape)
covmatrix = []

for i in range(correlation.shape[1]):
    tmp = []
    for j in range(correlation.shape[1]):
        tmp.append(correlation[i][j]*risk[i]*risk[j])
    covmatrix.append(tmp)
print(covmatrix)
covmatrix = np.array(covmatrix)
print(covmatrix)
toexelcovmatrix = pd.DataFrame(covmatrix)
toexelcovmatrix.to_csv('матрица ковариации.csv',index=False,decimal=',',header=None)
##########################################################################################
# расчет обратной матрицы ковариации

matrix = np.linalg.inv(covmatrix)
print(matrix)
toexelcovmatrix = pd.DataFrame(matrix)
toexelcovmatrix.to_csv('обратная матрица ковариации.csv',index=False,decimal=',',header=None)

##################################################################################################
# расчет индекса шарпа
revenue = pd.read_csv("revenues.csv",decimal=',',header=None)
print(f"{revenue} stroka 64")
anticovarmatrix= pd.read_csv("обратная матрица ковариации.csv",decimal=',',header=None)
anticovarmatrix= anticovarmatrix.values
print(anticovarmatrix)
print(anticovarmatrix.shape)
print(revenue.values)

mf = np.ones(revenue.values.shape)*noriskrev
print(mf)
mmf = (revenue.values-mf).T
print(mmf)

gg=np.dot(mmf,anticovarmatrix)

gg = np.dot(gg,mmf.T)[0][0]
print(gg)

###################################################################################################
# расчет вектора долей
x1 = (revpr - noriskrev)/gg
print(x1)
x2 = np.dot(anticovarmatrix,mmf.T)
print(x2)
vector = x1*x2
print('Ответ')
print(vector)
a=0
m=0
doli=[]
for i in range(vector.shape[0]):
    if vector[i][0]>=0:
        m+=1
        a= a+vector[i][0]
        print(f'Актив: {companies[i]} Доля - {vector[i][0]}')
        doli.append(vector[i][0])
print(f'Безрисковый актив - {1-a}')

print(a)
print(m)

