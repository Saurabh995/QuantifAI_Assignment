############################Code Base#####################################




















##############################Part-I Strategy ####################################

------------------------------------------Core Dependencies----------------------------------------
Python version:- 3.5
IDE:- Jupyter notebook
numpy
pandas 
matplotlib 
pykalman 
statsmodels
seaborn
datetime
quandl
yfinance
psycopg2             (for postgresql connection) 
concurrent            (thread pool can speedup data fetching)

--------------------------------------Other Dependencies--------------------------------------------
postgresql
createdb quantifai

------------------------------------------Hypothesis/Strategy----------------------------------------
I am going to use one of the most popular strategies of pair trading. The spread between a pair 
of stocks (using hedge ratio) can form a staionary time series which can be utilized for a mean 
reverting strategy.

------------------------------------------Stock Selection:---------------------------------------------
Stocks belonging to similar market sector and having similar market cap are likely to co integerate.
Some of the sectors can be pharma, cement, automobile. banking. I was able to identify 5 stocks 
from banking sector traded on NSE, - KOTAKBANK,HDFCBANK,ICICBANK,INDUSINDBK,AXISBANK. 
They have similar market cap , hence likely to co integerate

------------------------------------------Download Data----------------------------------------------
Most of the banks have gone stock splits. Therefore an adjusted closed data was required .  
Quandl wasn't giving adjusted close price on a free account. Yahoo finance offers it , but its 
support for pandas_datareader has deprecated. Although there is a hack by including yfinance module. 
The data is downloaded using it and null data is dropped(although data was pure).

------------------------------------------Find Cointegerated Pairs-----------------------------------
ADF test was ran on all unordered pair combination using statsmodels's coint method. The critical value 
used for hypothesis is 0.05. A p_value matrix is constructed for heatmap visualization showing only 2 pairs cointegeration.

A scatter plot between the identified stocks of the pairs , show a single regression fit cannot be used for hedge ratio,
so we need moving regression (online regression). Kalman filter can be used here

------------------------------------------Datasplit-----------------------------------------------------
Split into test and train data (50-50). Only train data to be used for back testing

------------------------------------------Assumptions------------------------------------------------
i)Sufficient balance to both short and long the position
ii)Cannot have multiple positions at a time
iii)Ability to maintain a portfolio of pairs
---------------------------------------------Strategy Backtesting---------------------------------------

A simple mean reversion strategy of bollinger bands crossover is used. Long the position when 
spread crosses lower band and short the position when spread crosses upper band .
Mean(middle band)= mean over a look back window of 'w' days
Upper band = mean + k*stddev
Lower band =mean -k*stdev
w=look back days
k= zscore 
Back testing logic considered performance of the strategy for each pair measured by its sharpe_ratio.
The (w,z) combination which resulted in best sharpe for all the pairs was chosen

Parameters optimized = w and k -----> w= 20 and k =1.8

------------------------------------------Trade Engine & PnL----------------------------------------------
Strategy was run for the test_data. Some performance visulization is done including equity curve.
ffn library is used for visualizing other metrics such as daily sharpe ratio, drawdown curve.
The performance evaluation is done again for the equiweighted portfolio.

-------------------------------------------Export Data to postgresql---------------------------------------
The dataframes of perfomance metrics is flused into postgresql. It is necessary to have psycopg2 installed
in conda environment to allow sqlalchemy to connect to the postgresql server.  Export the transpose for dataframe
for constructing two tables , one for the pair metrics and other for portfolio metrics

###############################Part-II Backend Interface ############################

-------------------------------------------dependencies----------------------------------------------------

node version - v8.11.3
"dependencies": {
    "body-parser": "^1.19.0",
    "express": "^4.17.1",
    "pg": "^7.11.0"
}

------------------------------------------------API----------------------------------------------------------
pairwise/metrics:-
        returns performance metrics of a pair based on filters and given two stocks 
portfolio/metrics:-
        returns perfomance metrics of portfolio based on filters 

-------------------------------------------------------Sample request and response --------------------------------------------

Request----------
curl -X POST \
http://localhost:3000/pairwise/metrics \
-H 'Content-Type: application/json' \
-H 'Postman-Token: dafc6021-e4dd-48b6-9ecc-fb8e8a5d6087' \
-H 'cache-control: no-cache' \
-d '{
"filters":["total_return","max_drawdown","daily_mean"],
"stock_1":"INDUSINDBK",
"stock_2":"HDFCBANK"
}'


-------------Response----------
[
{
"total_return": 0.468911021357632,
"max_drawdown": -0.0199074277608164,
"daily_mean": 0.132803000404871
}
]

---------------Request------------
curl -X POST \
http://localhost:3000/portfolio/metrics \
-H 'Content-Type: application/json' \
-H 'Postman-Token: cf007dff-01c0-48e3-8a2e-73685697ddc0' \
-H 'cache-control: no-cache' \
-d '{
"filters":["total_return"]
}'

----------------Response-------------
[
{
"total_return": 0.435329463173898
}
]

---------------Request-----------------
curl -X POST \
http://localhost:3000/portfolio/metrics \
-H 'Content-Type: application/json' \
-H 'Postman-Token: 5a932bad-9a75-496a-aa6e-4393f0cf5dc2' \
-H 'cache-control: no-cache' \
-d '{
"filters":["xyzzz"]
}'

----------------Response-----------------
{
"message": "Invalid filters"
}



