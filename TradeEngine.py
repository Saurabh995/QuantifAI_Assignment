#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
print(sys.version)


# In[2]:



import numpy as np
import pandas as pd
import matplotlib as mpl
from pykalman import KalmanFilter
from datetime import datetime
from numpy import log, polyfit, sqrt, std, subtract
import statsmodels.tsa.stattools as ts
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


from concurrent import futures


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


pd.core.common.is_list_like = pd.api.types.is_list_like
import pandas_datareader.data as web


# In[6]:


import quandl
quandl.ApiConfig.api_key='mPe9uNzhPMYcN5yoSpsV'


# In[7]:


#these stocks have similar market cap this would be my universe of stocks
stock_banking= ['KOTAKBANK','ICICIBANK','AXISBANK','HDFCBANK','INDUSINDBK']
start_date='2013-06-30'
end_date = '2019-06-30'


# In[8]:


#need to check whether quandl gives adjusted close data or not
data_test = quandl.get('NSE/KOTAKBANK', trim_start = "2013-06-30", trim_end = "2019-06-30", authtoken='mPe9uNzhPMYcN5yoSpsV')


# In[9]:


data_test['Close'].plot(figsize=(15,7),grid=True)


# In[10]:


#as there is a stock split , I need adjusted close price.Need to download from some other APIs
#using yfinance as hack for downloading from yahoo finance through datareader
import yfinance as yf
yf.pdr_override()


# In[11]:


data_test = web.get_data_yahoo("KOTAKBANK.NS", start=start_date, end=end_date)


# In[12]:


data_test.head()


# In[13]:


data_test['Adj Close'].plot(figsize=(15,7),grid=True)


# In[14]:


df_bank =pd.DataFrame() #initializing a dataframe to store the universe
error_ticker=[]


# In[15]:


def prepare_data(stock):
    try:
        print (stock+'.NS')
        df_bank[stock] = web.get_data_yahoo(stock+'.NS', start=start_date, end=end_date)['Adj Close']
        
    except BaseException as e:
        print (e.__doc__)
        print('failed to download %s' %(stock))
        error_ticker.append(stock)


# In[16]:


##########TO DO############### 
##add concurrency to download data  ########


# In[17]:


for x in stock_banking:
    prepare_data(x)


# In[18]:


df_bank.plot(figsize=(15,7),grid=True)


# In[19]:


#checking for null data
null_data = df_bank[df_bank.isnull().any(axis=1)]
null_data


# In[20]:


df_bank=df_bank.dropna()
df_bank.info()


# In[21]:


#finding the co integerated pairs by running adf test on the pairs

def find_cointegerated_pairs(dataframe, critical_level , keys):
    n=dataframe.shape[1]
    #pvalue matrix is a n*n matrix having all pvalues from adf test
    pvalue_matrix = np.ones((n,n));
    pairs=[]
    for i in range(n):
        for j in range (i+1,n):
            stock1= dataframe[keys[i]]
            stock2= dataframe[keys[j]]
            result = ts.coint(stock1,stock2)
            pvalue=result[1]
            pvalue_matrix[i][j]=pvalue
            if pvalue <= critical_level:
                pairs.append((keys[i],keys[j],pvalue))
    return pvalue_matrix, pairs


# In[22]:


#50-50 split for train and test
split =int(len(df_bank)*0.5)
train_bank_df = df_bank.iloc[:split]
test_bank_df= df_bank.iloc[split:]
pvalue_matrix,pairs = find_cointegerated_pairs(train_bank_df,critical_level=0.05, keys=stock_banking)


# In[23]:


pairs
#only two pairs are co integerated


# In[24]:


#plotting heat map to show p values of the pairs to visualize the p values
pvalue_matrix_df= pd.DataFrame(pvalue_matrix)
fig, ax = plt.subplots(figsize=(15,10))
sns.heatmap(pvalue_matrix_df,xticklabels=stock_banking,yticklabels=stock_banking,ax=ax)


# In[25]:


#evenly spaced 10 dates
(df_bank[::len(df_bank)//9].index)


# In[26]:


# Plot a pair on scatterplot just to get an idea of a regression fit
def scatterPlot(x,y):
    plt.figure()
    cm=plt.get_cmap('jet')
    colors = np.linspace(0.1,1,len(df_bank))# whole data is required for visualization not just train 
    sc=plt.scatter(x=df_bank[x],y=df_bank[y],s=30,c=colors,cmap=cm,edgecolors='k',alpha=0.7)
    cb=plt.colorbar(sc)
    cb.ax.set_yticklabels([str(p.date()) for p in df_bank[::len(df_bank)//9].index])
    plt.xlabel(x)
    plt.ylabel(y)
    
for x in pairs:
    scatterPlot(x[0],x[1])


# In[27]:


#for one of the pairs a single regression fit won't do can use kalman filter for an online regression fit
def kalmanFilterAverage(x):
    #create a filter
    kf = KalmanFilter(transition_matrices=[1], 
                      observation_matrices= [1], 
                      initial_state_mean= 0, 
                      initial_state_covariance=1,
                      observation_covariance=1,
                      transition_covariance=0.1)
    #calculate rolling mean
    
    state_means,_=kf.filter(x.values)
    state_means = pd.Series(state_means.flatten(),index=x.index)
    return state_means

# Kalman filter regression
def kalmanFilterRegression(x,y):
    delta = 1e-3
    trans_cov = delta / (1 - delta) * np.eye(2) # How much random walk wiggles
    obs_mat = np.expand_dims(np.vstack([[x], [np.ones(len(x))]]).T, axis=1)

    kf = KalmanFilter(n_dim_obs=1, n_dim_state=2, # y is 1-dimensional, (alpha, beta) is 2-dimensional
    initial_state_mean=[0,0],
    initial_state_covariance=np.ones((2, 2)),
    transition_matrices=np.eye(2),
    observation_matrices=obs_mat,
    observation_covariance=2,
    transition_covariance=trans_cov)

    # Use the observations y to get running estimates and errors for the state parameters
    state_means, state_covs = kf.filter(y.values)
    return state_means


# In[28]:


##########paramters to backtest = entry and exit zscore, window of rolling mean#####################
def backTest(s1,s2,df,entry_zscore=2,exit_zscore=0,window =20):
    #############################################################
    # INPUT:
    # df : dataframe containing prices
    # s1: the symbol of stock one
    # s2: the symbol of stock two
    # entry_zscore: the lower and upper bollinger bands
    # exit_zscore:  when spread crosses this exit or cover
    # window: custom window for calculating z score
    # OUTPUT:
    # df1['cum rets']: cumulative returns in pandas data frame
    # sharpe: Sharpe ratio
    # CAGR: Compound Annual Growth Rate
    
    x = df[s1]
    y = df[s2]

    # create a dataframe out of two stocks
    df1 = pd.DataFrame({'y':y,'x':x})
    df1.index = pd.to_datetime(df1.index)

    # run regression (including Kalman Filter) to find hedge ratio and then create spread series
    state_means = kalmanFilterRegression(kalmanFilterAverage(x),kalmanFilterAverage(y))
    
    # state means returns an array of array we need to convert into a series and also revert sign
    df1['hr'] = - state_means[:,0] 
    
    # Calculating the spread
    df1['spread'] = df1.y + (df1.x * df1.hr)

    # calculate z-score with window
    meanSpread = df1['spread'].rolling(window=window).mean()
    stdSpread = df1['spread'].rolling(window=window).std()
    df1['zscore'] = (df1['spread']-meanSpread)/stdSpread

############################################Trading Logic ###########################################################

    # For longposition  we use -Zscore 
    df1['long_entry'] = ((df1['zscore'] < - entry_zscore) & ( df1['zscore'].shift(1) > - entry_zscore))
    df1['long_exit'] = ((df1['zscore'] > - exit_zscore) & (df1['zscore'].shift(1) < - exit_zscore)) 


    # No of units long 
    df1['num_units_long'] = np.nan
    df1.loc[df1['long_entry'],'num_units_long'] = 1 # num_units_long is 1 on that position
    df1.loc[df1['long_exit'],'num_units_long'] = 0 

    #entry position must be 0(exit)
    df1['num_units_long'][0] = 0 

    # propagate last position forward
    df1['num_units_long'] = df1['num_units_long'].fillna(method='pad')  

    # For shortposition  we use +zscore 
    df1['short_entry'] = ((df1.zscore > entry_zscore) & ( df1.zscore.shift(1) < entry_zscore))
    df1['short_exit'] = ((df1.zscore < exit_zscore) &(df1.zscore.shift(1) > exit_zscore))

    # No of units short
    df1['num_units_short']=np.nan
    df1.loc[df1['short_entry'],'num_units_short'] = -1
    df1.loc[df1['short_exit'],'num_units_short'] = 0

    
    #entry position must be 0(exit)
    df1['num_units_short'][0] = 0
    
    # propagate last valid observation forward
    df1['num_units_short'] = df1['num_units_short'].fillna(method='pad')
    

    #total no of units at a time, it cannot be more than 1
    df1['num_units'] = df1['num_units_long'] + df1['num_units_short']
    

    
###################################################PnL###########################################3
    
    # caluclating spread %age change and using it to calculate cumsum=(spread(t)-Spread(t-1)/(Y+mx)) need to have (y+mx in the account at any time)   
    df1['spread_pct_ch'] = (df1['spread']-df1['spread'].shift(1))/(df1['y'].shift(1)+(df1['x'].shift(1)*abs(df1['hr'].shift(1))))
    df1['port_rets'] = df1['spread_pct_ch'] * df1['num_units'].shift(1)
    df1['cum_rets'] = df1['port_rets'].cumsum()
    df1['cum_rets'] = df1['cum_rets'] + 1

    # calucalte sharp ratio annualized
    try:
        sharpe = ((df1['port_rets'].mean() / df1['port_rets'].std()) * sqrt(252))
    except ZeroDivisionError:
        sharpe = 0.0
    
    #storing cum_rets to be used for graph plotting for returns
    df1[s1+ " "+s2] = df1['cum_rets']
    
    return df1[[s1+ " "+s2,'num_units_short','num_units_long']] ,sharpe
               


# In[29]:


returns,sharpe_ratio = backTest(s1=pairs[0][0],s2=pairs[0][1],df=train_bank_df,entry_zscore=2,
                                         exit_zscore=0,window=20)
print(sharpe_ratio)


# In[30]:


#Back testing will be done on annualized sharpe_ratio, for window size =20,30and entry_zscore=1.8, 2 on a equiweight
def backTest_Engine(window_sizes, entry_zscores):
    for w in window_sizes:
        for z in entry_zscores:
            print('BackTesting for windowsize: {} and entryzscore: {}\n'.format(w,z))
            for pair in pairs:
                returns,sharpe_ratio=backTest(s1=pair[0],s2=pair[1],df=train_bank_df,entry_zscore=z,
                                         exit_zscore=0,window=w)
                print('sharpe_ratio for {} and {} is: {}  '.format(pair[0],pair[1],sharpe_ratio))

                


# In[31]:


backTest_Engine([20,30],[1.8,2])


# In[32]:


#sharpe ratio comes out to be best for window size =20 and entryzscore = 1.8
window_size = 20
entry_zscore = 1.8


# In[33]:


########################Trade Engine#####################
#Will use ffn library to export certain useful stats to be used by backend servers
import ffn


# In[34]:


# For the backend interface following stats will be shown
# Pairwise stats for each pair in the portfolio and portfolio stats
# Portfolio time series data showing positions,and returns
# Flush the dataframes into the tables of postgresql
pairwise_stats_df = pd.DataFrame(); 
portfolio_stats_df = pd.DataFrame();
portfolio_series_df = pd.DataFrame();


# In[35]:


#TO DO add thread pool executor 
def trade_engine(test_df, test_pairs, window_size, entry_zscore):
    # test_df : dataframe consisting of instruments close price
    # test_pairs : pairs to be tested upon
    # plots and returns the returns dataframe having pair wise results
    returns_df =pd.DataFrame()
    print('Going to run the trade engine with window_size: {} and entry_zscore: {}'.format(window_size,entry_zscore))
    for pair in test_pairs:
        returns,sharpe_ratio = backTest(s1=pair[0],s2=pair[1],df=test_df,entry_zscore=2,
                                         exit_zscore=0,window=20)
        returns_df[pair[0]+ " "+pair[1]]=returns[pair[0]+ " "+pair[1]]
        print("The pair {} and {} produced a Sharpe Ratio of {} ".format(pair[0],pair[1],round(sharpe_ratio,2)))
        returns[pair[0]+ " "+pair[1]].plot(figsize=(20,15),legend=True)
    return returns_df
    


# In[36]:


returns_df=trade_engine(test_bank_df,pairs,window_size, entry_zscore)


# In[37]:


returns_df.head()


# In[38]:


###################Using ffn library to visualize some performance metrics####################
pairwise_stats = returns_df.calc_stats() 


# In[39]:


#equity curve
pairwise_stats.plot()


# In[40]:


#some more metrics
pairwise_stats.display()


# In[41]:


#draw down curve
for pair in pairs:
    plt.figure()
    ffn.to_drawdown_series(returns_df[pair[0]+" "+pair[1]]).plot(figsize=(15,7),grid=True,legend=True)


# In[42]:


pairwise_stats_df=pairwise_stats.stats


# In[43]:


#stores the df of stats
pairwise_stats_df.head()


# In[44]:


def portfolio_return_analysis(df):
    # df: returnsdf of pairs
    # Plots the equity curve
    
    df=df/(len(df.columns))
    portfolio_return = pd.DataFrame()
    # Add up the result to plot the equity curve
    portfolio_return["portfolio"] = df.sum(axis=1)
    
    #ignore 0 value as it will result it sharpe as inf
    portfolio_return = portfolio_return.loc[portfolio_return["portfolio"] !=0]
    
    return (portfolio_return)


# In[45]:


portfolio_return = portfolio_return_analysis(returns_df)


# In[46]:


portfolio_return.head()


# In[47]:


portfolio_stats = portfolio_return.calc_stats()


# In[48]:


############equity curve portfolio###############
portfolio_stats.plot()


# In[49]:


#######Some metrics of the portfolio##########
portfolio_stats.display()


# In[50]:


portfolio_stats_df = portfolio_stats.stats


# In[51]:


portfolio_stats_df.head()


# In[53]:


############################Exporting Engine#####################
# Exports metrics into postgresql db tables
# Metrics can be queried from the backend interface
from sqlalchemy import create_engine


# In[54]:


username = 'saurabh'
password = 'saurabh'
dbname = 'quantifai'
engine = create_engine('postgresql://{}:{}@localhost/{}'.format(username,password,dbname))


# In[55]:


pairwise_stats_df.head()


# In[56]:


#writing 
pairwise_stats_df.T.to_sql(name='pairwise_metrics', con=engine, if_exists = 'replace', index=True)
portfolio_stats_df.T.to_sql(name='portfolio_metrics', con=engine, if_exists = 'replace', index=True)


# In[60]:


data = pd.read_sql('SELECT * FROM pairwise_metrics', engine)


# In[61]:


data.head()

