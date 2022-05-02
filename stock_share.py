# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Team DreamTree (22.4.12 ~ ing).
#
# ### JPX Tokyo Stock Exchange Prediction
# - https://www.kaggle.com/competitions/jpx-tokyo-stock-exchange-prediction
# - April 4, 2022 - Start Date
# - June 28, 2022 - Entry deadline. You must accept the competition rules before this date in order to compete.
# - June 28, 2022 - Team Merger deadline. This is the last day participants may join or merge teams.
# - July 5, 2022 - Final submission deadline.
#
#

# ## References (시계열 데이터 분석 레퍼런스)
#
# ### Technical Analysis Library in Python
# - Technical Analysis library useful to do feature engineering from financial time series datasets 
# - https://github.com/bukosabino/ta
#
# ### Tidy Viewer (CSV viewer in linux system)
# - https://github.com/alexhallam/tv
#
# ~~~
# wget https://github.com/alexhallam/tv/releases/download/1.4.3/tidy-viewer_1.4.3_amd64.deb
# sudo dpkg -i tidy-viewer_1.4.3_amd64.deb
# # echo "alias tv='tidy-viewer'" >> ~/.bashrc
# source ~/.bashrc
# ~~~
#
# ~~~ 
# tv file.csv 
# ~~~
#
# ### Time Series Prediction Tutorial with EDA (Kaggle Notebook)
# - https://www.kaggle.com/code/kanncaa1/time-series-prediction-tutorial-with-eda/notebook
#
# ### Stock Prices Predictions-EDA,LSTM(DeepExploration)
# - https://www.kaggle.com/code/saurabhshahane/stock-prices-predictions-eda-lstm-deepexploration/notebook

# # EDA
#
# ### Data Cleaning
# - trn.dropna(subset=['Country'], inplace=True)
# ~~~
# drop_list = ['Mission ID','Unit ID']
# aerial.drop(columns=drop_list, inplace=True)
# ~~~

# # !pip uninstall -y plotly
# # !conda uninstall -y plotly
# # !pip uninstall -y chart-studio
# # !conda uninstall -y chart-studio
# # !pip install plotly==3.10.0
# # !source env/bin/activate
# # !pip install --upgrade requests
# # cd /home/smob/anaconda3/envs/tf2/bin/pytho3.8/site-packages 
# pip uninstall urllib3
# # !pip install charset-normalizer
# # !pip install jupytext --upgrade


import charset_normalizer

import seaborn as sns # visualization library
import matplotlib.pyplot as plt # visualization library
import pandas as pd
import numpy as np
import urllib3

# +
# https://stackoverflow.com/questions/57240949/importerror-the-plotly-future-module-must-be-imported-before-the-plotly-modul
# ImportError: The _plotly_future_ module must be imported before the plotly module
from _plotly_future_ import v4_subplots

import plotly.plotly as py # visualization library
import plotly.graph_objs as go # plotly graphical object
#import plotly.graph_objects as go
from plotly.graph_objs import Line
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode, iplot # plotly offline mode


init_notebook_mode(connected=True) 
# -

# ls

# ### (Sample) Stock Prices Predictions-EDA,LSTM(DeepExploration)

df = pd.read_pickle('./SIEMENS-15minute-Hist')
df = pd.DataFrame(df)
df['date'] = df['date'].apply(pd.to_datetime)
df.set_index('date',inplace=True)

fig = go.Figure(data=[go.Table(
    header=dict(values=list(['date','open','high','low','close','volume']),
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[df.index,df.open, df.high, df.low, df.close,df.volume],
               fill_color='lavender',
               align='left'))
])

# +
#fig.show()

# +
fig_ex = make_subplots(rows=4, cols=1,subplot_titles=('Open','High','Low','Close'))

fig_ex.add_trace(
    Line(x=df.index, y=df.open),
    row=1, col=1
)

fig_ex.add_trace(
    Line(x=df.index, y=df.high),
    row=2, col=1
)

fig_ex.add_trace(
    Line(x=df.index, y=df.low),
    row=3, col=1
)

fig_ex.add_trace(
    go.Line(x=df.index, y=df.close),
    row=4, col=1
)

fig_ex.layout.update(height=1400, width=1000, title_text="OHLC Line Plots")

#fig_ex.show()

# +
trn_ex = pd.read_csv('./example_test_files/stock_prices.csv')
#opt_ex = pd.read_csv('./example_test_files/options.csv')

trn = pd.read_csv('./train_files/stock_prices.csv')
#opt = pd.read_csv('./train_files/options.csv')
# -

trn_ex.head()

trn.head()

trn.shape, 1202*2000

# +
# unique 한 날짜, Securities Code 갯수 
date_list,code_list = trn['Date'].unique(),trn['SecuritiesCode'].unique()
print(len(date_list),len(code_list))

date_list,code_list = sorted(date_list), sorted(code_list)
# -

trn.info()

# Date to datetime, and set as index
trn['Date'] = trn['Date'].apply(pd.to_datetime)
trn.set_index('Date',inplace=True)

trn.head()

type(trn.index[0])
day_week = ['월','화','수','목','금','토','일']
day_week[trn.index[0].weekday()]

fig = go.Figure(data=[go.Table(
    header=dict(values=list(['Date','SecuritiesCode','Open','High','Low','Close','Volume']),
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[trn.index,trn.SecuritiesCode, trn.Open, trn.High, trn.Low, trn.Close,trn.Volume],
               fill_color='lavender',
               align='left'))
])

# +
#fig.show()
# -

trn_1301 = trn[trn['SecuritiesCode'] == 1301]
trn_1332 = trn[trn['SecuritiesCode'] == 1332]
trn_1333 = trn[trn['SecuritiesCode'] == 1333]

# +
fig = make_subplots(rows=3, cols=1,subplot_titles=('1301','1332','1333'))

fig.add_trace(
    Line(x=trn_1301.index, y=trn_1301.Close),
    row=1, col=1
)

fig.add_trace(
    Line(x=trn_1332.index, y=trn_1332.Close),
    row=2, col=1
)

fig.add_trace(
    Line(x=trn_1333.index, y=trn_1333.Close),
    row=3, col=1
)


#fig.update_layout(height=1400, width=1000, title_text=" Line Plots")
fig.layout.update(height=1400, width=1000, title_text=" Line Plots")

fig.show()
# -

for d in trn_1301.index:
    print(day_week[d.weekday()],d)
    if (day_week[d.weekday()]=='금'):
        print('-'*7)

# 모든 변수 미리보기
for col in trn.columns:
    print('{}\n'.format(trn[col].head()))

# ## Candle Stick 

# +
from datetime import datetime
data_ = trn_1301
fig = go.Figure(data=[go.Candlestick(x=data_.index,
                open=data_['Open'],
                high=data_['High'],
                low=data_['Low'],
                close=data_['Close'])])

fig.show()
# -

# # Long Short Term Memory Networks(LSTM)

# ## create Train / test data

# +
from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense


from sklearn.preprocessing import MinMaxScaler

# +

new_df = pd.DataFrame()
new_df = df['close']
new_df.index = df.index

scaler=MinMaxScaler(feature_range=(0,1))
final_dataset=new_df.values

train_data=final_dataset[0:20000,]
valid_data=final_dataset[20000:,]

train_df = pd.DataFrame()
valid_df = pd.DataFrame()
train_df['Close'] = train_data
train_df.index = new_df[0:20000].index
valid_df['Close'] = valid_data
valid_df.index = new_df[20000:].index

scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(final_dataset.reshape(-1,1))

x_train_data,y_train_data=[],[]

for i in range(60,len(train_data)):
    x_train_data.append(scaled_data[i-60:i,0])
    y_train_data.append(scaled_data[i,0])
    
x_train_data,y_train_data=np.array(x_train_data),np.array(y_train_data)

x_train_data=np.reshape(x_train_data,(x_train_data.shape[0],x_train_data.shape[1],1))


# -

# # 수치형 변수에 대한 데이터 특성 
# num_cols = [col for col in trn.columns[:] if trn[col].dtype in ['int64', 'float64']]
# trn[num_cols].describe()

### Visualization
~~~
aerial['Country'].value_counts()
# country
print(aerial['Country'].value_counts())
plt.figure(figsize=(22,10))
sns.countplot(aerial['Country'])
plt.show()
~~~


# +
plt.figure(figsize=(22,10))
plt.plot(weather_bin.Date,weather_bin.MeanTemp)
plt.title("Mean Temperature of Bindukuri Area")
plt.xlabel("Date")
plt.ylabel("Mean Temperature")
plt.show()

# Mean temperature of Bindikuri area
plt.figure(figsize=(22,10))
plt.plot(weather_bin.Date,weather_bin.MeanTemp)
plt.title("Mean Temperature of Bindukuri Area")
plt.xlabel("Date")
plt.ylabel("Mean Temperature")
plt.show()

# lets create time series from weather 
timeSeries = weather_bin.loc[:, ["Date","MeanTemp"]]
timeSeries.index = timeSeries.Date
ts = timeSeries.drop("Date",axis=1)

# +
df['time'] = df['time'].astype(float) # data type 이 현재 str 이므로 int로 변환
    
for column in df.columns:
    if column == 'time':
        continue
    #if column in ['Oil_tSwmp', 'Oil_pSwmp', 'CEngDsT_t','FuelT_t']:
    print(column,type(column))
    df[column] = df[column].astype(float) # data type 이 현재 str 이므로 int로 변환
    y_col_name = column
    x_col_name = 'time'
    y_col = df.loc[:,[y_col_name]]
    x_col = df.loc[:,[x_col_name]]
    plt.figure(figsize=(20,8))
    plt.plot(x_col, y_col)
    plt.xlabel(x_col_name)
    plt.ylabel(y_col_name)
    plt.savefig('png_original/'+str(column)+'.png')
    plt.show()

# -

# # Evaluation
# - https://www.kaggle.com/code/smeitoma/jpx-competition-metric-definition

# +
# import jpx_tokyo_market_prediction
# env = jpx_tokyo_market_prediction.make_env()   # initialize the environment
# iter_test = env.iter_test()    # an iterator which loops over the test files
# for (prices, options, financials, trades, secondary_prices, sample_prediction) in iter_test:
#     sample_prediction_df['Rank'] = np.arange(len(sample_prediction))  # make your predictions here
#     env.predict(sample_prediction_df)   # register your predictions

# +
import numpy as np
import pandas as pd

def calc_spread_return_sharpe(df: pd.DataFrame, portfolio_size: int = 200, toprank_weight_ratio: float = 2) -> float:
    """
    Args:
        df (pd.DataFrame): predicted results
        portfolio_size (int): # of equities to buy/sell
        toprank_weight_ratio (float): the relative weight of the most highly ranked stock compared to the least.
    Returns:
        (float): sharpe ratio
    """
    def _calc_spread_return_per_day(df, portfolio_size, toprank_weight_ratio):
        """
        Args:
            df (pd.DataFrame): predicted results
            portfolio_size (int): # of equities to buy/sell
            toprank_weight_ratio (float): the relative weight of the most highly ranked stock compared to the least.
        Returns:
            (float): spread return
        """
        assert df['Rank'].min() == 0
        assert df['Rank'].max() == len(df['Rank']) - 1
        weights = np.linspace(start=toprank_weight_ratio, stop=1, num=portfolio_size)
        purchase = (df.sort_values(by='Rank')['Target'][:portfolio_size] * weights).sum() / weights.mean()
        short = (df.sort_values(by='Rank', ascending=False)['Target'][:portfolio_size] * weights).sum() / weights.mean()
        return purchase - short

    buf = df.groupby('Date').apply(_calc_spread_return_per_day, portfolio_size, toprank_weight_ratio)
    sharpe_ratio = buf.mean() / buf.std()
    return sharpe_ratio
# -


