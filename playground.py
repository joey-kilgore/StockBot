# import stocker
# #from stocker.predict import tomorrow


# # error2 = tomorrow(stock, years=2)[1]
# # error3 = tomorrow(stock, years=3)[1]
# # print('Error by using 1 year of data:',error1,'%')
# # print('Error by using 2 years of data:',error2,'%')
# # print('Error by using 3 years of data:',error3,'%')

# # error1 = tomorrow(stock, steps=1)[1]
# # print("seeing what the error array looks like")
# # print(error1)
# # error2 = tomorrow(stock, steps=10)[1]
# # error3 = tomorrow(stock, steps=20)[1]
# # print('Error by using 1 previous day of data:',error1,'%')
# # print('Error by using 10 previous days of data:',error2,'%')
# # print('Error by using 20 previous days of data:',error3,'%')

# from keras.models import Sequential
# from keras.layers import Dense, LSTM, Dropout
# from sklearn.preprocessing import MinMaxScaler
# import numpy as np
# from stocker.error import get


# def data(df, features=[]):
#     columns = ['Close']
#     if len(features) > 0:
#         for i in range(len(features)):
#             columns.append(features[i])

#     df = df[columns]

#     return df


# def get_lstm_input(data, steps=1):
#     samples = []
#     for i in range(steps, data.shape[0]):
#         features = []
#         for j in range(steps):
#             features.append(data[i - steps + j, :])
#         features.append(data[i, :])
#         samples.append(features)

#     features = []
#     for j in range(steps + 1):
#         features.append(data[-1, :])

#     samples.append(features)
#     samples = np.asarray(samples)
#     return samples


# def run(df, features=[], steps=1, training=0.9, error_method='mape'):

#     new_df = data(df, features)

#     scaler = MinMaxScaler(feature_range=(0, 1))
#     scaled = scaler.fit_transform(new_df)
#     reframed = get_lstm_input(scaled, steps)

#     rows = round(len(df) * training)

#     train = reframed[:rows, :, :]
#     test = reframed[rows:, :, :]

#     train_x, train_y = train[:, :steps, :], train[:, steps, 0]
#     test_x, test_y = test[:, :steps, :], test[:-1, steps, 0]

    


#     # designing and fitting network
#     model = Sequential()
#     model.add(LSTM(50, input_shape=(train_x.shape[1], train_x.shape[2])))
#     model.add(Dropout(0.2))
#     model.add(Dense(1))
#     model.compile(loss='mae', optimizer='adam')
#     model.fit(train_x, train_y, epochs=100, batch_size=70, verbose=0)

#     mod1 = rows + steps - 1
#     mod2 = rows + steps

#     print("test_x")
#     print(test_x)
#     print(test_y)

#     # generate a prediction
#     prediction = model.predict(test_x)
#     new_scaled = np.copy(scaled)
#     for x in range(mod1, new_scaled.shape[0]):
#         new_scaled[x, 0] = prediction[x-mod1]

#     # invert normalized values
#     # for predictions
#     y_predicted = scaler.inverse_transform(new_scaled)
#     y_predicted = y_predicted[mod1:, 0]
#     # for real values
#     y = scaler.inverse_transform(scaled)
#     y = y[mod2:, 0]

#     finalprice = round(y_predicted[-1], 2)
#     y_predicted = y_predicted[:-1]

#     error = get(y, y_predicted, error_method)

#     result = [finalprice, error]

#     return result, y_predicted, new_df[-len(y):]


# from stocker.get_data import total
# import datetime as dt


# def tomorrow(stock, features=None, steps=1, training=0.9, period=14, years=1, error_method='mape', plot=False):
#     """
#     Function to predict the "close price" for the next day.
#     Arguments:
#         stock (str): stock label
#         features (list): ['Interest', 'Wiki_views', 'RSI', '%K', '%R']
#         steps (int): previous days to consider for generating the model.
#         training (float): fraction assigned for training the model
#         period (int): number of days considered for calculating indicators.
#         years (int or float): years of data to be considered
#         error_method (str): 'mape' or 'mse'
#         plot (bool): generate performance plot
#     Returns:
#         Result for the next business day. [price, error, date]
#     """

#     if features is None:
#         features = []

#     # GET ALL THE DATA:
#     stock_data = total(stock, years=years, interest='Interest' in features, wiki_views='Wiki_views' in features,
#                        indicators='RSI' and '%K' and '%R' in features, period=period)

#     # SPLIT DATA, CREATE THE MODEL, GENERATE AND CALCULATE THE ERROR:
#     result, y_predicted, df = run(stock_data, features, steps, training, error_method)

#     date = (dt.datetime.today() + dt.timedelta(days=1))
#     while date.weekday() == 5 or date.weekday() == 6:
#         date = date + dt.timedelta(days=1)
#     date = date.strftime('%Y-%m-%d')
#     result.append(date)

#     if not plot:
#         return result

#     if plot:
#         dates = df.index.tolist()
#         from pandas.plotting import register_matplotlib_converters
#         register_matplotlib_converters()
#         import matplotlib.pyplot as plt
#         plt.plot(dates, y_predicted)
#         plt.plot(dates, df.Close.tolist())
#         plt.title(stock + ' - %1.2f' % result[0] + ' - %1.3f' % result[1] + '% - ' + result[2])
#         plt.xlabel('Date')
#         plt.ylabel('Close price (USD)')
#         plt.legend(['Predicted', 'True'])
#         plt.gcf().autofmt_xdate()
#         plt.savefig('test.png')
#         plt.show()
        
#         return result


# stock = 'INTC'
# error1 = tomorrow(stock, years=1, plot=True)[1]

#Install the dependencies
# import numpy as np
# import pandas as pd
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# plt.style.use('bmh')

# from google.colab import files # Use to load data on Google Colab
# uploaded = files.upload() # Use to load data on Google Colab

# df = pd.read_csv('NFLX_Stock.csv')
# df.head(6)

import yfinance as yf
data = yf.download(tickers="MSFT", period="5d", interval="1m")

from datetime import datetime, timedelta
d = datetime.today() - timedelta(days=3)
newD = d.strftime("%Y-%m-%d %H:%M")+':00-04:00'

print(data.loc[[newD]].values[0][1])