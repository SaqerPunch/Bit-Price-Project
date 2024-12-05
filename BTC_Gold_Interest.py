##Import needed libraries##
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sqlalchemy import create_engine

def main():
   
    
    tickers = ['BTC-USD', '^TNX', 'GLD']
    data = yf.download(tickers,start='2018-09-04', end='2024-09-04', group_by='ticker')

    btc_close = data[('BTC-USD','Close')]
    btc_close.name = 'BTC-Close'

    tenyr_close = data[('^TNX','Close')]
    tenyr_close.name = '10YR-Close'

    gold_close = data[('GLD','Close')]
    gold_close.name = 'Gold-Close'

    data = pd.concat([btc_close,tenyr_close,gold_close], axis=1)
    data = data.dropna()
    print(data.info())

    # PostgreSQL connection details
    user = 'Hidden for Privacy Purposes'
    password = 'Hidden for Privacy Purposes'
    host = 'localhost'  # or your host
    port = '5432'       # default port
    database = 'your_database'

    # Create the connection string
    connection_string = f'postgresql://{user}:{password}@{host}:{port}/{database}'

    # Create SQLAlchemy engine
    engine = create_engine(connection_string)

    # Upload data to PostgreSQL
    table_name = 'financial_data'

    # Uploading data to PostgreSQL
    data.to_sql(table_name, engine, if_exists='replace', index=True)

    #Derviving Correlation and Obtaining Heatmap
    correlation = data.corr()
    plt.figure(figsize = (8,8))
    sns.heatmap(correlation,cbar=True, square=True,fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Blues')
    plt.show()

    #Split Data
    data = data.reset_index()    
    X = data.drop(['Date','BTC-Close'], axis=1)
    Y = data['BTC-Close']

    print()
    print(X)
    print(Y)
    

    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=.2, random_state=2)

    ##Model Training##
    regressor = RandomForestRegressor(n_estimators=100)
    regressor.fit(X_train, Y_train)

    ##Test Data##
    test_prediction = regressor.predict(X_test)

    ##Compare test and Model Data##
    error = metrics.r2_score(Y_test,test_prediction)
    print(f"R-squared error: {error}")

    ##Visual Representation of Test and Predicition:
    
    Y_test = list(Y_test)
    plt.plot(Y_test, color ='blue', label = 'Actual Value')
    plt.plot(test_prediction, color = 'green', label = 'Predicted Value')
    plt.title('Actual v Predicted')
    plt.xlabel('Number of Values')
    plt.ylabel('BTC Price')
    plt.legend()
    plt.show()
  

if __name__ == "__main__":
    main()
