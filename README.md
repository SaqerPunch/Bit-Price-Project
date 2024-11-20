**Bitcoin Returns Prediction Using Random Forest Regression**

This project analyzes the relationship between Bitcoin returns and two macroeconomic indicators—Gold Price and 10-Year Treasury Yield—using a Random Forest Regression model. The aim is to predict Bitcoin returns based on these features, providing insights into how these economic factors may impact the cryptocurrency market.

**Overview**
In this analysis, I used the following features:

Gold Price: The price of gold, often seen as a safe-haven asset.
10-Year Treasury Yield: The interest rate on U.S. Treasury bonds, which reflects investor sentiment and economic conditions.
Bitcoin Returns: The percentage change in the Bitcoin price, used as the target variable for prediction.
I utilized a Random Forest Regression model to predict Bitcoin returns based on historical data of the two features. This project aims to explore how traditional financial indicators like gold and treasury yields correlate with Bitcoin's performance.

**Requirements**

Python 3.x
pandas: For data manipulation and handling CSV files.
numpy: For numerical operations.
scikit-learn: For implementing the Random Forest Regression model.
matplotlib & seaborn: For data visualization.
yfinance (optional): To fetch real-time data for gold prices and treasury yields.
You can install the necessary libraries with:

bash
Copy code
pip install pandas numpy scikit-learn matplotlib seaborn yfinance
Project Structure
bash
Copy code
/project
│
├── data/
│   ├── bitcoin_data.csv        # Historical Bitcoin returns data
│   ├── gold_data.csv           # Historical Gold prices data
│   ├── treasury_yield_data.csv # Historical 10-Year Treasury Yield data
│
├── analysis/
│   ├── preprocess_data.py      # Data preprocessing and cleaning script
│   ├── feature_engineering.py   # Feature selection and transformation
│   ├── train_model.py          # Model training and evaluation
│
└── results/
    ├── predictions.csv         # Predicted Bitcoin returns
    ├── evaluation_metrics.txt  # Model evaluation metrics and results

**Steps Involved**
Data Collection:

Historical Bitcoin prices, Gold prices, and 10-Year Treasury Yield data were gathered. The data was cleaned and processed to ensure consistency.
Feature Engineering:

I transformed the features into a form suitable for model training, such as calculating returns for Bitcoin and ensuring all data points were aligned in time.
Model Training:

A Random Forest Regressor was chosen for its ability to capture complex relationships in the data. Hyperparameters were tuned to optimize performance.
Evaluation:

The model was evaluated using metrics like R-Squared to determine the accuracy of the predictions.
