#Results can be found in .ipynb colab file

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import yfinance as yf
from datetime import datetime

# Load existing CSV
file_path = '/content/drive/MyDrive/CPS 4150 Milestone Project/stock.csv'
existing_data = pd.read_csv(file_path)

# Define the stock symbols
symbols = ['AAPL', 'BA', 'T', 'MGM', 'AMZN', 'IBM', 'TSLA', 'GOOG', '^GSPC']

# Determine the last date in the existing data
last_date_str = existing_data['Date'].iloc[-1]

# Strip any time zone information and convert to date
try:
    last_date = datetime.strptime(last_date_str, '%Y-%m-%d')
except ValueError:
    last_date = datetime.strptime(last_date_str.split()[0], '%Y-%m-%d')

# Get today's date
today = datetime.now().date()

# Check if data is already up-to-date
if last_date.date() >= today:
    print("The CSV file is already up-to-date.")
else:
    # Fetch new data from the day after the last date to today
    start_date = last_date + pd.Timedelta(days=1)
    end_date = today

    # Download new data
    new_data = yf.download(symbols, start=start_date, end=end_date)

    # Adjust the column names
    new_data = new_data['Adj Close'].reset_index()
    new_data.columns = ['Date', 'AAPL', 'BA', 'T', 'MGM', 'AMZN', 'IBM', 'TSLA', 'GOOG', 'sp500']

    # Concatenate the existing and new data
    updated_data = pd.concat([existing_data, new_data], ignore_index=True)

    # Save the updated data back to the CSV
    updated_data.to_csv(file_path, index=False)

    print(f"CSV file updated successfully up to {end_date.strftime('%Y-%m-%d')}.")

import pandas as pd

# Loads in the stocks dataset
file_path = '/content/drive/MyDrive/CPS 4150 Milestone Project/stock.csv'
stock_data = pd.read_csv(file_path)

# This prints the first 5 rows
print("First 5 rows of the dataset:")
print(stock_data.head())


#Checkinng for errors within the dataset

# This shows if any values are missing in the dataset
print("\nMissing values in the dataset:")
print(stock_data.isnull().sum())
# Check for duplicate rows
print("\nNumber of duplicate rows in the dataset:")
print(stock_data.duplicated().sum())


# Print the last 10 rows as you did before
print("\nLast 10 rows of the dataset:")
print(stock_data.tail(10))


#Milestone 3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

#Load the dataset
file_path = '/content/drive/MyDrive/CPS 4150 Milestone Project/stock.csv'
stock_data = pd.read_csv(file_path)

#Display basic info about the dataset
print("Dataset Info:")
print(stock_data.info())

#Handle missing values
#For numerical columns, fill missing values with the mean
numerical_columns = ['AAPL', 'BA', 'T', 'MGM', 'AMZN', 'IBM', 'TSLA', 'GOOG', 'sp500']
imputer = SimpleImputer(strategy='mean')
stock_data[numerical_columns] = imputer.fit_transform(stock_data[numerical_columns])

#Check for missing values after imputation
print("Missing Values After Cleaning:")
print(stock_data.isnull().sum())

#Handle Outliers using IQR method
for col in numerical_columns:
    q1 = stock_data[col].quantile(0.25)
    q3 = stock_data[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    stock_data = stock_data[(stock_data[col] >= lower_bound) & (stock_data[col] <= upper_bound)]

#Feature Engineering - Date processing
#Converting 'Date' to datetime, ensuring any format inconsistencies are handled
stock_data['Date'] = pd.to_datetime(stock_data['Date'], errors='coerce')

#Extracting Year, Month, Day from Date
stock_data['Year'] = stock_data['Date'].dt.year
stock_data['Month'] = stock_data['Date'].dt.month
stock_data['Day'] = stock_data['Date'].dt.day

#Remove the 'Date' column
stock_data.drop('Date', axis=1, inplace=True)

#Normalize numerical data (Scaling)
scaler = StandardScaler()
stock_data[numerical_columns] = scaler.fit_transform(stock_data[numerical_columns])

#Split the data into training, validation, and test sets
X = stock_data.drop('sp500', axis=1)
y = stock_data['sp500']

#Split into 80% train, 10% validation, 10% test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

#Impute NaN values after splitting
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_val = imputer.transform(X_val)
X_test = imputer.transform(X_test)

#Train models and evaluate performance
#List of regression models to train
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest Regressor': RandomForestRegressor(),
    'Support Vector Regressor': SVR()
}

#Dictionary to store performance metrics
model_results = {}

for model_name, model in models.items():
    #Train the model
    model.fit(X_train, y_train)

    #Predict on validation set
    y_pred = model.predict(X_val)

    #Calculate performance metrics
    mae = mean_absolute_error(y_val, y_pred)
    mse = mean_squared_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)

    #Store the results
    model_results[model_name] = {
        'MAE': mae,
        'MSE': mse,
        'R2': r2
    }

#Print Model Evaluation Results
print("\nModel Evaluation Results on Validation Set:")
for model_name, metrics in model_results.items():
    print(f"\n{model_name}:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

#Final Model Evaluation on Test Set
best_model = models['Random Forest Regressor']  #Choose the best model based on validation performance (e.g., Random Forest)
y_test_pred = best_model.predict(X_test)

#Final performance on test set
test_mae = mean_absolute_error(y_test, y_test_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("\nFinal Model Evaluation on Test Set (Random Forest Regressor):")
print(f"MAE: {test_mae:.4f}")
print(f"MSE: {test_mse:.4f}")
print(f"R2: {test_r2:.4f}")

#Milestone 4 Hyperparameter opitimaztion

from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

#Load the dataset
file_path = '/content/drive/MyDrive/CPS 4150 Milestone Project/stock.csv'
stock_data = pd.read_csv(file_path)

#Display basic info about the dataset
print("Dataset Info:")
print(stock_data.info())

#Handle missing values
#For numerical columns, fill missing values with the mean
numerical_columns = ['AAPL', 'BA', 'T', 'MGM', 'AMZN', 'IBM', 'TSLA', 'GOOG', 'sp500']
imputer = SimpleImputer(strategy='mean')
stock_data[numerical_columns] = imputer.fit_transform(stock_data[numerical_columns])

#Check for missing values after imputation
print("Missing Values After Cleaning:")
print(stock_data.isnull().sum())

#Handle Outliers using IQR method
for col in numerical_columns:
    q1 = stock_data[col].quantile(0.25)
    q3 = stock_data[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    stock_data = stock_data[(stock_data[col] >= lower_bound) & (stock_data[col] <= upper_bound)]

#Feature Engineering - Date processing
#Converting 'Date' to datetime, ensuring any format inconsistencies are handled
stock_data['Date'] = pd.to_datetime(stock_data['Date'], errors='coerce')

#Extracting Year, Month, Day from Date
stock_data['Year'] = stock_data['Date'].dt.year
stock_data['Month'] = stock_data['Date'].dt.month
stock_data['Day'] = stock_data['Date'].dt.day

#Remove the 'Date' column
stock_data.drop('Date', axis=1, inplace=True)

#Normalize numerical data (Scaling)
scaler = StandardScaler()
stock_data[numerical_columns] = scaler.fit_transform(stock_data[numerical_columns])

#Split the data into training, validation, and test sets
X = stock_data.drop('sp500', axis=1)
y = stock_data['sp500']

#Split into 80% train, 10% validation, 10% test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

#Impute NaN values after splitting
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_val = imputer.transform(X_val)
X_test = imputer.transform(X_test)

#Train models and evaluate performance
#List of regression models to train
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest Regressor': RandomForestRegressor(),
    'Support Vector Regressor': SVR()
}

#Dictionary to store performance metrics
model_results = {}

for model_name, model in models.items():
    #Train the model
    model.fit(X_train, y_train)

    #Predict on validation set
    y_pred = model.predict(X_val)

    #Calculate performance metrics
    mae = mean_absolute_error(y_val, y_pred)
    mse = mean_squared_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)

    #Store the results
    model_results[model_name] = {
        'MAE': mae,
        'MSE': mse,
        'R2': r2
    }

#Print Model Evaluation Results
print("\nModel Evaluation Results on Validation Set:")
for model_name, metrics in model_results.items():
    print(f"\n{model_name}:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

#Final Model Evaluation on Test Set
best_model = models['Random Forest Regressor']  #Choose the best model based on validation performance (e.g., Random Forest)
y_test_pred = best_model.predict(X_test)

#Final performance on test set
test_mae = mean_absolute_error(y_test, y_test_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("\nFinal Model Evaluation on Test Set (Random Forest Regressor):")
print(f"MAE: {test_mae:.4f}")
print(f"MSE: {test_mse:.4f}")
print(f"R2: {test_r2:.4f}")

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=5, scoring='r2')
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
best_rf_model = grid_search.best_estimator_


# Re-train the Random Forest model
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_rf_model = grid_search.best_estimator_

# Predict on test set
y_test_pred = best_rf_model.predict(X_test)

# Residual Analysis
residuals = y_test - y_test_pred
plt.scatter(y_test, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.title("Residual Analysis")
plt.xlabel("True Values")
plt.ylabel("Residuals")
plt.show()

# Feature Importance Analysis
importances = best_rf_model.feature_importances_
feature_names = X_train.columns

# Sort feature importances in descending order
sorted_indices = np.argsort(importances)[::-1]

# Plot Feature Importances
plt.figure(figsize=(10, 6))
plt.bar(range(len(importances)), importances[sorted_indices], align='center')
plt.xticks(range(len(importances)), feature_names[sorted_indices], rotation=90)
plt.title("Feature Importances")
plt.xlabel("Features")
plt.ylabel("Importance Score")
plt.show()

# Additional Metrics
explained_variance = explained_variance_score(y_test, y_test_pred)
mape = mean_absolute_percentage_error(y_test, y_test_pred)
print(f"Explained Variance Score: {explained_variance:.4f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.4%}")

# QQ Plot
import scipy.stats as stats
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("QQ Plot of Residuals")
plt.show()

# Cross-Validation Scores
cv_scores = cross_val_score(best_rf_model, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
print(f"Cross-Validation Scores (R2): {cv_scores}")
print(f"Mean Cross-Validation R2: {np.mean(cv_scores):.4f}")


# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

# Load the most recent dataset
file_path = '/content/drive/MyDrive/CPS 4150 Milestone Project/stock.csv'
stock_data = pd.read_csv(file_path)

# Extract the latest feature values
numerical_columns = ['AAPL', 'BA', 'T', 'MGM', 'AMZN', 'IBM', 'TSLA', 'GOOG']
latest_data = stock_data.iloc[-1]
latest_features = {col: latest_data[col] for col in numerical_columns}

# Prepare data for training models
X = stock_data.drop(columns=['Date', 'sp500'])  # Features
y = stock_data['sp500']  # Target variable

# Standardize the numerical data
scaler = StandardScaler()
X[numerical_columns] = scaler.fit_transform(X[numerical_columns])

# Define and train models
models = {
    'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=30, min_samples_split=5, min_samples_leaf=1),
    'Linear Regression': LinearRegression(),
    'Support Vector Regressor': SVR()
}

trained_models = {}
for name, model in models.items():
    model.fit(X, y)
    trained_models[name] = model

# Prediction Function
def predict_stock_price(year, month, day, other_features, model):
    # Prepare new data with year, month, and day included
    new_data = {
        'Year': year,
        'Month': month,
        'Day': day,
        **other_features
    }

    # Match features from training
    feature_names = [col for col in stock_data.columns if col not in ['Date', 'sp500']]
    input_data = {feature: new_data.get(feature, 0) for feature in feature_names}
    input_df = pd.DataFrame([input_data])

    # Scale features using the pre-existing scaler
    input_df[numerical_columns] = scaler.transform(input_df[numerical_columns])

    # Predict using the specified model
    prediction = model.predict(input_df)[0]
    return prediction

# Predict for March 1, 2025, using each model
predictions = {}
for model_name, model in trained_models.items():
    predictions[model_name] = predict_stock_price(2025, 3, 1, latest_features, model)

# Output predictions
print("Predicted Stock Prices for March 1, 2025:")
for model_name, predicted_price in predictions.items():
    print(f"{model_name}: {predicted_price:.2f}")

# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

# Load the most recent dataset
file_path = '/content/drive/MyDrive/CPS 4150 Milestone Project/stock.csv'
stock_data = pd.read_csv(file_path)

# Feature Engineering: Extract Year, Month, Day from Date
stock_data['Date'] = stock_data['Date'].str.split(' ').str[0]  # Removes time and timezone
stock_data['Date'] = pd.to_datetime(stock_data['Date'], errors='coerce')
stock_data['Year'] = stock_data['Date'].dt.year
stock_data['Month'] = stock_data['Date'].dt.month
stock_data['Day'] = stock_data['Date'].dt.day

# Prepare features (X) and target (y)
numerical_columns = ['AAPL', 'BA', 'T', 'MGM', 'AMZN', 'IBM', 'TSLA', 'GOOG']
X = stock_data.drop(columns=['Date', 'sp500'])  # Features include Year, Month, Day
y = stock_data['sp500']  # Target variable

# Standardize numerical columns
scaler = StandardScaler()
X[numerical_columns] = scaler.fit_transform(X[numerical_columns])

# Define and train models
models = {
    'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=30, min_samples_split=5, min_samples_leaf=1),
    'Linear Regression': LinearRegression(),
    'Support Vector Regressor': SVR()
}

trained_models = {}
for name, model in models.items():
    model.fit(X, y)
    trained_models[name] = model

# Define prediction function
def predict_stock_price(year, month, day, other_features, model):
    # Prepare new data with year, month, and day included
    new_data = {
        'Year': year,
        'Month': month,
        'Day': day,
        **other_features
    }

    # Match features from training
    feature_names = [col for col in X.columns]
    input_data = {feature: new_data.get(feature, 0) for feature in feature_names}
    input_df = pd.DataFrame([input_data])

    # Scale features using the pre-existing scaler
    input_df[numerical_columns] = scaler.transform(input_df[numerical_columns])

    # Predict using the specified model
    prediction = model.predict(input_df)[0]
    return prediction

# Extract the latest feature values
latest_data = stock_data.iloc[-1]
latest_features = {col: latest_data[col] for col in numerical_columns}

# Predict for two dates: December 15, 2025, and July 1, 2026
dates_to_predict = [(2025, 12, 15), (2026, 7, 1)]
predictions = {}

for year, month, day in dates_to_predict:
    predictions[(year, month, day)] = {}
    for model_name, model in trained_models.items():
        predictions[(year, month, day)][model_name] = predict_stock_price(year, month, day, latest_features, model)

# Output predictions for each date
for date, models_predictions in predictions.items():
    print(f"\nPredicted S&P 500 Stock Prices for {date[1]}/{date[2]}/{date[0]}:")
    for model_name, predicted_price in models_predictions.items():
        print(f"{model_name}: {predicted_price:.2f}")