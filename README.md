# ML-Stock-Predictor

### Overview
The project focused on predicting the S&P 500 index using machine learning models. This research was conducted with 2 colleagues from Kean Unversity. The dataset was sourced from Yahoo Finance, and various preprocessing, modeling, and evaluation techniques were applied to build an accurate predictive model.

### Methodology
1. Data Preprocessing:
   - Cleaning: Missing values were handled using mean imputation. Outliers were identified and removed using the IQR method.
   - Feature Engineering: The `Date` column was processed into `Year`, `Month`, and `Day`. Numerical features were scaled using `StandardScaler`.
   - Splitting: Data was divided into training (80%), validation (10%), and test (10%) sets.

2. Modeling:
   - Three models were trained and evaluated:
     - Linear Regression
     - Random Forest Regressor
     - Support Vector Regressor (SVR)
   - Hyperparameter tuning was performed for the Random Forest Regressor using `GridSearchCV`.

3. Evaluation Metrics:
   - Models were assessed using Mean Absolute Error (MAE), Mean Squared Error (MSE), and R² Score.
   - The Random Forest Regressor consistently outperformed other models, achieving:
     - Validation MAE: 0.0115
     - Validation MSE: 0.0003
     - Validation R²: 0.9997
     - Test MAE: 0.0131
     - Test MSE: 0.0004
     - Test R²: 0.9995

4. Residual and Feature Importance Analysis:
   - Residual analysis confirmed minimal bias in predictions.
   - Feature importance analysis highlighted `T` (AT&T) and `AAPL` (Apple) as the most significant predictors.

5. Future Predictions:
   - Using trained models, predictions were made for future dates:
     - March 1, 2025: Random Forest predicted an S&P 500 value of 6051.79.
     - Linear Regression and SVR provided alternative estimates, though Random Forest showed superior reliability.

Insights and Resul
- Best Model: Random Forest Regressor was identified as the most accurate and reliable model.
- Predicted Trends: The model forecasts a slight decline in the S&P 500 over time, influenced by dataset trends and features.
- Challenges: SVR struggled with feature scaling and data sensitivity, resulting in lower prediction accuracy.
- Final Evaluation: The model achieved high performance across all metrics, demonstrating its robustness for stock market analysis.

### Key Features
- Dynamic Predictions: Users can input future dates and generate S&P 500 predictions.
- Interactive Analysis: Predictions are available for comparison across models, enabling flexible validation.
