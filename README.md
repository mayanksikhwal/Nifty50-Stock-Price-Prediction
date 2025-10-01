# Nifty50-Stock-Price-Prediction

## Project Description
This project focuses on predicting **NIFTY 50 stock prices** using both **Machine Learning (ML)** and **Deep Learning (DL)** approaches. The workflow involves preparing time-series data, training multiple models with varying time windows (30–250 days), and comparing their performance using MAE and RMSE. The project explores various ML models (Linear, Tree-based, SVR, KNN, XGBoost, LightGBM) and DL models (RNN, LSTM, GRU, Bidirectional LSTM) to identify the most effective approach for this time-series forecasting task.

## Tech Stack
*   **Python:** The primary programming language used for the analysis and model building.
*   **pandas:** Used for data loading, manipulation, and analysis.
*   **NumPy:** Used for numerical operations and data handling.
*   **scikit-learn:** Used for data splitting, various ML models (Linear Regression, Ridge, Lasso, RandomForestRegressor, GradientBoostingRegressor, SVR, KNeighborsRegressor), and evaluation metrics (mean_absolute_error, mean_squared_error).
*   **xgboost:** Used for XGBoost Regressor model.
*   **lightgbm:** Used for LightGBM Regressor model.
*   **TensorFlow/Keras:** Used for building and training Deep Learning models (SimpleRNN, LSTM, GRU, Bidirectional LSTM).
*   **matplotlib:** Used for creating visualizations.
*   **tqdm:** Used for displaying progress bars during model training.
*   **joblib:** Used for saving and loading trained models.
*   **Google Colab:** The environment where the notebook was developed and executed.

## How to Run
1.  **Open the notebook in Google Colab:** Upload or open the notebook file (.ipynb) in your Google Colab environment.
2.  **Install dependencies:** Ensure you have the necessary libraries installed. You can install them using pip (refer to the `requirements.txt` file generated earlier).
3.  **Run all cells:** Execute all the code cells in the notebook sequentially from top to bottom. This will perform the data loading, data preparation, model definition, training of all models across different time windows, evaluation, and saving of results and trained models.
4.  **Explore Results:** The notebook includes sections for visualizing the performance of the top models and analyzing the impact of time windows and target columns.

## Screenshots

**1. Performance Metrics of Models sorted by test MAE**

<img width="619" height="346" alt="image" src="https://github.com/user-attachments/assets/ef9b7651-59ef-47b8-b2e1-95554e25e04d" />

**2. Visualization of Performance Metrics of Various Models**

<img width="737" height="344" alt="image" src="https://github.com/user-attachments/assets/e403afd4-8019-4524-be5a-ef1deca43a84" />
