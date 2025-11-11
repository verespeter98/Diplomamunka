import sqlite3
import pandas as pd
import joblib
import numpy as np
from datetime import date
import torch  
from pytorch_forecasting.data.encoders import NaNLabelEncoder
from pytorch_forecasting import TimeSeriesDataSet, NHiTS

SENTIMENT_LABEL_MAPPING = {0 : "positive", 1 : "negative", 2: "neutral"}

SCALER = joblib.load("data_pipeline/scaler_model.pkl")

def calculate_dates(start_date: str, end_date: str):

    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)

    start_exclusive = start_ts + pd.Timedelta(days=1)

    business_days = pd.bdate_range(start_exclusive, end_ts)

    return business_days

def load_max_date(db_path: str, stock_ticker: str = "OTP_BD"):
    try:
        conn = sqlite3.connect(db_path)

        stock_data = pd.read_sql(f"SELECT max(Date) FROM {stock_ticker}", conn)
        
            
        if stock_data.empty:
            print(f"Warning: No data found for {stock_ticker} in the specified date range.")
        
        return pd.to_datetime(stock_data.iloc[0, 0])
        
    except sqlite3.Error as e:
        print(f"SQLite error: {e} - Could not read data from {db_path}")
        return date.today()
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        return date.today()

    

def load_data_from_sqlite(db_path: str, start_date: date, end_date: date, stock_ticker: str = "OTP_BD") -> pd.DataFrame:

    print(f"Attempting to load data for {stock_ticker} from {start_date} to {end_date} from {db_path}")

    try:
        conn = sqlite3.connect(db_path)

        stock_data = pd.read_sql(f"SELECT * FROM {stock_ticker} WHERE date <= '{start_date}'", conn)

        stock_data['Sentiment_labels'] = stock_data['sentiment_score'].map(SENTIMENT_LABEL_MAPPING)

        if stock_data.empty:
            print(f"Warning: No data found for {stock_ticker} in the specified date range.")
        
        return stock_data
        
    except sqlite3.Error as e:
        print(f"SQLite error: {e} - Could not read data from {db_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        return pd.DataFrame()



def create_prediction(stock_data: pd.DataFrame) -> np.ndarray:

    def load_model(dataset):
        nhits_model = NHiTS.from_dataset(
                dataset,
                learning_rate=1e-3,  
                log_interval=10,  
                log_val_interval=1,  
                weight_decay=0.001,  
                n_blocks=[3, 3, 3],  
                n_layers=2,  
                hidden_size=512,  
                dropout=0.1,  
            )
        return nhits_model
    
    stock_data['time_idx'] = range(len(stock_data))  
    stock_data["series"] = 0 

    max_encoder_length = 60
    max_prediction_length = 30

    common_params = dict(
        time_idx="time_idx",
        target="Scaled_price",
        categorical_encoders={"series": NaNLabelEncoder(add_nan=True).fit(stock_data.series)},
        group_ids=["series"],
        time_varying_unknown_reals=["Scaled_price"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_unknown_categoricals = ["Sentiment_labels"]
    )

    last_known_price = stock_data["Close"].iloc[-1]

    data = TimeSeriesDataSet(  
        stock_data,  
        **common_params 
    ) 

    future_df = pd.DataFrame(
        np.nan,  # Use NaN as the placeholder
        index=np.arange(max_prediction_length),  # The number of future steps
        columns=stock_data.columns              # The same columns
    )

    future_df["Scaled_price"] = 0

    timeseries = pd.concat([stock_data[- max_encoder_length:], future_df], ignore_index=True)

    timeseries['Sentiment_labels'] = timeseries['Sentiment_labels'].ffill()
    timeseries["time_idx"] = range(len(timeseries))
    timeseries["series"] = 0

    dataset = TimeSeriesDataSet.from_dataset(data, timeseries, predict=True, stop_randomization=True)
    dataloader = dataset.to_dataloader(train=False, batch_size=32, num_workers=0, shuffle=False)  

    model = load_model(dataset)

    model.load_state_dict(torch.load('forecast_model_training/models/nhits_weights_sentiment.pth')) 

    pred = model.predict(dataloader)  
    
    pred_transformed = SCALER.inverse_transform(pred)

    return pred_transformed.flatten()

# # --- Example Usage (for testing this file directly) ---
# if __name__ == "__main__":
    
#     # --- 1. Test Data Loading ---
#     print("\n--- Testing Data Loading ---")
#     DB_FILE = "example_stocks.db"
    
#     # Create a dummy database for testing
#     try:
#         with sqlite3.connect(DB_FILE) as conn:
#             cursor = conn.cursor()
#             cursor.execute("DROP TABLE IF EXISTS stock_prices")
#             cursor.execute("""
#                 CREATE TABLE stock_prices (
#                     id INTEGER PRIMARY KEY,
#                     ticker TEXT NOT_NULL,
#                     date TEXT NOT_NULL,
#                     Open REAL,
#                     High REAL,
#                     Low REAL,
#                     Close REAL,
#                     Volume INTEGER
#                 )
#             """)
#             # Add some dummy data
#             dummy_data = [
#                 ('OTP', '2023-01-01', 100, 102, 99, 101, 5000),
#                 ('OTP', '2023-01-02', 101, 103, 100, 102, 6000),
#                 ('OTP', '2023-01-03', 102, 105, 102, 104, 7000),
#                 ('MSFT', '2023-01-01', 200, 202, 199, 201, 2000),
#             ]
#             cursor.executemany("INSERT INTO stock_prices (ticker, date, Open, High, Low, Close, Volume) VALUES (?, ?, ?, ?, ?, ?, ?)", dummy_data)
#             conn.commit()
#         print(f"Dummy database '{DB_FILE}' created.")
#     except Exception as e:
#         print(f"Dummy DB creation error: {e}")

#     # Test the loader
#     data = load_data_from_sqlite(
#         db_path=DB_FILE, 
#         stock_ticker="OTP", 
#         start_date=date(2023, 1, 1), 
#         end_date=date(2023, 1, 3)
#     )
#     print("Loaded data:")
#     print(data)

#     # --- 2. Test Model Loading & Prediction (Mocked) ---
#     print("\n--- Testing Model Loading & Prediction ---")
    
#     # Create a dummy model and scaler file for testing
#     from sklearn.linear_model import LinearRegression
#     from sklearn.preprocessing import StandardScaler
    
#     MODEL_FILE = "dummy_model.joblib"
#     SCALER_FILE = "dummy_scaler.joblib"

#     try:
#         # Create and save a dummy model
#         dummy_model = LinearRegression()
#         dummy_model.fit(np.array([[1], [2]]), np.array([1, 2])) # Train on simple data
#         joblib.dump(dummy_model, MODEL_FILE)
#         print(f"Dummy model saved to {MODEL_FILE}")

#         # Create and save a dummy scaler
#         dummy_scaler = StandardScaler()
#         dummy_scaler.fit(np.array([[100], [102], [104]])) # Fit on sample data
#         joblib.dump(dummy_scaler, SCALER_FILE)
#         print(f"Dummy scaler saved to {SCALER_FILE}")
#     except Exception as e:
#         print(f"Dummy model/scaler saving error: {e}")

#     # Load the model and scaler
#     model = load_prediction_model(MODEL_FILE)
#     scaler = load_prediction_model(SCALER_FILE)

#     if model and scaler and not data.empty:
#         # 3. Test Preprocessing
#         processed_data = preprocess_for_prediction(data, scaler, tokenizer=None)
#         print("Processed data (sample):", processed_data[:5])
        
#         # 4. Test Prediction
#         # Note: This prediction will be nonsense, as the model and data are dummies.
#         # It just tests the *pipeline*.
#         predictions = create_prediction(model, processed_data)
#         print("Generated predictions:", predictions)


#         test_set['Sentiment_labels'] = test_set['sentiment_score'].map(SENTIMENT_LABEL_MAPPING)

# common_params = dict(
#     time_idx="time_idx",
#     target="Scaled_price",
#     categorical_encoders={"series": NaNLabelEncoder(add_nan=True).fit(stock_data.series)},
#     group_ids=["series"],
#     time_varying_unknown_reals=["Scaled_price"],
#     max_encoder_length=max_encoder_length,
#     max_prediction_length=max_prediction_length,
#     time_varying_unknown_categoricals=["Sentiment_labels"]
# )

# test_dataset = TimeSeriesDataSet(
#     test_set,
#     **common_params
# )

# test_dataloader = test_dataset.to_dataloader(train=False, batch_size=32, num_workers=0, shuffle=False)