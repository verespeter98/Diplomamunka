import sqlite3
import pandas as pd
import joblib
import numpy as np
from datetime import date
import torch  
from pytorch_forecasting.data.encoders import NaNLabelEncoder
from pytorch_forecasting import TimeSeriesDataSet, NHiTS
import io
from src.model_data import model_weights

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
        np.nan, 
        index=np.arange(max_prediction_length), 
        columns=stock_data.columns             
    )

    future_df["Scaled_price"] = 0

    timeseries = pd.concat([stock_data[- max_encoder_length:], future_df], ignore_index=True)

    timeseries['Sentiment_labels'] = timeseries['Sentiment_labels'].ffill()
    timeseries["time_idx"] = range(len(timeseries))
    timeseries["series"] = 0

    dataset = TimeSeriesDataSet.from_dataset(data, timeseries, predict=True, stop_randomization=True)
    dataloader = dataset.to_dataloader(train=False, batch_size=32, num_workers=0, shuffle=False)  

    model = load_model(dataset)

    buffer = io.BytesIO(model_weights)

    model.load_state_dict(torch.load(buffer))

    pred = model.predict(dataloader)  
    
    pred_transformed = SCALER.inverse_transform(pred)

    return pred_transformed.flatten()