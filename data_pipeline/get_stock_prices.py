import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import joblib
import os
from sklearn.preprocessing import StandardScaler

def download_stock_data(ticker, start_date, end_date):
    
    try:
        stock = yf.Ticker(ticker)
        

        df = stock.history(start=start_date, end=end_date)

        df = df.reset_index()
        

        
        return df
    
    except Exception as e:
        print(f"Error downloading data for {ticker}: {str(e)}")
        return None


def normalize_stock_data(data: pd.DataFrame, features=None, model_path="data_pipeline/scaler_model.pkl"):

    if features is None:
        features = ["Close"]

    # Ensure columns exist
    missing = [col for col in features if col not in data.columns]
    if missing:
        raise ValueError(f"Missing columns in input data: {missing}")

    if os.path.exists(model_path):
        print(f"✅ Loading existing scaler from '{model_path}'...")
        scaler = joblib.load(model_path)
    else:
        print("⚙️ No saved scaler found. Fitting a new one...")
        scaler = StandardScaler()
        scaler.fit(data[features])
        joblib.dump(scaler, model_path)
        print(f"💾 Scaler saved to '{model_path}'")

    scaled_values = scaler.transform(data[features])

    data["Scaled_price"] = scaled_values

    return data


def main():
    ticker = "OTP.BD"  
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')  

    df = download_stock_data(ticker, start_date, end_date)
    
    if df is not None:
        print("\nFirst few rows of the data:")
        print(df.head())

if __name__ == "__main__":
    main()